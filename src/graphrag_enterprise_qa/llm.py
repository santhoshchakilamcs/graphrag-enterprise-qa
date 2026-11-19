"""Optional LLM layer for the GraphRAG system (Qwen2.5-0.5B-Instruct).

Rule-based parsing is the default and always works with no model
download. LLM mode is opt-in (``--llm`` on run_eval.py, or the
``GRAPHRAG_USE_LLM`` env var) and does two things:

  1. Route: the LLM reads the question and returns JSON with an intent
     and entity names, replacing the rule-based parse_question. The
     output is validated against the known intents and the graph's
     entity names; anything invalid falls back to the rules.
  2. Compose: the LLM writes the final answer as one natural-language
     sentence from the extractive answer + evidence lines. The scored
     answer is always the extractive value from the graph, never the
     generated sentence - so the LLM cannot inflate the score by
     guessing, and a composed sentence that drops the value is
     reported as such.

The model downloads at runtime into the Hugging Face cache
(~/.cache/huggingface); nothing is stored in the repo. The import of
transformers/torch is lazy (inside load_llm) so the rest of the
package works without them installed.
"""

from __future__ import annotations

import json
import os
import re

from .data import Indexes
from .qa import UNKNOWN, _dispatch, parse_question

LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_FLAG = "GRAPHRAG_USE_LLM"

INTENT_DESCRIPTIONS = {
    "supplier_most_products_in_category": "which supplier supplies the most products in a given category",
    "supplier_most_products": "which supplier supplies the most products overall",
    "category_most_products": "which category has the most products",
    "employee_most_orders": "which employee handled the most orders, optionally in a given year",
    "top_customer_orders": "which customer placed the most orders",
    "top_customer_revenue": "which customer generated the most revenue",
    "top_product_revenue": "which product generated the most revenue",
    "revenue_category": "total revenue from a category",
    "revenue_product": "total revenue from a product",
    "revenue_supplier": "total revenue from a supplier's products",
    "revenue_customer": "total revenue from a customer",
    "revenue_total": "total revenue overall",
    "freight_shipper": "total freight charged by a shipper",
    "supplier_of_product": "who is the supplier of a product",
    "count_products_supplier": "how many products a supplier supplies",
    "count_orders_customer": "how many orders a customer placed",
    "count_orders_shipper": "how many orders a shipper shipped",
    "count_orders_year": "how many orders were placed in a year",
    "count_customers_country": "how many customers are based in a country",
    "list_products_category": "list the products in a category",
    "average_order_value": "the average order value",
    "count_discontinued": "how many products are discontinued",
}

INTENTS = list(INTENT_DESCRIPTIONS)


def llm_available() -> bool:
    """True if transformers + torch import (no model download)."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _route_prompt(question: str, idx: Indexes) -> str:
    entities = idx.find_names(question)
    by_type: dict[str, list[str]] = {}
    for entity_type, name in entities:
        by_type.setdefault(entity_type, []).append(name)
    candidates = "; ".join(
        f"{t}: {', '.join(names)}" for t, names in sorted(by_type.items())
    ) or "none"
    intents = "\n".join(f"- {name}: {desc}" for name, desc in INTENT_DESCRIPTIONS.items())
    return (
        "You route business questions to one intent and the entity names they mention.\n"
        "Reply with JSON only, no other text.\n\n"
        f"Valid intents:\n{intents}\n\n"
        f"Entity names mentioned in the question ({candidates or 'none'}). "
        "Use these exact names; do not invent names.\n\n"
        "Entity keys must be exactly one of: category, supplier, customer, "
        "product, employee, shipper. A year is not an entity - do not put "
        "years in entities.\n\n"
        "Format: {\"intent\": \"<one of the valid intents>\", "
        "\"entities\": {\"category\": [...], \"supplier\": [...], \"customer\": [...], "
        "\"product\": [...], \"employee\": [...], \"shipper\": [...]}} "
        "Include only entity types that appear, with empty lists omitted.\n\n"
        "Example:\n"
        "Question: How many orders did the customer Around the Horn place?\n"
        "{\"intent\": \"count_orders_customer\", \"entities\": {\"customer\": [\"Around the Horn\"]}}\n\n"
        f"Question: {question}\n"
        "JSON:"
    )


def _compose_prompt(question: str, answer: str, evidence: list[str]) -> str:
    evidence_text = "\n".join(f"- {line}" for line in evidence) or "- (no evidence)"
    return (
        "Answer the business question in one natural sentence, using "
        "exactly the answer and evidence below. Do not add facts.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Evidence:\n{evidence_text}\n\n"
        "One sentence:"
    )


def validate_parse(raw: dict, idx: Indexes) -> dict | None:
    """Validate an LLM-produced parse against the known intents/names.

    Returns a normalised parse dict (intent/entities/years/countries)
    or None if anything is invalid. Years and countries are taken from
    the rule-based extractors - they are unambiguous regex matches,
    so the LLM only routes intent + entities.
    """
    if not isinstance(raw, dict):
        return None
    intent = raw.get("intent")
    if intent not in INTENT_DESCRIPTIONS:
        return None
    entities_in = raw.get("entities")
    if not isinstance(entities_in, dict):
        return None
    lookup = {
        "category": idx.categories,
        "supplier": idx.suppliers,
        "customer": idx.customers,
        "product": idx.products,
        "employee": idx.employees,
        "shipper": idx.shippers,
    }
    entities: dict[str, list[str]] = {}
    for entity_type, names in entities_in.items():
        if entity_type not in lookup:
            return None
        # The model sometimes returns a bare string instead of a list.
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list):
            return None
        for name in names:
            if name not in lookup[entity_type]:
                return None
        if names:
            entities[entity_type] = list(names)
    return {"intent": intent, "entities": entities, "years": [], "countries": []}


class LLMRouter:
    """Qwen2.5-0.5B-Instruct on CPU, greedy decoding (deterministic)."""

    def __init__(self, model_name: str = LLM_MODEL, max_new_tokens: int = 160):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(0)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        )
        self.model.eval()

    def _generate(self, system: str, user: str, max_new_tokens: int) -> str | None:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def parse(self, question: str, idx: Indexes) -> dict | None:
        """LLM routing: intent + entities, or None (caller falls back)."""
        try:
            raw = self._generate(
                "You are a precise question router. Reply with JSON only.",
                _route_prompt(question, idx),
                max_new_tokens=160,
            )
        except Exception:
            return None
        if not raw:
            return None
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        validated = validate_parse(parsed, idx)
        if validated is None:
            return None
        # Years / countries are deterministic regex matches; fill them in.
        rule = parse_question(question, idx)
        validated["years"] = rule["years"]
        validated["countries"] = rule["countries"]
        return validated

    def compose(self, question: str, answer: str, evidence: list[str]) -> str | None:
        """One natural-language sentence from the extractive answer."""
        try:
            text = self._generate(
                "You write short, factual answers from given evidence.",
                _compose_prompt(question, answer, evidence),
                max_new_tokens=120,
            )
        except Exception:
            return None
        if not text:
            return None
        # Keep the first sentence only; the model sometimes continues.
        first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0].strip()
        return first or None


def load_llm(model_name: str = LLM_MODEL) -> LLMRouter | None:
    """Load the router, or None (with a printed reason) on any failure."""
    if not llm_available():
        print("LLM unavailable: transformers/torch not installed (pip install -r requirements.txt).")
        return None
    try:
        router = LLMRouter(model_name)
    except Exception as e:  # noqa: BLE001 - download / load can fail many ways
        print(f"LLM unavailable: could not load {model_name}: {e}")
        return None
    print(f"LLM loaded: {model_name}")
    return router


def _contains_answer(composed: str, answer: str) -> bool:
    """Does the composed sentence actually state the extractive answer?

    Single values: the value must appear as a whole word/phrase
    (money compared without $/commas). Lists: at least half the items
    must appear as whole words/phrases. Word-boundary matching, so a
    single letter inside another word does not count.
    """
    text = composed.lower().replace("$", "").replace(",", "")

    def _has(phrase: str) -> bool:
        phrase = phrase.strip().lower().replace("$", "").replace(",", "")
        if not phrase:
            return False
        return (
            re.search(r"(?<![0-9a-z])" + re.escape(phrase) + r"(?![0-9a-z])", text)
            is not None
        )

    if ", " in answer:
        items = [item.strip() for item in answer.split(",")]
        found = sum(1 for item in items if _has(item))
        return found * 2 >= len(items)
    return _has(answer)


def answer_question_llm(question: str, g, idx: Indexes, llm: LLMRouter | None) -> dict:
    """Answer via the LLM router (with rule-based fallback) + LLM compose.

    Returns a dict with the extractive answer (the scored value), the
    composed sentence, the evidence, and whether the LLM or the rules
    did the routing. With llm=None this is exactly the rule-based path.
    """
    parse_source = "fallback"
    parsed = None
    if llm is not None:
        parsed = llm.parse(question, idx)
        if parsed is not None:
            parse_source = "llm"
    if parsed is None:
        parsed = parse_question(question, idx)
    answer, evidence = _dispatch(parsed, g, idx)
    composed = None
    composed_contains = False
    if llm is not None and answer != UNKNOWN and evidence:
        composed = llm.compose(question, answer, evidence)
        if composed:
            composed_contains = _contains_answer(composed, answer)
    return {
        "answer": answer,
        "composed": composed,
        "evidence": evidence,
        "parse_source": parse_source,
        "intent": parsed["intent"],
        "composed_contains_answer": composed_contains,
    }


def llm_enabled() -> bool:
    """Opt-in flag: GRAPHRAG_USE_LLM=1 enables LLM mode."""
    return os.environ.get(ENV_FLAG, "").strip() in ("1", "true", "yes")
