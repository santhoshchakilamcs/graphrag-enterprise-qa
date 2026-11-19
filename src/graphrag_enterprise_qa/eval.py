"""Evaluation: ~20 business questions, ground truth from pandas.

Ground truth is computed from the CSVs with pandas at run time - it is
never hard-coded and never read from the QA systems. Both systems see
only the raw question text. Ties are broken the same way everywhere:
highest value, then alphabetically smallest name.
"""

from __future__ import annotations

import json
import os
import re

from .baseline import Baseline
from .data import load_data
from .graph_build import build_graph
from .qa import answer_question

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")


def load_questions(path: str | None = None) -> list[dict]:
    with open(path or QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _argmax_tiebreak(series) -> str:
    """Index label with the max value; ties -> alphabetically first."""
    best = series.max()
    winners = sorted(series[series == best].index.tolist())
    return winners[0]


def ground_truth(intent: str, params: dict, tables) -> str:
    """Compute the answer from the CSVs with pandas."""
    orders = tables["orders"]
    details = tables["order_details"]
    products = tables["products"]
    customers = tables["customers"]
    suppliers = tables["suppliers"]
    categories = tables["categories"]
    employees = tables["employees"]

    # Money in integer cents: float sums drift by a cent with summation
    # order, and the eval compares money to the cent.
    details = details.assign(
        line_total_cents=(
            details["unitPrice"] * details["quantity"] * (1.0 - details["discount"]) * 100
        ).round().astype(int)
    )

    if intent == "supplier_most_products_in_category":
        cat_id = int(categories.loc[categories["categoryName"] == params["category"], "categoryID"].iloc[0])
        counts = products[products["categoryID"] == cat_id].groupby("supplierID").size()
        counts.index = [str(suppliers.loc[suppliers["supplierID"] == s, "companyName"].iloc[0]) for s in counts.index]
        return _argmax_tiebreak(counts)

    if intent == "employee_most_orders":
        year = params["year"]
        counts = orders[orders["year"] == year].groupby("employeeID").size()
        counts.index = [
            f"{employees.loc[employees['employeeID'] == e, 'firstName'].iloc[0]} "
            f"{employees.loc[employees['employeeID'] == e, 'lastName'].iloc[0]}"
            for e in counts.index
        ]
        return _argmax_tiebreak(counts)

    if intent == "revenue_category":
        cat_id = int(categories.loc[categories["categoryName"] == params["category"], "categoryID"].iloc[0])
        merged = details.merge(products[["productID", "categoryID"]], on="productID")
        return _money(float(merged.loc[merged["categoryID"] == cat_id, "line_total_cents"].sum()) / 100.0)

    if intent == "top_customer_orders":
        counts = orders.groupby("customerID").size()
        counts.index = [str(customers.loc[customers["customerID"] == c, "companyName"].iloc[0]) for c in counts.index]
        return _argmax_tiebreak(counts)

    if intent == "top_customer_revenue":
        merged = details.merge(orders[["orderID", "customerID"]], on="orderID")
        revenue = merged.groupby("customerID")["line_total_cents"].sum() / 100.0
        revenue.index = [str(customers.loc[customers["customerID"] == c, "companyName"].iloc[0]) for c in revenue.index]
        return _argmax_tiebreak(revenue)

    if intent == "top_product_revenue":
        revenue = details.groupby("productID")["line_total_cents"].sum() / 100.0
        revenue.index = [str(products.loc[products["productID"] == p, "productName"].iloc[0]) for p in revenue.index]
        return _argmax_tiebreak(revenue)

    if intent == "category_most_products":
        counts = products.groupby("categoryID").size()
        counts.index = [str(categories.loc[categories["categoryID"] == c, "categoryName"].iloc[0]) for c in counts.index]
        return _argmax_tiebreak(counts)

    if intent == "supplier_of_product":
        supplier_id = int(products.loc[products["productName"] == params["product"], "supplierID"].iloc[0])
        return str(suppliers.loc[suppliers["supplierID"] == supplier_id, "companyName"].iloc[0])

    if intent == "count_products_supplier":
        supplier_id = int(suppliers.loc[suppliers["companyName"] == params["supplier"], "supplierID"].iloc[0])
        return str(int((products["supplierID"] == supplier_id).sum()))

    if intent == "count_orders_customer":
        cust_id = str(customers.loc[customers["companyName"] == params["customer"], "customerID"].iloc[0])
        return str(int((orders["customerID"] == cust_id).sum()))

    if intent == "count_orders_shipper":
        shipper_id = int(tables["shippers"].loc[tables["shippers"]["companyName"] == params["shipper"], "shipperID"].iloc[0])
        return str(int((orders["shipVia"] == shipper_id).sum()))

    if intent == "freight_shipper":
        shipper_id = int(tables["shippers"].loc[tables["shippers"]["companyName"] == params["shipper"], "shipperID"].iloc[0])
        return _money(float(orders.loc[orders["shipVia"] == shipper_id, "freight"].sum()))

    if intent == "list_products_category":
        cat_id = int(categories.loc[categories["categoryName"] == params["category"], "categoryID"].iloc[0])
        names = sorted(products.loc[products["categoryID"] == cat_id, "productName"].tolist())
        return ", ".join(names)

    if intent == "count_customers_country":
        return str(int((customers["country"] == params["country"]).sum()))

    if intent == "count_orders_year":
        return str(int((orders["year"] == params["year"]).sum()))

    if intent == "average_order_value":
        return _money(float(details["line_total_cents"].sum()) / 100.0 / len(orders))

    if intent == "count_discontinued":
        return str(int((products["discontinued"] == 1).sum()))

    raise ValueError(f"unknown intent: {intent}")


def answers_match(got: str, expected: str) -> bool:
    """Compare two answers: money within $0.01, ints exact, lists as sets."""
    got_n = " ".join(str(got).strip().split())
    exp_n = " ".join(str(expected).strip().split())
    if got_n.lower() == exp_n.lower():
        return True
    try:
        got_money = float(got_n.replace("$", "").replace(",", ""))
        exp_money = float(exp_n.replace("$", "").replace(",", ""))
        # Money is compared to the cent: 1-cent float noise is not an error.
        return round(got_money, 2) == round(exp_money, 2)
    except ValueError:
        pass
    if ", " in exp_n:
        return sorted(s.strip().lower() for s in got_n.split(",")) == sorted(
            s.strip().lower() for s in exp_n.split(",")
        )
    return False


def run_eval(data_dir: str | None = None, llm=None, embedding: bool = False) -> dict:
    """Run the systems over all questions and score them.

    Always runs rule-based GraphRAG and the TF-IDF baseline. With
    embedding=True it also runs the dense embedding baseline; with
    llm set to a loaded LLMRouter it also runs LLM-mode GraphRAG
    (LLM routing + LLM-composed answers, scored on the extractive
    graph value). The extra systems add their own keys; the
    rule-based keys are always present and unchanged.
    """
    tables = load_data(data_dir)
    graph, idx = build_graph(data_dir)
    baseline = Baseline(data_dir)
    emb = None
    if embedding:
        from .embedding_baseline import EmbeddingBaseline

        emb = EmbeddingBaseline(data_dir)
    llm_row_fn = None
    if llm is not None:
        from .llm import answer_question_llm

        llm_row_fn = answer_question_llm
    questions = load_questions()

    results = []
    for item in questions:
        question = item["question"]
        expected = ground_truth(item["intent"], item["params"], tables)
        graph_answer, graph_evidence = answer_question(question, graph, idx)
        base_answer, base_evidence = baseline.answer_question(question, idx)
        row = {
            "id": item["id"],
            "question": question,
            "intent": item["intent"],
            "ground_truth": expected,
            "graphrag_answer": graph_answer,
            "graphrag_correct": answers_match(graph_answer, expected),
            "graphrag_evidence": graph_evidence,
            "baseline_answer": base_answer,
            "baseline_correct": answers_match(base_answer, expected),
            "baseline_evidence": base_evidence,
        }
        if emb is not None:
            emb_answer, emb_evidence = emb.answer_question(question, idx)
            row["embedding_answer"] = emb_answer
            row["embedding_correct"] = answers_match(emb_answer, expected)
            row["embedding_evidence"] = emb_evidence
        if llm_row_fn is not None:
            llm_row = llm_row_fn(question, graph, idx, llm)
            row["llm_answer"] = llm_row["answer"]
            row["llm_correct"] = answers_match(llm_row["answer"], expected)
            row["llm_parse_source"] = llm_row["parse_source"]
            row["llm_composed"] = llm_row["composed"]
            row["llm_composed_contains_answer"] = llm_row["composed_contains_answer"]
            row["llm_evidence"] = llm_row["evidence"]
        results.append(row)

    total = len(results)
    graph_correct = sum(r["graphrag_correct"] for r in results)
    base_correct = sum(r["baseline_correct"] for r in results)
    summary = {
        "n_questions": total,
        "graphrag_correct": graph_correct,
        "graphrag_accuracy": graph_correct / total if total else 0.0,
        "baseline_correct": base_correct,
        "baseline_accuracy": base_correct / total if total else 0.0,
        "results": results,
    }
    if emb is not None:
        emb_correct = sum(r["embedding_correct"] for r in results)
        summary["embedding_baseline_correct"] = emb_correct
        summary["embedding_baseline_accuracy"] = emb_correct / total if total else 0.0
        summary["embedding_model"] = emb.model_name
    if llm_row_fn is not None:
        llm_correct = sum(r["llm_correct"] for r in results)
        summary["llm_correct"] = llm_correct
        summary["llm_accuracy"] = llm_correct / total if total else 0.0
        summary["llm_model"] = llm.model_name
        summary["llm_routing"] = {
            "parsed_by_llm": sum(1 for r in results if r["llm_parse_source"] == "llm"),
            "fallback_to_rules": sum(1 for r in results if r["llm_parse_source"] == "fallback"),
        }
        summary["llm_composed_contains_answer"] = sum(
            1 for r in results if r["llm_composed_contains_answer"]
        )
    return summary
