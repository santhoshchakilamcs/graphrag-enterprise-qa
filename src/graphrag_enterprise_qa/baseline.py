"""TF-IDF retrieval baseline over flattened Northwind row text.

The baseline is deliberately simple and honest: every CSV row becomes
one text chunk, the question retrieves the top-k chunks by TF-IDF
cosine similarity, and the answer is extracted from those chunks only
(by counting mentions or summing numbers found in them). It never sees
the graph, so it cannot aggregate over the full dataset - that is the
point of the comparison.

It shares the question parser with the GraphRAG system (same
entity/intent understanding) so the comparison isolates retrieval
vs graph traversal.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import Indexes, load_data
from .qa import _money, _top, parse_question

TOP_K = 50

_LINE_TOTAL_RE = re.compile(
    r"unit price (\d+(?:\.\d+)?), quantity (\d+), discount (\d+(?:\.\d+)?)"
)
_FREIGHT_RE = re.compile(r"freight (\d+(?:\.\d+)?)")
_ORDER_ID_RE = re.compile(r"order (\d+),")


def _is_order_chunk(chunk: str) -> bool:
    """True for an Order row chunk ("Order 10248: ..."), not an order-detail chunk."""
    return re.match(r"Order \d+:", chunk) is not None


def build_chunks(data_dir: str | None = None) -> list[str]:
    """One denormalised text chunk per CSV row."""
    tables = load_data(data_dir)
    idx = Indexes(tables)
    chunks: list[str] = []

    categories = {int(r.categoryID): str(r.categoryName) for r in tables["categories"].itertuples()}
    suppliers = {int(r.supplierID): str(r.companyName) for r in tables["suppliers"].itertuples()}
    customers = {str(r.customerID): str(r.companyName) for r in tables["customers"].itertuples()}
    employees = {
        int(r.employeeID): f"{r.firstName} {r.lastName}" for r in tables["employees"].itertuples()
    }
    shippers = {int(r.shipperID): str(r.companyName) for r in tables["shippers"].itertuples()}
    products = {int(r.productID): str(r.productName) for r in tables["products"].itertuples()}

    for row in tables["categories"].itertuples():
        chunks.append(f"Category {row.categoryName}: {row.description}")
    for row in tables["customers"].itertuples():
        chunks.append(
            f"Customer {row.companyName} ({row.customerID}): contact {row.contactName}, "
            f"{row.contactTitle}, city {row.city}, country {row.country}."
        )
    for row in tables["suppliers"].itertuples():
        chunks.append(
            f"Supplier {row.companyName}: contact {row.contactName}, "
            f"city {row.city}, country {row.country}."
        )
    for row in tables["employees"].itertuples():
        chunks.append(
            f"Employee {row.firstName} {row.lastName}: title {row.title}, "
            f"city {row.city}, country {row.country}."
        )
    for row in tables["shippers"].itertuples():
        chunks.append(f"Shipper {row.companyName}: phone {row.phone}.")
    for row in tables["products"].itertuples():
        disc = "yes" if int(row.discontinued) else "no"
        chunks.append(
            f"Product {row.productName}: supplier {suppliers[int(row.supplierID)]}, "
            f"category {categories[int(row.categoryID)]}, unit price {float(row.unitPrice):.2f}, "
            f"units in stock {int(row.unitsInStock)}, discontinued {disc}."
        )
    for row in tables["orders"].itertuples():
        chunks.append(
            f"Order {int(row.orderID)}: customer {customers[str(row.customerID)]}, "
            f"employee {employees[int(row.employeeID)]}, order date {str(row.orderDate)[:10]}, "
            f"shipper {shippers[int(row.shipVia)]}, freight {float(row.freight):.2f}."
        )
    products_by_id = tables["products"].set_index("productID")
    orders_by_id = tables["orders"].set_index("orderID")
    for row in tables["order_details"].itertuples():
        product_id = int(row.productID)
        product_name = products[product_id]
        product_row = products_by_id.loc[product_id]
        chunks.append(
            f"Order detail: order {int(row.orderID)}, "
            f"customer {customers[str(orders_by_id.loc[int(row.orderID), 'customerID'])]}, "
            f"product {product_name}, category {categories[int(product_row['categoryID'])]}, "
            f"supplier {suppliers[int(product_row['supplierID'])]}, "
            f"unit price {float(row.unitPrice):.2f}, quantity {int(row.quantity)}, "
            f"discount {float(row.discount)}."
        )
    return chunks


class Baseline:
    """TF-IDF retriever + extraction over the retrieved chunks."""

    def __init__(self, data_dir: str | None = None, top_k: int = TOP_K):
        self.chunks = build_chunks(data_dir)
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, question: str) -> list[str]:
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).flatten()
        order = np.argsort(-scores, kind="stable")[: self.top_k]
        return [self.chunks[i] for i in order if scores[i] > 0]

    def answer_question(self, question: str, idx: Indexes) -> tuple[str, list[str]]:
        parsed = parse_question(question, idx)
        intent = parsed["intent"]
        entities = parsed["entities"]
        years = parsed["years"]
        countries = parsed["countries"]
        chunks = self.retrieve(question)
        handler = _HANDLERS.get(intent)
        if handler is None or not chunks:
            return "I don't know", []
        try:
            return handler(chunks, idx, entities, years, countries)
        except (KeyError, ValueError, IndexError):
            return "I don't know", []


def _mentions(chunks: list[str], name: str) -> int:
    pattern = re.compile(r"\b" + re.escape(name.lower()) + r"\b")
    return sum(1 for c in chunks if pattern.search(c.lower()))


def _most_mentioned(chunks: list[str], candidates: list[str]) -> str:
    counts = Counter({name: _mentions(chunks, name) for name in candidates})
    counts = Counter({n: c for n, c in counts.items() if c > 0})
    if not counts:
        return "I don't know"
    return _top(counts)[0]


def _sum_line_totals(chunks: list[str]) -> float:
    # Money in integer cents, same as the graph and the ground truth.
    total_cents = 0
    for chunk in chunks:
        for price, qty, disc in _LINE_TOTAL_RE.findall(chunk):
            total_cents += int(round(float(price) * int(qty) * (1.0 - float(disc)) * 100))
    return total_cents / 100.0


def _line_totals_by(chunks: list[str], key_re: str) -> Counter:
    totals: Counter = Counter()
    for chunk in chunks:
        match = re.search(key_re, chunk)
        if not match:
            continue
        for price, qty, disc in _LINE_TOTAL_RE.findall(chunk):
            totals[match.group(1)] += int(round(float(price) * int(qty) * (1.0 - float(disc)) * 100))
    return Counter({k: v / 100.0 for k, v in totals.items()})


def _b_supplier_most_products_in_category(chunks, idx, entities, years, countries):
    category = entities["category"][0]
    scoped = [c for c in chunks if category.lower() in c.lower() and c.startswith("Product ")]
    name = _most_mentioned(scoped, list(idx.suppliers))
    return name, [f"Retrieved {len(scoped)} product chunks mentioning '{category}'."]


def _b_supplier_most_products(chunks, idx, entities, years, countries):
    scoped = [c for c in chunks if c.startswith("Product ")]
    return _most_mentioned(scoped, list(idx.suppliers)), [f"Retrieved {len(scoped)} product chunks."]


def _b_category_most_products(chunks, idx, entities, years, countries):
    scoped = [c for c in chunks if c.startswith("Product ")]
    return _most_mentioned(scoped, list(idx.categories)), [f"Retrieved {len(scoped)} product chunks."]


def _b_employee_most_orders(chunks, idx, entities, years, countries):
    scoped = [c for c in chunks if _is_order_chunk(c)]
    if years:
        scoped = [c for c in scoped if str(years[0]) in c]
    return _most_mentioned(scoped, list(idx.employees)), [f"Retrieved {len(scoped)} order chunks."]


def _b_top_customer_orders(chunks, idx, entities, years, countries):
    scoped = [c for c in chunks if _is_order_chunk(c)]
    return _most_mentioned(scoped, list(idx.customers)), [f"Retrieved {len(scoped)} order chunks."]


def _b_top_customer_revenue(chunks, idx, entities, years, countries):
    details = [c for c in chunks if c.startswith("Order detail:")]
    totals = _line_totals_by(details, r"customer ([^,]+), product")
    if not totals:
        return "I don't know", []
    name = max(sorted(totals), key=lambda n: totals[n])
    return name, [f"Revenue per customer over {len(details)} retrieved order-detail chunks (top 3): "
                  + ", ".join(f"{n} ({_money(totals[n])})" for n in sorted(totals, key=lambda n: -totals[n])[:3])]


def _b_top_product_revenue(chunks, idx, entities, years, countries):
    details = [c for c in chunks if c.startswith("Order detail:")]
    totals = _line_totals_by(details, r"product ([^,]+), category")
    if not totals:
        return "I don't know", []
    name = max(sorted(totals), key=lambda n: totals[n])
    return name, [f"Revenue per product over {len(details)} retrieved order-detail chunks (top 3): "
                  + ", ".join(f"{n} ({_money(totals[n])})" for n in sorted(totals, key=lambda n: -totals[n])[:3])]


def _b_revenue_category(chunks, idx, entities, years, countries):
    category = entities["category"][0]
    scoped = [c for c in chunks if c.startswith("Order detail:") and category.lower() in c.lower()]
    return _money(_sum_line_totals(scoped)), [f"Summed line totals over {len(scoped)} retrieved order-detail chunks mentioning '{category}'."]


def _b_revenue_product(chunks, idx, entities, years, countries):
    product = entities["product"][0]
    scoped = [c for c in chunks if c.startswith("Order detail:") and product.lower() in c.lower()]
    return _money(_sum_line_totals(scoped)), [f"Summed line totals over {len(scoped)} retrieved order-detail chunks mentioning '{product}'."]


def _b_revenue_supplier(chunks, idx, entities, years, countries):
    supplier = entities["supplier"][0]
    scoped = [c for c in chunks if c.startswith("Order detail:") and supplier.lower() in c.lower()]
    return _money(_sum_line_totals(scoped)), [f"Summed line totals over {len(scoped)} retrieved order-detail chunks mentioning '{supplier}'."]


def _b_revenue_customer(chunks, idx, entities, years, countries):
    customer = entities["customer"][0]
    scoped = [c for c in chunks if c.startswith("Order detail:") and customer.lower() in c.lower()]
    return _money(_sum_line_totals(scoped)), [f"Summed line totals over {len(scoped)} retrieved order-detail chunks mentioning '{customer}'."]


def _b_revenue_total(chunks, idx, entities, years, countries):
    details = [c for c in chunks if c.startswith("Order detail:")]
    return _money(_sum_line_totals(details)), [f"Summed line totals over {len(details)} retrieved order-detail chunks."]


def _b_freight_shipper(chunks, idx, entities, years, countries):
    shipper = entities["shipper"][0]
    scoped = [c for c in chunks if _is_order_chunk(c) and shipper.lower() in c.lower()]
    total = sum(float(m) for c in scoped for m in _FREIGHT_RE.findall(c))
    return _money(total), [f"Summed freight over {len(scoped)} retrieved order chunks mentioning '{shipper}'."]


def _b_supplier_of_product(chunks, idx, entities, years, countries):
    product = entities["product"][0]
    scoped = [c for c in chunks if c.startswith("Product ") and product.lower() in c.lower()]
    if not scoped:
        return "I don't know", []
    match = re.search(r"supplier ([^,]+), category", scoped[0])
    if not match:
        return "I don't know", []
    return match.group(1), [f"Top retrieved chunk: {scoped[0]}"]


def _b_count_products_supplier(chunks, idx, entities, years, countries):
    supplier = entities["supplier"][0]
    scoped = [c for c in chunks if c.startswith("Product ") and supplier.lower() in c.lower()]
    return str(len(scoped)), [f"Retrieved {len(scoped)} product chunks mentioning '{supplier}'."]


def _b_count_orders_customer(chunks, idx, entities, years, countries):
    customer = entities["customer"][0]
    scoped = [c for c in chunks if _is_order_chunk(c) and customer.lower() in c.lower()]
    return str(len(scoped)), [f"Retrieved {len(scoped)} order chunks mentioning '{customer}'."]


def _b_count_orders_shipper(chunks, idx, entities, years, countries):
    shipper = entities["shipper"][0]
    scoped = [c for c in chunks if _is_order_chunk(c) and shipper.lower() in c.lower()]
    return str(len(scoped)), [f"Retrieved {len(scoped)} order chunks mentioning '{shipper}'."]


def _b_count_orders_year(chunks, idx, entities, years, countries):
    year = years[0]
    scoped = [c for c in chunks if _is_order_chunk(c) and f"order date {year}" in c]
    return str(len(scoped)), [f"Retrieved {len(scoped)} order chunks dated {year}."]


def _b_count_customers_country(chunks, idx, entities, years, countries):
    country = countries[0]
    scoped = [c for c in chunks if c.startswith("Customer ") and country.lower() in c.lower()]
    return str(len(scoped)), [f"Retrieved {len(scoped)} customer chunks mentioning '{country}'."]


def _b_list_products_category(chunks, idx, entities, years, countries):
    category = entities["category"][0]
    scoped = [c for c in chunks if c.startswith("Product ") and category.lower() in c.lower()]
    names = sorted({re.match(r"Product ([^:]+):", c).group(1) for c in scoped})
    return ", ".join(names), [f"Retrieved {len(scoped)} product chunks mentioning '{category}'."]


def _b_average_order_value(chunks, idx, entities, years, countries):
    details = [c for c in chunks if c.startswith("Order detail:")]
    order_ids = {_ORDER_ID_RE.search(c).group(1) for c in details if _ORDER_ID_RE.search(c)}
    if not order_ids:
        return "I don't know", []
    total = _sum_line_totals(details)
    return _money(total / len(order_ids)), [f"Mean line-total sum over {len(order_ids)} distinct orders in {len(details)} retrieved order-detail chunks."]


def _b_count_discontinued(chunks, idx, entities, years, countries):
    scoped = [c for c in chunks if c.startswith("Product ") and "discontinued yes" in c]
    return str(len(scoped)), [f"Retrieved {len(scoped)} product chunks marked discontinued."]


_HANDLERS = {
    "supplier_most_products_in_category": _b_supplier_most_products_in_category,
    "supplier_most_products": _b_supplier_most_products,
    "category_most_products": _b_category_most_products,
    "employee_most_orders": _b_employee_most_orders,
    "top_customer_orders": _b_top_customer_orders,
    "top_customer_revenue": _b_top_customer_revenue,
    "top_product_revenue": _b_top_product_revenue,
    "revenue_category": _b_revenue_category,
    "revenue_product": _b_revenue_product,
    "revenue_supplier": _b_revenue_supplier,
    "revenue_customer": _b_revenue_customer,
    "revenue_total": _b_revenue_total,
    "freight_shipper": _b_freight_shipper,
    "supplier_of_product": _b_supplier_of_product,
    "count_products_supplier": _b_count_products_supplier,
    "count_orders_customer": _b_count_orders_customer,
    "count_orders_shipper": _b_count_orders_shipper,
    "count_orders_year": _b_count_orders_year,
    "count_customers_country": _b_count_customers_country,
    "list_products_category": _b_list_products_category,
    "average_order_value": _b_average_order_value,
    "count_discontinued": _b_count_discontinued,
}
