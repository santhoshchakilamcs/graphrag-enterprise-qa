"""GraphRAG question answering over the Northwind knowledge graph.

The question is parsed with rules (entity names matched against the
graph, intent from keywords), the answer is computed by 1-2 hop
traversal + aggregation over the graph, and the answer is extractive -
it comes from the graph itself.

Every answer is returned as (answer_string, evidence_lines).

The optional LLM layer lives in llm.py: it replaces parse_question
with an LLM router (with rule-based fallback) and composes the final
answer in natural language. This module stays rule-based, and
_dispatch is shared by both paths.
"""

from __future__ import annotations

import re
from collections import Counter

import networkx as nx

from .data import Indexes

UNKNOWN = "I don't know"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _top(counter: Counter, names: dict | None = None) -> tuple[str, int]:
    """Highest count, ties broken by alphabetically smallest name."""
    best_count = max(counter.values())
    winners = sorted(n for n, c in counter.items() if c == best_count)
    return winners[0], best_count


def parse_question(question: str, idx: Indexes) -> dict:
    """Rule-based parse: entity mentions, years, countries, intent."""
    q = question.lower()
    entities = idx.find_names(question)
    by_type: dict[str, list[str]] = {}
    for entity_type, name in entities:
        by_type.setdefault(entity_type, []).append(name)
    years = idx.find_years(question)
    countries = _find_countries(question, idx)

    intent = None
    if "average order" in q:
        intent = "average_order_value"
    elif "most revenue" in q:
        if "customer" in by_type or "which customer" in q:
            intent = "top_customer_revenue"
        else:
            intent = "top_product_revenue"
    elif "highest revenue" in q or "best-selling" in q or "top-selling" in q:
        intent = "top_product_revenue"
    elif "revenue" in q:
        if "category" in by_type:
            intent = "revenue_category"
        elif "product" in by_type:
            intent = "revenue_product"
        elif "supplier" in by_type:
            intent = "revenue_supplier"
        elif "customer" in by_type:
            intent = "revenue_customer"
        else:
            intent = "revenue_total"
    elif "freight" in q and "shipper" in by_type:
        intent = "freight_shipper"
    elif "most products" in q:
        if "which category" in q:
            intent = "category_most_products"
        elif "category" in by_type:
            intent = "supplier_most_products_in_category"
        else:
            intent = "supplier_most_products"
    elif "most orders" in q:
        if "customer" in by_type or "which customer" in q:
            intent = "top_customer_orders"
        else:
            intent = "employee_most_orders"
    elif "supplier of" in q or "who supplies" in q or ("which supplier" in q and "product" in by_type):
        intent = "supplier_of_product"
    elif "how many products" in q:
        if "discontinued" in q:
            intent = "count_discontinued"
        elif "supplier" in by_type:
            intent = "count_products_supplier"
    elif "how many orders" in q:
        if "customer" in by_type:
            intent = "count_orders_customer"
        elif "shipper" in by_type:
            intent = "count_orders_shipper"
        elif years:
            intent = "count_orders_year"
    elif "how many customers" in q:
        intent = "count_customers_country"
    elif "which products" in q and "category" in by_type:
        intent = "list_products_category"
    elif "discontinued" in q:
        intent = "count_discontinued"

    return {"intent": intent, "entities": by_type, "years": years, "countries": countries}


def _find_countries(text: str, idx: Indexes) -> list[str]:
    countries = set(idx.tables["customers"]["country"].dropna().unique())
    countries |= set(idx.tables["suppliers"]["country"].dropna().unique())
    lowered = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    found = []
    for country in sorted(countries, key=len, reverse=True):
        needle = " " + country.lower().strip() + " "
        if needle in lowered:
            found.append(str(country))
    return found


# --- graph traversal helpers -------------------------------------------------


def _in_edges(g: nx.MultiDiGraph, node, relation: str):
    return [(u, d) for u, _, d in g.in_edges(node, data=True) if d.get("relation") == relation]


def _out_edges(g: nx.MultiDiGraph, node, relation: str):
    return [(v, d) for _, v, d in g.out_edges(node, data=True) if d.get("relation") == relation]


def _products_in_category(g: nx.MultiDiGraph, idx: Indexes, category: str) -> list[tuple]:
    cat_id = idx.categories[category]
    return [u for u, _ in _in_edges(g, ("Category", cat_id), "IN_CATEGORY")]


def _suppliers_of_products(g: nx.MultiDiGraph, products: list[tuple]) -> list[tuple]:
    out = []
    for product in products:
        out.extend(v for v, _ in _out_edges(g, product, "SUPPLIED_BY"))
    return out


def _orders_of_employee(g: nx.MultiDiGraph, emp_id: int, year: int | None = None) -> list[tuple]:
    orders = [u for u, _ in _in_edges(g, ("Employee", emp_id), "MANAGED_BY")]
    if year is not None:
        orders = [o for o in orders if g.nodes[o].get("year") == year]
    return orders


def _orders_of_customer(g: nx.MultiDiGraph, cust_id: str) -> list[tuple]:
    return [v for v, _ in _out_edges(g, ("Customer", cust_id), "PLACED")]


def _orders_of_shipper(g: nx.MultiDiGraph, shipper_id: int) -> list[tuple]:
    return [u for u, _ in _in_edges(g, ("Shipper", shipper_id), "SHIPPED_BY")]


def _order_revenue(g: nx.MultiDiGraph, orders: list[tuple]) -> float:
    total_cents = 0
    for order in orders:
        for _, d in _out_edges(g, order, "CONTAINS"):
            total_cents += d.get("line_total_cents", 0)
    return total_cents / 100.0


def _product_revenue(g: nx.MultiDiGraph, product: tuple) -> float:
    return sum(d.get("line_total_cents", 0) for _, d in _in_edges(g, product, "CONTAINS")) / 100.0


def _name(g: nx.MultiDiGraph, node) -> str:
    return str(g.nodes[node].get("name", node))


# --- answer functions (graph traversal only) --------------------------------


def _dispatch(parsed: dict, g: nx.MultiDiGraph, idx: Indexes) -> tuple[str, list[str]]:
    """Run the handler for a parsed question (rule-based or LLM-routed)."""
    handler = _HANDLERS.get(parsed["intent"])
    if handler is None:
        return UNKNOWN, []
    try:
        return handler(g, idx, parsed["entities"], parsed["years"], parsed["countries"])
    except (KeyError, ValueError, IndexError):
        return UNKNOWN, []


def answer_question(question: str, g: nx.MultiDiGraph, idx: Indexes) -> tuple[str, list[str]]:
    return _dispatch(parse_question(question, idx), g, idx)


def _a_supplier_most_products_in_category(g, idx, entities, years, countries):
    category = entities["category"][0]
    products = _products_in_category(g, idx, category)
    counts = Counter(_name(g, s) for s in _suppliers_of_products(g, products))
    name, count = _top(counts)
    evidence = [
        f"Category '{category}' has {len(products)} products.",
        "Supplier product counts in this category: "
        + ", ".join(f"{n} ({c})" for n, c in counts.most_common(3)),
    ]
    return name, evidence


def _a_supplier_most_products(g, idx, entities, years, countries):
    counts = Counter()
    for node, data in g.nodes(data=True):
        if data["type"] == "Product":
            for supplier, _ in _out_edges(g, node, "SUPPLIED_BY"):
                counts[_name(g, supplier)] += 1
    name, count = _top(counts)
    return name, [f"Supplier product counts (top 3): {', '.join(f'{n} ({c})' for n, c in counts.most_common(3))}"]


def _a_category_most_products(g, idx, entities, years, countries):
    counts = Counter()
    for node, data in g.nodes(data=True):
        if data["type"] == "Product":
            for category, _ in _out_edges(g, node, "IN_CATEGORY"):
                counts[_name(g, category)] += 1
    name, count = _top(counts)
    return name, [f"Product counts per category (top 3): {', '.join(f'{n} ({c})' for n, c in counts.most_common(3))}"]


def _a_employee_most_orders(g, idx, entities, years, countries):
    year = years[0] if years else None
    counts = Counter()
    for emp_id in idx.employees.values():
        orders = _orders_of_employee(g, emp_id, year)
        counts[_name(g, ("Employee", emp_id))] = len(orders)
    name, count = _top(counts)
    scope = f" in {year}" if year else ""
    return name, [f"Orders handled per employee{scope} (top 3): {', '.join(f'{n} ({c})' for n, c in counts.most_common(3))}"]


def _a_top_customer_orders(g, idx, entities, years, countries):
    counts = Counter()
    for cust_id in idx.customers.values():
        counts[_name(g, ("Customer", cust_id))] = len(_orders_of_customer(g, cust_id))
    name, count = _top(counts)
    return name, [f"Orders placed per customer (top 3): {', '.join(f'{n} ({c})' for n, c in counts.most_common(3))}"]


def _a_top_customer_revenue(g, idx, entities, years, countries):
    revenues = Counter()
    for cust_id in idx.customers.values():
        revenues[_name(g, ("Customer", cust_id))] = _order_revenue(g, _orders_of_customer(g, cust_id))
    name = max(sorted(revenues), key=lambda n: revenues[n])
    return name, [f"Revenue per customer (top 3): {', '.join(f'{n} ({_money(revenues[n])})' for n in sorted(revenues, key=lambda n: -revenues[n])[:3])}"]


def _a_top_product_revenue(g, idx, entities, years, countries):
    revenues = Counter()
    for node, data in g.nodes(data=True):
        if data["type"] == "Product":
            revenues[_name(g, node)] = _product_revenue(g, node)
    name = max(sorted(revenues), key=lambda n: revenues[n])
    return name, [f"Revenue per product (top 3): {', '.join(f'{n} ({_money(revenues[n])})' for n in sorted(revenues, key=lambda n: -revenues[n])[:3])}"]


def _a_revenue_category(g, idx, entities, years, countries):
    category = entities["category"][0]
    products = _products_in_category(g, idx, category)
    product_set = set(products)
    total_cents = 0
    for node, data in g.nodes(data=True):
        if data["type"] != "Order":
            continue
        for product, d in _out_edges(g, node, "CONTAINS"):
            if product in product_set:
                total_cents += d.get("line_total_cents", 0)
    return _money(total_cents / 100.0), [f"Summed line totals (unit price x quantity x (1 - discount)) over {len(products)} products in '{category}' across all orders."]


def _a_revenue_product(g, idx, entities, years, countries):
    product = entities["product"][0]
    node = ("Product", idx.products[product])
    return _money(_product_revenue(g, node)), [f"Summed line totals for product '{product}' across all orders."]


def _a_revenue_supplier(g, idx, entities, years, countries):
    supplier = entities["supplier"][0]
    node = ("Supplier", idx.suppliers[supplier])
    products = [u for u, _ in _in_edges(g, node, "SUPPLIED_BY")]
    total = sum(_product_revenue(g, p) for p in products)
    return _money(total), [f"Summed line totals over {len(products)} products supplied by '{supplier}'."]


def _a_revenue_customer(g, idx, entities, years, countries):
    customer = entities["customer"][0]
    orders = _orders_of_customer(g, idx.customers[customer])
    return _money(_order_revenue(g, orders)), [f"Summed line totals over {len(orders)} orders placed by '{customer}'."]


def _a_revenue_total(g, idx, entities, years, countries):
    total_cents = 0
    for node, data in g.nodes(data=True):
        if data["type"] == "Order":
            for _, d in _out_edges(g, node, "CONTAINS"):
                total_cents += d.get("line_total_cents", 0)
    return _money(total_cents / 100.0), ["Summed line totals across all orders."]


def _a_freight_shipper(g, idx, entities, years, countries):
    shipper = entities["shipper"][0]
    orders = _orders_of_shipper(g, idx.shippers[shipper])
    total = sum(float(g.nodes[o].get("freight", 0.0)) for o in orders)
    return _money(total), [f"Summed freight over {len(orders)} orders shipped by '{shipper}'."]


def _a_supplier_of_product(g, idx, entities, years, countries):
    product = entities["product"][0]
    node = ("Product", idx.products[product])
    suppliers = [v for v, _ in _out_edges(g, node, "SUPPLIED_BY")]
    return _name(g, suppliers[0]), [f"Product '{product}' --SUPPLIED_BY--> '{_name(g, suppliers[0])}'."]


def _a_count_products_supplier(g, idx, entities, years, countries):
    supplier = entities["supplier"][0]
    node = ("Supplier", idx.suppliers[supplier])
    products = [u for u, _ in _in_edges(g, node, "SUPPLIED_BY")]
    return str(len(products)), [f"Supplier '{supplier}' --SUPPLIED_BY-- {len(products)} products."]


def _a_count_orders_customer(g, idx, entities, years, countries):
    customer = entities["customer"][0]
    orders = _orders_of_customer(g, idx.customers[customer])
    return str(len(orders)), [f"Customer '{customer}' --PLACED--> {len(orders)} orders."]


def _a_count_orders_shipper(g, idx, entities, years, countries):
    shipper = entities["shipper"][0]
    orders = _orders_of_shipper(g, idx.shippers[shipper])
    return str(len(orders)), [f"Shipper '{shipper}' --SHIPPED_BY-- {len(orders)} orders."]


def _a_count_orders_year(g, idx, entities, years, countries):
    year = years[0]
    count = sum(1 for _, d in g.nodes(data=True) if d["type"] == "Order" and d.get("year") == year)
    return str(count), [f"{count} Order nodes have year == {year}."]


def _a_count_customers_country(g, idx, entities, years, countries):
    country = countries[0]
    count = sum(1 for _, d in g.nodes(data=True) if d["type"] == "Customer" and d.get("country") == country)
    return str(count), [f"{count} Customer nodes have country == '{country}'."]


def _a_list_products_category(g, idx, entities, years, countries):
    category = entities["category"][0]
    products = sorted(_name(g, p) for p in _products_in_category(g, idx, category))
    return ", ".join(products), [f"{len(products)} products --IN_CATEGORY--> '{category}'."]


def _a_average_order_value(g, idx, entities, years, countries):
    orders = [n for n, d in g.nodes(data=True) if d["type"] == "Order"]
    total = _order_revenue(g, orders)
    return _money(total / len(orders)), [f"Total revenue {_money(total)} over {len(orders)} orders."]


def _a_count_discontinued(g, idx, entities, years, countries):
    count = sum(1 for _, d in g.nodes(data=True) if d["type"] == "Product" and d.get("discontinued"))
    return str(count), [f"{count} Product nodes have discontinued == True."]


_HANDLERS = {
    "supplier_most_products_in_category": _a_supplier_most_products_in_category,
    "supplier_most_products": _a_supplier_most_products,
    "category_most_products": _a_category_most_products,
    "employee_most_orders": _a_employee_most_orders,
    "top_customer_orders": _a_top_customer_orders,
    "top_customer_revenue": _a_top_customer_revenue,
    "top_product_revenue": _a_top_product_revenue,
    "revenue_category": _a_revenue_category,
    "revenue_product": _a_revenue_product,
    "revenue_supplier": _a_revenue_supplier,
    "revenue_customer": _a_revenue_customer,
    "revenue_total": _a_revenue_total,
    "freight_shipper": _a_freight_shipper,
    "supplier_of_product": _a_supplier_of_product,
    "count_products_supplier": _a_count_products_supplier,
    "count_orders_customer": _a_count_orders_customer,
    "count_orders_shipper": _a_count_orders_shipper,
    "count_orders_year": _a_count_orders_year,
    "count_customers_country": _a_count_customers_country,
    "list_products_category": _a_list_products_category,
    "average_order_value": _a_average_order_value,
    "count_discontinued": _a_count_discontinued,
}
