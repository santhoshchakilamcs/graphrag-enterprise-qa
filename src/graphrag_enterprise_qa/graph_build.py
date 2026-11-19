"""Knowledge-graph builder: Northwind CSVs -> NetworkX MultiDiGraph.

Node ids are (entity_type, id) tuples. Edge labels are the relation
names; traversal uses them to walk 1-2 hops from a matched entity.
"""

from __future__ import annotations

import networkx as nx

from .data import Indexes, load_data

NODE_TYPES = [
    "Customer",
    "Product",
    "Supplier",
    "Employee",
    "Category",
    "Order",
    "Shipper",
]

EDGE_TYPES = ["PLACED", "CONTAINS", "SUPPLIED_BY", "MANAGED_BY", "IN_CATEGORY", "SHIPPED_BY"]


def build_graph(data_dir: str | None = None) -> tuple[nx.MultiDiGraph, Indexes]:
    """Build the graph from the CSVs and return (graph, indexes)."""
    tables = load_data(data_dir)
    idx = Indexes(tables)
    g = nx.MultiDiGraph()

    for row in tables["categories"].itertuples():
        g.add_node(("Category", int(row.categoryID)), type="Category", name=str(row.categoryName))
    for row in tables["customers"].itertuples():
        g.add_node(
            ("Customer", str(row.customerID)),
            type="Customer",
            name=str(row.companyName),
            country=str(row.country),
            city=str(row.city),
        )
    for row in tables["employees"].itertuples():
        g.add_node(
            ("Employee", int(row.employeeID)),
            type="Employee",
            name=f"{row.firstName} {row.lastName}",
            title=str(row.title),
        )
    for row in tables["shippers"].itertuples():
        g.add_node(("Shipper", int(row.shipperID)), type="Shipper", name=str(row.companyName))
    for row in tables["suppliers"].itertuples():
        g.add_node(
            ("Supplier", int(row.supplierID)),
            type="Supplier",
            name=str(row.companyName),
            country=str(row.country),
        )
    for row in tables["products"].itertuples():
        g.add_node(
            ("Product", int(row.productID)),
            type="Product",
            name=str(row.productName),
            unit_price=float(row.unitPrice),
            discontinued=bool(int(row.discontinued)),
        )
        g.add_edge(
            ("Product", int(row.productID)),
            ("Supplier", int(row.supplierID)),
            relation="SUPPLIED_BY",
        )
        g.add_edge(
            ("Product", int(row.productID)),
            ("Category", int(row.categoryID)),
            relation="IN_CATEGORY",
        )

    for row in tables["orders"].itertuples():
        g.add_node(
            ("Order", int(row.orderID)),
            type="Order",
            name=f"Order {int(row.orderID)}",
            year=int(row.year),
            freight=float(row.freight),
        )
        g.add_edge(("Customer", str(row.customerID)), ("Order", int(row.orderID)), relation="PLACED")
        g.add_edge(("Order", int(row.orderID)), ("Employee", int(row.employeeID)), relation="MANAGED_BY")
        g.add_edge(("Order", int(row.orderID)), ("Shipper", int(row.shipVia)), relation="SHIPPED_BY")

    for row in tables["order_details"].itertuples():
        order_id = int(row.orderID)
        product_id = int(row.productID)
        # Money is kept in integer cents: float sums drift by a cent
        # depending on summation order, and the eval compares to the cent.
        line_total_cents = int(
            round(float(row.unitPrice) * int(row.quantity) * (1.0 - float(row.discount)) * 100)
        )
        g.add_edge(
            ("Order", order_id),
            ("Product", product_id),
            relation="CONTAINS",
            quantity=int(row.quantity),
            unit_price=float(row.unitPrice),
            discount=float(row.discount),
            line_total_cents=line_total_cents,
        )

    return g, idx


def graph_stats(g: nx.MultiDiGraph) -> dict:
    """Node/edge counts by type, for the README and the tests."""
    nodes = {t: sum(1 for _, d in g.nodes(data=True) if d["type"] == t) for t in NODE_TYPES}
    edges = {r: sum(1 for _, _, d in g.edges(data=True) if d["relation"] == r) for r in EDGE_TYPES}
    return {"nodes": nodes, "edges": edges, "total_nodes": g.number_of_nodes(), "total_edges": g.number_of_edges()}
