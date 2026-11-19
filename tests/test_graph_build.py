"""Tests for the knowledge-graph builder."""

from graphrag_enterprise_qa.graph_build import build_graph, graph_stats


def test_node_counts():
    graph, _ = build_graph()
    stats = graph_stats(graph)
    assert stats["nodes"] == {
        "Customer": 91,
        "Product": 77,
        "Supplier": 29,
        "Employee": 9,
        "Category": 8,
        "Order": 830,
        "Shipper": 3,
    }
    assert stats["total_nodes"] == 1047


def test_edge_counts():
    graph, _ = build_graph()
    stats = graph_stats(graph)
    assert stats["edges"] == {
        "PLACED": 830,
        "CONTAINS": 2155,
        "SUPPLIED_BY": 77,
        "MANAGED_BY": 830,
        "IN_CATEGORY": 77,
        "SHIPPED_BY": 830,
    }
    assert stats["total_edges"] == 4799


def test_beverages_products():
    graph, idx = build_graph()
    cat_id = idx.categories["Beverages"]
    products = [
        u for u, _, d in graph.in_edges(("Category", cat_id), data=True)
        if d["relation"] == "IN_CATEGORY"
    ]
    assert len(products) == 12
    names = {graph.nodes[p]["name"] for p in products}
    assert "Chai" in names and "Steeleye Stout" in names


def test_chai_supplier_edge():
    graph, idx = build_graph()
    product = ("Product", idx.products["Chai"])
    suppliers = [
        v for _, v, d in graph.out_edges(product, data=True)
        if d["relation"] == "SUPPLIED_BY"
    ]
    assert len(suppliers) == 1
    assert graph.nodes[suppliers[0]]["name"] == "Exotic Liquids"


def test_order_edges_have_line_totals_in_cents():
    graph, _ = build_graph()
    contains = [
        d for _, _, d in graph.edges(data=True) if d["relation"] == "CONTAINS"
    ]
    assert len(contains) == 2155
    assert all(isinstance(d["line_total_cents"], int) and d["line_total_cents"] >= 0 for d in contains)
    # Order 10248, product 11 (Queso Cabrales): 14.00 x 12 x (1 - 0) = $168.00
    order_10248 = [
        d for u, v, d in graph.edges(data=True)
        if d["relation"] == "CONTAINS" and u == ("Order", 10248) and v == ("Product", 11)
    ]
    assert len(order_10248) == 1
    assert order_10248[0]["line_total_cents"] == 16800
