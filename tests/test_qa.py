"""Tests for the GraphRAG question answering."""

from graphrag_enterprise_qa.data import load_data
from graphrag_enterprise_qa.eval import ground_truth, load_questions
from graphrag_enterprise_qa.graph_build import build_graph
from graphrag_enterprise_qa.qa import UNKNOWN, answer_question, parse_question

EXPECTED_INTENTS = {
    "q01": "supplier_most_products_in_category",
    "q02": "supplier_most_products_in_category",
    "q03": "employee_most_orders",
    "q04": "employee_most_orders",
    "q05": "revenue_category",
    "q06": "revenue_category",
    "q07": "top_customer_orders",
    "q08": "top_customer_revenue",
    "q09": "top_product_revenue",
    "q10": "category_most_products",
    "q11": "supplier_of_product",
    "q12": "count_products_supplier",
    "q13": "count_orders_customer",
    "q14": "count_orders_shipper",
    "q15": "freight_shipper",
    "q16": "list_products_category",
    "q17": "count_customers_country",
    "q18": "count_orders_year",
    "q19": "average_order_value",
    "q20": "count_discontinued",
}


def test_parse_intents():
    _, idx = build_graph()
    for item in load_questions():
        parsed = parse_question(item["question"], idx)
        assert parsed["intent"] == EXPECTED_INTENTS[item["id"]], item["id"]


def test_answers_match_pandas_ground_truth():
    tables = load_data()
    graph, idx = build_graph()
    for item in load_questions():
        expected = ground_truth(item["intent"], item["params"], tables)
        answer, evidence = answer_question(item["question"], graph, idx)
        assert answer == expected, (item["id"], answer, expected)
        assert evidence, item["id"]


def test_unknown_question():
    graph, idx = build_graph()
    answer, _ = answer_question("What is the weather in Austin today?", graph, idx)
    assert answer == UNKNOWN


def test_money_answers_are_cent_exact():
    graph, idx = build_graph()
    answer, _ = answer_question("What is the total revenue from the Beverages category?", graph, idx)
    assert answer == "$267,868.18"
    answer, _ = answer_question("What is the average order value?", graph, idx)
    assert answer == "$1,525.05"
