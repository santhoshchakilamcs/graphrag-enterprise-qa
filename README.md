# GraphRAG Enterprise QA

A personal project by Santhosh Chakilam (data scientist). GraphRAG
question answering over ERP-style business data: build a knowledge
graph from sales tables, answer business questions by traversing the
graph, and compare against a plain TF-IDF retrieval baseline.

**No LLM is used anywhere in this project.** Questions are parsed with
rules, answers are computed by graph traversal and aggregation, and
every answer is extractive — it comes from the graph. The README says
so plainly because that is the point: this demo shows what the graph
adds over retrieval alone, not what a language model can generate.

## Data

Northwind Traders sample data — the classic ERP-style dataset:
customers, orders, order details, products, suppliers, employees,
categories, shippers. The CSVs live in `data/` (220 KB total) and come
from the public Northwind CSV mirror in the
[graphql-compose examples repo](https://github.com/graphql-compose/graphql-compose-examples/tree/master/examples/northwind/data/csv).

**Northwind is a sample dataset, NOT real SAP data.** Nothing here
touches SAP systems, S/4HANA, or SuccessFactors, and no claim is made
about SAP data. It is used because its schema looks like ERP data
(orders, products, suppliers, employees) and it is small enough to
run anywhere.

## What is built

**Knowledge graph** (`src/graphrag_enterprise_qa/graph_build.py`):
a NetworkX `MultiDiGraph` with 1,047 nodes and 4,799 edges.

| Node type | Count | Edge type | Count |
|---|---|---|---|
| Customer | 91 | PLACED (Customer → Order) | 830 |
| Product | 77 | CONTAINS (Order → Product) | 2,155 |
| Supplier | 29 | SUPPLIED_BY (Product → Supplier) | 77 |
| Employee | 9 | MANAGED_BY (Order → Employee) | 830 |
| Category | 8 | IN_CATEGORY (Product → Category) | 77 |
| Order | 830 | SHIPPED_BY (Order → Shipper) | 830 |
| Shipper | 3 | | |

Order lines carry `line_total_cents` (integer cents — money is never
summed in floats, so totals are exact to the cent).

**GraphRAG QA** (`src/graphrag_enterprise_qa/qa.py`): rule-based
entity extraction (names matched against the graph), intent detection
from keywords, then 1–2 hop traversal and aggregation. Example:
"Which supplier provides the most products in the Beverages category?"
→ find the Beverages category node → its 12 products via
`IN_CATEGORY` → their suppliers via `SUPPLIED_BY` → count → max
(Bigfoot Breweries, 3). Answers come with evidence lines describing
the traversal.

**Baseline** (`src/graphrag_enterprise_qa/baseline.py`): every CSV row
flattened to one text chunk (3,202 chunks), TF-IDF retrieval
(top-50), and the answer extracted from those chunks only — by
counting mentions or summing numbers found in them. It shares the
question parser with GraphRAG, so the comparison isolates retrieval
vs graph traversal. It never sees the graph, so it cannot aggregate
over the full dataset.

**Evaluation** (`src/graphrag_enterprise_qa/eval.py`): 20 business
questions in `src/graphrag_enterprise_qa/questions.json`. Ground truth
is computed from the CSVs with pandas at run time — never hard-coded,
never read from either system. Money is compared to the cent, counts
exactly, lists as sets.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_eval.py          # runs the eval, prints the table, writes results.json
python verify_results.py    # re-runs the eval and checks results.json + README
python -m pytest tests/ -q  # 18 tests
```

Try a single question:

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from graphrag_enterprise_qa.graph_build import build_graph
from graphrag_enterprise_qa.qa import answer_question
g, idx = build_graph()
print(answer_question('Which employee handled the most orders in 1997?', g, idx)[0])
"
```

## Results

Measured by `python run_eval.py` on 20 questions. Every number below
comes from a run of the code — see `results.json` for the full
per-question output, and `verify_results.py` to reproduce it.

- GraphRAG: 20/20 = 100.0%
- Baseline (TF-IDF): 5/20 = 25.0%

| # | Question | Ground truth | GraphRAG | Baseline |
|---|---|---|---|---|
| q01 | Which supplier provides the most products in the Beverages category? | Bigfoot Breweries | ✓ | ✗ (no answer) |
| q02 | Which supplier provides the most products in the Seafood category? | Svensk Sjöföda AB | ✓ | ✗ New England Seafood Cannery |
| q03 | Which employee handled the most orders in 1997? | Margaret Peacock | ✓ | ✗ Janet Leverling |
| q04 | Which employee handled the most orders in 1996? | Margaret Peacock | ✓ | ✓ |
| q05 | What is the total revenue from the Beverages category? | $267,868.18 | ✓ | ✗ $14,499.39 |
| q06 | What is the total revenue from the Dairy Products category? | $234,507.29 | ✓ | ✗ $14,086.87 |
| q07 | Which customer placed the most orders? | Save-a-lot Markets | ✓ | ✗ (no answer) |
| q08 | Which customer generated the most revenue? | QUICK-Stop | ✓ | ✓ |
| q09 | Which product generated the highest revenue? | Côte de Blaye | ✓ | ✗ Konbu |
| q10 | Which category has the most products? | Confections | ✓ | ✗ Dairy Products |
| q11 | Who is the supplier of the product Chai? | Exotic Liquids | ✓ | ✓ |
| q12 | How many products does the supplier Exotic Liquids supply? | 3 | ✓ | ✗ 2 |
| q13 | How many orders did the customer Alfreds Futterkiste place? | 6 | ✓ | ✓ |
| q14 | How many orders did United Package ship? | 326 | ✓ | ✗ 49 |
| q15 | What is the total freight charged by Federal Shipping? | $20,512.51 | ✓ | ✗ $1,209.61 |
| q16 | Which products are in the Seafood category? | Boston Crab Meat, Carnarvon Tigers, Escargots de Bourgogne, Gravad lax, Ikura, Inlagd Sill, Jack's New England Clam Chowder, Konbu, Nord-Ost Matjeshering, Rogede sild, Röd Kaviar, Spegesild | ✓ | ✗ Boston Crab Meat (only 1 of 12) |
| q17 | How many customers are based in Germany? | 11 | ✓ | ✓ |
| q18 | How many orders were placed in 1998? | 270 | ✓ | ✗ 50 |
| q19 | What is the average order value? | $1,525.05 | ✓ | ✗ $241.68 |
| q20 | How many products are discontinued? | 8 | ✓ | ✗ 3 |

### Reading the result honestly

The baseline is not useless — it gets 5/20, and all 5 are simple
single-hop lookups (who supplies Chai, how many orders one customer
placed, how many customers are in Germany). Where it fails is
aggregation: revenue, freight, and counts over the whole dataset.
Its top-50 retrieval only ever sees a sample, so it undercounts
(Beverages revenue $14,499 vs the true $267,868; 49 of United
Package's 326 orders; 50 of 1998's 270 orders) or answers from a
sample (Janet Leverling instead of Margaret Peacock for 1997).

GraphRAG is 20/20 on this set, but that number needs its context:
the 20 questions were written to cover the intents the rule-based
parser supports (supplier/product/employee/customer/revenue/counts).
It is a closed question set, not open-ended QA — a question phrased
outside the covered intents returns "I don't know". The 100% says
the graph traversal is correct for these question shapes; it does
not say the system understands arbitrary business questions.

## Honest limits

- **In-memory NetworkX only.** There is no Neo4j server, no
  persistent graph store, and no Cypher. The graph is rebuilt from
  the CSVs on every run. It fits this dataset (1,047 nodes); it
  would not fit a production ERP database.
- **Rule-based question parsing, not an LLM.** Intent comes from
  keyword rules and entity names matched against the graph. Paraphrases
  outside the covered intents are not understood.
- **No LLM generation.** Answers are extractive values computed from
  the graph. Nothing here generates text.
- **Small sample dataset.** Northwind has 830 orders and 77 products.
  Nothing about scale, latency at production size, or real ERP data
  quality is demonstrated.
- **Retrieval baseline is deliberately simple.** Top-50 TF-IDF over
  flattened rows is a weak baseline by design; a stronger baseline
  (larger k, hybrid retrieval, SQL generation) would score higher.
  The comparison shows the graph's advantage on aggregation, not a
  ceiling for retrieval.

## Project layout

```
graphrag-enterprise-qa/
├── README.md
├── requirements.txt
├── run_eval.py            # run the eval, write results.json
├── verify_results.py      # re-run the eval, check results.json + README
├── results.json           # measured results (generated by run_eval.py)
├── data/                  # Northwind CSVs (8 tables, 220 KB)
├── src/graphrag_enterprise_qa/
│   ├── data.py            # CSV loading + name indexes
│   ├── graph_build.py     # CSVs -> NetworkX MultiDiGraph
│   ├── qa.py              # GraphRAG: parse + traverse + answer
│   ├── baseline.py        # TF-IDF retrieval baseline
│   ├── eval.py            # pandas ground truth + scoring
│   └── questions.json     # the 20 eval questions
└── tests/                 # 18 pytest tests
```
