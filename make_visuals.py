"""Generate result visualizations from results.json as UTF-8 text files.

Outputs (all plain text, safe for the GitHub push path which accepts
text files only - no PNG/JPEG anywhere):
  results/accuracy_comparison.svg   bar chart of the four systems
  results/per_question_correctness.svg  heatmap of per-question correctness
  results/report.html               self-contained HTML report (SVGs inlined)

Regenerate after any eval run:
    python run_eval.py --llm --embedding
    python make_visuals.py
"""

from __future__ import annotations

import html
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Text stays as <text> elements in the SVG (searchable, and
# verify_results.py can check the numbers are in the file).
matplotlib.rcParams["svg.fonttype"] = "none"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(PROJECT_DIR, "results.json")
OUT_DIR = os.path.join(PROJECT_DIR, "results")

SYSTEMS = [
    ("graphrag", "GraphRAG (rule-based)", "#2c7fb0"),
    ("llm", "GraphRAG (LLM mode)", "#41b6c4"),
    ("embedding", "Baseline (embedding RAG)", "#fdae61"),
    ("baseline", "Baseline (TF-IDF)", "#d73027"),
]


def load_results(path: str = RESULTS_JSON) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def systems_from(summary: dict) -> list[dict]:
    """The systems present in this results file, with their scores."""
    total = summary["n_questions"]
    found = []
    for key, label, color in SYSTEMS:
        if key == "graphrag":
            correct = summary["graphrag_correct"]
        elif key == "llm":
            if "llm_correct" not in summary:
                continue
            correct = summary["llm_correct"]
        elif key == "embedding":
            if "embedding_baseline_correct" not in summary:
                continue
            correct = summary["embedding_baseline_correct"]
        else:
            correct = summary["baseline_correct"]
        found.append(
            {
                "key": key,
                "label": label,
                "color": color,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )
    return found


def accuracy_chart(summary: dict, out_path: str) -> None:
    systems = sorted(systems_from(summary), key=lambda s: s["accuracy"], reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    labels = [s["label"] for s in systems]
    values = [s["accuracy"] * 100 for s in systems]
    colors = [s["color"] for s in systems]
    bars = ax.barh(labels, values, color=colors, height=0.62)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%) - correct answers out of 20 questions")
    ax.set_title("Accuracy comparison (20 questions, ground truth from pandas)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for bar, s in zip(bars, systems):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{s['correct']}/{s['total']} ({s['accuracy'] * 100:.0f}%)",
            va="center",
            ha="left",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def correctness_heatmap(summary: dict, out_path: str) -> None:
    systems = systems_from(summary)
    rows = summary["results"]
    keys = [s["key"] for s in systems]
    labels = [s["label"] for s in systems]
    # 1 = correct, 0 = wrong; missing system keys are not present.
    matrix = []
    for row in rows:
        matrix.append(
            [
                1 if row.get(f"{k}_correct" if k != "embedding" else "embedding_correct") else 0
                for k in keys
            ]
        )
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    cmap = ListedColormap(["#fdae61", "#1a9850"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["id"] for r in rows], fontsize=9)
    ax.set_title("Per-question correctness (green = correct, orange = wrong)")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                "\u2713" if value else "\u2717",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
            )
    # Thin grid lines between cells.
    ax.set_xticks([x - 0.5 for x in range(1, len(labels))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def _svg_inline(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def html_report(summary: dict, out_path: str, svg_paths: dict[str, str]) -> None:
    systems = systems_from(summary)
    rows = summary["results"]
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>GraphRAG Enterprise QA - results</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;"
        "padding:0 1em;color:#222}",
        "table{border-collapse:collapse;width:100%;font-size:14px}",
        "th,td{border:1px solid #ccc;padding:6px 8px;text-align:left;vertical-align:top}",
        "th{background:#f0f0f0}",
        ".ok{color:#1a9850;font-weight:bold}.bad{color:#d73027;font-weight:bold}",
        "svg{max-width:100%;height:auto}",
        "</style></head><body>",
        "<h1>GraphRAG Enterprise QA - results</h1>",
        "<p>20 business questions over the Northwind sample data. "
        "Ground truth is computed from the CSVs with pandas at run time. "
        "GraphRAG answers are scored on the extractive graph value; "
        "LLM-mode answers are composed in natural language by the model "
        "but scored on the same value.</p>",
        "<h2>Accuracy</h2>",
        "<table><tr><th>System</th><th>Correct</th><th>Accuracy</th></tr>",
    ]
    for s in systems:
        parts.append(
            f"<tr><td>{html.escape(s['label'])}</td>"
            f"<td>{s['correct']}/{s['total']}</td>"
            f"<td>{s['accuracy'] * 100:.1f}%</td></tr>"
        )
    parts.append("</table>")
    if "llm_routing" in summary:
        r = summary["llm_routing"]
        parts.append(
            f"<p>LLM mode ({html.escape(summary.get('llm_model', ''))}): "
            f"the model routed {r['parsed_by_llm']} questions itself and "
            f"fell back to the rule-based parser for {r['fallback_to_rules']}. "
            f"Composed answers contained the scored answer in "
            f"{summary.get('llm_composed_contains_answer', '?')} of 20 cases.</p>"
        )
    if "embedding_model" in summary:
        parts.append(
            f"<p>Embedding baseline model: "
            f"{html.escape(summary['embedding_model'])}.</p>"
        )
    parts.append("<h2>Accuracy comparison</h2>")
    parts.append(_svg_inline(svg_paths["accuracy"]))
    parts.append("<h2>Per-question correctness</h2>")
    parts.append(_svg_inline(svg_paths["heatmap"]))
    parts.append("<h2>Per-question results</h2>")
    parts.append(
        "<table><tr><th>ID</th><th>Question</th><th>Ground truth</th>"
        + "".join(f"<th>{html.escape(s['label'])}</th>" for s in systems)
        + "</tr>"
    )
    for row in rows:
        cells = [f"<td>{row['id']}</td>",
                 f"<td>{html.escape(row['question'])}</td>",
                 f"<td>{html.escape(str(row['ground_truth']))}</td>"]
        for s in systems:
            key = s["key"]
            ans_key = f"{key}_answer" if key != "embedding" else "embedding_answer"
            ok_key = f"{key}_correct" if key != "embedding" else "embedding_correct"
            answer = row.get(ans_key, "-")
            ok = row.get(ok_key)
            cls = "ok" if ok else "bad"
            mark = "\u2713" if ok else "\u2717"
            cells.append(
                f'<td class="{cls}">{mark} {html.escape(str(answer))}</td>'
            )
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</table>")
    parts.append(
        "<h2>Honest limits</h2>"
        "<ul>"
        "<li>Closed 20-question set: 20/20 is on questions the parser "
        "supports, not open-ended QA.</li>"
        "<li>The 0.5B LLM mis-routes some questions; its composed answers "
        "are natural language over the graph evidence, never computed values.</li>"
        "<li>Both baselines are deliberately simple (top-50 retrieval); "
        "they retrieve chunks and take the best value they find, they do "
        "not aggregate over the whole dataset.</li>"
        "</ul></body></html>"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    results_path = args[0] if args else RESULTS_JSON
    summary = load_results(results_path)
    os.makedirs(OUT_DIR, exist_ok=True)
    accuracy_path = os.path.join(OUT_DIR, "accuracy_comparison.svg")
    heatmap_path = os.path.join(OUT_DIR, "per_question_correctness.svg")
    report_path = os.path.join(OUT_DIR, "report.html")
    accuracy_chart(summary, accuracy_path)
    correctness_heatmap(summary, heatmap_path)
    html_report(
        summary,
        report_path,
        {"accuracy": accuracy_path, "heatmap": heatmap_path},
    )
    for s in systems_from(summary):
        print(f"{s['label']}: {s['correct']}/{s['total']} ({s['accuracy'] * 100:.1f}%)")
    print(f"wrote {accuracy_path}")
    print(f"wrote {heatmap_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
