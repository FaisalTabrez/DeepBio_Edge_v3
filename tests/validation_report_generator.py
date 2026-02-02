"""Generate validation reports with plots (Markdown/PDF).

Consumes validation_metrics.json produced by pipeline_validator.py and
outputs a Markdown report with embedded confusion matrix and rarefaction curve.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _save_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (Lineage)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_rarefaction_curve(sample_sizes: Sequence[int], diversity: Sequence[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sample_sizes, diversity, marker="o")
    ax.set_title("Rarefaction Curve (Mock Community)")
    ax.set_xlabel("Sequences Sampled")
    ax.set_ylabel("Unique Lineages")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _compute_rarefaction(labels: List[str]) -> Dict[str, List[int] | List[float]]:
    if not labels:
        return {"sample_sizes": [], "diversity": []}

    unique_counts = []
    sample_sizes = []
    max_n = len(labels)
    step = max(5, max_n // 10)

    for size in range(step, max_n + 1, step):
        sample = labels[:size]
        unique_counts.append(len(set(sample)))
        sample_sizes.append(size)

    return {"sample_sizes": sample_sizes, "diversity": unique_counts}


def generate_markdown_report(
    metrics: Dict[str, Any],
    output_dir: Path,
    confusion_matrix_path: Optional[Path],
    rarefaction_path: Optional[Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "pipeline_validation_report.md"

    report_lines = [
        "# Pipeline Validation Report (Mock Community)",
        "",
        "## Summary Metrics",
        f"- Precision: **{metrics.get('precision', 0.0):.3f}**",
        f"- Recall: **{metrics.get('recall', 0.0):.3f}**",
        f"- F1-Score: **{metrics.get('f1_score', 0.0):.3f}**",
        f"- Accuracy: **{metrics.get('accuracy', 0.0):.3f}**",
        "",
        "## Taxonomic Depth Accuracy",
        "",
    ]

    depth = metrics.get("taxonomic_depth_accuracy", {})
    for rank, score in depth.items():
        report_lines.append(f"- {rank.title()}: **{score:.3f}**")

    if confusion_matrix_path:
        report_lines.extend(
            [
                "",
                "## Confusion Matrix",
                f"![Confusion Matrix]({confusion_matrix_path.name})",
            ]
        )

    if rarefaction_path:
        report_lines.extend(
            [
                "",
                "## Rarefaction Curve",
                f"![Rarefaction Curve]({rarefaction_path.name})",
            ]
        )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def generate_pdf_report(
    metrics: Dict[str, Any],
    output_dir: Path,
    confusion_matrix_path: Optional[Path],
    rarefaction_path: Optional[Path],
) -> Optional[Path]:
    """Generate a simple PDF summary if reportlab is available."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
    except Exception:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "pipeline_validation_report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    y = height - 0.75 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.75 * inch, y, "Pipeline Validation Report (Mock Community)")
    y -= 0.5 * inch

    c.setFont("Helvetica", 10)
    for line in [
        f"Precision: {metrics.get('precision', 0.0):.3f}",
        f"Recall: {metrics.get('recall', 0.0):.3f}",
        f"F1-Score: {metrics.get('f1_score', 0.0):.3f}",
        f"Accuracy: {metrics.get('accuracy', 0.0):.3f}",
    ]:
        c.drawString(0.75 * inch, y, line)
        y -= 0.25 * inch

    if confusion_matrix_path and confusion_matrix_path.exists():
        y -= 0.2 * inch
        c.drawImage(str(confusion_matrix_path), 0.75 * inch, y - 3.0 * inch, width=5.5 * inch, height=3.0 * inch)
        y -= 3.2 * inch

    if rarefaction_path and rarefaction_path.exists():
        y -= 0.2 * inch
        c.drawImage(str(rarefaction_path), 0.75 * inch, y - 2.6 * inch, width=5.0 * inch, height=2.6 * inch)

    c.showPage()
    c.save()
    return pdf_path


def generate_report_from_json(metrics_path: Path, output_dir: Path) -> Path:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    labels = payload.get("labels", [])
    confusion = payload.get("confusion", [])

    confusion_matrix_path = None
    if confusion and labels:
        cm_array = np.array(confusion)
        confusion_matrix_path = output_dir / "confusion_matrix.png"
        _save_confusion_matrix(cm_array, labels, confusion_matrix_path)

    rarefaction = _compute_rarefaction(labels)
    rarefaction_path = None
    if rarefaction["sample_sizes"]:
        rarefaction_path = output_dir / "rarefaction_curve.png"
        _save_rarefaction_curve(
            [int(v) for v in rarefaction["sample_sizes"]],
            [float(v) for v in rarefaction["diversity"]],
            rarefaction_path,
        )

    report_path = generate_markdown_report(metrics, output_dir, confusion_matrix_path, rarefaction_path)
    generate_pdf_report(metrics, output_dir, confusion_matrix_path, rarefaction_path)
    return report_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate validation report")
    parser.add_argument(
        "--metrics",
        default=str(Path("data/test/validation_reports/validation_metrics.json")),
        help="Path to validation_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("data/test/validation_reports")),
        help="Output directory",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    report_path = generate_report_from_json(Path(args.metrics), Path(args.output_dir))
    logger.info("Report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
