"""Utility helpers for persisting textual run summaries."""
from pathlib import Path
from typing import Iterable, Tuple, Mapping


def write_metrics_report(
    output_dir,
    run_name: str,
    metadata: Iterable[Tuple[str, object]],
    test_loss: float,
    test_metrics: Mapping[str, float],
) -> Path:
    """Write a TXT report with metadata and test metrics for a training run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"{run_name}_test_metrics.txt"

    lines = [f"Run: {run_name}", ""]

    metadata_list = list(metadata)
    if metadata_list:
        lines.append("Run metadata")
        lines.append("------------")
        for label, value in metadata_list:
            lines.append(f"{label}: {value}")
        lines.append("")

    lines.append("Test evaluation")
    lines.append("---------------")
    lines.append(f"Loss: {test_loss:.4f}")
    for name in sorted(test_metrics):
        value = test_metrics[name]
        lines.append(f"{name}: {value:.4f}")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
