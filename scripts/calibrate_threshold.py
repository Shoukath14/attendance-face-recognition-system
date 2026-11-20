"""
Read ROC data from reports/eval_results_*.json and suggest threshold.
Usage: .venv\\Scripts\\python.exe scripts/calibrate_threshold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _find_threshold_at_eer(fpr_list, tpr_list, thresh_list) -> tuple[float, float, float]:
    """Return (eer, fpr_at_eer, threshold_at_eer)."""
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    thresholds = np.array(thresh_list)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(fpr[idx]), float(thresholds[idx])


def _find_threshold_at_far(fpr_list, tpr_list, thresh_list, target_far: float) -> tuple[float, float]:
    """Return (tar, threshold) at given FAR."""
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    thresholds = np.array(thresh_list)
    idx = np.searchsorted(fpr, target_far, side="right") - 1
    idx = max(0, min(int(idx), len(tpr) - 1))
    return float(tpr[idx]), float(thresholds[idx])


def main() -> None:
    reports_dir = Path("reports")
    json_files = sorted(reports_dir.glob("eval_results_*.json")) if reports_dir.exists() else []

    if not json_files:
        print("Run evaluate.py first")
        sys.exit(0)

    # Header
    print(f"\n{'Model':<20} {'Operating Point':<22} {'Threshold':>12} {'TAR':>8} {'FAR':>8}")
    print("-" * 76)

    recommendations: list[dict] = []

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        model = data.get("model", jf.stem.replace("eval_results_", ""))
        roc = data.get("roc", {})

        if not roc or "fpr" not in roc:
            print(f"{model:<20} {'[no ROC data]':<22}")
            continue

        fpr_list = roc["fpr"]
        tpr_list = roc["tpr"]
        thresh_list = roc["thresholds"]

        # EER operating point
        eer, fpr_at_eer, thresh_eer = _find_threshold_at_eer(fpr_list, tpr_list, thresh_list)
        tar_at_eer = 1.0 - eer
        print(f"{model:<20} {'EER':<22} {thresh_eer:>12.4f} {tar_at_eer:>8.4f} {fpr_at_eer:>8.4f}")

        # FAR = 0.01 operating point
        tar_001, thresh_001 = _find_threshold_at_far(fpr_list, tpr_list, thresh_list, 0.01)
        print(f"{'':<20} {'FAR=0.01':<22} {thresh_001:>12.4f} {tar_001:>8.4f} {'0.0100':>8}")

        # FAR = 0.001 operating point
        tar_0001, thresh_0001 = _find_threshold_at_far(fpr_list, tpr_list, thresh_list, 0.001)
        print(f"{'':<20} {'FAR=0.001':<22} {thresh_0001:>12.4f} {tar_0001:>8.4f} {'0.0010':>8}")

        print()

        recommendations.append(
            {
                "model": model,
                "recommended_threshold": thresh_eer,
                "note": "EER threshold — balanced between FAR and FRR",
                "strict_threshold": thresh_001,
                "strict_note": "FAR=0.01 — fewer false accepts, more false rejects",
            }
        )

    if recommendations:
        print("=== Recommendation ===")
        for rec in recommendations:
            print(f"  [{rec['model']}]")
            print(f"    Balanced  threshold: {rec['recommended_threshold']:.4f}  ({rec['note']})")
            print(f"    Strict    threshold: {rec['strict_threshold']:.4f}  ({rec['strict_note']})")
            print()
        print("Update recognition.threshold in config.yaml with the chosen value.")


if __name__ == "__main__":
    main()
