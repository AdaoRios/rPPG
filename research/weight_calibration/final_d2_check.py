"""Final, definitive A vs fixed-D2 verification using exported intermediates.

This is the last scientific decision of the project. It does NOT reprocess
videos, does NOT search parameters, and does NOT introduce any new weight
alternative: only configuration A (the pre-freeze production weights) and the
single proposed configuration D2 (frozen algorithm weights + TESTA 0.40 /
BOCH_ESQ 0.20 / BOCH_DIR 0.30 / GLABELA 0.10) are reconstructed. D2 won this
comparison and is now the frozen production configuration in config.py; A is
kept here only as the historical baseline for reproducibility.

Reconstruction reuses the exported 16 per-method signals and the frozen
production final filter, FFT (Hann, no zero-padding, no interpolation),
spectral selection, harmonic policy, confidence, and ambiguity rules.

Decision rule (documented engineering convention, applied once):
  melhora global coerente =
      MAE menor que A
      E mediana menor que A
      E MAE sem os 3 maiores erros menor que A
  sem regressao relevante =
      RMSE <= A
      E erro maximo nao piora mais de 3.0 bpm (margem documentada)
      E casos ambiguos <= A
      E confidence=low <= A
      E folds piores <= folds melhores
  D2 somente se (melhora global coerente E sem regressao relevante).

Outputs: final_d2_check.json in the intermediates directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .loocv import aggregate_metrics, mae
    from .reconstruct import load_capture, reconstruct
except ImportError:
    from rPPG.research.weight_calibration.loocv import aggregate_metrics, mae
    from rPPG.research.weight_calibration.reconstruct import load_capture, reconstruct

A_METHOD_WEIGHTS = {"chrom": 0.30, "pos": 0.30, "ica": 0.30, "green": 0.10}
A_ROI_WEIGHTS = {
    "testa": 0.36,
    "bochecha_esquerda": 0.27,
    "bochecha_direita": 0.27,
    "glabela": 0.10,
}
D2_ROI_WEIGHTS = {
    "testa": 0.40,
    "bochecha_esquerda": 0.20,
    "bochecha_direita": 0.30,
    "glabela": 0.10,
}
MAX_ERROR_REGRESSION_MARGIN_BPM = 3.0
HR_MATCH_TOLERANCE_BPM = 1e-6


def evaluate(captures, method_weights, roi_weights):
    results = []
    for capture in captures:
        result = reconstruct(capture, method_weights, roi_weights)
        result["error_bpm"] = float(
            result["hr_bpm"] - float(result["reference_hr_bpm"])
        )
        result["absolute_error_bpm"] = abs(result["error_bpm"])
        results.append(result)
    errors = [result["error_bpm"] for result in results]
    absolute = np.abs(errors)
    order = np.argsort(absolute)[::-1]
    keep = np.ones(len(errors), dtype=bool)
    keep[order[:3]] = False
    metrics = aggregate_metrics(errors)
    return {
        "metrics": metrics,
        "mae_excluding_own_top3_bpm": mae(np.asarray(errors)[keep]),
        "ambiguous_count": sum(1 for result in results if result["ambiguous"]),
        "harmonic_detected_count": sum(
            1 for result in results if result["harmonic_detected"]
        ),
        "harmonic_supported_count": sum(
            1 for result in results if result["harmonic_supported"]
        ),
        "confidence_counts": {
            level: sum(1 for result in results if result["confidence"] == level)
            for level in ("high", "medium", "low")
        },
        "per_capture": [
            {
                "capture_id": result["capture_id"],
                "reference_hr_bpm": result["reference_hr_bpm"],
                "hr_bpm": result["hr_bpm"],
                "error_bpm": result["error_bpm"],
                "confidence": result["confidence"],
                "ambiguous": result["ambiguous"],
                "harmonic_detected": result["harmonic_detected"],
                "harmonic_supported": result["harmonic_supported"],
                "decision_reason": result["decision_reason"],
            }
            for result in results
        ],
    }


def decide(baseline, candidate):
    base = baseline["metrics"]
    cand = candidate["metrics"]
    wins = ties = losses = 0
    for base_row, cand_row in zip(baseline["per_capture"], candidate["per_capture"]):
        base_abs = abs(base_row["error_bpm"])
        cand_abs = abs(cand_row["error_bpm"])
        if cand_abs < base_abs - 1e-9:
            wins += 1
        elif cand_abs > base_abs + 1e-9:
            losses += 1
        else:
            ties += 1

    improvement = {
        "mae_improved": cand["mae_bpm"] < base["mae_bpm"],
        "median_improved": (
            cand["median_absolute_error_bpm"] < base["median_absolute_error_bpm"]
        ),
        "mae_excl_top3_improved": (
            candidate["mae_excluding_own_top3_bpm"]
            < baseline["mae_excluding_own_top3_bpm"]
        ),
    }
    no_relevant_regression = {
        "rmse_not_worse": cand["rmse_bpm"] <= base["rmse_bpm"] + 1e-9,
        "max_error_within_margin": (
            cand["max_absolute_error_bpm"]
            <= base["max_absolute_error_bpm"] + MAX_ERROR_REGRESSION_MARGIN_BPM
        ),
        "ambiguous_not_higher": (
            candidate["ambiguous_count"] <= baseline["ambiguous_count"]
        ),
        "low_confidence_not_higher": (
            candidate["confidence_counts"]["low"]
            <= baseline["confidence_counts"]["low"]
        ),
        "folds_worse_not_above_better": losses <= wins,
    }
    consistent = all(improvement.values()) and all(no_relevant_regression.values())
    return {
        "decision": "CONGELAR D2" if consistent else "MANTER A",
        "melhora_global_coerente": all(improvement.values()),
        "sem_regressao_relevante": all(no_relevant_regression.values()),
        "criteria": {**improvement, **no_relevant_regression},
        "folds_better": wins,
        "folds_equal": ties,
        "folds_worse": losses,
        "deltas_vs_A": {
            "mae_bpm": cand["mae_bpm"] - base["mae_bpm"],
            "rmse_bpm": cand["rmse_bpm"] - base["rmse_bpm"],
            "bias_bpm": cand["bias_bpm"] - base["bias_bpm"],
            "median_abs_bpm": (
                cand["median_absolute_error_bpm"] - base["median_absolute_error_bpm"]
            ),
            "max_abs_bpm": (
                cand["max_absolute_error_bpm"] - base["max_absolute_error_bpm"]
            ),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Final A vs fixed-D2 verification on exported intermediates."
    )
    parser.add_argument(
        "--intermediates-dir",
        default=r"C:\rPPG\data\weight_calibration_intermediates",
    )
    args = parser.parse_args(argv)

    intermediates_dir = Path(args.intermediates_dir)
    captures = load_all(intermediates_dir)
    print(f"Capturas carregadas: {len(captures)}")
    if len(captures) != 25:
        print(f"ERRO: esperadas 25 capturas, encontradas {len(captures)}.")
        return 2

    baseline = evaluate(captures, A_METHOD_WEIGHTS, A_ROI_WEIGHTS)
    # Gate: the frozen A reconstruction must reproduce the recorded final HRs.
    for row in baseline["per_capture"]:
        capture = next(
            item for item in captures if item["capture_id"] == row["capture_id"]
        )
        recorded = capture["recorded_final_hr_bpm"]
        if recorded is None or abs(row["hr_bpm"] - float(recorded)) > HR_MATCH_TOLERANCE_BPM:
            print(
                f"ERRO: A nao reproduz o HR registrado em {row['capture_id']}: "
                f"reconstruido={row['hr_bpm']!r} registrado={recorded!r}"
            )
            return 3
    print("Gate OK: A reproduz os 25 HRs registrados.")

    candidate = evaluate(captures, A_METHOD_WEIGHTS, D2_ROI_WEIGHTS)
    decision = decide(baseline, candidate)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Final single comparison A vs fixed D2 on exported intermediates; "
            "no video reprocessing; no additional weight alternatives"
        ),
        "weights": {
            "method_weights": A_METHOD_WEIGHTS,
            "A_roi_weights": A_ROI_WEIGHTS,
            "D2_roi_weights": D2_ROI_WEIGHTS,
        },
        "decision_rule": {
            "melhora_global_coerente": [
                "MAE menor que A",
                "mediana menor que A",
                "MAE sem os 3 maiores erros menor que A",
            ],
            "sem_regressao_relevante": [
                "RMSE <= A",
                f"erro maximo nao piora mais de "
                f"{MAX_ERROR_REGRESSION_MARGIN_BPM} bpm",
                "casos ambiguos <= A",
                "confidence=low <= A",
                "folds piores <= folds melhores",
            ],
        },
        "A_current": baseline,
        "D2_fixed": candidate,
        "decision": decision,
        "disclaimer": (
            "Retrospective comparison on 25 captures with an external "
            "observational reference; not clinical validation; no claim of "
            "universal optimality."
        ),
    }
    out_path = intermediates_dir / "final_d2_check.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    for name, data in (("A", baseline), ("D2", candidate)):
        metrics = data["metrics"]
        print(
            f"{name}: MAE={metrics['mae_bpm']:.4f} RMSE={metrics['rmse_bpm']:.4f} "
            f"bias={metrics['bias_bpm']:+.4f} std={metrics['std_error_bpm']:.4f} "
            f"max={metrics['max_absolute_error_bpm']:.2f} "
            f"mediana={metrics['median_absolute_error_bpm']:.4f} "
            f"ambiguos={data['ambiguous_count']} "
            f"confidence={data['confidence_counts']}"
        )
    print(f"Criterios: {decision['criteria']}")
    print(
        f"Folds melhor/igual/pior: {decision['folds_better']}/"
        f"{decision['folds_equal']}/{decision['folds_worse']}"
    )
    print(f"DECISAO: {decision['decision']}")
    print(f"Resultados: {out_path}")
    return 0


def load_all(intermediates_dir: Path):
    capture_dirs = sorted(
        path for path in intermediates_dir.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    captures = [load_capture(path) for path in capture_dirs]
    captures.sort(key=lambda item: item["capture_id"])
    return captures


if __name__ == "__main__":
    sys.exit(main())