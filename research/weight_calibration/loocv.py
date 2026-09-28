"""Leave-one-out weight calibration over the exported fusion intermediates.

Methodology
-----------
25 folds (LOOCV). In each fold 24 captures are the development set and one
capture is the test set. Alternative weights are chosen ONLY from the
development errors; the test capture never participates in the choice. The
final HR for every candidate is obtained by rebuilding the signal in the time
domain from the exported intermediates and then running the FROZEN production
final filter, FFT (Hann, no zero-padding, no interpolation), spectral
candidate selection, and harmonic policy. No new HR-selection rule is added.

Configurations
--------------
A  current production weights (fixed in every fold)
B  GREEN reduced to 0.05 and to 0.00, redistributed to CHROM/POS/ICA (fixed)
C  per-fold selection among a small rational set of algorithm-weight shapes
   that reduce GREEN and favor POS/CHROM/ICA (chosen on development only)
D  per-fold selection among a small rational set of ROI-weight shapes that
   favor BOCHECHA_DIREITA/TESTA without eliminating the others (development
   only)

Decision rule (conservative): recommend a change only if the configuration
improves MAE AND does not worsen RMSE AND wins more folds than it loses
against A AND keeps a better MAE after excluding each configuration's own
three largest errors. Otherwise the current weights are kept. Recommendations
are proposals for review only: production code is not modified.

Outputs (NEW directory, never a historical results directory):
- loocv_results.csv       per-fold, per-configuration individual results
- summary.json            aggregate metrics and robustness analysis
- weight_calibration_report.md  full report (sections 1-9)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .reconstruct import load_capture, reconstruct
except ImportError:
    from rPPG.research.weight_calibration.reconstruct import load_capture, reconstruct

CURRENT_METHOD_WEIGHTS = {"chrom": 0.30, "pos": 0.30, "ica": 0.30, "green": 0.10}
CURRENT_ROI_WEIGHTS = {
    "testa": 0.36,
    "bochecha_esquerda": 0.27,
    "bochecha_direita": 0.27,
    "glabela": 0.10,
}

METHOD_CANDIDATES = {
    # Order matters only for exact-tie breakage: current weights come first.
    "A_current": {"chrom": 0.30, "pos": 0.30, "ica": 0.30, "green": 0.10},
    "B_green005": {
        "chrom": 0.95 / 3, "pos": 0.95 / 3, "ica": 0.95 / 3, "green": 0.05,
    },
    "B_green000": {"chrom": 1.0 / 3, "pos": 1.0 / 3, "ica": 1.0 / 3, "green": 0.0},
    "C_pos_favored": {"chrom": 0.35, "pos": 0.40, "ica": 0.25, "green": 0.0},
    "C_chrom_pos_green005": {
        "chrom": 0.35, "pos": 0.35, "ica": 0.25, "green": 0.05,
    },
}

ROI_CANDIDATES = {
    "D0_current": {
        "testa": 0.36, "bochecha_esquerda": 0.27,
        "bochecha_direita": 0.27, "glabela": 0.10,
    },
    "D1_dir_boost": {
        "testa": 0.33, "bochecha_esquerda": 0.24,
        "bochecha_direita": 0.33, "glabela": 0.10,
    },
    "D2_testa_boost": {
        "testa": 0.40, "bochecha_esquerda": 0.20,
        "bochecha_direita": 0.30, "glabela": 0.10,
    },
}

# name -> (method candidate list or fixed, roi candidate list or fixed, selection axis)
CONFIGURATIONS = {
    "A_current": (["A_current"], ["D0_current"], None),
    "B_green005": (["B_green005"], ["D0_current"], None),
    "B_green000": (["B_green000"], ["D0_current"], None),
    "C_algorithm_selection": (list(METHOD_CANDIDATES), ["D0_current"], "method"),
    "D_roi_selection": (["A_current"], list(ROI_CANDIDATES), "roi"),
}

HR_MATCH_TOLERANCE_BPM = 1e-6


def aggregate_metrics(errors) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    absolute = np.abs(errors)
    return {
        "n": int(len(errors)),
        "mae_bpm": float(np.mean(absolute)),
        "rmse_bpm": float(np.sqrt(np.mean(errors ** 2))),
        "bias_bpm": float(np.mean(errors)),
        "std_error_bpm": float(np.std(errors)),
        "max_absolute_error_bpm": float(np.max(absolute)),
        "median_absolute_error_bpm": float(np.median(absolute)),
    }


def mae(errors) -> float:
    return float(np.mean(np.abs(np.asarray(errors, dtype=np.float64))))


def load_all_captures(intermediates_dir: Path):
    capture_dirs = sorted(path for path in intermediates_dir.iterdir()
                          if path.is_dir() and (path / "metadata.json").is_file())
    captures = [load_capture(path) for path in capture_dirs]
    captures.sort(key=lambda item: item["capture_id"])
    return captures


def precompute(captures, pairs):
    """Reconstruct every unique (method candidate, roi candidate) pair once."""
    cache = {}
    for method_name, roi_name in pairs:
        method_weights = METHOD_CANDIDATES[method_name]
        roi_weights = ROI_CANDIDATES[roi_name]
        results = []
        for capture in captures:
            result = reconstruct(capture, method_weights, roi_weights)
            if result["reference_hr_bpm"] is None:
                raise RuntimeError(
                    f"reference_hr_bpm auscente em {capture['capture_id']}"
                )
            result["error_bpm"] = float(
                result["hr_bpm"] - float(result["reference_hr_bpm"])
            )
            result["absolute_error_bpm"] = abs(result["error_bpm"])
            results.append(result)
        cache[(method_name, roi_name)] = results
    return cache


def validate_reconstruction(captures, cache) -> list:
    """Configuration A must reproduce the recorded final HR of every capture."""
    problems = []
    baseline = cache[("A_current", "D0_current")]
    for result in baseline:
        recorded = result["recorded_final_hr_bpm"]
        if recorded is None:
            problems.append(f"{result['capture_id']}: HR registrado ausente")
            continue
        if abs(result["hr_bpm"] - float(recorded)) > HR_MATCH_TOLERANCE_BPM:
            problems.append(
                f"{result['capture_id']}: reconstruido={result['hr_bpm']!r} "
                f"registrado={recorded!r}"
            )
    return problems


def run_loocv(captures, cache):
    """Return per-fold rows for every configuration using development-only choice."""
    capture_ids = [capture["capture_id"] for capture in captures]
    rows = []
    for config_name, (method_options, roi_options, axis) in CONFIGURATIONS.items():
        for test_index, test_id in enumerate(capture_ids):
            development = [
                index for index in range(len(capture_ids)) if index != test_index
            ]
            chosen_method = method_options[0]
            chosen_roi = roi_options[0]
            if axis == "method":
                best = None
                for candidate in method_options:
                    errors = [
                        cache[(candidate, roi_options[0])][index]["error_bpm"]
                        for index in development
                    ]
                    score = mae(errors)
                    if best is None or score < best[0]:
                        best = (score, candidate)
                chosen_method = best[1]
            elif axis == "roi":
                best = None
                for candidate in roi_options:
                    errors = [
                        cache[(method_options[0], candidate)][index]["error_bpm"]
                        for index in development
                    ]
                    score = mae(errors)
                    if best is None or score < best[0]:
                        best = (score, candidate)
                chosen_roi = best[1]

            result = cache[(chosen_method, chosen_roi)][test_index]
            rows.append({
                "configuration": config_name,
                "capture_id": test_id,
                "chosen_method_candidate": chosen_method,
                "chosen_roi_candidate": chosen_roi,
                "hr_bpm": result["hr_bpm"],
                "reference_hr_bpm": result["reference_hr_bpm"],
                "error_bpm": result["error_bpm"],
                "absolute_error_bpm": result["absolute_error_bpm"],
                "confidence": result["confidence"],
                "ambiguous": result["ambiguous"],
                "harmonic_detected": result["harmonic_detected"],
                "harmonic_supported": result["harmonic_supported"],
                "decision_reason": result["decision_reason"],
                "snr": result["snr"],
                "spectral_concentration": result["spectral_concentration"],
            })
    return rows


def summarize(rows) -> dict:
    configs = sorted({row["configuration"] for row in rows})
    capture_ids = sorted({row["capture_id"] for row in rows})
    by_config = {
        config: {row["capture_id"]: row for row in rows
                 if row["configuration"] == config}
        for config in configs
    }
    summary = {}
    for config in configs:
        config_rows = [by_config[config][capture_id] for capture_id in capture_ids]
        errors = [row["error_bpm"] for row in config_rows]
        absolute = np.abs(errors)
        order = np.argsort(absolute)[::-1]
        keep_mask = np.ones(len(errors), dtype=bool)
        keep_mask[order[:3]] = False
        summary[config] = {
            "metrics": aggregate_metrics(errors),
            "mae_excluding_own_top3_bpm": mae(np.asarray(errors)[keep_mask]),
            "ambiguous_count": sum(1 for row in config_rows if row["ambiguous"]),
            "harmonic_detected_count": sum(
                1 for row in config_rows if row["harmonic_detected"]
            ),
            "harmonic_supported_count": sum(
                1 for row in config_rows if row["harmonic_supported"]
            ),
            "confidence_counts": {
                level: sum(1 for row in config_rows if row["confidence"] == level)
                for level in ("high", "medium", "low")
            },
            "top3_errors": [
                {
                    "capture_id": config_rows[index]["capture_id"],
                    "reference": config_rows[index]["reference_hr_bpm"],
                    "hr": config_rows[index]["hr_bpm"],
                    "absolute_error": float(absolute[index]),
                }
                for index in order[:3]
            ],
        }

    baseline = by_config["A_current"]
    for config in configs:
        if config == "A_current":
            continue
        wins = ties = losses = 0
        for capture_id in capture_ids:
            current_abs = by_config[config][capture_id]["absolute_error_bpm"]
            baseline_abs = baseline[capture_id]["absolute_error_bpm"]
            if current_abs < baseline_abs - 1e-9:
                wins += 1
            elif current_abs > baseline_abs + 1e-9:
                losses += 1
            else:
                ties += 1
        summary[config]["vs_A_current"] = {
            "folds_better": wins,
            "folds_equal": ties,
            "folds_worse": losses,
            "delta_mae_bpm": (
                summary[config]["metrics"]["mae_bpm"]
                - summary["A_current"]["metrics"]["mae_bpm"]
            ),
            "delta_rmse_bpm": (
                summary[config]["metrics"]["rmse_bpm"]
                - summary["A_current"]["metrics"]["rmse_bpm"]
            ),
            "delta_bias_bpm": (
                summary[config]["metrics"]["bias_bpm"]
                - summary["A_current"]["metrics"]["bias_bpm"]
            ),
        }
    return summary


def recommend(summary, frequencies) -> dict:
    """Conservative consistency rule; never implemented automatically."""
    baseline = summary["A_current"]
    evaluated = []
    consistent = []
    for config, data in summary.items():
        if config == "A_current":
            continue
        comparison = data["vs_A_current"]
        criteria = {
            "mae_improved": data["metrics"]["mae_bpm"] < baseline["metrics"]["mae_bpm"],
            "rmse_not_worse": (
                data["metrics"]["rmse_bpm"] <= baseline["metrics"]["rmse_bpm"] + 1e-9
            ),
            "more_wins_than_losses": (
                comparison["folds_better"] > comparison["folds_worse"]
            ),
            "mae_excl_top3_improved": (
                data["mae_excluding_own_top3_bpm"]
                < baseline["mae_excluding_own_top3_bpm"]
            ),
        }
        entry = {
            "configuration": config,
            "criteria": criteria,
            "consistent_improvement": all(criteria.values()),
        }
        evaluated.append(entry)
        if entry["consistent_improvement"]:
            consistent.append(config)

    if not consistent:
        return {
            "decision": "MANTER OS PESOS ATUAIS",
            "proposed_weights": None,
            "reason": (
                "Nenhuma configuracao apresentou melhora consistente em MAE, "
                "RMSE, vitorias por fold e MAE sem os tres maiores erros."
            ),
            "evaluated": evaluated,
        }
    best = min(
        consistent,
        key=lambda config: summary[config]["metrics"]["mae_bpm"],
    )
    method_options, roi_options, axis = CONFIGURATIONS[best]

    def modal(candidates, counts):
        return max(
            candidates,
            key=lambda name: (counts.get(name, 0), -candidates.index(name)),
        )

    if axis == "method":
        chosen_method = modal(method_options, frequencies["method"])
        chosen_roi = roi_options[0]
    elif axis == "roi":
        chosen_method = method_options[0]
        chosen_roi = modal(roi_options, frequencies["roi"])
    else:
        chosen_method = method_options[0]
        chosen_roi = roi_options[0]
    return {
        "decision": "PROPOR (nao implementado; aguarda revisao)",
        "configuration": best,
        "proposed_weights": {
            "method_candidate": chosen_method,
            "roi_candidate": chosen_roi,
            "method_weights": METHOD_CANDIDATES[chosen_method],
            "roi_weights": ROI_CANDIDATES[chosen_roi],
            "note": (
                "As configuracoes C/D escolhem pesos por fold usando apenas o "
                "conjunto de desenvolvimento; os pesos exibidos sao o candidato "
                "modal entre os vencedores dos folds, apresentado como "
                "PROPOSTA fixa sujeita a revisao. A metrica LOOCV avaliou a "
                "selecao por fold, nao esta proposta fixa."
            ),
        },
        "reason": (
            "Todos os criterios de consistencia foram atendidos; a mudanca "
            "permanece uma PROPOSTA sujeita a revisao humana."
        ),
        "evaluated": evaluated,
    }


def selection_frequencies(rows) -> dict:
    frequencies = {"method": {}, "roi": {}}
    for row in rows:
        if row["configuration"] == "C_algorithm_selection":
            key = row["chosen_method_candidate"]
            frequencies["method"][key] = frequencies["method"].get(key, 0) + 1
        if row["configuration"] == "D_roi_selection":
            key = row["chosen_roi_candidate"]
            frequencies["roi"][key] = frequencies["roi"].get(key, 0) + 1
    return frequencies


def write_csv(rows, path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value) -> str:
    return f"{value:.4f}"


def write_report(path: Path, summary, rows, recommendation, frequencies,
                 export_summary) -> None:
    capture_ids = sorted({row["capture_id"] for row in rows})
    by_config = {
        config: {row["capture_id"]: row for row in rows
                 if row["configuration"] == config}
        for config in summary
    }
    lines = []
    lines.append("# Calibracao de Pesos (LOOCV) - Relatorio de Revisao\n")
    lines.append(
        f"Gerado em (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        "Esta analise usa somente os resultados de teste da validacao cruzada. "
        "Nao e validacao clinica e nao afirma otimalidade universal dos pesos. "
        "Nenhum peso de producao foi alterado.\n"
    )

    lines.append("## 1. Pesos atuais de producao\n")
    lines.append("| Algoritmo | Peso |")
    lines.append("|---|---:|")
    for name in ("chrom", "pos", "ica", "green"):
        lines.append(f"| {name.upper()} | {CURRENT_METHOD_WEIGHTS[name]:.2f} |")
    lines.append("\n| ROI | Peso |")
    lines.append("|---|---:|")
    for name in ("testa", "bochecha_esquerda", "bochecha_direita", "glabela"):
        lines.append(f"| {name.upper()} | {CURRENT_ROI_WEIGHTS[name]:.2f} |")
    lines.append("")

    lines.append("## 2. Configuracoes avaliadas\n")
    lines.append("| Configuracao | Descricao | Selecao |")
    lines.append("|---|---|---|")
    descriptions = {
        "A_current": ("Pesos atuais (controle)", "Fixa"),
        "B_green005": ("GREEN=0.05; CHROM/POS/ICA=0.3167 cada", "Fixa"),
        "B_green000": ("GREEN=0.00; CHROM/POS/ICA=0.3333 cada", "Fixa"),
        "C_algorithm_selection": (
            "Selecao por fold entre " + ", ".join(METHOD_CANDIDATES),
            "Desenvolvimento (24 capturas)",
        ),
        "D_roi_selection": (
            "Selecao por fold entre " + ", ".join(ROI_CANDIDATES),
            "Desenvolvimento (24 capturas)",
        ),
    }
    for config in summary:
        description, selection = descriptions[config]
        lines.append(f"| {config} | {description} | {selection} |")
    lines.append("")
    lines.append("Detalhe dos candidatos de algoritmo:\n")
    lines.append("| Candidato | CHROM | POS | ICA | GREEN |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, weights in METHOD_CANDIDATES.items():
        lines.append(
            f"| {name} | {weights['chrom']:.4f} | {weights['pos']:.4f} | "
            f"{weights['ica']:.4f} | {weights['green']:.4f} |"
        )
    lines.append("\nDetalhe dos candidatos de ROI:\n")
    lines.append("| Candidato | TESTA | BOCH_ESQ | BOCH_DIR | GLABELA |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, weights in ROI_CANDIDATES.items():
        lines.append(
            f"| {name} | {weights['testa']:.4f} | "
            f"{weights['bochecha_esquerda']:.4f} | "
            f"{weights['bochecha_direita']:.4f} | "
            f"{weights['glabela']:.4f} |"
        )
    lines.append("")

    lines.append("## 3. Metodologia LOOCV\n")
    lines.append(
        "- 25 folds; em cada fold: 24 capturas de desenvolvimento e 1 de teste.\n"
        "- Os pesos alternativos sao escolhidos SOMENTE pelo erro MAE das 24 "
        "capturas de desenvolvimento; a captura de teste nunca participa.\n"
        "- O HR final de cada candidato vem da reconstrucao no dominio do tempo "
        "dos 16 sinais intermediarios exportados, seguida do filtro final, FFT "
        "(Hann, sem zero-padding, sem interpolacao) e politica de harmonicos "
        "de producao, inalterados.\n"
        "- A referencia externa (Apple Watch) e usada apenas para avaliacao, "
        "nunca para selecionar HR ou pesos.\n"
        f"- Exportacoes validadas: {len(export_summary.get('captures', []))} "
        "capturas reproduziram exatamente o HR final registrado com os pesos "
        "atuais.\n"
    )

    lines.append("## 4. Resultados por fold\n")
    header = "| Captura | Ref |"
    for config in summary:
        header += f" {config} HR | {config} erro |"
    lines.append(header)
    lines.append("|---" + "|---:---:" * len(summary) + "|")
    for capture_id in capture_ids:
        reference = summary_reference(by_config, capture_id)
        line = f"| {capture_id} | {reference:.2f} |"
        for config in summary:
            row = by_config[config][capture_id]
            line += f" {row['hr_bpm']:.2f} | {row['error_bpm']:+.2f} |"
        lines.append(line)
    lines.append("")

    lines.append("## 5. Metricas agregadas (25 resultados de teste)\n")
    lines.append(
        "| Configuracao | MAE | RMSE | Bias | Std | Max | Mediana | "
        "Ambiguos | Harmonicos det. | MAE s/ top3 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for config, data in summary.items():
        metrics = data["metrics"]
        lines.append(
            f"| {config} | {_fmt(metrics['mae_bpm'])} | "
            f"{_fmt(metrics['rmse_bpm'])} | {_fmt(metrics['bias_bpm'])} | "
            f"{_fmt(metrics['std_error_bpm'])} | "
            f"{_fmt(metrics['max_absolute_error_bpm'])} | "
            f"{_fmt(metrics['median_absolute_error_bpm'])} | "
            f"{data['ambiguous_count']} | "
            f"{data['harmonic_detected_count']} | "
            f"{_fmt(data['mae_excluding_own_top3_bpm'])} |"
        )
    lines.append("")

    lines.append("## 6. Comparacao entre configuracoes (vs A)\n")
    lines.append(
        "| Configuracao | dMAE | dRMSE | dBias | Folds melhor | Igual | Pior |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for config, data in summary.items():
        if config == "A_current":
            continue
        comparison = data["vs_A_current"]
        lines.append(
            f"| {config} | {comparison['delta_mae_bpm']:+.4f} | "
            f"{comparison['delta_rmse_bpm']:+.4f} | "
            f"{comparison['delta_bias_bpm']:+.4f} | "
            f"{comparison['folds_better']} | {comparison['folds_equal']} | "
            f"{comparison['folds_worse']} |"
        )
    lines.append("")
    lines.append("Frequencia de selecao nos folds (C e D):\n")
    lines.append(f"- C (algoritmos): {frequencies['method']}")
    lines.append(f"- D (ROIs): {frequencies['roi']}")
    lines.append("")

    lines.append("## 7. Analise dos outliers\n")
    for config, data in summary.items():
        top = ", ".join(
            f"{item['capture_id']} ({item['absolute_error']:.2f} bpm)"
            for item in data["top3_errors"]
        )
        lines.append(
            f"- {config}: maiores erros = {top}; "
            f"MAE sem os 3 maiores = {_fmt(data['mae_excluding_own_top3_bpm'])} bpm."
        )
    lines.append("")

    lines.append("## 8. Recomendacao dos pesos finais\n")
    lines.append(f"**Decisao: {recommendation['decision']}**\n")
    lines.append(f"Motivo: {recommendation['reason']}\n")
    lines.append("Criterios de consistencia por configuracao:\n")
    lines.append("| Configuracao | MAE melhor | RMSE nao pior | "
                 "mais vitorias que derrotas | MAE s/ top3 melhor | Consistente |")
    lines.append("|---|---|---|---|---|---|")
    for entry in recommendation["evaluated"]:
        criteria = entry["criteria"]
        lines.append(
            f"| {entry['configuration']} | {criteria['mae_improved']} | "
            f"{criteria['rmse_not_worse']} | "
            f"{criteria['more_wins_than_losses']} | "
            f"{criteria['mae_excl_top3_improved']} | "
            f"{entry['consistent_improvement']} |"
        )
    lines.append("")
    if recommendation.get("proposed_weights"):
        winning = recommendation["configuration"]
        winning_metrics = summary[winning]["metrics"]
        winning_comparison = summary[winning]["vs_A_current"]
        baseline_max = summary["A_current"]["metrics"]["max_absolute_error_bpm"]
        lines.append(
            f"Configuracao vencedora ({winning}): "
            f"MAE={winning_metrics['mae_bpm']:.4f} "
            f"(dMAE={winning_comparison['delta_mae_bpm']:+.4f}), "
            f"RMSE={winning_metrics['rmse_bpm']:.4f} "
            f"(dRMSE={winning_comparison['delta_rmse_bpm']:+.4f}), "
            f"erro max={winning_metrics['max_absolute_error_bpm']:.2f} "
            f"(delta={winning_metrics['max_absolute_error_bpm'] - baseline_max:+.2f} "
            f"bpm versus A), folds melhor/igual/pior = "
            f"{winning_comparison['folds_better']}/"
            f"{winning_comparison['folds_equal']}/"
            f"{winning_comparison['folds_worse']}.\n"
        )
        lines.append(
            "Atencao: os ganhos sao retrospectivos, pequenos e obtidos em 25 "
            "capturas de um unico conjunto; o erro maximo pode piorar mesmo "
            "com MAE melhor. Nao implementar sem revisao humana e nova "
            "validacao.\n"
        )
        lines.append("Pesos propostos (PARA REVISAO; nao implementados):\n")
        lines.append(
            f"```json\n{json.dumps(recommendation['proposed_weights'], indent=2)}\n```\n"
        )
    else:
        lines.append(
            "Nenhum peso alternativo foi proposto; os pesos atuais permanecem.\n"
        )

    lines.append("## 9. Limitacoes\n")
    lines.append(
        "- Os resultados sao retrospectivos para 25 capturas de um unico "
        "conjunto; nao ha garantia de generalizacao.\n"
        "- A referencia externa nao e ground truth clinico.\n"
        "- Os 25 casos atuais estao todos com confidence=low, o que limita a "
        "interpretacao de qualquer melhora.\n"
        "- A selecao por fold em C e D e uma escolha de desenvolvimento; com "
        "apenas 25 capturas, vies de selecao e variabilidade entre folds devem "
        "ser considerados.\n"
        "- Esta etapa nao altera producao: qualquer mudanca de pesos depende de "
        "revisao humana explicita.\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_reference(by_config, capture_id):
    any_config = next(iter(by_config))
    return float(by_config[any_config][capture_id]["reference_hr_bpm"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="LOOCV weight calibration over exported intermediates."
    )
    parser.add_argument(
        "--intermediates-dir",
        default=r"C:\rPPG\data\weight_calibration_intermediates",
    )
    parser.add_argument(
        "--out-dir",
        default=r"C:\rPPG\data\weight_calibration_intermediates",
    )
    args = parser.parse_args(argv)

    intermediates_dir = Path(args.intermediates_dir)
    out_dir = Path(args.out_dir)
    forbidden = {Path(r"C:\rPPG\data\results").resolve(),
                 Path(r"C:\rPPG\data\results_final_validation").resolve()}
    if out_dir.resolve() in forbidden:
        print("ERRO: --out-dir nao pode ser um diretorio historico.")
        return 2

    captures = load_all_captures(intermediates_dir)
    print(f"Capturas carregadas: {len(captures)}")
    if len(captures) < 25:
        print("ERRO: sao necessarias as 25 capturas exportadas.")
        return 2

    pairs = sorted({
        (method_name, roi_name)
        for method_options, roi_options, _ in CONFIGURATIONS.values()
        for method_name in method_options
        for roi_name in roi_options
    })
    print(f"Reconstruindo {len(pairs)} pares x {len(captures)} capturas...")
    cache = precompute(captures, pairs)

    problems = validate_reconstruction(captures, cache)
    if problems:
        print("ERRO: a reconstrucao com os pesos atuais nao reproduz os HRs registrados:")
        for problem in problems:
            print(f"  - {problem}")
        return 3
    print("Gate de validacao OK: config A reproduz os 25 HRs registrados.")

    rows = run_loocv(captures, cache)
    summary = summarize(rows)
    frequencies = selection_frequencies(rows)
    recommendation = recommend(summary, frequencies)
    if recommendation.get("proposed_weights") is not None:
        recommendation["proposed_weights"]["selection_frequency"] = frequencies

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "loocv_results.csv")

    export_summary = {}
    export_summary_path = intermediates_dir / "export_summary.json"
    if export_summary_path.is_file():
        with open(export_summary_path, "r", encoding="utf-8") as handle:
            export_summary = json.load(handle)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": "LOOCV 25 folds; development-only weight selection; "
                       "frozen final filter/FFT/harmonic policy",
        "configurations": {
            config: {
                "method_candidates": method_options,
                "roi_candidates": roi_options,
                "selection_axis": axis,
            }
            for config, (method_options, roi_options, axis) in CONFIGURATIONS.items()
        },
        "method_candidates": METHOD_CANDIDATES,
        "roi_candidates": ROI_CANDIDATES,
        "summary": summary,
        "selection_frequencies": frequencies,
        "recommendation": recommendation,
        "disclaimer": (
            "Retrospective cross-validation on 25 captures; not clinical "
            "validation; no claim of universal optimality; production weights "
            "unchanged."
        ),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    write_report(
        out_dir / "weight_calibration_report.md",
        summary, rows, recommendation, frequencies, export_summary,
    )

    print("\nMetricas agregadas (25 testes):")
    for config, data in summary.items():
        metrics = data["metrics"]
        print(
            f"  {config}: MAE={metrics['mae_bpm']:.4f} "
            f"RMSE={metrics['rmse_bpm']:.4f} bias={metrics['bias_bpm']:+.4f} "
            f"max={metrics['max_absolute_error_bpm']:.2f} "
            f"ambiguos={data['ambiguous_count']}"
        )
    print(f"\nDecisao: {recommendation['decision']}")
    print(f"Relatorio: {out_dir / 'weight_calibration_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())