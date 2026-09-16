"""Small dependency-free HTML report for calibration observations."""

from __future__ import annotations

from html import escape
from pathlib import Path


def _write_plots(output_dir, records, summary):
    """Create lightweight optional plots when matplotlib is already present."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    root = Path(output_dir)
    images = []
    paired = [item for item in records if item.get("reference_hr_bpm") is not None]
    if paired:
        fig, axis = plt.subplots()
        axis.scatter([item["reference_hr_bpm"] for item in paired], [item["final_hr_bpm"] for item in paired])
        axis.set(xlabel="Apple Watch reference (bpm)", ylabel="Estimated HR (bpm)", title="Reference × estimated HR")
        path = root / "reference_vs_estimated.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); images.append(path.name)
    for group_type, filename, title in (("algorithm", "error_by_algorithm.png", "MAE by algorithm"), ("roi", "error_by_roi.png", "MAE by ROI")):
        rows = [item for item in summary if item["group_type"] == group_type and item["mae_bpm"] is not None]
        if rows:
            fig, axis = plt.subplots(); axis.bar([item["group"] for item in rows], [item["mae_bpm"] for item in rows]); axis.set(ylabel="MAE (bpm)", title=title); fig.autofmt_xdate()
            path = root / filename; fig.savefig(path, bbox_inches="tight"); plt.close(fig); images.append(path.name)
    light = [item for item in paired if item.get("lighting", {}).get("bright_pixel_ratio") is not None and item.get("absolute_error_bpm") is not None]
    if light:
        fig, axis = plt.subplots(); axis.scatter([item["lighting"]["bright_pixel_ratio"] for item in light], [item["absolute_error_bpm"] for item in light]); axis.set(xlabel="bright_pixel_ratio", ylabel="Absolute error (bpm)", title="Error × bright pixel ratio")
        path = root / "error_vs_bright_pixel_ratio.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); images.append(path.name)
    return images


def write_report(output_dir, records, summary):
    valid = [item for item in records if item.get("reference_hr_bpm") is not None]
    final = next((item for item in summary if item["group_type"] == "final"), {})
    largest = sorted((item for item in records if item.get("absolute_error_bpm") is not None),
                     key=lambda item: item["absolute_error_bpm"], reverse=True)[:10]
    headers = ["group_type", "group", "mae_bpm", "rmse_bpm", "mean_error_bpm", "std_error_bpm", "n_valid"]
    table = "".join("<tr>" + "".join(f"<td>{escape(str(row.get(h, '')))}</td>" for h in headers) + "</tr>" for row in summary)
    errors = "".join(f"<li>{escape(item['capture_id'])}: {item['absolute_error_bpm']:.2f} bpm</li>" for item in largest)
    plots = "".join(f'<img src="{escape(name)}" alt="{escape(name)}" style="max-width:48%">' for name in _write_plots(output_dir, records, summary))
    html = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>rPPG calibration</title>
<style>body{{font-family:Arial;margin:2rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.4rem}}</style></head><body>
<h1>rPPG calibration report</h1><p>Captures: {len(records)} | external references valid: {len(valid)} | final MAE: {final.get('mae_bpm')} | final RMSE: {final.get('rmse_bpm')}</p>
<p>References are external Apple Watch observations, not clinical ground truth. No production weights were changed.</p>
<h2>Performance</h2><table><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr>{table}</table>
<h2>Largest final errors</h2><ol>{errors}</ol></body></html>"""
    html = html.replace("</body>", f"<h2>Plots</h2>{plots}</body>")
    path = Path(output_dir) / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
