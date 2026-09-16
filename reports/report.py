"""Console reporting functions."""

import numpy as np


def _format_value(value):
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.2f}"


def _print_reference_comparison(algorithms, reference_hr):
    """Print validation-only benchmark deltas when an external reference exists."""
    if reference_hr is None:
        return
    candidates = {
        name: metrics.get("hr_bpm")
        for name, metrics in algorithms.items()
        if metrics.get("hr_bpm") is not None and np.isfinite(metrics["hr_bpm"])
    }
    if not candidates:
        return
    closest = min(candidates, key=lambda name: abs(candidates[name] - reference_hr))
    print(f"Reference HR (validation only): {reference_hr:.2f} bpm")
    print(
        f"Closest benchmark HR: {closest} = {candidates[closest]:.2f} bpm "
        f"(absolute error {abs(candidates[closest] - reference_hr):.2f} bpm)"
    )


def print_algorithm_benchmark(
    chrom_metrics, pos_metrics, green_metrics, ica_metrics=None, reference_hr=None
):
    """Print the per-algorithm benchmark without changing any weights."""
    algorithms = {"CHROM": chrom_metrics, "POS": pos_metrics, "GREEN": green_metrics}
    if ica_metrics is not None:
        algorithms["ICA"] = ica_metrics
    labels = (("hr_bpm", "HR (bpm)"), ("snr", "SNR (dB)"),
              ("spectral_concentration", "Spectral Concentration"),
              ("fft_peak", "FFT Peak"), ("peak_ratio", "Peak Ratio"),
              ("std", "Signal Std"), ("amplitude", "Signal Amplitude"),
              ("energy", "Signal Energy"))
    print("\n================ Algorithm Benchmark ================")
    for name, metrics in algorithms.items():
        print(name)
        for key, label in labels:
            print(f"  {label:<28}: {_format_value(metrics.get(key))}")
        print()
    for key, label in labels[1:4]:
        candidates = {name: metrics.get(key) for name, metrics in algorithms.items()
                      if metrics.get(key) is not None and np.isfinite(metrics[key])}
        if candidates:
            print(f"Highest {label:<21}: {max(candidates, key=candidates.get)}")
    _print_reference_comparison(algorithms, reference_hr)
    print("=======================================================\n")


def print_report(result):
    """Print the final analysis report from an :class:`AnalysisResult`."""
    metrics = result.signal_metrics
    print("\n================ ANALYSIS REPORT ================")
    print("Heart Rate")
    print(f"  Heart Rate (bpm)              : {_format_value(result.heart_rate)}")
    hr_fft = (result.audit or {}).get("hr_fft")
    if hr_fft:
        selected = hr_fft["selected_peak"]
        second = hr_fft["second_local_peak"]
        print("\nFFT Diagnostics")
        print(f"  Samples                       : {hr_fft['samples']}")
        print(f"  FPS                           : {_format_value(hr_fft['fps'])}")
        print(f"  Frequency resolution (Hz)     : {_format_value(hr_fft['frequency_resolution_hz'])}")
        print(f"  BPM resolution                : {_format_value(hr_fft['bpm_resolution'])}")
        print(f"  Selected peak (Hz)            : {_format_value(selected['frequency_hz'])}")
        print(f"  Selected FFT bin              : {selected['fft_bin']}")
        print(f"  Second local peak (Hz)        : {_format_value(second['frequency_hz'])}")
        print(f"  Primary/second ratio           : {_format_value(second['primary_to_second_ratio'])}")
        print(f"  Peak separation (bpm)         : {_format_value(second['separation_bpm'])}")
        print(f"  Cardiac-band power             : {_format_value(hr_fft['cardiac_band_power'])}")
        print(f"  Total spectral power           : {_format_value(hr_fft['total_spectral_power'])}")
        print(f"  Cardiac-band power ratio       : {_format_value(hr_fft['cardiac_band_power_ratio'])}")
        print(f"  Selection                      : {hr_fft['selection_method']}")
    reference = (result.audit or {}).get("reference_validation")
    if reference:
        print("\nExternal Reference Validation (does not alter production HR)")
        print(f"  Reference HR (bpm)            : {_format_value(reference['reference_hr_bpm'])}")
        print(f"  Absolute Error (bpm)          : {_format_value(reference['absolute_error_bpm'])}")
        print(f"  Relative Error (%)             : {_format_value(reference['relative_error_percent'])}")
    print("\nHeart Rate Variability")
    print(f"  SDNN (ms)                     : {_format_value(result.hrv.get('SDNN_ms'))}")
    print(f"  RMSSD (ms)                    : {_format_value(result.hrv.get('RMSSD_ms'))}")
    print(f"  pNN50 (%)                     : {_format_value(result.hrv.get('pNN50_%'))}")
    if result.hrv.get("aviso"):
        print(f"  Aviso                          : {result.hrv['aviso']}")
    print("\nSignal Metrics")
    for key, label in (("snr", "SNR (dB)"),
                       ("spectral_concentration", "Spectral Concentration"),
                       ("fft_peak", "FFT Peak"), ("amplitude", "Signal Amplitude"),
                       ("std", "Signal Standard Deviation"), ("energy", "Signal Energy")):
        print(f"  {label:<30}: {_format_value(metrics.get(key))}")
    print("\nCapture Information")
    print(f"  Effective FPS                 : {_format_value(result.fps)}")
    print(f"  Valid Frames                  : {_format_value(result.valid_frames)}")
    print(f"  Capture Duration (s)          : {_format_value(result.duration)}")
    if result.audit:
        final_source = result.audit.get("final_biomarker_signal", {}).get("source")
        if final_source:
            print(f"  HR/HRV/metrics source         : {final_source}")
    print("===================================================\n")


def print_roi_benchmark(roi_results, reference_hr=None):
    """Print benchmark metrics grouped by ROI."""

    labels = (
        ("hr_bpm", "HR (bpm)"),
        ("snr", "SNR (dB)"),
        ("spectral_concentration", "Spectral Concentration"),
        ("fft_peak", "FFT Peak"),
    )

    print("\n================ ROI BENCHMARK ================\n")

    for roi_name, algorithms in roi_results.items():

        print(f"{roi_name.upper()}")

        for algorithm_name, metrics in algorithms.items():

            print(f"  {algorithm_name}")

            for key, label in labels:
                print(f"    {label:<24}: {_format_value(metrics.get(key))}")

            print()

    if reference_hr is not None:
        candidates = {
            f"{roi_name.upper()}/{algorithm_name}": metrics.get("hr_bpm")
            for roi_name, algorithms in roi_results.items()
            for algorithm_name, metrics in algorithms.items()
            if metrics.get("hr_bpm") is not None and np.isfinite(metrics["hr_bpm"])
        }
        if candidates:
            closest = min(candidates, key=lambda name: abs(candidates[name] - reference_hr))
            print(f"Reference HR (validation only): {reference_hr:.2f} bpm")
            print(
                f"Closest ROI/algorithm HR: {closest} = {candidates[closest]:.2f} bpm "
                f"(absolute error {abs(candidates[closest] - reference_hr):.2f} bpm)"
            )

    print("================================================\n")
