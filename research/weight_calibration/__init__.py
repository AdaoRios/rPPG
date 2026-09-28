"""Weight-calibration research utilities.

This package does not change production behavior: it reuses the frozen
pipeline primitives to export intermediate fusion inputs and to evaluate
alternative weight configurations offline. Production weights, harmonic
policy, cardiac band, windowing, and thresholds are untouched.
"""