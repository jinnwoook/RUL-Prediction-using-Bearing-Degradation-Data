"""
Analysis module for bearing vibration frequency analysis.

This module provides:
- frequency_analysis: STFT, envelope spectrum, and fault frequency detection
"""

from .frequency_analysis import (
    FrequencyAnalyzer,
    BearingFaultFrequencies,
    compute_envelope_stft,
    compute_envelope_psd,
    extract_target_frequency_amplitude,
)

__all__ = [
    "FrequencyAnalyzer",
    "BearingFaultFrequencies",
    "compute_envelope_stft",
    "compute_envelope_psd",
    "extract_target_frequency_amplitude",
]
