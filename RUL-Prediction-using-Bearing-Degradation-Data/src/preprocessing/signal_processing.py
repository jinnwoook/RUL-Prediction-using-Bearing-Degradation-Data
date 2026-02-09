"""
Signal processing utilities for bearing vibration analysis.

This module provides filtering and envelope extraction functions
for vibration signal processing.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth band-pass filter to the signal.

    Band-pass filtering isolates the frequency range of interest,
    which is essential for extracting bearing fault frequencies
    (typically 1000-5000Hz for bearing defects).

    Args:
        data: Input signal array
        fs: Sampling frequency in Hz
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        order: Filter order (default: 4)

    Returns:
        Filtered signal array

    Example:
        >>> signal = np.random.randn(25600)
        >>> filtered = bandpass_filter(signal, fs=25600, lowcut=1000, highcut=5000)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def highpass_filter(
    data: np.ndarray,
    fs: float,
    cutoff: float = 25,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth high-pass filter to the signal.

    High-pass filtering removes low-frequency noise and DC offset,
    which is useful for envelope analysis preprocessing.

    Args:
        data: Input signal array
        fs: Sampling frequency in Hz
        cutoff: Cutoff frequency in Hz (default: 25)
        order: Filter order (default: 4)

    Returns:
        Filtered signal array
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)


def compute_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Compute the envelope of a signal using Hilbert transform.

    Envelope analysis is effective for detecting bearing defects
    by extracting the amplitude modulation caused by periodic
    impacts during fault progression.

    Args:
        signal: Input signal array

    Returns:
        Envelope (analytic signal magnitude) array

    Note:
        The Hilbert transform-based envelope analysis is particularly
        useful for identifying low-energy periodic impacts in noisy
        vibration signals.
    """
    analytic_signal = hilbert(signal)
    return np.abs(analytic_signal)


def compute_envelope_spectrum(
    signal: np.ndarray,
    fs: float,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the envelope spectrum of a signal.

    The envelope spectrum reveals the characteristic fault frequencies
    that may be masked in the raw vibration spectrum.

    Args:
        signal: Input signal array
        fs: Sampling frequency in Hz
        lowcut: Optional lower cutoff for band-pass filtering
        highcut: Optional upper cutoff for band-pass filtering

    Returns:
        Tuple of (frequencies, amplitude spectrum)
    """
    # Remove DC offset
    signal = signal - np.mean(signal)

    # Apply band-pass filter if specified
    if lowcut is not None and highcut is not None:
        signal = bandpass_filter(signal, fs, lowcut, highcut)

    # Compute envelope
    envelope = compute_envelope(signal)

    # FFT of envelope
    n = len(envelope)
    yf = np.fft.rfft(envelope)
    xf = np.fft.rfftfreq(n, 1/fs)
    amplitude = np.abs(yf) / n * 2

    return xf, amplitude


def compute_rms(signal: np.ndarray) -> float:
    """
    Compute the Root Mean Square (RMS) of a signal.

    RMS is a robust indicator of signal energy and increases
    progressively as bearing damage develops.

    Args:
        signal: Input signal array

    Returns:
        RMS value as float32
    """
    return np.sqrt(np.mean(signal ** 2)).astype(np.float32)


def segment_signal(
    signal: np.ndarray,
    window_size: int,
    overlap: int = 0
) -> np.ndarray:
    """
    Segment a signal into windows.

    Args:
        signal: Input signal array
        window_size: Number of samples per window
        overlap: Number of overlapping samples (default: 0)

    Returns:
        2D array of shape (n_segments, window_size)
    """
    step = window_size - overlap
    n_segments = (len(signal) - window_size) // step + 1

    segments = np.zeros((n_segments, window_size), dtype=signal.dtype)
    for i in range(n_segments):
        start = i * step
        segments[i] = signal[start:start + window_size]

    return segments
