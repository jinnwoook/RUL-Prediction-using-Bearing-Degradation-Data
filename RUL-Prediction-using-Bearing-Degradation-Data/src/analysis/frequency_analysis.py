"""
Frequency analysis module for bearing fault detection.

This module provides tools for analyzing bearing vibration signals
in the frequency domain, including:
- STFT-based envelope spectrum analysis
- Bearing fault frequency extraction (BPFI, BPFO, BSF, FTF)
- Time-frequency feature extraction
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from ..preprocessing.signal_processing import bandpass_filter, highpass_filter


@dataclass
class BearingFaultFrequencies:
    """
    Bearing characteristic fault frequencies.

    These frequencies are determined by bearing geometry and
    shaft rotation speed.

    Attributes:
        BPFI: Ball Pass Frequency Inner race (inner race defect)
        BPFO: Ball Pass Frequency Outer race (outer race defect)
        BSF: Ball Spin Frequency (rolling element defect)
        FTF: Fundamental Train Frequency (cage defect)
    """
    BPFI: float = 140.0
    BPFO: float = 93.0
    BSF: float = 73.0
    FTF: float = 6.7

    def as_dict(self) -> Dict[str, float]:
        """Return fault frequencies as dictionary."""
        return {
            "BPFI": self.BPFI,
            "BPFO": self.BPFO,
            "BSF": self.BSF,
            "FTF": self.FTF
        }


@dataclass
class FrequencyBand:
    """Frequency band definition."""
    lowcut: float
    highcut: float

    def __str__(self) -> str:
        return f"{int(self.lowcut)}-{int(self.highcut)}Hz"


def compute_envelope_stft(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute envelope spectrum using STFT-like processing.

    Process:
    1. Remove DC offset
    2. Apply band-pass filter
    3. Compute Hilbert envelope
    4. Perform FFT on envelope

    Args:
        signal: Input vibration signal
        fs: Sampling frequency in Hz
        lowcut: Lower cutoff frequency for band-pass filter
        highcut: Upper cutoff frequency for band-pass filter

    Returns:
        Tuple of (frequency array, amplitude spectrum)
    """
    # Remove DC offset
    signal = signal - np.mean(signal)

    # Band-pass filter
    filtered = bandpass_filter(signal, fs, lowcut, highcut)

    # Hilbert envelope
    envelope = np.abs(hilbert(filtered))

    # FFT
    n = len(envelope)
    yf = np.fft.rfft(envelope)
    xf = np.fft.rfftfreq(n, 1/fs)
    amplitude = np.abs(yf) / n * 2

    return xf, amplitude


def compute_envelope_psd(
    signal: np.ndarray,
    fs: float,
    cutoff: float = 25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute envelope Power Spectral Density (PSD).

    Uses high-pass filtering followed by Hilbert transform
    to extract the envelope, then computes the PSD.

    Args:
        signal: Input vibration signal
        fs: Sampling frequency in Hz
        cutoff: High-pass filter cutoff (default: 25 Hz)

    Returns:
        Tuple of (frequency array, PSD values)
    """
    # High-pass filter
    filtered = highpass_filter(signal, fs, cutoff=cutoff)

    # Hilbert envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    # PSD via FFT
    n = len(envelope)
    fft_vals = np.fft.rfft(envelope)
    fft_freqs = np.fft.rfftfreq(n, d=1/fs)
    psd = (np.abs(fft_vals) ** 2) / n

    return fft_freqs, psd


def extract_target_frequency_amplitude(
    signal: np.ndarray,
    fs: float,
    target_freq: float,
    lowcut: float,
    highcut: float
) -> float:
    """
    Extract amplitude at a specific target frequency from envelope spectrum.

    Args:
        signal: Input vibration signal
        fs: Sampling frequency in Hz
        target_freq: Target frequency to extract (e.g., BPFI)
        lowcut: Lower cutoff for band-pass filter
        highcut: Upper cutoff for band-pass filter

    Returns:
        Amplitude at the target frequency
    """
    xf, amplitude = compute_envelope_stft(signal, fs, lowcut, highcut)
    idx_target = np.argmin(np.abs(xf - target_freq))
    return amplitude[idx_target]


def extract_stft_features(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float
) -> Dict[str, float]:
    """
    Extract frequency domain features from envelope spectrum.

    Features:
    - Dominant frequency (excluding DC)
    - Maximum amplitude
    - Total band energy

    Args:
        signal: Input vibration signal
        fs: Sampling frequency in Hz
        lowcut: Lower cutoff frequency
        highcut: Upper cutoff frequency

    Returns:
        Dictionary with extracted features
    """
    xf, amplitude = compute_envelope_stft(signal, fs, lowcut, highcut)

    # Dominant frequency (excluding DC)
    dom_freq = xf[1:][np.argmax(amplitude[1:])]

    # Maximum amplitude
    max_amp = np.max(amplitude)

    # Total band energy
    band_energy = np.sum(amplitude ** 2)

    return {
        "dominant_freq": dom_freq,
        "max_amplitude": max_amp,
        "band_energy": band_energy
    }


class FrequencyAnalyzer:
    """
    Comprehensive frequency analyzer for bearing vibration signals.

    Provides methods for extracting fault-related frequency features
    and visualizing frequency spectra.

    Attributes:
        fs: Sampling frequency in Hz
        fault_freqs: Bearing fault frequencies
        band_ranges: List of frequency bands to analyze

    Example:
        >>> analyzer = FrequencyAnalyzer(fs=25600)
        >>> features = analyzer.extract_all_features(signal)
    """

    DEFAULT_BAND_RANGES = [
        FrequencyBand(50, 500),
        FrequencyBand(100, 800),
        FrequencyBand(200, 1000),
        FrequencyBand(300, 1500),
    ]

    def __init__(
        self,
        fs: float = 25600,
        fault_freqs: Optional[BearingFaultFrequencies] = None,
        band_ranges: Optional[List[FrequencyBand]] = None,
        window_size: Optional[int] = None
    ):
        """
        Initialize the frequency analyzer.

        Args:
            fs: Sampling frequency in Hz
            fault_freqs: Bearing fault frequencies (default: uses standard)
            band_ranges: Frequency bands to analyze
            window_size: Window size for windowed analysis
        """
        self.fs = fs
        self.fault_freqs = fault_freqs or BearingFaultFrequencies()
        self.band_ranges = band_ranges or self.DEFAULT_BAND_RANGES
        self.window_size = window_size or int(fs * 0.5)  # 0.5 second default

    def extract_fault_frequency_amplitudes(
        self,
        signal: np.ndarray,
        band: FrequencyBand
    ) -> Dict[str, float]:
        """
        Extract amplitudes at all fault frequencies.

        Args:
            signal: Input vibration signal
            band: Frequency band for analysis

        Returns:
            Dictionary mapping fault name to amplitude
        """
        results = {}
        for name, freq in self.fault_freqs.as_dict().items():
            amp = extract_target_frequency_amplitude(
                signal, self.fs, freq, band.lowcut, band.highcut
            )
            results[name] = amp
        return results

    def extract_all_features(
        self,
        signal: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract all frequency features across all bands.

        Args:
            signal: Input vibration signal

        Returns:
            Nested dictionary with features for each band
        """
        all_features = {}

        for band in self.band_ranges:
            band_key = str(band)

            # Fault frequency amplitudes
            fault_amps = self.extract_fault_frequency_amplitudes(signal, band)

            # General STFT features
            stft_features = extract_stft_features(
                signal, self.fs, band.lowcut, band.highcut
            )

            all_features[band_key] = {
                **{f"{name}_amp": amp for name, amp in fault_amps.items()},
                **stft_features
            }

        return all_features

    def analyze_windowed(
        self,
        signal: np.ndarray
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Perform windowed frequency analysis.

        Args:
            signal: Input vibration signal

        Returns:
            Dictionary with time-series of features for each band
        """
        results: Dict[str, List[Dict[str, float]]] = {
            str(band): [] for band in self.band_ranges
        }

        for start in range(0, len(signal) - self.window_size, self.window_size):
            segment = signal[start:start + self.window_size]

            for band in self.band_ranges:
                band_key = str(band)

                # Fault frequencies
                fault_amps = self.extract_fault_frequency_amplitudes(segment, band)

                # STFT features
                stft_features = extract_stft_features(
                    segment, self.fs, band.lowcut, band.highcut
                )

                window_features = {
                    "time": start / self.fs,
                    **{f"{name}_amp": amp for name, amp in fault_amps.items()},
                    **stft_features
                }

                results[band_key].append(window_features)

        return results

    def get_time_axis(self, signal_length: int) -> np.ndarray:
        """
        Get time axis for windowed analysis.

        Args:
            signal_length: Length of the signal in samples

        Returns:
            Array of time values for each window
        """
        return np.array([
            start / self.fs
            for start in range(0, signal_length - self.window_size, self.window_size)
        ])


def batch_frequency_analysis(
    vib_data_dict: Dict[str, np.ndarray],
    channels: List[str],
    analyzer: FrequencyAnalyzer
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """
    Perform frequency analysis on multiple channels.

    Args:
        vib_data_dict: Dictionary mapping channel names to signal arrays
        channels: List of channel names to analyze
        analyzer: FrequencyAnalyzer instance

    Returns:
        Nested dictionary with analysis results for each channel
    """
    results = {}

    for channel in channels:
        if channel in vib_data_dict:
            signal = vib_data_dict[channel]
            results[channel] = analyzer.analyze_windowed(signal)

    return results
