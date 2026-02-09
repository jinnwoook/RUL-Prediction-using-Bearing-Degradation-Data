"""
Feature extraction module for bearing vibration analysis.

This module provides wavelet-based and envelope-based feature extraction
for Remaining Useful Life (RUL) prediction of bearings.

Key Features:
- Wavelet decomposition (Daubechies db4) for D4/D5 RMS and entropy
- Envelope analysis for band-pass filtered signals
- Support for parallel processing of large datasets
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pywt

from .signal_processing import bandpass_filter, compute_envelope, compute_rms


@dataclass
class WaveletConfig:
    """Configuration for wavelet feature extraction."""
    wavelet: str = "db4"
    level: int = 5
    target_scales: Tuple[str, ...] = ("D4", "D5")


@dataclass
class EnvelopeConfig:
    """Configuration for envelope feature extraction."""
    fs: float = 25600
    lowcut: float = 1000
    highcut: float = 5000
    window_size: int = 12800  # 0.5 seconds at 25600 Hz


def extract_wavelet_params(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 5
) -> Dict[str, Optional[np.float32]]:
    """
    Extract wavelet-based parameters from a vibration signal.

    Uses Daubechies 4 (db4) wavelet decomposition to extract RMS and
    entropy values from D4 and D5 detail coefficients. These parameters
    are effective indicators for bearing fault progression.

    Args:
        signal: Input vibration signal (1D numpy array)
        wavelet: Wavelet type (default: "db4")
        level: Decomposition level (default: 5)

    Returns:
        Dictionary containing:
        - D4_RMS: RMS of D4 detail coefficients
        - D5_RMS: RMS of D5 detail coefficients
        - D5_Entropy: Shannon entropy of D5 detail coefficients

    Note:
        D4 and D5 scales correspond to frequency bands that capture
        bearing fault frequencies and their harmonics (140Hz, 280Hz, etc.)

    References:
        [1] Kumar et al. (2013), Wavelet transform for bearing condition
            monitoring and fault diagnosis: A review
        [2] Rafia Nishat Toma et al. (2020), Bearing Fault Classification
            of Induction Motors Using DWT and Ensemble ML Algorithms
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    param_dict: Dict[str, Optional[np.float32]] = {
        "D4_RMS": None,
        "D5_RMS": None,
        "D5_Entropy": None
    }

    for i in range(1, len(coeffs)):
        scale = f"D{level - i + 1}"
        if scale not in ["D4", "D5"]:
            continue

        detail = coeffs[i].astype(np.float32)
        rms = np.sqrt(np.mean(detail ** 2)).astype(np.float32)

        if scale == "D4":
            param_dict["D4_RMS"] = rms
        elif scale == "D5":
            # Compute Shannon entropy
            prob_density, _ = np.histogram(detail, bins=64, density=True)
            prob_density = prob_density[prob_density > 0]
            entropy = -np.sum(prob_density * np.log2(prob_density)).astype(np.float32)

            param_dict["D5_RMS"] = rms
            param_dict["D5_Entropy"] = entropy

    return param_dict


def extract_envelope_params(
    signal: np.ndarray,
    fs: float = 25600,
    lowcut: float = 1000,
    highcut: float = 5000,
    window_size: int = 12800
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract envelope-based parameters from a vibration signal.

    Applies band-pass filtering followed by Hilbert transform-based
    envelope analysis. The 1000-5000Hz band is selected because bearing
    fault frequencies and their harmonics are concentrated in this range.

    Args:
        signal: Input vibration signal (1D numpy array)
        fs: Sampling frequency in Hz (default: 25600)
        lowcut: Lower cutoff frequency (default: 1000)
        highcut: Upper cutoff frequency (default: 5000)
        window_size: Window size for RMS calculation (default: 12800)

    Returns:
        Tuple of:
        - bpf_rms: RMS of band-pass filtered signal per window
        - envelope_rms: RMS of envelope signal per window

    Note:
        These parameters show progressive increase as bearing damage
        develops, making them effective for RUL prediction.
    """
    signal = signal.astype(np.float32)
    signal -= np.mean(signal)

    # Apply band-pass filter
    filtered = bandpass_filter(signal, fs, lowcut, highcut)

    # Compute envelope
    envelope = compute_envelope(filtered)

    # Calculate windowed RMS
    n_windows = len(filtered) // window_size
    trimmed_filtered = filtered[:n_windows * window_size].reshape(-1, window_size)
    trimmed_envelope = envelope[:n_windows * window_size].reshape(-1, window_size)

    bpf_rms = np.sqrt(np.mean(trimmed_filtered ** 2, axis=1)).astype(np.float32)
    envelope_rms = np.sqrt(np.mean(trimmed_envelope ** 2, axis=1)).astype(np.float32)

    # Repeat values to match original signal length
    bpf_expanded = np.repeat(bpf_rms, window_size)
    envelope_expanded = np.repeat(envelope_rms, window_size)

    return bpf_expanded, envelope_expanded


class WaveletFeatureExtractor:
    """
    Class for extracting wavelet-based features from vibration signals.

    This extractor uses Daubechies 4 (db4) wavelet for decomposition,
    which is effective for capturing transient bearing fault signatures.

    Attributes:
        config: WaveletConfig instance with extraction parameters

    Example:
        >>> extractor = WaveletFeatureExtractor()
        >>> features = extractor.extract(signal, window_size=12800)
    """

    def __init__(self, config: Optional[WaveletConfig] = None):
        """
        Initialize the wavelet feature extractor.

        Args:
            config: WaveletConfig instance (default: None, uses defaults)
        """
        self.config = config or WaveletConfig()

    def extract(
        self,
        signal: np.ndarray,
        window_size: int = 12800
    ) -> Dict[str, np.ndarray]:
        """
        Extract wavelet features from signal windows.

        Args:
            signal: Input vibration signal
            window_size: Window size for feature extraction

        Returns:
            Dictionary with feature arrays (D4_RMS, D5_RMS, D5_Entropy)
        """
        n_windows = len(signal) // window_size
        trimmed = signal[:n_windows * window_size].reshape(-1, window_size)

        d4_rms_list: List[np.float32] = []
        d5_rms_list: List[np.float32] = []
        d5_entropy_list: List[np.float32] = []

        for segment in trimmed:
            params = extract_wavelet_params(
                segment,
                wavelet=self.config.wavelet,
                level=self.config.level
            )
            d4_rms_list.append(params["D4_RMS"])
            d5_rms_list.append(params["D5_RMS"])
            d5_entropy_list.append(params["D5_Entropy"])

        return {
            "D4_RMS": np.array(d4_rms_list, dtype=np.float32),
            "D5_RMS": np.array(d5_rms_list, dtype=np.float32),
            "D5_Entropy": np.array(d5_entropy_list, dtype=np.float32)
        }

    def extract_expanded(
        self,
        signal: np.ndarray,
        window_size: int = 12800
    ) -> Dict[str, np.ndarray]:
        """
        Extract wavelet features and expand to match signal length.

        Args:
            signal: Input vibration signal
            window_size: Window size for feature extraction

        Returns:
            Dictionary with expanded feature arrays
        """
        features = self.extract(signal, window_size)
        return {
            key: np.repeat(values, window_size)
            for key, values in features.items()
        }


class EnvelopeFeatureExtractor:
    """
    Class for extracting envelope-based features from vibration signals.

    Envelope analysis is effective for detecting bearing defects by
    revealing periodic impact patterns through amplitude demodulation.

    Attributes:
        config: EnvelopeConfig instance with extraction parameters

    Example:
        >>> extractor = EnvelopeFeatureExtractor()
        >>> bpf, envelope = extractor.extract(signal)
    """

    def __init__(self, config: Optional[EnvelopeConfig] = None):
        """
        Initialize the envelope feature extractor.

        Args:
            config: EnvelopeConfig instance (default: None, uses defaults)
        """
        self.config = config or EnvelopeConfig()

    def extract(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract envelope features from a signal.

        Args:
            signal: Input vibration signal

        Returns:
            Tuple of (BPF RMS, Envelope RMS) arrays
        """
        return extract_envelope_params(
            signal,
            fs=self.config.fs,
            lowcut=self.config.lowcut,
            highcut=self.config.highcut,
            window_size=self.config.window_size
        )


def extract_all_features(
    vib_data: Dict[str, np.ndarray],
    channels: List[str],
    fs: float = 25600,
    window_size: int = 12800,
    env_target_channel: str = "CH2",
    band_range: Tuple[float, float] = (1000, 5000)
) -> Dict[str, np.ndarray]:
    """
    Extract all features from multi-channel vibration data.

    Combines wavelet and envelope features for comprehensive
    bearing health monitoring.

    Args:
        vib_data: Dictionary mapping channel names to signal arrays
        channels: List of channel names to process
        fs: Sampling frequency in Hz
        window_size: Window size for feature extraction
        env_target_channel: Channel for envelope analysis (default: "CH2")
        band_range: Frequency band for envelope analysis

    Returns:
        Dictionary containing all extracted features
    """
    features: Dict[str, np.ndarray] = {}

    wavelet_extractor = WaveletFeatureExtractor()
    envelope_extractor = EnvelopeFeatureExtractor(
        EnvelopeConfig(fs=fs, lowcut=band_range[0], highcut=band_range[1], window_size=window_size)
    )

    # Extract wavelet features for all channels
    for ch in channels:
        signal = vib_data[ch]
        wavelet_features = wavelet_extractor.extract_expanded(signal, window_size)

        features[f"{ch}_D4_RMS"] = wavelet_features["D4_RMS"]
        features[f"{ch}_D5_RMS"] = wavelet_features["D5_RMS"]
        features[f"{ch}_D5_Entropy"] = wavelet_features["D5_Entropy"]

    # Extract envelope features for target channel
    if env_target_channel in vib_data:
        bpf, envelope = envelope_extractor.extract(vib_data[env_target_channel])
        features[f"{env_target_channel}_BPF_RMS"] = bpf
        features[f"{env_target_channel}_Envelope_RMS"] = envelope

    return features
