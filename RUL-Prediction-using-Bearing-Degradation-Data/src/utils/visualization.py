"""
Visualization utilities for bearing vibration analysis.

This module provides plotting functions for:
- Raw vibration signals
- Envelope spectra
- Fault frequency tracking
- Feature trends over time
- Training history
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from ..analysis.frequency_analysis import BearingFaultFrequencies


def plot_signal(
    signal: np.ndarray,
    fs: float = 25600,
    title: str = "Vibration Signal",
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a vibration signal in time domain.

    Args:
        signal: Input vibration signal array
        fs: Sampling frequency in Hz
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    time = np.arange(len(signal)) / fs

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, signal, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_envelope_spectrum(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    fault_freqs: Optional[BearingFaultFrequencies] = None,
    freq_limit: float = 500,
    title: str = "Envelope Spectrum",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot envelope spectrum with fault frequency markers.

    Args:
        frequencies: Frequency array from FFT
        amplitudes: Amplitude/PSD array
        fault_freqs: Bearing fault frequencies to mark
        freq_limit: Upper frequency limit for display
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot spectrum
    mask = frequencies <= freq_limit
    ax.plot(frequencies[mask], amplitudes[mask], 'k-', linewidth=0.8, label='Spectrum')

    # Mark fault frequencies
    if fault_freqs:
        colors = ['r', 'g', 'b', 'm']
        for (name, freq), color in zip(fault_freqs.as_dict().items(), colors):
            if freq <= freq_limit:
                ax.axvline(x=freq, color=color, linestyle='--', alpha=0.7, label=f'{name} ({freq}Hz)')
                ax.text(freq + 2, ax.get_ylim()[1] * 0.9, name,
                       color=color, fontsize=9, rotation=90, va='top')

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, freq_limit)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_fault_frequency_amplitudes(
    time_axis: np.ndarray,
    fault_amplitudes: Dict[str, np.ndarray],
    band_label: str = "",
    title: str = "Fault Frequency Amplitudes Over Time",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot fault frequency amplitudes over time.

    Args:
        time_axis: Time values array
        fault_amplitudes: Dictionary mapping fault names to amplitude arrays
        band_label: Frequency band label
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    n_faults = len(fault_amplitudes)
    fig, axes = plt.subplots(n_faults, 1, figsize=figsize, sharex=True)

    if n_faults == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_faults))

    for ax, (fault_name, amplitudes), color in zip(axes, fault_amplitudes.items(), colors):
        ax.plot(time_axis, amplitudes, color=color, linewidth=0.8)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{fault_name} {band_label}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_feature_trends(
    time_axis: np.ndarray,
    features: Dict[str, np.ndarray],
    title: str = "Feature Trends",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple feature trends over time.

    Args:
        time_axis: Time values array
        features: Dictionary mapping feature names to value arrays
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)

    if n_features == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, features.items()):
        ax.plot(time_axis, values, linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation loss history.

    Args:
        history: Dictionary with 'train_loss' and optionally 'val_loss'
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)

    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_channel_comparison(
    vib_data: Dict[str, np.ndarray],
    channels: List[str],
    fs: float = 25600,
    title: str = "Channel Comparison",
    figsize: Tuple[int, int] = (14, 10),
    max_samples: int = 256000,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple channels for comparison.

    Args:
        vib_data: Dictionary mapping channel names to signal arrays
        channels: List of channel names to plot
        fs: Sampling frequency in Hz
        title: Plot title
        figsize: Figure size
        max_samples: Maximum samples to plot
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)

    if n_channels == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        if ch in vib_data:
            signal = vib_data[ch][:max_samples]
            time = np.arange(len(signal)) / fs

            ax.plot(time, signal, linewidth=0.3)
            ax.set_ylabel(ch)
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction_results(
    filenames: List[str],
    actual_rul: List[float],
    predicted_rul: List[float],
    title: str = "RUL Prediction Results",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot actual vs predicted RUL values.

    Args:
        filenames: List of sample identifiers
        actual_rul: List of actual RUL values
        predicted_rul: List of predicted RUL values
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(filenames))
    width = 0.35

    ax.bar(x - width/2, actual_rul, width, label='Actual RUL', color='steelblue')
    ax.bar(x + width/2, predicted_rul, width, label='Predicted RUL', color='coral')

    ax.set_xlabel("Sample")
    ax.set_ylabel("RUL (seconds)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(filenames, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_analysis_report(
    signal: np.ndarray,
    channel: str,
    fs: float,
    fault_freqs: BearingFaultFrequencies,
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    save_dir: str = "./"
) -> None:
    """
    Create a comprehensive analysis report with multiple plots.

    Args:
        signal: Input vibration signal
        channel: Channel name
        fs: Sampling frequency
        fault_freqs: Bearing fault frequencies
        frequencies: Frequency array for spectrum
        amplitudes: Amplitude array for spectrum
        save_dir: Directory to save plots
    """
    import os

    # Time domain plot
    plot_signal(
        signal,
        fs=fs,
        title=f"{channel} - Time Domain Signal",
        save_path=os.path.join(save_dir, f"{channel}_time_domain.png")
    )

    # Envelope spectrum
    plot_envelope_spectrum(
        frequencies,
        amplitudes,
        fault_freqs=fault_freqs,
        title=f"{channel} - Envelope Spectrum",
        save_path=os.path.join(save_dir, f"{channel}_envelope_spectrum.png")
    )

    plt.close('all')
    print(f"Analysis report saved to {save_dir}")
