"""
Data loading utilities for TDMS vibration files.

This module provides functions and classes for loading TDMS format
vibration data files used in the KSPHM-KIMM bearing challenge.
"""

import os
import re
import glob
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from nptdms import TdmsFile


@dataclass
class DataConfig:
    """Configuration for data loading."""
    fs: float = 25600
    channels: Tuple[str, ...] = ("CH1", "CH2", "CH3", "CH4")
    expected_row_count: int = 256000


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from TDMS filename.

    TDMS files are named with a 14-digit timestamp in format YYYYMMDDHHMMSS.

    Args:
        filename: TDMS filename

    Returns:
        Parsed datetime object or None if not found

    Example:
        >>> extract_timestamp_from_filename("KIMM_20160321050739.tdms")
        datetime.datetime(2016, 3, 21, 5, 7, 39)
    """
    match = re.search(r"(\d{14})", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    return None


def safe_find_operation_value(
    op_dict: Dict[str, Any],
    key_keyword: str
) -> float:
    """
    Safely find an operation parameter value by keyword.

    Searches for a key containing the keyword (case-insensitive)
    in the operation parameter dictionary.

    Args:
        op_dict: Dictionary of operation parameters
        key_keyword: Keyword to search for in keys

    Returns:
        Parameter value or NaN if not found
    """
    for key in op_dict:
        if key_keyword.lower() in key.lower():
            return op_dict[key]
    print(f"[Warning] Operation parameter with keyword '{key_keyword}' not found")
    return np.nan


def load_tdms_file(
    filepath: str,
    config: Optional[DataConfig] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Load a single TDMS file and extract vibration and operation data.

    Args:
        filepath: Path to the TDMS file
        config: DataConfig instance (default: None, uses defaults)

    Returns:
        Tuple of:
        - vib_data: Dictionary mapping channel names to vibration arrays
        - op_data: Dictionary of operation parameters (torque, temperature)

    Example:
        >>> vib, op = load_tdms_file("path/to/file.tdms")
        >>> print(vib["CH1"].shape)
        (256000,)
    """
    config = config or DataConfig()

    tdms = TdmsFile.read(filepath)

    # Get group names
    groups = tdms.groups()
    vib_group = groups[0].name
    op_group = groups[1].name if len(groups) > 1 else None

    # Extract vibration data
    row_count = len(tdms[vib_group]["CH1"].data)
    vib_data: Dict[str, np.ndarray] = {}

    for ch in config.channels:
        vib_data[ch] = tdms[vib_group][ch].data[:row_count].astype(np.float32)

    # Extract operation parameters
    op_data: Dict[str, float] = {}
    if op_group:
        op_dict = {ch.name.strip(): ch.data[0] for ch in tdms[op_group].channels()}
        op_data["torque"] = safe_find_operation_value(op_dict, "torque")
        op_data["temp_front"] = safe_find_operation_value(op_dict, "front")
        op_data["temp_rear"] = safe_find_operation_value(op_dict, "rear")

    return vib_data, op_data


def load_tdms_segments(
    folder_path: str,
    config: Optional[DataConfig] = None
) -> Tuple[List[pd.DataFrame], List[datetime]]:
    """
    Load all TDMS files from a folder as segments.

    Args:
        folder_path: Path to folder containing TDMS files
        config: DataConfig instance

    Returns:
        Tuple of:
        - segments: List of DataFrames with vibration data
        - timestamps: List of timestamps extracted from filenames
    """
    config = config or DataConfig()
    segments: List[pd.DataFrame] = []
    timestamps: List[datetime] = []

    tdms_files = sorted(glob.glob(os.path.join(folder_path, "*.tdms")))

    for filepath in tdms_files:
        tdms = TdmsFile.read(filepath)
        vib_group = tdms.groups()[0].name

        vib_data = {ch.name: ch.data for ch in tdms[vib_group].channels()}
        df = pd.DataFrame(vib_data)
        segments.append(df)

        ts = extract_timestamp_from_filename(os.path.basename(filepath))
        if ts:
            timestamps.append(ts)

    return segments, timestamps


class TDMSDataLoader:
    """
    Data loader class for TDMS vibration files.

    Provides methods for loading individual files or entire datasets,
    with support for parallel processing.

    Attributes:
        config: DataConfig instance
        base_dir: Base directory for data files

    Example:
        >>> loader = TDMSDataLoader(base_dir="./data")
        >>> df = loader.load_and_process_file("path/to/file.tdms")
    """

    def __init__(
        self,
        base_dir: str = "./data",
        config: Optional[DataConfig] = None
    ):
        """
        Initialize the TDMS data loader.

        Args:
            base_dir: Base directory for data files
            config: DataConfig instance
        """
        self.config = config or DataConfig()
        self.base_dir = base_dir

    def load_file(
        self,
        filepath: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Load a single TDMS file.

        Args:
            filepath: Path to the TDMS file

        Returns:
            Tuple of (vibration data dict, operation data dict)
        """
        return load_tdms_file(filepath, self.config)

    def load_and_process_file(
        self,
        filepath: str,
        feature_extractor=None
    ) -> pd.DataFrame:
        """
        Load a TDMS file and create a processed DataFrame.

        Args:
            filepath: Path to the TDMS file
            feature_extractor: Optional feature extractor to apply

        Returns:
            DataFrame with raw signals and extracted features
        """
        vib_data, op_data = self.load_file(filepath)
        row_count = len(vib_data["CH1"])

        # Create time axis
        timestamps = (np.arange(0, row_count) / self.config.fs).astype(np.float32)

        # Build DataFrame
        df = pd.DataFrame({
            "Time (s)": timestamps,
            **{ch: vib_data[ch] for ch in self.config.channels},
            "Torque[Nm]": np.repeat(op_data.get("torque", np.nan), row_count).astype(np.float32),
            "TC SP Front[C]": np.repeat(op_data.get("temp_front", np.nan), row_count).astype(np.float32),
            "TC SP Rear[C]": np.repeat(op_data.get("temp_rear", np.nan), row_count).astype(np.float32),
        })

        return df

    def get_dataset_paths(
        self,
        dataset_type: str = "train",
        indices: Optional[List[int]] = None
    ) -> Dict[int, List[str]]:
        """
        Get all TDMS file paths for a dataset.

        Args:
            dataset_type: "train" or "validation"
            indices: List of dataset indices (e.g., [1, 2, 3])

        Returns:
            Dictionary mapping index to list of file paths
        """
        if indices is None:
            indices = list(range(1, 9)) if dataset_type == "train" else list(range(1, 7))

        base_path = os.path.join(self.base_dir, f"{dataset_type.capitalize()} Set")
        paths: Dict[int, List[str]] = {}

        for idx in indices:
            folder = os.path.join(base_path, f"{dataset_type.capitalize()}{idx}")
            if os.path.exists(folder):
                paths[idx] = sorted(glob.glob(os.path.join(folder, "*.tdms")))

        return paths

    def concatenate_channel_data(
        self,
        segments: List[pd.DataFrame],
        channel: str
    ) -> np.ndarray:
        """
        Concatenate channel data from multiple segments.

        Args:
            segments: List of DataFrames with segment data
            channel: Channel name to concatenate

        Returns:
            Concatenated numpy array
        """
        return np.concatenate([seg[channel].values for seg in segments])


class PreprocessedDatasetLoader:
    """
    Loader for preprocessed CSV datasets.

    Used for loading feature-extracted data for model training/inference.

    Attributes:
        root_dir: Root directory containing preprocessed data
        normalization_params: Parameters for feature normalization
    """

    # Normalization parameters from validation data
    DEFAULT_NORM_PARAMS = {
        "torque_mean": -6.08,
        "torque_std": 1.68,
        "temp1_mean": 102.80,
        "temp1_std": 12.30,
        "temp2_mean": 115.0,
        "temp2_std": 16.53
    }

    def __init__(
        self,
        root_dir: str,
        normalization_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the preprocessed dataset loader.

        Args:
            root_dir: Root directory with preprocessed CSV files
            normalization_params: Normalization parameters (default: validation-based)
        """
        self.root_dir = root_dir
        self.normalization_params = normalization_params or self.DEFAULT_NORM_PARAMS

    def load_csv(
        self,
        filepath: str,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Load a preprocessed CSV file.

        Args:
            filepath: Path to CSV file
            normalize: Whether to apply normalization

        Returns:
            DataFrame with loaded data
        """
        df = pd.read_csv(filepath)

        # Remove time column if present
        if "Time (s)" in df.columns:
            df.drop(columns=["Time (s)"], inplace=True)

        if normalize:
            df = self._normalize(df)

        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mean-std normalization to operational parameters.

        Args:
            df: Input DataFrame

        Returns:
            Normalized DataFrame
        """
        params = self.normalization_params

        if 'Torque[Nm]' in df.columns:
            df['Torque[Nm]'] = (df['Torque[Nm]'] - params["torque_mean"]) / params["torque_std"]
        if 'TC SP Front[C]' in df.columns:
            df['TC SP Front[C]'] = (df['TC SP Front[C]'] - params["temp1_mean"]) / params["temp1_std"]
        if 'TC SP Rear[C]' in df.columns:
            df['TC SP Rear[C]'] = (df['TC SP Rear[C]'] - params["temp2_mean"]) / params["temp2_std"]

        return df

    def load_all_from_folder(
        self,
        normalize: bool = True
    ) -> List[Tuple[pd.DataFrame, str]]:
        """
        Load all CSV files from the root directory.

        Args:
            normalize: Whether to apply normalization

        Returns:
            List of (DataFrame, filename) tuples
        """
        results: List[Tuple[pd.DataFrame, str]] = []

        for folder in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith(".csv"):
                    filepath = os.path.join(folder_path, fname)
                    df = self.load_csv(filepath, normalize)
                    results.append((df, fname))

        return results
