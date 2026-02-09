# Bearing RUL Prediction using CNN-LSTM

KSPHM-KIMM 2025 Bearing Remaining Useful Life (RUL) Prediction Competition

## Overview

This project implements a deep learning-based approach for predicting the Remaining Useful Life (RUL) of bearings using vibration signal analysis. The solution combines traditional signal processing techniques with a CNN-LSTM neural network architecture.

## Key Features

- **Wavelet-based Feature Extraction**: Uses Daubechies 4 (db4) wavelet decomposition to extract D4/D5 RMS and entropy features
- **Envelope Analysis**: Band-pass filtering (1000-5000Hz) with Hilbert transform for fault frequency detection
- **CNN-LSTM Model**: Hybrid architecture combining CNN for local feature extraction and LSTM for temporal pattern learning
- **Bearing Fault Frequencies**: Analysis of BPFI, BPFO, BSF, and FTF fault signatures

## Project Structure

```
RUL-Prediction-using-Bearing-Degradation-Data/
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── feature_extraction.py    # Wavelet and envelope feature extraction
│   │   ├── signal_processing.py     # Band-pass filtering, Hilbert transform
│   │   └── data_loader.py           # TDMS file loading utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_lstm.py              # CNN-LSTM model architecture
│   │   └── train.py                 # Training utilities
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── frequency_analysis.py    # STFT and fault frequency analysis
│   └── utils/
│       ├── __init__.py
│       └── visualization.py         # Plotting functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── configs/
│   └── config.yaml
├── weights/
│   └── best_model_0.66.pth
├── requirements.txt
└── setup.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/RUL-Prediction-using-Bearing-Degradation-Data.git
cd RUL-Prediction-using-Bearing-Degradation-Data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### 1. Feature Extraction

```python
from src.preprocessing import TDMSDataLoader, WaveletFeatureExtractor, EnvelopeFeatureExtractor

# Load TDMS data
loader = TDMSDataLoader(base_dir="./data")
vib_data, op_data = loader.load_file("path/to/file.tdms")

# Extract wavelet features
wavelet_extractor = WaveletFeatureExtractor()
wavelet_features = wavelet_extractor.extract(vib_data["CH2"], window_size=12800)

# Extract envelope features
envelope_extractor = EnvelopeFeatureExtractor()
bpf_rms, envelope_rms = envelope_extractor.extract(vib_data["CH2"])
```

### 2. Model Training

```python
from src.models import CNNLSTM, train_model, TrainingConfig

# Configure training
config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-3,
    epochs=100
)

# Train model
model, history = train_model(
    train_data_dir="./data/preprocessed",
    rul_labels=rul_dict,
    model_save_path="./weights/model.pth",
    config=config
)
```

### 3. Inference

```python
from src.models import load_model, evaluate_model
from torch.utils.data import DataLoader

# Load trained model
model = load_model("./weights/best_model_0.66.pth", input_channels=15)

# Run inference
predictions = evaluate_model(model, test_loader)
```

### 4. Frequency Analysis

```python
from src.analysis import FrequencyAnalyzer, BearingFaultFrequencies

# Initialize analyzer
analyzer = FrequencyAnalyzer(
    fs=25600,
    fault_freqs=BearingFaultFrequencies()
)

# Analyze signal
features = analyzer.extract_all_features(signal)
```

## Data Requirements

- **Format**: TDMS files with vibration data
- **Channels**: CH1, CH2, CH3, CH4 (vibration sensors)
- **Sampling Rate**: 25,600 Hz
- **Sample Duration**: 10 seconds (256,000 samples per file)

## Model Architecture

```
Input (batch, time_steps, 15 channels)
    │
    ▼
Conv1D (64 filters, kernel=5) + ReLU
    │
    ▼
MaxPool1d (kernel=2)
    │
    ▼
LSTM (hidden_size=64)
    │
    ▼
FC (64 → 32) + ReLU
    │
    ▼
FC (32 → 1)
    │
    ▼
Output (RUL prediction)
```

## Feature Selection Rationale

### Wavelet Features (D4_RMS, D5_RMS, D5_Entropy)
- D4/D5 scales capture bearing fault frequencies and harmonics (140Hz, 280Hz, etc.)
- RMS provides robust energy indicator with low noise sensitivity
- Entropy captures signal complexity increase during fault progression

### Envelope Features (1000-5000Hz BPF + Envelope RMS)
- Band-pass filtering isolates fault-related frequency content
- Envelope analysis via Hilbert transform extracts amplitude modulation
- CH2 shows best correlation with degradation across all training sets

## Results

- Best model validation score: 0.66
- Log-transformed RUL prediction with expm1 inverse transformation

## References

1. Kumar et al. (2013), "Wavelet transform for bearing condition monitoring and fault diagnosis: A review"
2. Rafia Nishat Toma et al. (2020), "Bearing Fault Classification of Induction Motors Using DWT and Ensemble ML Algorithms"
3. C K E Nizwana et al. (2016), "A wavelet decomposition analysis of vibration signal for bearing fault detection"

## License

MIT License

## Team

KIST Bearing Team - KSPHM-KIMM 2025 Data Challenge
