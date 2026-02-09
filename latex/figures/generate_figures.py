#!/usr/bin/env python3
"""
논문용 그림 생성 스크립트
KSPHM-KIMM 2025 베어링 RUL 예측 프로젝트
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.patches import ConnectionPatch
import matplotlib.lines as mlines
import numpy as np
from scipy import signal
from scipy.signal import hilbert

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# 색상 팔레트
COLORS = {
    'primary': '#2E86AB',      # 파란색
    'secondary': '#A23B72',    # 보라색
    'accent': '#F18F01',       # 주황색
    'success': '#C73E1D',      # 빨간색
    'dark': '#3C3C3C',         # 진회색
    'light': '#E8E8E8',        # 연회색
    'highlight': '#FFD700',    # 금색
    'cnn': '#4ECDC4',          # 청록색
    'lstm': '#FF6B6B',         # 코랄
    'ensemble': '#95E1D3',     # 민트
}


def create_rounded_box(ax, x, y, width, height, text, color, fontsize=9,
                       text_color='white', alpha=1.0, linewidth=1.5):
    """둥근 모서리 박스 생성"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='black',
                         linewidth=linewidth, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', wrap=True)
    return box


def draw_arrow(ax, start, end, color='black', style='->', lw=1.5):
    """화살표 그리기"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# =============================================================================
# 그림 1: 전체 방법론 파이프라인
# =============================================================================
def create_methodology_pipeline():
    """전체 방법론 파이프라인 다이어그램"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 제목
    ax.text(7, 7.5, 'Bearing RUL Prediction Pipeline', fontsize=16,
            ha='center', fontweight='bold', color=COLORS['dark'])

    # 1. 원본 데이터 (왼쪽)
    create_rounded_box(ax, 1.2, 4, 1.8, 1.2, 'TDMS\nRaw Data',
                       COLORS['dark'], fontsize=10)
    ax.text(1.2, 3.1, 'Vibration (25.6kHz)\n4 Channels',
            ha='center', va='top', fontsize=7, color=COLORS['dark'])

    # 화살표 1
    draw_arrow(ax, (2.2, 4), (3.0, 4), COLORS['dark'])

    # 2. 신호처리 블록 (중앙-왼쪽)
    # 신호처리 박스 배경
    signal_bg = FancyBboxPatch((2.8, 2.0), 3.4, 4.0,
                                boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor='#F0F8FF', edgecolor=COLORS['primary'],
                                linewidth=2, alpha=0.5)
    ax.add_patch(signal_bg)
    ax.text(4.5, 5.7, 'Signal Processing', fontsize=11, ha='center',
            fontweight='bold', color=COLORS['primary'])

    # BPF
    create_rounded_box(ax, 4.5, 5.0, 2.4, 0.7, 'Band-Pass Filter\n(1000-5000Hz)',
                       COLORS['primary'], fontsize=8)

    # Envelope
    create_rounded_box(ax, 4.5, 4.0, 2.4, 0.7, 'Envelope Analysis\n(Hilbert Transform)',
                       COLORS['primary'], fontsize=8)

    # Wavelet
    create_rounded_box(ax, 4.5, 3.0, 2.4, 0.7, 'Wavelet Transform\n(Db4, Level 5)',
                       COLORS['primary'], fontsize=8)

    # 화살표들
    draw_arrow(ax, (4.5, 4.55), (4.5, 4.35), COLORS['primary'])
    draw_arrow(ax, (4.5, 3.55), (4.5, 3.35), COLORS['primary'])

    # 화살표 2 (신호처리 -> 특징추출)
    draw_arrow(ax, (6.3, 4), (7.0, 4), COLORS['dark'])

    # 3. 특징 추출 블록
    feature_bg = FancyBboxPatch((6.8, 2.5), 2.0, 3.0,
                                 boxstyle="round,pad=0.02,rounding_size=0.2",
                                 facecolor='#FFF8E7', edgecolor=COLORS['accent'],
                                 linewidth=2, alpha=0.5)
    ax.add_patch(feature_bg)
    ax.text(7.8, 5.2, 'Features', fontsize=11, ha='center',
            fontweight='bold', color=COLORS['accent'])

    # 특징들
    features = ['BPF_RMS', 'Envelope_RMS', 'D4_RMS', 'D5_RMS', 'D5_Entropy']
    for i, feat in enumerate(features):
        y_pos = 4.6 - i * 0.45
        create_rounded_box(ax, 7.8, y_pos, 1.6, 0.35, feat,
                          COLORS['accent'], fontsize=7)

    # 화살표 3 (특징 -> 모델)
    draw_arrow(ax, (8.9, 4), (9.5, 4), COLORS['dark'])

    # 4. 모델 블록 (CNN-LSTM)
    model_bg = FancyBboxPatch((9.3, 2.0), 2.8, 4.0,
                               boxstyle="round,pad=0.02,rounding_size=0.2",
                               facecolor='#E8F5E9', edgecolor=COLORS['cnn'],
                               linewidth=2, alpha=0.5)
    ax.add_patch(model_bg)
    ax.text(10.7, 5.7, 'Deep Learning Model', fontsize=11, ha='center',
            fontweight='bold', color='#2E7D32')

    # CNN
    create_rounded_box(ax, 10.7, 5.0, 2.2, 0.7, 'CNN\n(Conv1D + MaxPool)',
                       COLORS['cnn'], fontsize=8, text_color='black')

    # LSTM
    create_rounded_box(ax, 10.7, 4.0, 2.2, 0.7, 'LSTM\n(Bidirectional)',
                       COLORS['lstm'], fontsize=8, text_color='white')

    # FC
    create_rounded_box(ax, 10.7, 3.0, 2.2, 0.7, 'Fully Connected\n(64 -> 32 -> 1)',
                       '#9C27B0', fontsize=8)

    draw_arrow(ax, (10.7, 4.55), (10.7, 4.35), '#2E7D32')
    draw_arrow(ax, (10.7, 3.55), (10.7, 3.35), '#2E7D32')

    # 화살표 4 (모델 -> RUL)
    draw_arrow(ax, (12.2, 4), (12.8, 4), COLORS['dark'])

    # 5. 최종 RUL 출력
    create_rounded_box(ax, 13.2, 4, 1.0, 1.2, 'RUL\nOutput',
                       COLORS['success'], fontsize=10)

    # 하단 설명
    ax.text(7, 1.2, 'Data Flow: TDMS (10 sec @ 10 min interval) → Signal Processing → Feature Extraction (15 channels) → CNN-LSTM → RUL Prediction',
            ha='center', va='center', fontsize=9, color=COLORS['dark'],
            style='italic', bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/srv2/jinwook/bearing/latex/figures/methodology_pipeline.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: methodology_pipeline.png")


# =============================================================================
# 그림 2: 신호처리 개념도
# =============================================================================
def create_signal_processing_concept():
    """신호처리 개념도"""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    # 샘플 신호 생성
    np.random.seed(42)
    t = np.linspace(0, 0.1, 2560)  # 0.1초, 25.6kHz

    # 원본 신호: 베어링 결함 특성 + 노이즈
    fault_freq = 140  # BPFI 근사
    carrier_freq = 3000  # 공진 주파수

    # 결함 신호 (진폭 변조)
    modulation = 1 + 0.5 * np.sin(2 * np.pi * fault_freq * t)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    noise = 0.3 * np.random.randn(len(t))
    low_freq_noise = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)

    raw_signal = modulation * carrier + noise + low_freq_noise

    # BPF 적용 (1000-5000Hz)
    fs = 25600
    nyq = fs / 2
    low = 1000 / nyq
    high = 5000 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, raw_signal)

    # Envelope 추출
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)

    # RMS 계산 (윈도우)
    window_size = 256
    rms_values = []
    for i in range(0, len(envelope) - window_size, window_size):
        rms = np.sqrt(np.mean(envelope[i:i+window_size]**2))
        rms_values.extend([rms] * window_size)
    rms_values.extend([rms_values[-1]] * (len(envelope) - len(rms_values)))
    rms_signal = np.array(rms_values)

    # ========= 상단 행: BPF + Envelope 경로 =========
    # 원본 신호
    axes[0, 0].plot(t*1000, raw_signal, 'b-', linewidth=0.5)
    axes[0, 0].set_title('Raw Vibration Signal', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Time (ms)', fontsize=9)
    axes[0, 0].set_ylabel('Amplitude', fontsize=9)
    axes[0, 0].set_xlim([0, 20])
    axes[0, 0].grid(True, alpha=0.3)

    # BPF 적용 후
    axes[0, 1].plot(t*1000, filtered_signal, 'g-', linewidth=0.5)
    axes[0, 1].set_title('After Band-Pass Filter\n(1000-5000 Hz)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Time (ms)', fontsize=9)
    axes[0, 1].set_ylabel('Amplitude', fontsize=9)
    axes[0, 1].set_xlim([0, 20])
    axes[0, 1].grid(True, alpha=0.3)

    # Envelope
    axes[0, 2].plot(t*1000, filtered_signal, 'g-', linewidth=0.3, alpha=0.5, label='Filtered')
    axes[0, 2].plot(t*1000, envelope, 'r-', linewidth=1.0, label='Envelope')
    axes[0, 2].set_title('Envelope Analysis\n(Hilbert Transform)', fontsize=11, fontweight='bold')
    axes[0, 2].set_xlabel('Time (ms)', fontsize=9)
    axes[0, 2].set_ylabel('Amplitude', fontsize=9)
    axes[0, 2].set_xlim([0, 20])
    axes[0, 2].legend(loc='upper right', fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # RMS 추적
    axes[0, 3].plot(t*1000, envelope, 'r-', linewidth=0.3, alpha=0.5, label='Envelope')
    axes[0, 3].plot(t*1000, rms_signal, 'purple', linewidth=2.0, label='RMS')
    axes[0, 3].set_title('RMS Feature Extraction', fontsize=11, fontweight='bold')
    axes[0, 3].set_xlabel('Time (ms)', fontsize=9)
    axes[0, 3].set_ylabel('Amplitude', fontsize=9)
    axes[0, 3].set_xlim([0, 20])
    axes[0, 3].legend(loc='upper right', fontsize=8)
    axes[0, 3].grid(True, alpha=0.3)

    # ========= 하단 행: Wavelet 경로 =========
    # 원본 신호 (복사)
    axes[1, 0].plot(t*1000, raw_signal, 'b-', linewidth=0.5)
    axes[1, 0].set_title('Raw Vibration Signal', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Time (ms)', fontsize=9)
    axes[1, 0].set_ylabel('Amplitude', fontsize=9)
    axes[1, 0].set_xlim([0, 20])
    axes[1, 0].grid(True, alpha=0.3)

    # Wavelet 분해 시각화
    # 간단한 시뮬레이션 (실제 Db4 대신)
    # D4: 800-1600 Hz, D5: 400-800 Hz
    b_d4, a_d4 = signal.butter(4, [800/nyq, 1600/nyq], btype='band')
    b_d5, a_d5 = signal.butter(4, [400/nyq, 800/nyq], btype='band')

    d4_signal = signal.filtfilt(b_d4, a_d4, raw_signal)
    d5_signal = signal.filtfilt(b_d5, a_d5, raw_signal)

    axes[1, 1].plot(t*1000, d4_signal, 'orange', linewidth=0.5)
    axes[1, 1].set_title('Wavelet D4 Coefficients\n(~800-1600 Hz)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Time (ms)', fontsize=9)
    axes[1, 1].set_ylabel('Amplitude', fontsize=9)
    axes[1, 1].set_xlim([0, 20])
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(t*1000, d5_signal, 'brown', linewidth=0.5)
    axes[1, 2].set_title('Wavelet D5 Coefficients\n(~400-800 Hz)', fontsize=11, fontweight='bold')
    axes[1, 2].set_xlabel('Time (ms)', fontsize=9)
    axes[1, 2].set_ylabel('Amplitude', fontsize=9)
    axes[1, 2].set_xlim([0, 20])
    axes[1, 2].grid(True, alpha=0.3)

    # 최종 특징
    features = ['CH2_BPF_RMS', 'CH2_Env_RMS', 'D4_RMS', 'D5_RMS', 'D5_Entropy']
    values = [0.42, 0.38, 0.31, 0.28, 2.85]
    colors = [COLORS['primary'], COLORS['secondary'], 'orange', 'brown', COLORS['accent']]

    bars = axes[1, 3].bar(range(len(features)), values, color=colors, edgecolor='black')
    axes[1, 3].set_xticks(range(len(features)))
    axes[1, 3].set_xticklabels(features, rotation=45, ha='right', fontsize=8)
    axes[1, 3].set_title('Extracted Features', fontsize=11, fontweight='bold')
    axes[1, 3].set_ylabel('Feature Value', fontsize=9)
    axes[1, 3].grid(True, alpha=0.3, axis='y')

    # 화살표 추가
    for i in range(3):
        axes[0, i].annotate('', xy=(1.15, 0.5), xytext=(1.02, 0.5),
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='black', lw=2))

    for i in range(3):
        axes[1, i].annotate('', xy=(1.15, 0.5), xytext=(1.02, 0.5),
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='black', lw=2))

    plt.tight_layout()
    plt.savefig('/srv2/jinwook/bearing/latex/figures/signal_processing_concept.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: signal_processing_concept.png")


# =============================================================================
# 그림 3: 주파수 대역 선택 근거
# =============================================================================
def create_frequency_band_selection():
    """주파수 대역 선택 근거 다이어그램"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 주파수 스펙트럼 시뮬레이션
    np.random.seed(42)
    freq = np.linspace(0, 10000, 5000)

    # 기본 스펙트럼 (1/f 특성 + 노이즈)
    base_spectrum = 10 / (freq + 100) + 0.5 * np.random.randn(len(freq)) * 0.1

    # 고장 주파수 피크 추가
    fault_freqs = {
        'BPFI': 140,
        'BPFO': 93,
        'BSF': 73,
        'FTF': 6.7
    }

    # 배수 주파수 추가 (1000-5000Hz 대역에서 더 강하게)
    for name, base_freq in fault_freqs.items():
        for harmonic in range(1, 50):
            f = base_freq * harmonic
            if f < 10000:
                # 1000-5000Hz 대역에서 더 강한 피크
                if 1000 <= f <= 5000:
                    peak_height = 3.0 / (harmonic ** 0.3)
                else:
                    peak_height = 1.0 / (harmonic ** 0.5)

                idx = int(f / 2)
                if idx < len(base_spectrum):
                    width = 20
                    for j in range(max(0, idx-width), min(len(base_spectrum), idx+width)):
                        base_spectrum[j] += peak_height * np.exp(-((j-idx)**2) / (2*(width/3)**2))

    # 스펙트럼 플롯
    ax.fill_between(freq, 0, base_spectrum, alpha=0.3, color='gray', label='Spectrum')
    ax.plot(freq, base_spectrum, 'b-', linewidth=0.5, alpha=0.7)

    # 대역 표시
    # 0-1000 Hz (노이즈 대역)
    ax.axvspan(0, 1000, alpha=0.2, color='red', label='Low-freq Noise (0-1000 Hz)')

    # 1000-5000 Hz (목표 대역)
    ax.axvspan(1000, 5000, alpha=0.3, color='green', label='Target Band (1000-5000 Hz)')

    # 5000+ Hz (고주파 노이즈)
    ax.axvspan(5000, 10000, alpha=0.2, color='orange', label='High-freq Noise (5000+ Hz)')

    # 고장 주파수 표시
    fault_colors = {'BPFI': 'red', 'BPFO': 'blue', 'BSF': 'purple', 'FTF': 'brown'}
    y_offset = {'BPFI': 4.5, 'BPFO': 4.0, 'BSF': 3.5, 'FTF': 3.0}

    for name, base_freq in fault_freqs.items():
        # 기본 주파수
        ax.axvline(x=base_freq, color=fault_colors[name], linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.text(base_freq + 30, y_offset[name], f'{name}\n({base_freq}Hz)',
                fontsize=8, color=fault_colors[name], fontweight='bold')

        # 배수 표시 (1000-5000Hz 대역 내)
        for harmonic in [7, 14, 21, 28, 35]:  # BPFI 배수 예시
            f = fault_freqs['BPFI'] * harmonic
            if 1000 <= f <= 5000:
                ax.axvline(x=f, color='red', linestyle=':', linewidth=0.8, alpha=0.5)

    # 대역 설명 텍스트
    ax.text(500, 5.5, 'Mechanical\nStructure\nVibration', ha='center', va='center',
            fontsize=9, color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(3000, 5.5, 'Fault Frequency\nHarmonics\n(High SNR)', ha='center', va='center',
            fontsize=9, color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(7500, 5.5, 'High-frequency\nNoise\nSensitive', ha='center', va='center',
            fontsize=9, color='darkorange', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 축 설정
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude (a.u.)', fontsize=12, fontweight='bold')
    ax.set_title('Frequency Band Selection for Bearing Fault Detection', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 10000])
    ax.set_ylim([0, 6.5])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 하단 표 추가
    table_data = [
        ['BPFI', '140 Hz', 'Inner race fault frequency'],
        ['BPFO', '93 Hz', 'Outer race fault frequency'],
        ['BSF', '73 Hz', 'Ball spin frequency'],
        ['FTF', '6.7 Hz', 'Cage frequency']
    ]

    table = ax.table(cellText=table_data,
                     colLabels=['Fault Type', 'Frequency', 'Description'],
                     loc='lower right',
                     cellLoc='center',
                     bbox=[0.65, 0.02, 0.33, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    plt.tight_layout()
    plt.savefig('/srv2/jinwook/bearing/latex/figures/frequency_band_selection.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: frequency_band_selection.png")


# =============================================================================
# 그림 4: CNN-LSTM 모델 구조도
# =============================================================================
def create_cnn_lstm_architecture():
    """CNN-LSTM 모델 구조도"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 제목
    ax.text(7, 7.5, 'CNN-LSTM Architecture for RUL Prediction', fontsize=16,
            ha='center', fontweight='bold', color=COLORS['dark'])

    # 레이어 위치
    y_center = 4.0

    # 1. Input Layer
    input_box = FancyBboxPatch((0.3, y_center - 1.5), 1.5, 3.0,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor='#E3F2FD', edgecolor=COLORS['primary'],
                                linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.05, y_center + 0.8, 'Input', fontsize=11, ha='center', fontweight='bold',
            color=COLORS['primary'])
    ax.text(1.05, y_center + 0.2, '(Batch, Time,\n15 channels)', fontsize=9,
            ha='center', color=COLORS['dark'])
    ax.text(1.05, y_center - 0.8, 'Features:\nBPF_RMS\nEnv_RMS\nD4_RMS\nD5_RMS\nD5_Entropy\n...',
            fontsize=7, ha='center', color='gray')

    # 화살표
    draw_arrow(ax, (1.9, y_center), (2.5, y_center), COLORS['dark'], lw=2)

    # 2. Conv1D Layer
    conv_box = FancyBboxPatch((2.5, y_center - 1.2), 2.0, 2.4,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=COLORS['cnn'], edgecolor='black',
                               linewidth=2, alpha=0.8)
    ax.add_patch(conv_box)
    ax.text(3.5, y_center + 0.6, 'Conv1D', fontsize=11, ha='center', fontweight='bold',
            color='black')
    ax.text(3.5, y_center, 'Filters: 64\nKernel: 5\nActivation: ReLU', fontsize=8,
            ha='center', color='black')
    ax.text(3.5, y_center - 0.8, 'Local Feature\nExtraction', fontsize=7,
            ha='center', color='gray', style='italic')

    # 화살표
    draw_arrow(ax, (4.6, y_center), (5.2, y_center), COLORS['dark'], lw=2)

    # 3. MaxPool Layer
    pool_box = FancyBboxPatch((5.2, y_center - 0.8), 1.3, 1.6,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='#B2EBF2', edgecolor='black',
                               linewidth=2, alpha=0.8)
    ax.add_patch(pool_box)
    ax.text(5.85, y_center + 0.3, 'MaxPool1D', fontsize=10, ha='center', fontweight='bold')
    ax.text(5.85, y_center - 0.3, 'Kernel: 2', fontsize=8, ha='center')

    # 화살표
    draw_arrow(ax, (6.6, y_center), (7.2, y_center), COLORS['dark'], lw=2)

    # 4. LSTM Layer
    lstm_box = FancyBboxPatch((7.2, y_center - 1.2), 2.2, 2.4,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=COLORS['lstm'], edgecolor='black',
                               linewidth=2, alpha=0.8)
    ax.add_patch(lstm_box)
    ax.text(8.3, y_center + 0.6, 'LSTM', fontsize=11, ha='center', fontweight='bold',
            color='white')
    ax.text(8.3, y_center, 'Hidden: 64\nBidirectional', fontsize=9,
            ha='center', color='white')
    ax.text(8.3, y_center - 0.8, 'Temporal\nDependency', fontsize=7,
            ha='center', color='white', style='italic')

    # 화살표
    draw_arrow(ax, (9.5, y_center), (10.1, y_center), COLORS['dark'], lw=2)

    # 5. FC Layer 1
    fc1_box = FancyBboxPatch((10.1, y_center - 0.8), 1.3, 1.6,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='#CE93D8', edgecolor='black',
                              linewidth=2, alpha=0.8)
    ax.add_patch(fc1_box)
    ax.text(10.75, y_center + 0.3, 'Dense', fontsize=10, ha='center', fontweight='bold')
    ax.text(10.75, y_center - 0.3, '64 -> 32\nReLU', fontsize=8, ha='center')

    # 화살표
    draw_arrow(ax, (11.5, y_center), (12.1, y_center), COLORS['dark'], lw=2)

    # 6. Output Layer
    output_box = FancyBboxPatch((12.1, y_center - 0.8), 1.4, 1.6,
                                 boxstyle="round,pad=0.02,rounding_size=0.1",
                                 facecolor=COLORS['success'], edgecolor='black',
                                 linewidth=2)
    ax.add_patch(output_box)
    ax.text(12.8, y_center + 0.3, 'Output', fontsize=10, ha='center', fontweight='bold',
            color='white')
    ax.text(12.8, y_center - 0.3, 'RUL\n(1 unit)', fontsize=9, ha='center', color='white')

    # 하단 정보 박스
    info_box = FancyBboxPatch((1, 0.8), 12, 1.4,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='#F5F5F5', edgecolor='gray',
                               linewidth=1)
    ax.add_patch(info_box)

    # 하단 정보 텍스트
    info_text = """Training Configuration:    Optimizer: AdamW (weight decay=0.01)  |  Learning Rate: 0.001 (Cosine Annealing)  |  Batch Size: 32  |  Epochs: 100  |  Early Stopping: 15
Loss Function: Asymmetric MSE (Late prediction penalty α=1.5)    |    Dropout: 0.3    |    Sequence Length: 50 time steps"""
    ax.text(7, 1.5, info_text, ha='center', va='center', fontsize=8,
            color=COLORS['dark'], family='monospace')

    # 상단 데이터 흐름 표시
    ax.text(1.05, 6.2, 'Batch × T × 15', fontsize=8, ha='center', color='gray')
    ax.text(3.5, 6.2, 'Batch × T × 64', fontsize=8, ha='center', color='gray')
    ax.text(5.85, 6.2, 'Batch × T/2 × 64', fontsize=8, ha='center', color='gray')
    ax.text(8.3, 6.2, 'Batch × 64', fontsize=8, ha='center', color='gray')
    ax.text(10.75, 6.2, 'Batch × 32', fontsize=8, ha='center', color='gray')
    ax.text(12.8, 6.2, 'Batch × 1', fontsize=8, ha='center', color='gray')

    plt.tight_layout()
    plt.savefig('/srv2/jinwook/bearing/latex/figures/cnn_lstm_architecture.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: cnn_lstm_architecture.png")


# =============================================================================
# 그림 5: 데이터 취득 조건 다이어그램
# =============================================================================
def create_data_acquisition():
    """데이터 취득 조건 다이어그램"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1.5, 1]})

    # 상단: 타임라인 다이어그램
    ax1 = axes[0]
    ax1.set_xlim(0, 35)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # 제목
    ax1.text(17.5, 9.5, 'Data Acquisition Timeline', fontsize=14, ha='center', fontweight='bold')

    # 타임라인 기준선
    ax1.axhline(y=5, xmin=0.05, xmax=0.95, color=COLORS['dark'], linewidth=2)

    # 10분 주기 표시
    cycle_starts = [2, 12, 22, 32]
    for i, start in enumerate(cycle_starts[:3]):
        # 10초 측정 구간 (초록색)
        rect = Rectangle((start, 4), 1, 2, facecolor=COLORS['cnn'],
                         edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.add_patch(rect)
        ax1.text(start + 0.5, 5, '10s\nData', fontsize=8, ha='center', va='center',
                fontweight='bold')

        # 9분 50초 대기 구간 (회색)
        if i < 2:
            rect2 = Rectangle((start + 1, 4.3), 9, 1.4, facecolor='#E0E0E0',
                             edgecolor='gray', linewidth=1, alpha=0.6, linestyle='--')
            ax1.add_patch(rect2)
            ax1.text(start + 5.5, 5, '9m 50s Wait', fontsize=8, ha='center', va='center',
                    color='gray', style='italic')

    # 시간 표시
    time_labels = ['0', '10s', '10min', '10min\n10s', '20min', '20min\n10s']
    time_positions = [2, 3, 12, 13, 22, 23]
    for pos, label in zip(time_positions, time_labels):
        ax1.plot([pos, pos], [3.5, 4], 'k-', linewidth=1)
        ax1.text(pos, 3, label, fontsize=8, ha='center', va='top')

    # 점선 (연속 표시)
    ax1.text(30, 5, '...', fontsize=20, ha='center', va='center')

    # 범례
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['cnn'], edgecolor='black', label='Data Acquisition (10 sec)'),
        mpatches.Patch(facecolor='#E0E0E0', edgecolor='gray', linestyle='--', label='Wait Period (9 min 50 sec)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # 10분 주기 화살표
    ax1.annotate('', xy=(12, 7.5), xytext=(2, 7.5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=2))
    ax1.text(7, 8, '10 min cycle', fontsize=10, ha='center', va='bottom',
            color=COLORS['primary'], fontweight='bold')

    # 하단: 샘플링 조건 표
    ax2 = axes[1]
    ax2.axis('off')

    # 표 데이터
    table_data = [
        ['Signal Type', 'Sampling Rate', 'Points per File', 'Duration', 'Channels'],
        ['Vibration', '25,600 Hz', '256,000', '10 sec', '4 (CH1-CH4)'],
        ['Temperature', '0.1 Hz', '1', '10 sec', '1'],
        ['Torque', '0.1 Hz', '1', '10 sec', '1']
    ]

    colors = [['#E3F2FD']*5,
              [COLORS['cnn'], 'white', 'white', 'white', 'white'],
              ['#FFE0B2', 'white', 'white', 'white', 'white'],
              ['#E1BEE7', 'white', 'white', 'white', 'white']]

    table = ax2.table(cellText=table_data,
                      cellColours=colors,
                      loc='center',
                      cellLoc='center',
                      bbox=[0.1, 0.2, 0.8, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 헤더 굵게
    for j in range(5):
        table[(0, j)].set_text_props(fontweight='bold')

    ax2.set_title('Sampling Configuration', fontsize=12, fontweight='bold', pad=20)

    # 추가 정보 텍스트
    info_text = """Data Format: TDMS (Technical Data Management Streaming)
Total Training Sets: 8 (Run-to-failure experiments)  |  Validation Sets: 6
Each file contains synchronized multi-channel sensor data"""
    ax2.text(0.5, 0.05, info_text, ha='center', va='top', fontsize=9,
            transform=ax2.transAxes, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig('/srv2/jinwook/bearing/latex/figures/data_acquisition.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: data_acquisition.png")


# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == "__main__":
    print("Generating figures for IEEE paper...")
    print("=" * 50)

    create_methodology_pipeline()
    create_signal_processing_concept()
    create_frequency_band_selection()
    create_cnn_lstm_architecture()
    create_data_acquisition()

    print("=" * 50)
    print("All figures generated successfully!")
    print("\nOutput directory: /srv2/jinwook/bearing/latex/figures/")
