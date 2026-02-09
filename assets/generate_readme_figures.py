#!/usr/bin/env python3
"""
README용 다이어그램 생성 스크립트
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'

# 색상 팔레트
COLORS = {
    'input': '#3498DB',      # 파란색
    'wavelet': '#9B59B6',    # 보라색
    'bpf': '#E74C3C',        # 빨간색
    'envelope': '#F39C12',   # 주황색
    'feature': '#27AE60',    # 초록색
    'model': '#2C3E50',      # 진회색
    'output': '#1ABC9C',     # 청록색
    'cnn': '#3498DB',
    'lstm': '#E74C3C',
    'fc': '#9B59B6',
}


def create_box(ax, x, y, width, height, text, color, fontsize=10, text_color='white'):
    """박스 생성"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.03,rounding_size=0.15",
                         facecolor=color, edgecolor='white',
                         linewidth=2, alpha=0.95)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', wrap=True)


def draw_arrow(ax, start, end, color='#34495E', lw=2.5):
    """화살표 그리기"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))


def create_signal_processing_pipeline():
    """신호처리 파이프라인 다이어그램"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')

    # 제목
    ax.text(7, 9.5, 'Signal Processing Pipeline', fontsize=18,
            ha='center', fontweight='bold', color='#2C3E50')

    # 1. 입력 데이터
    create_box(ax, 7, 8.2, 5, 0.9, 'Raw Vibration Signal\n(4 channels, 25.6kHz)',
               COLORS['input'], fontsize=11)

    # 화살표 (입력 -> 3개 분기)
    draw_arrow(ax, (7, 7.7), (7, 7.2))

    # 분기점
    ax.plot([3, 11], [7.2, 7.2], color='#34495E', lw=2.5)
    ax.plot([3, 3], [7.2, 6.8], color='#34495E', lw=2.5)
    ax.plot([7, 7], [7.2, 6.8], color='#34495E', lw=2.5)
    ax.plot([11, 11], [7.2, 6.8], color='#34495E', lw=2.5)

    # 2. 세 가지 처리 경로
    # Wavelet
    create_box(ax, 3, 6.2, 3.2, 1.0, 'Wavelet Transform\n(DWT, db4, Level 5)',
               COLORS['wavelet'], fontsize=10)
    draw_arrow(ax, (3, 5.6), (3, 5.0))
    create_box(ax, 3, 4.4, 2.8, 0.9, 'D4_RMS\nD5_RMS\nD5_Entropy',
               COLORS['wavelet'], fontsize=9, text_color='white')

    # BPF
    create_box(ax, 7, 6.2, 3.2, 1.0, 'Band-Pass Filter\n(1000~5000Hz)',
               COLORS['bpf'], fontsize=10)
    draw_arrow(ax, (7, 5.6), (7, 5.0))
    create_box(ax, 7, 4.4, 2.8, 0.9, 'CH2_BPF_RMS',
               COLORS['bpf'], fontsize=10, text_color='white')

    # Envelope
    create_box(ax, 11, 6.2, 3.2, 1.0, 'Envelope Analysis\n(Hilbert Transform)',
               COLORS['envelope'], fontsize=10)
    draw_arrow(ax, (11, 5.6), (11, 5.0))
    create_box(ax, 11, 4.4, 2.8, 0.9, 'Envelope_RMS',
               COLORS['envelope'], fontsize=10, text_color='white')

    # 합류
    ax.plot([3, 3], [3.9, 3.5], color='#34495E', lw=2.5)
    ax.plot([7, 7], [3.9, 3.5], color='#34495E', lw=2.5)
    ax.plot([11, 11], [3.9, 3.5], color='#34495E', lw=2.5)
    ax.plot([3, 11], [3.5, 3.5], color='#34495E', lw=2.5)
    draw_arrow(ax, (7, 3.5), (7, 3.0))

    # 3. 특징 벡터
    create_box(ax, 7, 2.4, 4.5, 0.9, 'Feature Vector (15 dims)\n+ Torque, Temperature',
               COLORS['feature'], fontsize=10)

    draw_arrow(ax, (7, 1.9), (7, 1.4))

    # 4. 정규화
    create_box(ax, 7, 0.9, 3.5, 0.7, 'Normalization (Mean-Std)',
               '#7F8C8D', fontsize=10)

    plt.tight_layout()
    plt.savefig('/srv2/jinwook/bearing/assets/signal_processing_pipeline.png',
                dpi=150, bbox_inches='tight', facecolor='#F8F9FA', edgecolor='none')
    plt.close()
    print("Created: signal_processing_pipeline.png")


def create_cnn_lstm_architecture():
    """CNN-LSTM 아키텍처 다이어그램"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')

    # 제목
    ax.text(7, 7.5, 'CNN-LSTM Architecture', fontsize=18,
            ha='center', fontweight='bold', color='#2C3E50')

    y_center = 4.0

    # 1. Input
    create_box(ax, 1.5, y_center, 2.2, 1.8, 'Input\n(batch, T, 15)',
               '#34495E', fontsize=11)
    ax.text(1.5, y_center - 1.3, 'Features', fontsize=9, ha='center', color='#7F8C8D')

    draw_arrow(ax, (2.7, y_center), (3.3, y_center))

    # 2. Conv1D
    create_box(ax, 4.3, y_center, 2.2, 1.8, 'Conv1D\n64 filters\nkernel=5',
               COLORS['cnn'], fontsize=10)
    ax.text(4.3, y_center - 1.3, 'Local Pattern', fontsize=9, ha='center', color='#7F8C8D')

    draw_arrow(ax, (5.5, y_center), (6.1, y_center))

    # 3. MaxPool
    create_box(ax, 6.8, y_center, 1.4, 1.4, 'MaxPool\nk=2',
               '#95A5A6', fontsize=10)

    draw_arrow(ax, (7.6, y_center), (8.2, y_center))

    # 4. LSTM
    create_box(ax, 9.2, y_center, 2.2, 1.8, 'LSTM\nhidden=64\nbatch_first',
               COLORS['lstm'], fontsize=10)
    ax.text(9.2, y_center - 1.3, 'Temporal', fontsize=9, ha='center', color='#7F8C8D')

    draw_arrow(ax, (10.4, y_center), (11.0, y_center))

    # 5. FC
    create_box(ax, 11.8, y_center, 1.8, 1.8, 'FC\n64→32→1',
               COLORS['fc'], fontsize=10)
    ax.text(11.8, y_center - 1.3, 'Regression', fontsize=9, ha='center', color='#7F8C8D')

    draw_arrow(ax, (12.8, y_center), (13.4, y_center))

    # 6. Output
    create_box(ax, 13.7, y_center, 0.8, 1.2, 'RUL',
               COLORS['output'], fontsize=11)

    # 하단 정보
    info_box = FancyBboxPatch((1, 1.2), 12, 1.2,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='white', edgecolor='#BDC3C7',
                               linewidth=1.5)
    ax.add_patch(info_box)

    info_text = ('Optimizer: AdamW (lr=0.001)  |  Loss: Asymmetric MSE  |  '
                 'Epochs: 100  |  Batch: 32  |  Dropout: 0.3')
    ax.text(7, 1.8, info_text, ha='center', va='center', fontsize=10, color='#2C3E50')

    plt.tight_layout()
    plt.savefig('/srv2/jinwook/bearing/assets/cnn_lstm_architecture.png',
                dpi=150, bbox_inches='tight', facecolor='#F8F9FA', edgecolor='none')
    plt.close()
    print("Created: cnn_lstm_architecture.png")


def create_ensemble_pipeline():
    """앙상블 파이프라인 다이어그램"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')

    # 제목
    ax.text(7, 8.5, 'CNN-LSTM + BearLLM Ensemble', fontsize=18,
            ha='center', fontweight='bold', color='#2C3E50')
    ax.text(7, 8.0, '70팀 중 7등 달성', fontsize=12,
            ha='center', color='#E74C3C', fontweight='bold')

    # 1. 입력
    create_box(ax, 2, 6.5, 2.5, 1.2, 'TDMS Data\n(Vibration)',
               '#34495E', fontsize=11)

    # 화살표 분기
    draw_arrow(ax, (3.3, 6.5), (4.5, 6.5))

    # 신호처리
    create_box(ax, 5.5, 6.5, 2.2, 1.0, 'Signal\nProcessing',
               '#3498DB', fontsize=10)

    draw_arrow(ax, (6.7, 6.5), (7.5, 6.5))

    # 특징
    create_box(ax, 8.3, 6.5, 1.8, 1.0, 'Features',
               '#27AE60', fontsize=10)

    # 분기
    ax.plot([9.3, 10], [6.5, 6.5], color='#34495E', lw=2.5)
    ax.plot([10, 10], [6.5, 5.5], color='#34495E', lw=2.5)
    ax.plot([10, 10], [6.5, 7.5], color='#34495E', lw=2.5)

    # 상단: CNN-LSTM
    draw_arrow(ax, (10, 7.5), (10.8, 7.5))
    create_box(ax, 11.8, 7.5, 2.2, 0.9, 'CNN-LSTM',
               COLORS['cnn'], fontsize=11)
    draw_arrow(ax, (13, 7.5), (13, 6.2))

    # 하단: BearLLM
    draw_arrow(ax, (10, 5.5), (10.8, 5.5))
    create_box(ax, 11.8, 5.5, 2.2, 0.9, 'BearLLM',
               '#9B59B6', fontsize=11)
    draw_arrow(ax, (13, 5.5), (13, 6.2))

    # RUL Initial
    ax.text(13.5, 7.5, 'RUL', fontsize=9, ha='left', va='center', color='#2C3E50')

    # Wear Rate
    ax.text(13.5, 5.5, 'Wear Rate', fontsize=9, ha='left', va='center', color='#2C3E50')

    # 보정 공식
    formula_box = FancyBboxPatch((5, 3.8), 6, 1.4,
                                  boxstyle="round,pad=0.02,rounding_size=0.1",
                                  facecolor='#FFF9C4', edgecolor='#F9A825',
                                  linewidth=2)
    ax.add_patch(formula_box)
    ax.text(8, 4.8, 'RUL Correction Formula', fontsize=10, ha='center',
            fontweight='bold', color='#F57F17')
    ax.text(8, 4.2, r'$RUL_{corrected} = RUL_{initial} \times e^{wear\_rate}$',
            fontsize=12, ha='center', color='#2C3E50', fontfamily='serif')

    # 앙상블
    draw_arrow(ax, (8, 3.7), (8, 2.8))
    create_box(ax, 8, 2.2, 2.5, 1.0, 'Ensemble\nFinal RUL',
               '#1ABC9C', fontsize=11)

    plt.tight_layout()
    plt.savefig('/srv2/jinwook/bearing/assets/ensemble_pipeline.png',
                dpi=150, bbox_inches='tight', facecolor='#F8F9FA', edgecolor='none')
    plt.close()
    print("Created: ensemble_pipeline.png")


if __name__ == "__main__":
    import os
    os.makedirs('/srv2/jinwook/bearing/assets', exist_ok=True)

    print("Generating README figures...")
    create_signal_processing_pipeline()
    create_cnn_lstm_architecture()
    create_ensemble_pipeline()
    print("All figures generated!")
