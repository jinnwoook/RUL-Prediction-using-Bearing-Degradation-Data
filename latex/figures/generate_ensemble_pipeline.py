#!/usr/bin/env python3
"""
BearLLM 앙상블 파이프라인 그림 생성
KSPHM-KIMM 2025 베어링 RUL 예측 프로젝트
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'

# 색상 팔레트
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'dark': '#3C3C3C',
    'cnn': '#4ECDC4',
    'lstm': '#FF6B6B',
    'bearllm': '#9B59B6',
    'ensemble': '#27AE60',
}


def create_rounded_box(ax, x, y, width, height, text, color, fontsize=9,
                       text_color='white', alpha=1.0, linewidth=1.5):
    """둥근 모서리 박스 생성"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor='black',
                         linewidth=linewidth, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', wrap=True)
    return box


def draw_arrow(ax, start, end, color='black', style='->', lw=2):
    """화살표 그리기"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def create_ensemble_pipeline():
    """BearLLM 앙상블 파이프라인 다이어그램"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 제목
    ax.text(8, 9.5, 'CNN-LSTM + BearLLM Ensemble Pipeline for RUL Prediction',
            fontsize=16, ha='center', fontweight='bold', color=COLORS['dark'])
    ax.text(8, 9.0, 'KSPHM-KIMM 2025 Challenge - 7th Place (70 Teams)',
            fontsize=11, ha='center', color='gray', style='italic')

    # ========== 1. 입력 데이터 ==========
    create_rounded_box(ax, 1.5, 5.5, 2.2, 1.5, 'TDMS\nRaw Data\n(4 Channels)',
                       COLORS['dark'], fontsize=10)
    ax.text(1.5, 4.4, 'Vibration: 25.6kHz\n10 sec / 10 min',
            ha='center', fontsize=8, color='gray')

    # 화살표 (데이터 -> 신호처리)
    draw_arrow(ax, (2.7, 5.5), (3.5, 5.5), COLORS['dark'])

    # ========== 2. 신호처리 ==========
    signal_bg = FancyBboxPatch((3.3, 3.8), 2.8, 3.8,
                                boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor='#E3F2FD', edgecolor=COLORS['primary'],
                                linewidth=2, alpha=0.6)
    ax.add_patch(signal_bg)
    ax.text(4.7, 7.3, 'Signal Processing', fontsize=11, ha='center',
            fontweight='bold', color=COLORS['primary'])

    create_rounded_box(ax, 4.7, 6.3, 2.2, 0.7, 'BPF (1000-5000Hz)',
                       COLORS['primary'], fontsize=9)
    create_rounded_box(ax, 4.7, 5.4, 2.2, 0.7, 'Envelope (Hilbert)',
                       COLORS['primary'], fontsize=9)
    create_rounded_box(ax, 4.7, 4.5, 2.2, 0.7, 'Wavelet (Db4)',
                       COLORS['primary'], fontsize=9)

    # 화살표 (신호처리 -> 분기)
    draw_arrow(ax, (6.2, 5.5), (7.0, 5.5), COLORS['dark'])

    # ========== 3. 특징 추출 ==========
    feature_bg = FancyBboxPatch((6.8, 4.0), 1.8, 3.0,
                                 boxstyle="round,pad=0.02,rounding_size=0.2",
                                 facecolor='#FFF8E1', edgecolor=COLORS['accent'],
                                 linewidth=2, alpha=0.6)
    ax.add_patch(feature_bg)
    ax.text(7.7, 6.7, 'Features', fontsize=10, ha='center',
            fontweight='bold', color=COLORS['accent'])

    features = ['BPF_RMS', 'Env_RMS', 'D4_RMS', 'D5_RMS', 'D5_Ent']
    for i, feat in enumerate(features):
        y = 6.1 - i * 0.45
        create_rounded_box(ax, 7.7, y, 1.4, 0.35, feat,
                          COLORS['accent'], fontsize=7)

    # ========== 4. 두 갈래 분기 ==========
    # 상단: CNN-LSTM 경로
    draw_arrow(ax, (8.7, 5.8), (9.5, 6.8), COLORS['cnn'], lw=2.5)
    # 하단: BearLLM 경로
    draw_arrow(ax, (8.7, 5.2), (9.5, 4.2), COLORS['bearllm'], lw=2.5)

    # ========== 5. CNN-LSTM 브랜치 (상단) ==========
    cnn_bg = FancyBboxPatch((9.3, 5.8), 2.6, 2.4,
                             boxstyle="round,pad=0.02,rounding_size=0.2",
                             facecolor='#E0F7FA', edgecolor=COLORS['cnn'],
                             linewidth=2, alpha=0.6)
    ax.add_patch(cnn_bg)
    ax.text(10.6, 7.9, 'CNN-LSTM Model', fontsize=10, ha='center',
            fontweight='bold', color='#00796B')

    create_rounded_box(ax, 10.6, 7.2, 2.0, 0.6, 'Conv1D + MaxPool',
                       COLORS['cnn'], fontsize=8, text_color='black')
    create_rounded_box(ax, 10.6, 6.4, 2.0, 0.6, 'LSTM (64 units)',
                       COLORS['lstm'], fontsize=8)

    # CNN-LSTM 출력
    draw_arrow(ax, (12.0, 6.8), (12.8, 6.8), COLORS['dark'])
    create_rounded_box(ax, 13.5, 6.8, 1.4, 0.8, 'RUL\nInitial',
                       '#00897B', fontsize=9)

    # ========== 6. BearLLM 브랜치 (하단) ==========
    llm_bg = FancyBboxPatch((9.3, 2.6), 2.6, 2.4,
                             boxstyle="round,pad=0.02,rounding_size=0.2",
                             facecolor='#F3E5F5', edgecolor=COLORS['bearllm'],
                             linewidth=2, alpha=0.6)
    ax.add_patch(llm_bg)
    ax.text(10.6, 4.7, 'BearLLM', fontsize=10, ha='center',
            fontweight='bold', color=COLORS['bearllm'])
    ax.text(10.6, 4.2, '(Pretrained LLM)', fontsize=8, ha='center',
            color='gray', style='italic')

    create_rounded_box(ax, 10.6, 3.5, 2.2, 0.6, 'Feature Encoder',
                       COLORS['bearllm'], fontsize=8)
    create_rounded_box(ax, 10.6, 2.9, 2.2, 0.5, 'Finetuned on CH2',
                       '#7B1FA2', fontsize=7)

    # BearLLM 출력
    draw_arrow(ax, (12.0, 3.5), (12.8, 3.5), COLORS['dark'])
    create_rounded_box(ax, 13.5, 3.5, 1.4, 0.8, 'Wear\nRate',
                       COLORS['bearllm'], fontsize=9)

    # ========== 7. RUL 보정 ==========
    # 화살표 (RUL Initial -> 보정)
    draw_arrow(ax, (14.3, 6.8), (14.3, 5.5), COLORS['dark'])
    # 화살표 (Wear Rate -> 보정)
    draw_arrow(ax, (14.3, 3.5), (14.3, 4.5), COLORS['dark'])

    # 보정 공식 박스
    formula_bg = FancyBboxPatch((12.8, 4.3), 3.0, 1.4,
                                 boxstyle="round,pad=0.02,rounding_size=0.15",
                                 facecolor='#FFFDE7', edgecolor='#F57F17',
                                 linewidth=2, alpha=0.8)
    ax.add_patch(formula_bg)
    ax.text(14.3, 5.3, 'RUL Correction', fontsize=9, ha='center',
            fontweight='bold', color='#F57F17')
    ax.text(14.3, 4.7, r'$RUL_{corrected} = RUL_{initial} \times e^{wear\_rate}$',
            fontsize=9, ha='center', color=COLORS['dark'],
            fontfamily='serif', style='italic')

    # ========== 8. 앙상블 ==========
    draw_arrow(ax, (14.3, 4.3), (14.3, 2.8), COLORS['ensemble'], lw=2.5)

    ensemble_bg = FancyBboxPatch((13.0, 1.2), 2.6, 1.4,
                                  boxstyle="round,pad=0.02,rounding_size=0.2",
                                  facecolor=COLORS['ensemble'], edgecolor='black',
                                  linewidth=2, alpha=0.9)
    ax.add_patch(ensemble_bg)
    ax.text(14.3, 1.9, 'Ensemble', fontsize=12, ha='center',
            fontweight='bold', color='white')
    ax.text(14.3, 1.4, 'Final RUL', fontsize=10, ha='center', color='white')

    # ========== 하단 설명 박스 ==========
    info_box = FancyBboxPatch((0.5, 0.2), 15, 0.8,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='#F5F5F5', edgecolor='gray',
                               linewidth=1)
    ax.add_patch(info_box)

    info_text = ('CNN-LSTM: Local feature extraction + Temporal pattern learning  |  '
                 'BearLLM: Pretrained multimodal LLM for wear rate estimation  |  '
                 'Ensemble: Combines both predictions for robust RUL estimation')
    ax.text(8, 0.6, info_text, ha='center', va='center', fontsize=8,
            color=COLORS['dark'])

    plt.tight_layout()
    plt.savefig('/srv2/jinwook/bearing/latex/figures/ensemble_pipeline.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: ensemble_pipeline.png")


if __name__ == "__main__":
    create_ensemble_pipeline()
    print("Ensemble pipeline figure generated!")
