#!/usr/bin/env python3
"""
데이터 수집 조건 다이어그램 생성
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


def create_data_acquisition_diagram():
    """데이터 수집 주기 다이어그램"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')

    # 제목
    ax.text(6, 4.5, 'Data Acquisition Cycle', fontsize=16,
            ha='center', fontweight='bold', color='#2C3E50')

    # 시간 축
    ax.annotate('', xy=(11.5, 1.5), xytext=(0.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    ax.text(11.7, 1.5, 'Time', fontsize=12, ha='left', va='center',
            color='#2C3E50', style='italic')

    # 첫 번째 데이터 수집 구간 (10초)
    box1 = FancyBboxPatch((1.5, 1.8), 1.2, 1.8,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#3498DB', edgecolor='#2980B9',
                          linewidth=2, alpha=0.9)
    ax.add_patch(box1)
    ax.text(2.1, 3.9, '10 sec', fontsize=14, ha='center', fontweight='bold',
            color='#2C3E50')
    ax.text(2.1, 2.7, 'Data\nAcquisition', fontsize=10, ha='center',
            color='white', fontweight='bold')

    # 간격 표시 (9분 50초)
    ax.annotate('', xy=(8.3, 2.7), xytext=(2.8, 2.7),
                arrowprops=dict(arrowstyle='<->', color='#E74C3C', lw=2.5))
    ax.text(5.55, 3.0, '9 min 50 sec', fontsize=14, ha='center',
            fontweight='bold', color='#E74C3C')
    ax.text(5.55, 2.3, '(No Data Collection)', fontsize=10, ha='center',
            color='#7F8C8D', style='italic')

    # 두 번째 데이터 수집 구간 (10초)
    box2 = FancyBboxPatch((8.3, 1.8), 1.2, 1.8,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='#3498DB', edgecolor='#2980B9',
                          linewidth=2, alpha=0.9)
    ax.add_patch(box2)
    ax.text(8.9, 3.9, '10 sec', fontsize=14, ha='center', fontweight='bold',
            color='#2C3E50')
    ax.text(8.9, 2.7, 'Data\nAcquisition', fontsize=10, ha='center',
            color='white', fontweight='bold')

    # 점선으로 반복 표시
    ax.plot([10, 10.5], [2.7, 2.7], 'k--', lw=2, alpha=0.5)
    ax.text(10.8, 2.7, '...', fontsize=20, ha='center', va='center', color='#7F8C8D')

    # 하단 설명
    info_box = FancyBboxPatch((1, 0.3), 10, 0.8,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='#FFF9C4', edgecolor='#F9A825',
                               linewidth=1.5, alpha=0.8)
    ax.add_patch(info_box)
    ax.text(6, 0.7, 'Total Cycle: 10 min  |  Vibration: 25.6 kHz  |  Other Signals: 0.1 Hz',
            ha='center', va='center', fontsize=11, color='#2C3E50', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/srv2/jinwook/bearing/assets/data_acquisition_cycle.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Created: data_acquisition_cycle.png")


if __name__ == "__main__":
    create_data_acquisition_diagram()
    print("Data acquisition diagram generated!")
