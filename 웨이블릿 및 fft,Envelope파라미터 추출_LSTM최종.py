# -*- coding: utf-8 -*-
"""
KSPHM-KIMM 2025 베어링 수명 예측 공모전 제출용 코드 (병렬 처리 포함)

[주요 기능 요약]
- /data 디렉토리 기준으로 Train/Validation 데이터 자동 탐색
- 각 TDMS 파일에서 원시 신호 및 파라미터 추출 (256000 row)
- 웨이블릿 (D4/D5) 및 인벨롭 (CH2 BPF, RMS) 파라미터 포함
- 수치 데이터는 float32 형식으로 저장
- 결과는 각 세트별 폴더(/data/웨이블릿 및 fft_TrainX_LSTM최종 등)에 저장
- TDMS 파일은 병렬로 처리하여 속도 향상
- 실행 환경 정보 및 라이브러리 버전 출력 포함

[제출 요건 체크리스트]
1. 입출력 경로: /data 기준
2. 확장자: .py/.ipynb
3. 인코딩: UTF-8
4. 수치형: float32
5. 실행 오류 없음
6. 주석 포함 및 가독성 확보
7. 운영체제 및 라이브러리 버전 명시

**validation, train, both 선택기능 43번 줄에 존재**
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
import pywt
from nptdms import TdmsFile
from joblib import Parallel, delayed

print("Python version:", sys.version)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scipy:", pd.__version__)
print("pywt:", pywt.__version__)

# --------------------------- 설정 -----------------------------
mode = "train"  # "train", "validation", "both" 중 선택
train_range = range(1, 9) # 1이상 9미만
val_range = range(1, 7) # 1이상 7미만

fs = 25600
window_sec = 0.5
window_size = int(fs * window_sec)
wavelet_function = "db4"
band_range = (1000, 5000)
channel_list = ["CH1", "CH2", "CH3", "CH4"]
env_target_channel = "CH2"

base_dir = os.path.join(os.getcwd(), "data")
base_paths = {
    "train": os.path.join(base_dir, "Train Set"),
    "validation": os.path.join(base_dir, "Validation Set")
}
range_dict = {
    "train": train_range,
    "validation": val_range
}

# --------------------- 파라미터 추출 함수 ----------------------
def extract_wavelet_params(signal, wavelet="db4", level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    param_dict = {"D4_RMS": None, "D5_RMS": None, "D5_Entropy": None}
    for i in range(1, len(coeffs)):
        scale = f"D{level - i + 1}"
        if scale not in ["D4", "D5"]:
            continue
        detail = coeffs[i].astype(np.float32)
        rms = np.sqrt(np.mean(detail**2)).astype(np.float32)
        if scale == "D4":
            param_dict["D4_RMS"] = rms
        elif scale == "D5":
            prob_density, _ = np.histogram(detail, bins=64, density=True)
            prob_density = prob_density[prob_density > 0]
            entropy = -np.sum(prob_density * np.log2(prob_density)).astype(np.float32)
            param_dict["D5_RMS"] = rms
            param_dict["D5_Entropy"] = entropy
    return param_dict

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def compute_envelope(signal):
    return np.abs(hilbert(signal))

def extract_envelope_params(signal):
    signal = signal.astype(np.float32)
    signal -= np.mean(signal)
    filtered = bandpass_filter(signal, fs, band_range[0], band_range[1])
    envelope = compute_envelope(filtered)
    trimmed = filtered[:len(filtered)//window_size*window_size].reshape(-1, window_size)
    trimmed_env = envelope[:len(envelope)//window_size*window_size].reshape(-1, window_size)
    bpf = np.sqrt(np.mean(trimmed ** 2, axis=1)).astype(np.float32)
    rms_env = np.sqrt(np.mean(trimmed_env ** 2, axis=1)).astype(np.float32)
    return np.repeat(bpf, window_size), np.repeat(rms_env, window_size)

def safe_find(op_dict, key_keyword):
    for k in op_dict:
        if key_keyword.lower() in k.lower():
            return op_dict[k]
    print(f"[경고] '{key_keyword}' 키워드 operation 채널 없음")
    return np.nan

# ----------------- 단일 TDMS 파일 처리 -----------------
def process_single_tdms_file(tdms_path, save_dir):
    tdms = TdmsFile.read(tdms_path)
    group_vib = tdms.groups()[0].name
    group_op = tdms.groups()[1].name
    row_count = len(tdms[group_vib]["CH1"].data)
    timestamps = (np.arange(0, row_count) / fs).astype(np.float32)

    vib_data = {
        ch: tdms[group_vib][ch].data[:row_count].astype(np.float32) for ch in channel_list
    }
    op_dict = {ch.name.strip(): ch.data[0] for ch in tdms[group_op].channels()}

    df = pd.DataFrame({
        "Time (s)": timestamps,
        "CH1": vib_data["CH1"],
        "CH2": vib_data["CH2"],
        "CH3": vib_data["CH3"],
        "CH4": vib_data["CH4"],
        "Torque[Nm]": np.repeat(safe_find(op_dict, "torque"), row_count).astype(np.float32),
        "TC SP Front[℃]": np.repeat(safe_find(op_dict, "front"), row_count).astype(np.float32),
        "TC SP Rear[℃]": np.repeat(safe_find(op_dict, "rear"), row_count).astype(np.float32),
    })

    for ch in channel_list:
        trimmed = vib_data[ch].reshape(-1, window_size)
        d4_rms, d5_rms, d5_entropy = [], [], []
        for seg in trimmed:
            p = extract_wavelet_params(seg)
            d4_rms.append(p["D4_RMS"])
            d5_rms.append(p["D5_RMS"])
            d5_entropy.append(p["D5_Entropy"])
        df[f"{ch}_D4_RMS"] = np.repeat(d4_rms, window_size)
        df[f"{ch}_D5_RMS"] = np.repeat(d5_rms, window_size)
        df[f"{ch}_D5_Entropy"] = np.repeat(d5_entropy, window_size)

    bpf, rms_env = extract_envelope_params(vib_data[env_target_channel])
    df["CH2_1000-5000Hz_BPF"] = bpf
    df["CH2_1000-5000Hz_RMS_Envelope"] = rms_env

    file_name = os.path.splitext(os.path.basename(tdms_path))[0] + ".csv"
    save_path = os.path.join(save_dir, file_name)
    df.to_csv(save_path, index=False)
    print(f"저장 완료 → {save_path}")

# -------------------- 전체 반복 처리 --------------------
def process_all_tdms():
    modes = ["train", "validation"] if mode == "both" else [mode]
    for m in modes:
        base_path = base_paths[m]
        for idx in range_dict[m]:
            folder_path = os.path.join(base_path, f"{m.capitalize()}{idx}")
            tdms_files = glob.glob(os.path.join(folder_path, "*.tdms"))
            if not tdms_files:
                print(f"[스킵] {folder_path} - TDMS 없음")
                continue
            save_dir = os.path.join(base_dir, f"웨이블릿 및 fft_{m.capitalize()}{idx}_LSTM최종")
            os.makedirs(save_dir, exist_ok=True)
            print(f"{folder_path} 내 {len(tdms_files)}개 파일 병렬 처리 시작...")
            Parallel(n_jobs=-1)(
                delayed(process_single_tdms_file)(tdms_path, save_dir)
                for tdms_path in tdms_files
            )

# ✅ 실행
process_all_tdms()
