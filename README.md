<div align="center">

# 🏆 RUL-Prediction-using-Bearing-Degradation-Data

### KSPHM-KIMM 2025 베어링 수명 예측 챌린지

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

**🏅 70팀 중 8등 달성** (성능 7등)

<img src="베어링 대회 순위.png" width="600" alt="Competition Ranking"/>

<br>

### 예측 파이프라인

<img src="pipeline.png" width="750" alt="Prediction Pipeline"/>

</div>

---

### 🔍 파이프라인 방법론

본 프로젝트는 **두 개의 ConV-LSTM 브랜치**와 **BearLLM**을 결합한 3단계 앙상블 예측 파이프라인을 사용합니다.

**STEP 1 — ConV-LSTM 기반 RUL 초기 예측**

TDMS 형식의 베어링 진동 신호를 ConV-LSTM에 입력해 잔여 수명(RUL) 초기값을 예측합니다. Conv1D로 진동 파형의 결함 패턴(충격 임펄스, 주파수 특징)을 추출하고, LSTM으로 시간축 열화 추이를 학습합니다. 서로 다른 시점의 신호를 처리하는 **두 개의 독립 브랜치**로 구성됩니다.

**STEP 2 — BearLLM 마모율 추정 및 RUL 보정**

BearLLM이 전체 진동 신호 시퀀스를 입력받아 **마지막 시퀀스의 Wear_rate(마모율)** 을 추정합니다. 이를 ConV-LSTM의 초기 예측값에 지수함수적으로 보정합니다:

$$RUL_{corrected} = RUL_{initial} \times e^{wear\\_rate}$$

**STEP 3 — 최종 앙상블**

BearLLM으로 보정된 RUL과 두 번째 ConV-LSTM 브랜치의 RUL을 앙상블하여 **최종 RUL**을 산출합니다.

| 구성요소 | 역할 |
|:---|:---|
| ConV-LSTM Branch 1 | 초기 시점 진동 신호 → 1차 RUL 예측 |
| BearLLM | 마모율 추정 → 지수 보정 적용 |
| ConV-LSTM Branch 2 | 최근 시점 진동 신호 → 2차 RUL 예측 |
| Ensemble | 보정 RUL + Branch 2 RUL → **최종 RUL** |

---

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 특징](#-주요-특징)
- [디렉토리 구조](#-디렉토리-구조)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [방법론](#-방법론)
- [실험 결과](#-실험-결과)
- [Technical Report](#-technical-report)
- [참고문헌](#-참고문헌)
- [팀 정보](#-팀-정보)
- [라이센스](#-라이센스)

---

## 🎯 프로젝트 소개

본 프로젝트는 **KSPHM-KIMM 2025 데이터 챌린지**에서 베어링의 **잔여 수명(RUL, Remaining Useful Life)**을 예측하기 위해 개발되었습니다.

### 대회 개요

| 항목 | 내용 |
|:---:|:---|
| 🏛️ **주최** | KSPHM (한국PHM학회), KIMM (한국기계연구원) |
| 📅 **기간** | 2025년 4월 ~ 6월 |
| 🎯 **목표** | 진동 센서 데이터를 활용한 베어링 잔여 수명 예측 |
| 📊 **데이터** | TDMS 형식의 다채널 진동/온도/토크 시계열 데이터 |
| 🏆 **결과** | **70팀 중 8등** 달성 (성능 7등) |

### 대회 일정

| 진행 내용 | 날짜 |
|:---|:---:|
| 데이터 공개 1차 (Training Set) | 4/14 (월) |
| 팀 등록 마감 | 5/7 (수) |
| 데이터 공개 2차 (Validation Set) | 5/19 (월) |
| Validation 최종 제출 | 5/30 (금) |
| 결과 발표 | 6/3 (화) |
| 발표 평가 | 6/23 (월) |
| 우수 팀 시상 | 6/24 (화) |

### 📋 데이터 취득 조건

#### 데이터 샘플링 레이트 (Sampling Rate)
| 신호 유형 | 샘플링 레이트 |
|:---:|:---:|
| 진동 신호 | **25.6 kHz** |
| 이외 신호 (온도, 토크 등) | **0.1 Hz** |

#### 데이터 수집 주기
- **10분 주기**로 **10초씩** 데이터 취득
- 테스트베드는 연속적으로 운전

#### 시험 중단 조건과 데이터 특성
- 베어링이 **중단 조건**에 도달하면 실험이 종료됨
- 데이터 **측정 중** 고장 발생 시 → 고장 시점의 데이터가 **포함**
- 데이터 **미측정 중** 고장 발생 시 → 고장 시점의 데이터가 **불포함**

> ⚠️ **주의:** 실제 고장 시점과 마지막 데이터 측정 시간은 **일치하지 않을 수 있으며**, 이는 예측 모델에서 고려되어야 할 중요한 변수입니다.

#### 평가 방식
- Validation Set에 대한 RUL 예측 정확도
- 발표 평가를 통한 최종 순위 결정

---

## ✨ 주요 특징

### 🔬 신호 처리 기법

- **Discrete Wavelet Transform (DWT)**
  - Daubechies 4 (db4) 웨이블릿 사용
  - D4, D5 스케일에서 RMS 및 Entropy 특징 추출
  - 저주파 대역의 결함 신호 효과적 분리

- **Band-Pass Filtering (BPF)**
  - 1000~5000Hz 대역 필터링
  - 고장 주파수 성분 분리 및 잡음 제거
  - CH2 채널에서 최적 성능 확인

- **Envelope Analysis**
  - Hilbert Transform 기반 포락선 분석
  - 충격 임펄스 신호의 주기적 패턴 검출
  - 결함 반복 주파수 추출

### 🧠 딥러닝 모델

- **CNN-LSTM Hybrid Architecture**
  - Conv1D: 시계열의 지역적 특징 추출
  - LSTM: 시간적 의존성 학습
  - Fully Connected: RUL 회귀 예측

---

## 📁 디렉토리 구조

```
📦 RUL-Prediction-using-Bearing-Degradation-Data/
├── 📜 Bearing(KIST_베어링_신_제자들).ipynb              # 전처리 및 EDA 코드
├── 📜 Bearing(KIST_베어링_신_제자들)_modeling_ver.ipynb # 모델링 및 앙상블 코드
├── 📜 KIST_베어링신의_제자들_code.ipynb                  # 메인 통합 코드
├── 📜 웨이블릿 및 fft,Envelope파라미터 추출_LSTM최종.py  # 특징 추출 코드
├── 📜 STFT(고장 주파수).ipynb                           # 주파수 분석
├── 📜 pipeline.png                                      # 예측 파이프라인 다이어그램
├── 📜 KIST 베어링 신의 제자들_report.pdf                # 보고서
├── 📜 베어링 대회 순위.png                              # 대회 순위
├── 📜 info.txt                                          # 정보
└── 📜 README.md
```

---

## ⚙️ 설치 방법

### 요구사항

- Python 3.8+
- CUDA 11.0+ (GPU 사용 시)

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/RUL-Prediction-using-Bearing-Degradation-Data.git
cd RUL-Prediction-using-Bearing-Degradation-Data

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 주요 라이브러리

| 라이브러리 | 버전 | 용도 |
|:---:|:---:|:---|
| `torch` | 2.0+ | 딥러닝 프레임워크 |
| `numpy` | 1.20+ | 수치 연산 |
| `pandas` | 1.3+ | 데이터 처리 |
| `scipy` | 1.7+ | 신호 처리 |
| `pywt` | 1.3+ | 웨이블릿 변환 |
| `nptdms` | 1.6+ | TDMS 파일 읽기 |
| `joblib` | 1.1+ | 병렬 처리 |
| `matplotlib` | 3.5+ | 시각화 |

---

## 🚀 사용 방법

### 1️⃣ 데이터 전처리

TDMS 파일에서 특징을 추출하고 CSV 형식으로 저장합니다.

```python
# 웨이블릿 및 fft,Envelope파라미터 추출_LSTM최종.py 실행
python "웨이블릿 및 fft,Envelope파라미터 추출_LSTM최종.py"
```

**주요 설정:**
```python
mode = "train"          # "train", "validation", "both" 중 선택
fs = 25600              # 샘플링 주파수 (Hz)
window_sec = 0.5        # 윈도우 크기 (초)
wavelet_function = "db4"  # 웨이블릿 함수
band_range = (1000, 5000) # 밴드패스 필터 범위
```

### 2️⃣ 모델 학습

```python
import torch
from models import CNNLSTM

# 모델 초기화
model = CNNLSTM(input_channels=15)

# 학습 설정
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# 학습 루프
for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
```

### 3️⃣ 추론

```python
import torch
from models import CNNLSTM

# 모델 로드
model = CNNLSTM(input_channels=15)
model.load_state_dict(torch.load("./model_weights/best_model_0.66.pth"))
model.eval()

# 예측
with torch.no_grad():
    for x_test, fname in test_loader:
        pred = model(x_test)
        rul_sec = torch.expm1(pred).numpy()  # log 스케일 역변환
        print(f"{fname} → 예측 RUL: {rul_sec:.2f}초")
```

---

## 🔧 방법론

### ConV-LSTM + BearLLM 앙상블 파이프라인

본 프로젝트의 핵심은 **두 개의 ConV-LSTM 브랜치**와 **BearLLM**을 결합한 3단계 앙상블 예측 파이프라인입니다.

---

#### STEP 1. ConV-LSTM 기반 RUL 초기 예측

베어링 진동 센서에서 수집된 **TDMS 형식의 시계열 신호**를 ConV-LSTM 모델에 입력해 잔여 수명(RUL) 초기값을 예측합니다.

- **Conv1D**: 진동 파형에서 결함 관련 지역 특징(주파수 패턴, 충격 임펄스) 추출
- **LSTM**: 시간 순서에 따른 열화 추이를 학습해 장기 의존성 모델링
- 서로 다른 시점의 진동 신호를 처리하는 **두 개의 독립 브랜치**로 구성

---

#### STEP 2. BearLLM 기반 마모율(Wear Rate) 추정 및 RUL 보정

**BearLLM**은 전체 진동 신호 시퀀스를 입력받아 **마지막 시퀀스의 Wear_rate(마모율)** 을 추정합니다.

이를 활용해 ConV-LSTM의 초기 RUL 예측값을 지수함수적으로 보정합니다:

$$RUL_{corrected} = RUL_{initial} \times e^{wear\\_rate}$$

- `wear_rate > 0`: 마모가 빠르게 진행 중 → RUL 상향 보정 (아직 수명이 남음)
- `wear_rate < 0`: 급격한 열화 → RUL 하향 보정 (조기 고장 위험)

---

#### STEP 3. 최종 앙상블

BearLLM으로 보정된 RUL과 두 번째 ConV-LSTM 브랜치의 RUL 예측값을 **앙상블(평균)** 하여 최종 RUL을 산출합니다.

| 구성요소 | 역할 |
|:---|:---|
| **ConV-LSTM (Branch 1)** | 초기 시점 진동 신호 → 1차 RUL 예측 |
| **BearLLM** | 전체 시퀀스 마모율 추정 → 지수 보정 적용 |
| **ConV-LSTM (Branch 2)** | 최근 시점 진동 신호 → 2차 RUL 예측 |
| **Ensemble** | BearLLM 보정 RUL + Branch 2 RUL → **최종 RUL** |

---

### 특징 추출 (Feature Extraction)

| 특징 | 설명 | 물리적 의미 |
|:---|:---|:---|
| `D4_RMS` | 웨이블릿 D4 스케일 RMS | 중고주파 대역 에너지 |
| `D5_RMS` | 웨이블릿 D5 스케일 RMS | 저주파 대역 에너지 |
| `D5_Entropy` | 웨이블릿 D5 스케일 엔트로피 | 신호 복잡도/불규칙성 |
| `CH2_BPF_RMS` | CH2 밴드패스 필터 후 RMS | 결함 주파수 대역 에너지 |
| `Envelope_RMS` | 포락선 분석 후 RMS | 충격 임펄스 강도 |

---

## 📊 실험 결과

### 검증 데이터셋 성능

| 데이터셋 | 예측 RUL (초) | 비고 |
|:---:|:---:|:---|
| Validation 1 | 51,730.23 | - |
| Validation 2 | 30,379.84 | - |

### 모델 학습 정보

| 항목 | 값 |
|:---|:---:|
| Best Validation Score | **0.66** |
| Input Features | 15 |
| Model Parameters | ~165KB |
| Training Device | CUDA GPU |

### 특징 선정 근거

- **CH2 채널 선정**: 모든 Train 파일에서 1000-5000Hz 대역에서 고장 시간에 가까워질수록 점진적으로 상승하는 패턴 확인
- **저주파 대역 (D4, D5)**: 베어링 결함 주파수(~140Hz)와 그 고조파가 해당 대역에 집중
- **Envelope 분석**: 결함에 의한 충격 신호의 반복 패턴을 효과적으로 검출

---

## 📝 Technical Report

**📄 보고서:** [KIST 베어링 신의 제자들_report.pdf](KIST%20베어링%20신의%20제자들_report.pdf)

<details>
<summary><b>📄 논문 보기 (클릭하여 펼치기)</b></summary>

<br>

**📥 다운로드:** [논문_한국어.pdf](논문_한국어.pdf)

| 섹션 | 내용 |
|:---|:---|
| 1. 서론 | 베어링 RUL 예측의 중요성 및 연구 배경 |
| 2. 관련 연구 | 기존 방법론 분석 및 한계점 |
| 3. 제안 방법 | CNN-LSTM + BearLLM 앙상블 아키텍처 |
| 4. 실험 | 데이터셋, 전처리, 학습 설정 |
| 5. 결과 | 성능 분석 및 비교 |
| 6. 결론 | 요약 및 향후 연구 방향 |

</details>

---

## 📚 참고문헌

1. Kumar et al. (2013), *"Wavelet transform for bearing condition monitoring and fault diagnosis: A review"*

2. Rafia Nishat Toma et al. (2020), *"Bearing Fault Classification of Induction Motors Using Discrete Wavelet Transform and Ensemble Machine Learning Algorithms"*

3. C K E Nizwana et al. (2016), *"A wavelet decomposition analysis of vibration signal for bearing fault detection"*

4. *"ANN Based Fault Detection Scheme for Bearing Condition Monitoring in SRIMs using FFT, DWT and Band-pass Filters"* - 밴드패스 필터 참고

5. *"음향방출 신호를 이용한 복합결함 특성분석"* - Envelope Analysis 참고

---

## 👥 팀 정보

<div align="center">

### 🏅 KIST 베어링 신의 제자들

| 이름 | 소속 | 역할 |
|:---:|:---|:---|
| **김진욱** | 서울과학기술대학교 | 신호처리 및 모델링 |
| **서승일** | 서울과학기술대학교 | 데이터 분석 및 특징 추출 |

</div>

---

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

```
MIT License

Copyright (c) 2025 KIST 베어링 신의 제자들

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

**Made with ❤️ by KIST 베어링 신의 제자들**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/your-username/RUL-Prediction-using-Bearing-Degradation-Data)

</div>
