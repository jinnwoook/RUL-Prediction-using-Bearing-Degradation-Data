# KSPHM-KIMM 2025 베어링 RUL 예측 프로젝트 상세 분석 보고서

## 프로젝트 개요
- **대회명**: KSPHM-KIMM 2025 베어링 수명 예측 공모전
- **팀명**: KIST 베어링신의 제자들
- **성과**: 70팀 중 7등 달성
- **주제**: RUL(Remaining Useful Life) Prediction using Bearing Degradation Data

---

## 1. 신호처리 기법 분석

### 1.1 BPF (Band Pass Filter) - 1000~5000Hz
**적용 목적**:
- 베어링 고장의 특징 주파수 대역을 효과적으로 분리
- 고장 주파수의 배수 주파수들이 1000~5000Hz에 집중된다는 전역적 특성 활용

**기술적 근거**:
```
- 1-1000Hz: 저주파 대역 - 기계적 구조 진동, 잡음 많음
- 1000-5000Hz: 목표 대역 - 고장 신호의 배수 주파수 집중
- 8000Hz 이상: 고주파 대역 - 노이즈에 취약
```

**구현**:
- Butterworth 필터, Order=4 사용
- 정규화 주파수: [lowcut/nyq, highcut/nyq]
- filtfilt() 적용으로 Phase Distortion 제거

### 1.2 Envelope Analysis (포락선 분석)
**신호처리 파이프라인**:
1. BPF 필터링된 신호 입력
2. Hilbert Transform 적용
3. 복소 신호의 절댓값으로 포락선 추출: `envelope = |hilbert(signal)|`
4. RMS 값으로 추적

**장점**:
- 저에너지 신호를 효과적으로 검출
- 고주파 신호의 진폭 변화만을 분리하여 저주파 영역의 결함 반복 패턴 추출
- 신호가 잡음이 많고 복잡해도 결함 신호를 뚜렷하게 표현

### 1.3 Wavelet Transform (이산 웨이블릿 변환)
**Daubechies 4 (Db4) 선택 근거**:
- 베어링 결함 진단에 가장 널리 사용되는 웨이블릿
- 시간-주파수 분해를 통해 고장 시점 포착 및 열화 추적 용이

**분해 설정**:
- Level: 5 (D1~D5 + C5)
- 각 레벨에서 Detail coefficients 분석

### 1.4 STFT (Short-Time Fourier Transform)
**특징**:
- Window size: 0.5초 (12,800 samples @ 25.6kHz)
- 4개 필터 대역: 50-500Hz, 100-800Hz, 200-1000Hz, 300-1500Hz
- 각 시간 윈도우에서 주파수 영역 분석

**추출 피처**:
1. Target Frequency Amplitude (BPFI, BPFO, BSF, FTF)
2. Dominant Frequency
3. Max Amplitude
4. Total Band Energy

---

## 2. 추출된 피처 및 근거

### 2.1 최종 선정 피처

#### 피처 1: CH2_D4_RMS
- **출처**: Wavelet Level 4 (D4) Detail Coefficients
- **계산**: RMS of |D4 coefficients|
- **근거**: 저주파수 대역의 안정적인 지표, 고장 진행에 따라 점진적 상승

#### 피처 2: CH2_D5_RMS
- **출처**: Wavelet Level 5 (D5) Detail Coefficients
- **계산**: RMS of |D5 coefficients|
- **근거**: 결함 주파수가 기본적으로 저주파(~140Hz)에 위치, 하모닉으로 D4, D5에 전달

#### 피처 3: CH2_D5_Entropy
- **출처**: Wavelet Level 5 Entropy
- **계산**: `-sum(p * log2(p))` where p = normalized histogram bins
- **근거**: 진동 신호의 불확실성/복잡도 반영, 결함 발생 시 불규칙성 증가

#### 추가 피처: CH2_1000-5000Hz_BPF & CH2_1000-5000Hz_RMS_Envelope
- **채널 선택 근거**: CH2에서 모든 Train 파일에서 점진적 상승 관찰
- **대역 선택 근거**: 고장 신호의 특성 주파수 대역

---

## 3. 모델 아키텍처 분석

### 3.1 CNN-LSTM 하이브리드 구조

```
입력 (Batch, Time Steps, 15 Channels)
        ↓
    Conv1D (in=15, out=64, kernel=5) + ReLU
        ↓
    MaxPooling1D (kernel=2)
        ↓
    LSTM (input=64, hidden=64)
        ↓
    마지막 시점 선택 x[:, -1, :]
        ↓
    FC (64 → 32) + ReLU
        ↓
    FC (32 → 1) → RUL 예측
```

### 3.2 각 계층의 역할

| 계층 | 역할 |
|------|------|
| **Conv1D** | 지역 특징 추출 (진동의 작은 급변 패턴 감지) |
| **MaxPool** | 계산 효율 향상, 중요 특징 강조 |
| **LSTM** | 시간적 패턴, 추세, 변화 학습 |
| **FC** | RUL 회귀 출력 |

---

## 4. 데이터 파이프라인

### 4.1 전처리 단계

```
원본 TDMS 파일
    ↓
TDMS 읽기 (4채널 진동 + 운영 파라미터)
    ↓
윈도우 분할 (0.5초 = 12,800 샘플)
    ↓
피처 추출 (Wavelet + BPF + Envelope)
    ↓
정규화 (Mean-Std)
    ↓
CSV 저장 (Float32)
```

### 4.2 데이터 수집 조건
- 진동 신호 샘플링율: 25.6 kHz
- 온도, 토크 데이터: 0.1 Hz (10초당 1포인트)
- 수집 주기: 10분 간격, 각 10초 동안

---

## 5. 주요 하이퍼파라미터

### 5.1 신호처리 파라미터

| 파라미터 | 값 |
|---------|-----|
| Sampling Rate | 25,600 Hz |
| Window Size | 0.5초 (12,800 샘플) |
| BPF 대역 | 1,000~5,000 Hz |
| BPF Order | 4 |
| Wavelet | db4, Level 5 |

### 5.2 모델 파라미터

| 파라미터 | 값 |
|---------|-----|
| 입력 채널 | 15 |
| Conv1D Out/Kernel | 64 / 5 |
| LSTM Hidden | 64 |
| FC 출력 | 32 → 1 |

---

## 6. RUL 예측 평가 메트릭

### 에러율 (Error rate)
$$Er_i = 100 \times \frac{ActRUL_i - PredRUL_i}{ActRUL_i}$$

### 점수 함수 (비대칭 손실)
$$A_{RUL} = \begin{cases} \exp(-\ln(0.5) \times \frac{Er}{20}), & \text{if } Er \leq 0 \\ \exp(+\ln(0.5) \times \frac{Er}{50}), & \text{if } Er > 0 \end{cases}$$

### 최종 점수
$$FinalScore = \frac{1}{N} \sum_{i=1}^{N} A_{RUL}^{(i)}$$

---

## 7. 참고 문헌

1. Kumar et al. (2013), Wavelet transform for bearing condition monitoring and fault diagnosis: A review
2. Rafia Nishat Toma et al. (2020), Bearing Fault Classification of Induction Motors Using Discrete Wavelet Transform and Ensemble Machine Learning Algorithms
3. C K E Nizwana et al. (2016), A wavelet decomposition analysis of vibration signal for bearing fault detection

---

**분석 완료일**: 2026-02-09
