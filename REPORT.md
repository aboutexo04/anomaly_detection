# TEP Anomaly Detection Competition Report

## 1. 대회 개요

### 1.1 문제 정의
Tennessee Eastman Process(TEP) 시뮬레이션 데이터에서 **run 단위 이상 탐지**를 수행하는 대회.
화학 공정의 52개 센서 데이터를 분석하여, 각 시뮬레이션 run이 정상인지 고장인지 판별한다.

### 1.2 평가 지표
- **F1 Score** (primary)
- **Accuracy** (secondary)

### 1.3 최종 성적
| 지표 | 점수 |
|---|---|
| **F1 Score** | **0.9524** |
| **Accuracy** | **0.9703** |
| **순위** | **단독 1위** |

---

## 2. 데이터 분석 (EDA)

### 2.1 데이터 구조

| | Train | Test |
|---|---|---|
| Shape | 250,000 × 55 | 710,400 × 54 |
| Runs | 500 | 740 |
| Samples/Run | 500 | 960 |
| faultNumber | 0 (전부 정상) | 없음 (예측 대상) |
| 결측치 | 0 | 0 |

### 2.2 피처 구성 (52개 센서)

| 구분 | 센서 | 개수 | 설명 |
|---|---|---|---|
| Measurements | xmeas_1 ~ xmeas_41 | 41 | 공정 측정값 (온도, 압력, 유량, 조성 등) |
| Manipulated Variables | xmv_1 ~ xmv_11 | 11 | 제어 변수 (밸브 개도 등) |

### 2.3 주요 센서 통계 (Train 기준)

| 센서 | 평균 | 표준편차 | 특성 |
|---|---|---|---|
| xmeas_1 | 0.250 | 0.031 | A feed (stream 1) |
| xmeas_7 | 2705.0 | 7.53 | Reactor pressure |
| xmeas_9 | 120.4 | 0.019 | Reactor temperature (매우 안정) |
| xmeas_19 | 232.2 | 10.36 | Stripper steam flow (변동 큼) |
| xmv_3 | 24.6 | 3.04 | A feed flow valve (변동 큼) |
| xmv_7 | 38.1 | 2.97 | Purge valve (변동 큼) |

### 2.4 핵심 관찰
- **Train**: 500개 run이 **모두 정상** (faultNumber=0). 정상 패턴의 baseline 역할
- **Test**: 740개 run 중 일부가 고장. 고장 run에서는 fault가 sample 161부터 시작 (앞 160개는 정상)
- **라벨 구조**: Ground truth는 **run 단위** (한 run 내 모든 sample이 동일 라벨)
- **최적 이상 비율**: 740개 run 중 약 218개(29.5%)가 이상 → F1 최적

---

## 3. 접근 방법론

### 3.1 전체 파이프라인

```
[Phase 1] 대회 Train 데이터 → 정상 baseline 구축 (Z-score, PCA, ECOD, 상관행렬)
     ↓
[Phase 2] 대회 Test 데이터 → 8개 run-level 이상 메트릭 계산
     ↓
[Phase 3] 외부 Rieth TEP 데이터 (10,500 labeled runs) → 동일 메트릭 계산
     ↓
[Phase 4] RF + HGB 지도학습 모델 학습
     ↓
[Phase 5] Base8 투표 + 지도학습 borderline 재판정 → 최종 예측
```

### 3.2 8개 핵심 메트릭 (Base8)

| # | 메트릭 | 계산 방식 | 직관 |
|---|---|---|---|
| 1 | z_count3_mean | \|Z\| > 3인 센서 수의 run 평균 | 극단적 이탈 센서 수 |
| 2 | z_count2_mean | \|Z\| > 2인 센서 수의 run 평균 | 중간 이탈 센서 수 |
| 3 | z_sum_sq_mean | Z² 합의 run 평균 | 전체 이탈 에너지 |
| 4 | t2_mean | Hotelling's T² run 평균 | PCA 주성분 공간 이탈 |
| 5 | spe_mean | SPE(잔차 제곱합) run 평균 | PCA 잔차 공간 이탈 |
| 6 | ecod_mean | ECOD 이상 점수 run 평균 | 경험적 분포 기반 이탈 |
| 7 | corr_dist | 센서 상관행렬 유클리드 거리 | 센서 간 관계 변화 |
| 8 | trend_half_spe | SPE(후반부) - SPE(전반부) | 시간에 따른 이탈 증가 추세 |

### 3.3 Majority Vote (2of8)
- 각 메트릭의 74th percentile을 threshold로 사용
- 8개 중 **2개 이상** threshold 초과 → 이상으로 판정
- 결과: **216개** anomaly run (29.2%)

### 3.4 외부 데이터 지도학습

#### Rieth TEP 데이터 (Harvard Dataverse)
| 구분 | Runs | Samples/Run | 총 Samples |
|---|---|---|---|
| Normal (fault_free_testing) | 500 | 960 | 480,000 |
| Faulty (faulty_testing) | 10,000 (20 fault × 500) | 960 | 9,600,000 |
| **합계** | **10,500** | - | **10,080,000** |

- 대회 test와 동일한 960 samples/run 구조 사용
- 8개 메트릭을 동일 baseline으로 계산 → 라벨과 함께 지도학습

#### 모델
| 모델 | 설정 | 역할 |
|---|---|---|
| RandomForest | n_estimators=300, max_depth=20, balanced | 앙상블 확률 50% |
| HistGradientBoosting | max_iter=500, lr=0.05, max_depth=6 | 앙상블 확률 50% |

#### Feature Importance (RF)
| 메트릭 | 중요도 |
|---|---|
| spe_mean | **0.252** |
| t2_mean | 0.190 |
| z_count3_mean | 0.161 |
| z_sum_sq_mean | 0.157 |
| z_count2_mean | 0.092 |
| trend_half_spe | 0.077 |
| corr_dist | 0.040 |
| ecod_mean | 0.029 |

### 3.5 최종 전략: Base8 + Borderline Expansion

```
Base8 votes ≥ 2  →  확정 이상 (216개)
Base8 votes = 1  →  "경계선" run
  └─ supervised probability > 70th percentile  →  이상으로 승격 (+2개)
Base8 votes = 0  →  확정 정상

최종 결과: 218개 이상 run
```

---

## 4. 실험 이력

### 4.1 스코어 진행표

| 단계 | F1 | Acc | 방법 | 이상 수 |
|---|---|---|---|---|
| 1 | 0.6193 | 0.8207 | 초기 baseline | - |
| 2 | 0.6722 | 0.8398 | 개선 baseline | - |
| 3 | 0.8571 | 0.9189 | Z-score 기반 | - |
| 4 | 0.8732 | 0.9270 | Output 75% threshold | - |
| 5 | 0.8899 | 0.9351 | 74% threshold | - |
| 6 | 0.8991 | 0.9405 | Enhanced features | - |
| 7 | 0.9041 | 0.9432 | RF+ECOD+PseudoLabeling | ~216 |
| 8 | 0.9339 | 0.9595 | **Majority Vote 2of7** | 216 |
| 9 | 0.9345 | 0.9595 | 2of8 (+time-trend z) | 216 |
| 10 | 0.9391 | 0.9622 | 2of8 (+T² trend) | 216 |
| 11 | 0.9478 | 0.9676 | **2of8 (+SPE trend)** | 216 |
| 12 | 0.9520 | 0.9703 | **Rieth sup_top216** | 216 |
| **13** | **0.9524** | **0.9703** | **Rieth border_sup70** | **218** |

### 4.2 단계별 실험 스토리

---

#### Stage 1: Baseline ~ RF+ECOD (F1 0.62 → 0.90)

가장 기본적인 접근부터 시작했다. Train 데이터(정상 500 runs)의 평균/표준편차로 Test 데이터를 Z-score 변환하고, 이탈이 큰 run을 이상으로 판별했다.

**시도한 것들:**
- 단순 Z-score 기반 이상 탐지 → threshold를 75%에서 74%로 조정하며 점수 향상
- ECOD(경험적 분포 기반 이상탐지)를 추가하고 RandomForest로 앙상블
- Pseudo Labeling (자기학습) 기법으로 반복 학습

**결과:** F1 0.9041까지 도달. 하지만 여기서 **벽**에 부딪혔다.
어떤 모델을 쓰든, 하이퍼파라미터를 바꾸든 0.9041을 넘을 수 없었다.

> 교훈: 단일 모델의 한계. 하나의 이상 점수에 의존하면 천장이 있다.

---

#### Stage 2: Majority Vote 돌파 (F1 0.90 → 0.94)

발상을 전환했다. 하나의 점수 대신, **여러 독립적인 이상 메트릭을 만들어 투표**시키면 어떨까?

7가지 서로 다른 관점의 이상 메트릭을 설계했다:
1. Z-score 기반 3종 (극단적 이탈, 중간 이탈, 이탈 에너지)
2. PCA 기반 2종 (주성분 이탈 T², 잔차 이탈 SPE)
3. ECOD (경험적 분포 이탈)
4. 상관 거리 (센서 간 관계 변화)

각 메트릭의 상위 26%(74th percentile)를 이상 후보로 놓고, **7개 중 2개 이상이 동의하면 이상**으로 판정 → 216개 run 선택.

**결과:** F1 0.9041 → **0.9339** (대점프!)

이후 8번째 메트릭을 추가하는 실험을 반복했다:
- Z-score time-trend (+0.0006) → 미미
- T² time-trend (+0.0052) → 소폭 향상
- **SPE time-trend (+0.0087)** → 가장 큰 향상

SPE의 시간 추세(run 후반부가 전반부보다 SPE가 높은지)가 **새로운 정보**를 제공했다.

**최종:** 2of8 투표 → F1 **0.9478** (이 시점에서 2위)

> 교훈: 다양한 관점의 투표 > 단일 강력한 모델. 8번째 메트릭은 "시간적 패턴"이라는 기존과 다른 차원의 정보여야 효과가 있었다.

---

#### Stage 3: 벽을 넘기 위한 수많은 시도 (전부 실패)

0.9478에서 1위(0.9520)까지 F1 0.004 차이. 이 작은 갭을 좁히기 위해 다양한 방법을 시도했지만, **전부 실패**했다.

**새로운 메트릭 시도 (결과: 동일한 216개)**
- CUSUM (누적합 관리도): Z-score와 동일한 run을 선택
- Mahalanobis 거리: 역시 동일
- Autoencoder (PyTorch): 학습된 복원 오류도 동일한 순위
- Kernel PCA, One-Class SVM: 마찬가지

> 발견: 진폭(amplitude) 기반의 모든 방법은 결국 같은 run 순위를 만든다. 이 데이터에서는 "얼마나 크게 이탈했는가"라는 질문에 모든 방법이 같은 답을 한다.

**다른 패러다임 시도 (결과: 악화)**
- FFT 주파수 영역 분석: 다른 run을 선택하지만 **틀린** run → 0.9351
- LSTM 시계열 예측: 동일한 216개 (temporal 패턴도 같은 정보)
- 센서별 독립 투표: noise 증폭 → 0.8089 (대폭락)
- Sample-level 라벨링: ground truth가 run-level이라 부적합 → 0.8489

**TEP 도메인 지식 활용 (결과: 악화)**
- 공정 섹션별(반응기/분리기/스트리퍼 등) 독립 PCA → 정보 손실로 악화
- 제어변수(xmv) 변동성 분석 → 동일
- 물리적 센서 쌍 관계 모니터링 → 동일
- 핵심 센서 가중 Z-score → 동일

> 교훈: 비지도학습만으로는 0.9478이 천장. 새로운 메트릭을 아무리 추가해도 같은 216개 run이 나온다. 근본적으로 다른 접근이 필요했다.

---

#### Stage 4: 외부 라벨 데이터로 돌파 (F1 0.95 → 0.95!)

대회 규칙상 **외부 데이터셋 사용이 허용**되어 있었다. TEP는 널리 연구된 벤치마크이므로, 라벨이 있는 외부 데이터로 지도학습을 할 수 있었다.

**1차 시도: Braatz 데이터 (21개 labeled runs)**
- 고전적인 TEP 데이터셋 (fault당 1개 run)
- RF+HGB로 학습 → base8과 2개 run만 다름
- 제출 결과: 0.9432 (**악화**). 2개 swap이 **틀렸다**.

**2차 시도: Rieth 데이터 (10,500개 labeled runs)**
- Harvard Dataverse에서 대규모 TEP 시뮬레이션 데이터 다운로드
- 500개 정상 run + 20개 fault type × 500 runs = **10,500개 라벨 데이터**
- 대회 test와 동일한 960 samples/run 구조 사용
- 동일한 8개 메트릭으로 run-level feature 추출 후 RF+HGB 학습

결과:
- **sup_top216** (3개 run swap): 제출 → F1 **0.9520** (1등 동점!)
- **base8_border_sup70** (base8 + borderline 2개 추가): 제출 → F1 **0.9524** (단독 1위!)

> 교훈: 같은 지도학습이라도 데이터 양이 결정적이었다. Braatz 21개 → 잘못된 swap. Rieth 10,500개(500배) → 올바른 swap. 충분한 데이터가 있어야 정확하게 교정할 수 있다.

---

## 5. 핵심 인사이트

### 5.1 "216의 법칙"
- 2of8 투표에서 74% threshold → 정확히 216개 anomaly run
- 216에서 벗어나면 급격히 악화: 212개(-4) → 0.9427, 219개(+3) → 0.9397
- 최종 218개(+2)는 지도학습으로 검증된 borderline만 추가했기에 성공

### 5.2 Amplitude 기반 방법의 수렴
- Z-score, PCA T²/SPE, ECOD, CUSUM, Mahalanobis, AE, LSTM 등 **모든 magnitude 기반 방법이 동일한 run 순위를 생성**
- 새로운 방법을 추가해도 기존 투표 결과와 동일
- **투표 다양성**이 핵심: 수학적으로 다른 패러다임 (Z-score vs PCA vs ECOD vs 상관)

### 5.3 데이터 양의 결정적 중요성
- Braatz 21개 labeled runs → 2개 잘못된 swap → 0.9432 (악화)
- Rieth 10,500개 labeled runs → 3개 올바른 swap → 0.9520 (1등 동점)
- **500배 더 많은 데이터**가 정확도를 결정

### 5.4 Per-fault 탐지 성능 (Rieth RF 기준)
| Fault | Detection Rate | 난이도 |
|---|---|---|
| Fault 3 | 89.8% | 어려움 |
| Fault 9 | 93.2% | 어려움 |
| Fault 15 | 95.8% | 약간 어려움 |
| 나머지 17개 | 100.0% | 쉬움 |
| Normal | 0.0% (정상 판정) | 정확 |

---

## 6. 환경 및 재현

### 6.1 실행 환경
- Platform: macOS (Darwin)
- Python: conda env `nlp_new`
- `KMP_DUPLICATE_LIB_OK=TRUE` (PyTorch+sklearn 공존)
- `n_jobs=4` (MacBook 메모리 안전)

### 6.2 주요 라이브러리
- pandas, numpy, scikit-learn
- pyod (ECOD)
- pyreadr (RData 파일 로드)

### 6.3 외부 데이터 출처
- **Rieth et al. (2017)**: "Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation"
  - Harvard Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1
  - 4개 RData 파일, 총 ~1.3GB

### 6.4 실행 방법
```bash
conda activate nlp_new
export KMP_DUPLICATE_LIB_OK=TRUE
cd /path/to/anomaly_detection
python main.py
```

### 6.5 코드 백업 파일
| 파일 | 점수 |
|---|---|
| `experiments/main_9041_9432_74.py` | 0.9041/0.9432 |
| `experiments/main_9339_9595_2of7_best.py` | 0.9339/0.9595 |
| `experiments/main_9478_9676_2of8_spe_trend.py` | 0.9478/0.9676 |
| `experiments/main_9520_9703_rieth_sup_top216.py` | 0.9520/0.9703 |
| `experiments/main_9524_9703_rieth_border_sup70.py` | **0.9524/0.9703 (최종)** |
