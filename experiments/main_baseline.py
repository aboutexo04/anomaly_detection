"""
화학 공정 이상 탐지 (Anomaly Detection) 베이스라인 코드
- 학습 데이터: 정상 데이터만 존재 (faultNumber = 0)
- 테스트 데이터: 정상/이상 모두 존재
- 평가 지표: F1-Score
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 로드
# ============================================================
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

# 데이터 경로 설정 (다운로드 후 압축 해제한 경로로 수정)
DATA_PATH = "./data/"

train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {train.columns.tolist()[:10]}...")
print(f"\nTrain faultNumber 분포:\n{train['faultNumber'].value_counts()}")

# ============================================================
# 2. 데이터 전처리
# ============================================================
print("\n" + "=" * 60)
print("2. 데이터 전처리")
print("=" * 60)

# 피처 컬럼 추출 (xmeas_*, xmv_* 변수들)
id_cols = ['faultNumber', 'simulationRun', 'sample']
feature_cols = [col for col in train.columns if col not in id_cols]

print(f"Feature 개수: {len(feature_cols)}")
print(f"Feature 예시: {feature_cols[:5]}")

# 학습/테스트 피처 추출
X_train = train[feature_cols].values
X_test = test[feature_cols].values

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nX_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

# ============================================================
# 3. Isolation Forest 모델 학습
# ============================================================
print("\n" + "=" * 60)
print("3. Isolation Forest 모델 학습")
print("=" * 60)

# Isolation Forest: 정상 데이터만으로 학습 (Unsupervised Anomaly Detection)
# contamination: 테스트 데이터에서 예상되는 이상치 비율 (튜닝 필요)
model = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.1,  # 이상치 비율 추정값 (조절 필요)
    max_features=1.0,
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

print("모델 학습 중...")
model.fit(X_train_scaled)
print("모델 학습 완료!")

# ============================================================
# 4. 예측 및 결과 생성
# ============================================================
print("\n" + "=" * 60)
print("4. 예측 및 결과 생성")
print("=" * 60)

# 예측 (Isolation Forest: 1=정상, -1=이상)
predictions = model.predict(X_test_scaled)

# 레이블 변환: -1(이상) -> 1, 1(정상) -> 0
# 대회 형식: 이상=1, 정상=0
test_predictions = np.where(predictions == -1, 1, 0)

print(f"예측 분포:")
print(f"  정상(0): {np.sum(test_predictions == 0)}")
print(f"  이상(1): {np.sum(test_predictions == 1)}")
print(f"  이상 비율: {np.mean(test_predictions):.4f}")

# ============================================================
# 5. 제출 파일 생성 (sample_submission 참고)
# ============================================================
print("\n" + "=" * 60)
print("5. 제출 파일 생성")
print("=" * 60)

# sample_submission.csv 로드하여 형식 확인
sample_submission = pd.read_csv(f"{DATA_PATH}sample_submission.csv")
print(f"Sample submission shape: {sample_submission.shape}")
print(f"Sample submission columns: {sample_submission.columns.tolist()}")
print(f"Sample submission 미리보기:\n{sample_submission.head()}")

# 행 개수 검증
assert len(test) == len(sample_submission), \
    f"행 개수 불일치! test: {len(test)}, sample: {len(sample_submission)}"

# sample_submission 복사 후 예측값 채우기
output = sample_submission.copy()
target_col = sample_submission.columns[-1]  # 마지막 컬럼이 타겟일 가능성 높음
print(f"\n타겟 컬럼: {target_col}")

output[target_col] = test_predictions

# 저장
output.to_csv('output.csv', index=False)
print(f"\n제출 파일 저장 완료: output.csv")
print(f"제출 파일 shape: {output.shape}")
print(f"\n제출 파일 미리보기:")
print(output.head(10))

# ============================================================
# 6. (선택) 검증용 코드 - 테스트 데이터에 정답이 있는 경우
# ============================================================
print("\n" + "=" * 60)
print("6. 추가 분석")
print("=" * 60)

# Anomaly Score 분석
anomaly_scores = model.decision_function(X_test_scaled)
print(f"Anomaly Score 통계:")
print(f"  Mean: {np.mean(anomaly_scores):.4f}")
print(f"  Std: {np.std(anomaly_scores):.4f}")
print(f"  Min: {np.min(anomaly_scores):.4f}")
print(f"  Max: {np.max(anomaly_scores):.4f}")

# Score 기반으로 threshold 조절 가능
# 더 높은 threshold = 더 적은 이상 탐지
# 더 낮은 threshold = 더 많은 이상 탐지

print("\n" + "=" * 60)
print("완료!")
print("=" * 60)
print("""
개선 방향:
1. contamination 파라미터 조절 (이상치 비율 추정)
2. 다른 모델 시도: One-Class SVM, AutoEncoder, Local Outlier Factor
3. 피처 엔지니어링: 시계열 특성 활용 (rolling mean, std 등)
4. 앙상블: 여러 모델의 예측 결합
5. threshold 최적화: anomaly score 기반 임계값 조정
""")