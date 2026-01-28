import pandas as pd
import numpy as np
from pyod.models.ecod import ECOD
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 로드 및 전처리
# ============================================================
print("데이터 로딩 중...")
DATA_PATH = "./data/"
train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")

# 피처 컬럼 추출 (ID성 컬럼 제외)
id_cols = ['faultNumber', 'simulationRun', 'sample']
feature_cols = [col for col in train.columns if col not in id_cols]

X_train = train[feature_cols].values
X_test = test[feature_cols].values

# 1) RobustScaler: 이상치에 강한 스케일링 적용
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2) PCA: 변수 간 상관관계를 이용해 노이즈 제거 (설명력 95% 유지)
# 화학 공정 데이터의 다중공선성 문제를 해결해줍니다.
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA 적용 후 피처 개수: {X_train_pca.shape[1]}")

# ============================================================
# 2. ECOD 모델 학습
# ============================================================
# 훈련 데이터(정상)의 분포를 학습합니다.
print("ECOD 모델 학습 시작...")
model = ECOD(contamination=0.1) # 예상되는 이상치 비율
model.fit(X_train_pca)

# ============================================================
# 3. 임계값(Threshold) 최적화 및 예측
# ============================================================
# 훈련 데이터에서 모델이 산출한 '이상치 점수'를 가져옵니다.
train_scores = model.decision_scores_
test_scores = model.decision_function(X_test_pca)

# [전략] 점수 상위 10% 지점을 임계값으로 설정 (필요시 90을 85~95 사이로 조정)
threshold = np.percentile(train_scores, 90) 
test_predictions = (test_scores > threshold).astype(int)

print(f"예측된 이상치 개수: {np.sum(test_predictions)} / 전체 {len(test_predictions)}")

# ============================================================
# 4. 제출 파일 생성
# ============================================================
sample_submission = pd.read_csv(f"{DATA_PATH}sample_submission.csv")
output = sample_submission.copy()

# 샘플 제출 파일의 마지막 컬럼명에 맞게 예측값 대입
target_col = sample_submission.columns[-1]
output[target_col] = test_predictions

output.to_csv('main_improved_output.csv', index=False)
print("제출 파일 저장 완료: main_improved_output.csv")