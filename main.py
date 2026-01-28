import pandas as pd
import numpy as np
from pyod.models.ecod import ECOD
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 로드 및 시계열 Windowing 함수
# ============================================================
def create_rolling_windows(df, window_size=5):
    """Run별로 그룹화하여 이전 시점의 데이터를 옆으로 붙여주는 함수"""
    id_cols = ['faultNumber', 'simulationRun', 'sample']
    features = [col for col in df.columns if col not in id_cols]
    
    df_result = df.copy()
    
    # 주요 센서 위주로 과거 데이터(Lag) 생성
    # 모든 변수를 다 붙이면 차원이 너무 커지므로 핵심 센서 위주로 시도하세요.
    for col in features:
        for i in range(1, window_size + 1):
            df_result[f'{col}_lag_{i}'] = df.groupby('simulationRun')[col].shift(i)
    
    # 결측치는 각 Run의 첫 데이터로 채움 (또는 0으로 채움)
    return df_result.fillna(method='bfill')

print("데이터 로딩 및 시계열 처리 중...")
DATA_PATH = "./data/"
train = pd.read_csv(f"{DATA_PATH}train.csv") #
test = pd.read_csv(f"{DATA_PATH}test.csv")   #

# 윈도우 생성 (과거 5개 시점 추가)
train_win = create_rolling_windows(train, window_size=5)
test_win = create_rolling_windows(test, window_size=5)

# 피처 추출 (ID 컬럼 제외)
id_cols = ['faultNumber', 'simulationRun', 'sample']
feature_cols = [col for col in train_win.columns if col not in id_cols]

X_train = train_win[feature_cols].values
X_test = test_win[feature_cols].values

# ============================================================
# 2. 전처리 (Scaling & PCA)
# ============================================================
# 윈도우 생성으로 늘어난 피처의 노이즈를 PCA로 압축합니다.
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.98, random_state=42) # 정보 손실을 줄이기 위해 0.98 사용
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"최종 학습 피처 개수: {X_train_pca.shape[1]}")

# ============================================================
# 3. ECOD 모델 학습 및 임계값 튜닝
# ============================================================
print("최종 모델 학습 시작...")
model = ECOD(contamination=0.1) #
model.fit(X_train_pca)

# 점수 기반 임계값 설정
train_scores = model.decision_scores_
test_scores = model.decision_function(X_test_pca)

# [꿀팁] 61점대에서 더 올리려면 90퍼센타일을 85~92 사이로 미세하게 바꿔보세요.
threshold = np.percentile(train_scores, 88) 
test_predictions = (test_scores > threshold).astype(int)

# ============================================================
# 4. 제출 파일 생성
# ============================================================
sample_submission = pd.read_csv(f"{DATA_PATH}sample_submission.csv")
output = sample_submission.copy()
output[sample_submission.columns[-1]] = test_predictions #

output.to_csv('main_windowed_output.csv', index=False)
print(f"제출 파일 저장 완료! 예측된 이상치 비율: {np.mean(test_predictions):.4f}")