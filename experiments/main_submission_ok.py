"""
화학 공정 이상 탐지 - 고성능 버전
- 시계열 윈도우 + 통계 피처
- ECOD + Isolation Forest + LOF 앙상블
- 다양한 threshold 실험
"""

import pandas as pd
import numpy as np
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 로드
# ============================================================
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

DATA_PATH = "./data/"
train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")
sample_submission = pd.read_csv(f"{DATA_PATH}sample_submission.csv", index_col=0)

# Unnamed: 0 컬럼 제거 (train, test만)
if 'Unnamed: 0' in train.columns:
    train = train.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in test.columns:
    test = test.drop(columns=['Unnamed: 0'])

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

id_cols = ['faultNumber', 'simulationRun', 'sample']
base_feature_cols = [col for col in train.columns if col not in id_cols]

# ============================================================
# 2. 피처 엔지니어링 (강화)
# ============================================================
print("\n" + "=" * 60)
print("2. 피처 엔지니어링")
print("=" * 60)

def create_features(df, base_features):
    """종합 피처 엔지니어링"""
    df_new = df.copy()
    
    # xmeas, xmv 분리
    xmeas_cols = [col for col in base_features if 'xmeas' in col]
    xmv_cols = [col for col in base_features if 'xmv' in col]
    
    # 1. 행별 통계
    df_new['row_mean'] = df[base_features].mean(axis=1)
    df_new['row_std'] = df[base_features].std(axis=1)
    df_new['row_max'] = df[base_features].max(axis=1)
    df_new['row_min'] = df[base_features].min(axis=1)
    df_new['row_range'] = df_new['row_max'] - df_new['row_min']
    df_new['row_median'] = df[base_features].median(axis=1)
    df_new['row_q25'] = df[base_features].quantile(0.25, axis=1)
    df_new['row_q75'] = df[base_features].quantile(0.75, axis=1)
    df_new['row_iqr'] = df_new['row_q75'] - df_new['row_q25']
    
    # 2. xmeas / xmv 그룹 통계
    df_new['xmeas_mean'] = df[xmeas_cols].mean(axis=1)
    df_new['xmeas_std'] = df[xmeas_cols].std(axis=1)
    df_new['xmeas_max'] = df[xmeas_cols].max(axis=1)
    df_new['xmeas_min'] = df[xmeas_cols].min(axis=1)
    
    df_new['xmv_mean'] = df[xmv_cols].mean(axis=1)
    df_new['xmv_std'] = df[xmv_cols].std(axis=1)
    df_new['xmv_max'] = df[xmv_cols].max(axis=1)
    df_new['xmv_min'] = df[xmv_cols].min(axis=1)
    
    # 3. 그룹 간 관계
    df_new['xmeas_xmv_ratio'] = df_new['xmeas_mean'] / (df_new['xmv_mean'] + 1e-8)
    df_new['xmeas_xmv_diff'] = df_new['xmeas_mean'] - df_new['xmv_mean']
    df_new['xmeas_xmv_std_ratio'] = df_new['xmeas_std'] / (df_new['xmv_std'] + 1e-8)
    
    return df_new

def create_rolling_features(df, base_features, window_size=3):
    """시계열 Rolling 피처 (Run별)"""
    df_new = df.copy()
    
    # 주요 센서만 선택 (너무 많으면 노이즈)
    key_sensors = base_features[:20]  # 상위 20개 센서
    
    for col in key_sensors:
        # Lag 피처
        for lag in [1, 2, 3]:
            df_new[f'{col}_lag{lag}'] = df.groupby('simulationRun')[col].shift(lag)
        
        # Rolling 통계
        df_new[f'{col}_roll_mean'] = df.groupby('simulationRun')[col].transform(
            lambda x: x.rolling(window_size, min_periods=1).mean()
        )
        df_new[f'{col}_roll_std'] = df.groupby('simulationRun')[col].transform(
            lambda x: x.rolling(window_size, min_periods=1).std()
        )
        
        # 현재값 - Rolling 평균 (변화량)
        df_new[f'{col}_diff_from_mean'] = df[col] - df_new[f'{col}_roll_mean']
    
    return df_new.fillna(method='bfill').fillna(method='ffill').fillna(0)

print("기본 피처 생성 중...")
train_fe = create_features(train, base_feature_cols)
test_fe = create_features(test, base_feature_cols)

print("시계열 피처 생성 중...")
train_fe = create_rolling_features(train_fe, base_feature_cols, window_size=3)
test_fe = create_rolling_features(test_fe, base_feature_cols, window_size=3)

# 최종 피처
feature_cols = [col for col in train_fe.columns if col not in id_cols]
print(f"최종 피처 수: {len(feature_cols)}")

X_train = train_fe[feature_cols].values
X_test = test_fe[feature_cols].values

# ============================================================
# 3. 스케일링 + PCA
# ============================================================
print("\n" + "=" * 60)
print("3. 스케일링 + PCA")
print("=" * 60)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA로 차원 축소 (노이즈 제거 + 속도 향상)
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA 후 피처 수: {X_train_pca.shape[1]}")

# ============================================================
# 4. 다중 모델 앙상블
# ============================================================
print("\n" + "=" * 60)
print("4. 모델 학습 (앙상블)")
print("=" * 60)

# 모델 정의 (KNN 제외 - 너무 느림)
models = {
    'ECOD': ECOD(contamination=0.1),
    'COPOD': COPOD(contamination=0.1),
    'IForest1': IForest(n_estimators=300, contamination=0.1, random_state=42),
    'IForest2': IForest(n_estimators=300, contamination=0.12, max_features=0.7, random_state=123),
}

# 각 모델 학습 및 스코어 수집
all_scores = {}
for name, model in models.items():
    print(f"학습 중: {name}...")
    model.fit(X_train_pca)
    scores = model.decision_function(X_test_pca)
    # 정규화
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    all_scores[name] = scores_norm
    print(f"  완료! Score range: [{scores.min():.4f}, {scores.max():.4f}]")

# ============================================================
# 5. 앙상블 스코어 및 Threshold 최적화
# ============================================================
print("\n" + "=" * 60)
print("5. 앙상블 및 Threshold 최적화")
print("=" * 60)

# 가중 앙상블 (KNN 제외)
weights = {
    'ECOD': 0.3,
    'COPOD': 0.3,
    'IForest1': 0.25,
    'IForest2': 0.15,
}

ensemble_scores = np.zeros(len(X_test_pca))
for name, score in all_scores.items():
    ensemble_scores += weights[name] * score

print(f"앙상블 스코어 - Mean: {ensemble_scores.mean():.4f}, Std: {ensemble_scores.std():.4f}")

# 다양한 threshold 테스트
print("\nThreshold별 이상 비율:")
for pct in [80, 82, 85, 87, 88, 90, 92]:
    threshold = np.percentile(ensemble_scores, pct)
    preds = (ensemble_scores > threshold).astype(int)
    print(f"  {pct}th percentile: 이상 비율 = {preds.mean():.4f} ({preds.sum()}개)")

# ============================================================
# 6. 최종 예측 및 제출
# ============================================================
print("\n" + "=" * 60)
print("6. 제출 파일 생성")
print("=" * 60)

# Threshold 선택 (여러 값 시도해보세요!)
THRESHOLD_PCT = 85  # 80, 82, 85, 87, 88, 90 등 시도

threshold = np.percentile(ensemble_scores, THRESHOLD_PCT)
test_predictions = (ensemble_scores > threshold).astype(int)

print(f"선택 Threshold: {THRESHOLD_PCT}th percentile = {threshold:.4f}")
print(f"예측 - 정상: {(test_predictions==0).sum()}, 이상: {(test_predictions==1).sum()}")
print(f"이상 비율: {test_predictions.mean():.4f}")

# 저장 (인덱스 포함)
output = sample_submission.copy()
target_col = sample_submission.columns[-1]
output[target_col] = test_predictions
output.to_csv('output.csv', index=True)

print(f"\n제출 파일 저장: output.csv")
print(output.head())

print("\n" + "=" * 60)
print("완료!")
print("=" * 60)
print(f"""
점수가 안 나오면 THRESHOLD_PCT를 조절하세요:
- 현재: {THRESHOLD_PCT}
- F1 낮으면: 80, 82로 낮춰보기 (이상 더 많이 잡기)
- Precision 낮으면: 88, 90으로 높여보기 (이상 적게 잡기)
""")