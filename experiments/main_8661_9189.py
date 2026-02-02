!pip install lightgbm
# 설치 안 되면 터미널에 pip install lightgbm 입력

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from pyod.models.ecod import ECOD
import warnings
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 1. 데이터 로드 (성공한 코드와 동일하게 구성)
# ============================================================
print("1. 데이터 로드 및 최적화...")
DATA_PATH = "./data/"
train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")

# sample_submission 로드 안 함 (오염 원천 차단)
# 메모리 절약을 위한 32비트 변환
id_cols = ['faultNumber', 'simulationRun', 'sample']
feature_cols = [col for col in train.columns if col not in id_cols]

def reduce_mem_usage(df):
    for col in feature_cols:
        df[col] = df[col].astype(np.float32)
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# ============================================================
# 2. 시계열 피처 추가 (90점대를 위한 LightGBM용 피처)
# ============================================================
print("2. 피처 엔지니어링 (Lag & Rolling)...")

def add_time_features(df):
    # Lag-1 & Rolling Mean
    df[[f'{c}_lag1' for c in feature_cols]] = df.groupby('simulationRun')[feature_cols].shift(1).fillna(method='bfill')
    df[[f'{c}_rmean' for c in feature_cols]] = df.groupby('simulationRun')[feature_cols].transform(lambda x: x.rolling(5, min_periods=1).mean())
    return df

train = add_time_features(train)
test = add_time_features(test)

# 피처 리스트 업데이트
new_feature_cols = [col for col in train.columns if col not in id_cols]

# ============================================================
# 3. 가상 고장 주입 (Hard Mode)
# ============================================================
print("3. 가상 고장 데이터 생성...")

def inject_hard_faults(df, ratio=0.5):
    synthetic = df.copy()
    n_anomalies = int(len(df) * ratio)
    idx = np.random.choice(df.index, n_anomalies, replace=False)
    
    # Lag 피처 제외하고 원본 xmeas만 선택 (에러 방지용 필터)
    cols_to_distort = [c for c in df.columns if 'xmeas' in c and not c.endswith(('_lag1', '_rmean'))]
    if not cols_to_distort: cols_to_distort = [c for c in df.columns if 'xmeas' in c]

    for i, r_idx in enumerate(idx):
        col = np.random.choice(cols_to_distort)
        op = i % 3
        if op == 0: synthetic.loc[r_idx, col] *= 1.3
        elif op == 1: synthetic.loc[r_idx, col] += synthetic[col].std() * 2
        elif op == 2: synthetic.loc[r_idx, col] = synthetic[col].mean()
    return synthetic.loc[idx]

X_normal = train[new_feature_cols]
X_anomaly = inject_hard_faults(train[new_feature_cols], ratio=0.5)

X_train = pd.concat([X_normal, X_anomaly])
y_train = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])

del train, X_normal, X_anomaly
gc.collect()

# ============================================================
# 4. LightGBM 학습 (점수 상승 핵심)
# ============================================================
print("4. LightGBM 학습 중...")
lgbm = LGBMClassifier(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42
)
lgbm.fit(X_train, y_train)
score_lgbm = lgbm.predict_proba(test[new_feature_cols])[:, 1]

print("   ECOD 보조 학습 중...")
ecod = ECOD(n_jobs=-1)
# ECOD는 원본 피처만 사용
ecod.fit(X_train[feature_cols][:len(y_train)//2])
score_ecod = ecod.decision_function(test[feature_cols])

def normalize(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
final_score = 0.75 * normalize(score_lgbm) + 0.25 * normalize(score_ecod)

# ============================================================
# 5. Run 단위 판정
# ============================================================
print("5. 결과 집계...")
test_temp = test[['simulationRun']].copy()
test_temp['score'] = final_score

run_stats = test_temp.groupby('simulationRun')['score'].agg(['mean', 'max', 'std']).reset_index()
run_stats['run_score'] = (
    0.4 * normalize(run_stats['mean']) + 
    0.4 * normalize(run_stats['max']) + 
    0.2 * normalize(run_stats['std'].fillna(0))
)

threshold = np.percentile(run_stats['run_score'], 79) 
run_stats['is_anomaly'] = (run_stats['run_score'] > threshold).astype(int)

# ============================================================
# 6. 제출 파일 생성 (★방금 성공한 코드와 100% 동일★)
# ============================================================
print("6. 제출 파일 저장 중 (성공한 방식 적용)...")
final_map = test[['simulationRun']].merge(run_stats[['simulationRun', 'is_anomaly']], on='simulationRun')
predictions = final_map['is_anomaly'].values

# [핵심] 보여주신 성공 코드 그대로 사용합니다.
# 1. 예측값만으로 DataFrame 생성
# 2. index=True로 저장 (이렇게 하면 자동으로 0,1,2... 인덱스가 생기면서 콤마가 붙습니다)
output = pd.DataFrame({'faultNumber': predictions})
output.to_csv('submission_lgbm_success.csv', index=True)

print("-" * 60)
print("성공! 'submission_lgbm_success.csv' 파일이 생성되었습니다.")
print("이전 코드와 똑같은 방식으로 저장했으므로 이번엔 무조건 됩니다.")
print("-" * 60)