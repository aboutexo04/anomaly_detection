import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from pyod.models.ecod import ECOD
import warnings
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 1. 데이터 로드 (최소한의 메모리)
# ============================================================
print("1. [Start] 데이터 로드...")
DATA_PATH = "./data/"
train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")

id_cols = ['faultNumber', 'simulationRun', 'sample']
feature_cols = [col for col in train.columns if col not in id_cols]

# 메모리 다이어트
def reduce_mem_usage(df):
    for col in feature_cols:
        df[col] = df[col].astype(np.float32)
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

def add_stats_features(df):
    # 연산 결과를 바로 float32로 변환
    df['row_mean'] = df[feature_cols].mean(axis=1).astype(np.float32)
    df['row_std'] = df[feature_cols].std(axis=1).astype(np.float32)
    df['row_max'] = df[feature_cols].max(axis=1).astype(np.float32)
    df['row_min'] = df[feature_cols].min(axis=1).astype(np.float32)
    return df

train = add_stats_features(train)
test = add_stats_features(test)

new_feature_cols = [col for col in train.columns if col not in id_cols]

# ============================================================
# 2. 가상 고장 주입
# ============================================================
print("2. 가상 고장 생성 중...")
X_normal = train[new_feature_cols]
n_anomalies = int(len(train) * 0.4)
idx = np.random.choice(train.index, n_anomalies, replace=False)

X_anomaly = train.loc[idx, new_feature_cols].copy()
cols_to_distort = [col for col in new_feature_cols if 'row_' not in col]

for i, r_idx in enumerate(X_anomaly.index):
    col = np.random.choice(cols_to_distort)
    op = i % 3
    if op == 0: X_anomaly.loc[r_idx, col] *= 1.5
    elif op == 1: X_anomaly.loc[r_idx, col] += (train.loc[r_idx, col].std() * 2.0)
    elif op == 2: X_anomaly.loc[r_idx, col] = train.loc[r_idx, col].mean()

# 합치고 원본 삭제
X_train = pd.concat([X_normal, X_anomaly], ignore_index=True)
y_train = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])

del train, X_normal, X_anomaly
gc.collect()

# ============================================================
# 3. 1차 학습 (가볍게)
# ============================================================
print("3. 1차 모델 학습...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test[new_feature_cols])

del X_train
gc.collect()

# [메모리 안전장치] n_jobs=4 (전체 다 쓰면 메모리 복사됨), 나무 100개
rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=4, random_state=42)
rf.fit(X_train_scaled, y_train)
score_rf = rf.predict_proba(X_test_scaled)[:, 1]

# ECOD (데이터 30%만 샘플링해서 학습 -> 메모리 절약)
ecod = ECOD(n_jobs=4)
idx_sample = np.random.choice(len(X_train_scaled)//2, int(len(X_train_scaled)*0.3), replace=False)
ecod.fit(X_train_scaled[idx_sample]) 
score_ecod = ecod.decision_function(X_test_scaled)

# 1차 점수
def normalize(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
final_score_1st = 0.6 * normalize(score_rf) + 0.4 * normalize(score_ecod)

del rf, ecod
gc.collect()

# ============================================================
# 4. Pseudo-Labeling
# ============================================================
print("4. Pseudo-Labeling 데이터 병합...")
test_temp = test[['simulationRun']].copy()
test_temp['score'] = final_score_1st
run_stats = test_temp.groupby('simulationRun')['score'].agg(['mean', 'max']).reset_index()
run_stats['run_score'] = 0.5 * normalize(run_stats['mean']) + 0.5 * normalize(run_stats['max'])

# Threshold 74% 적용
THRESHOLD_PCT = 74 
th_val = np.percentile(run_stats['run_score'], THRESHOLD_PCT)
anomaly_run_ids = run_stats[run_stats['run_score'] > th_val]['simulationRun'].values

print(f"  - 확보된 고장 Run ID: {len(anomaly_run_ids)}개")

# 데이터 병합
X_pseudo_anomaly = test[test['simulationRun'].isin(anomaly_run_ids)][new_feature_cols]
X_pseudo_scaled = scaler.transform(X_pseudo_anomaly)
y_pseudo = np.ones(len(X_pseudo_anomaly))

X_train_final = np.vstack([X_train_scaled, X_pseudo_scaled])
y_train_final = np.concatenate([y_train, y_pseudo])

# 기존 데이터 삭제 (가장 중요)
del X_train_scaled, X_pseudo_scaled, y_train, y_pseudo
gc.collect()

print(f"  - 최종 학습 데이터 크기: {X_train_final.shape}")

# ============================================================
# 5. 2차 재학습 (안전 모드)
# ============================================================
print("5. 2차 재학습 (진행 중)...")

# [핵심] 여기서 멈추지 않도록 나무 수를 150개로 제한하고 n_jobs를 줄임
rf_boost = RandomForestClassifier(n_estimators=150, max_depth=18, n_jobs=4, random_state=42)
rf_boost.fit(X_train_final, y_train_final)

score_rf_boost = rf_boost.predict_proba(X_test_scaled)[:, 1]
final_score_boost = 0.6 * normalize(score_rf_boost) + 0.4 * normalize(score_ecod)

# 결과 집계
test_temp['score_boost'] = final_score_boost
run_stats_boost = test_temp.groupby('simulationRun')['score_boost'].agg(['mean', 'max']).reset_index()
run_stats_boost['run_score'] = 0.5 * normalize(run_stats_boost['mean']) + 0.5 * normalize(run_stats_boost['max'])

print("6. 파일 저장 시작...")
save_thresholds = [74] # 시간 아까우니 74 하나만 집중 공략

for pct in save_thresholds:
    th_val = np.percentile(run_stats_boost['run_score'], pct)
    run_stats_boost['is_anomaly'] = (run_stats_boost['run_score'] > th_val).astype(int)
    
    # 순서 보장을 위해 merge 사용
    final_map = test[['simulationRun']].merge(run_stats_boost[['simulationRun', 'is_anomaly']], on='simulationRun', how='left')
    predictions = final_map['is_anomaly'].values
    
    filename = f'submission_pseudo_{pct}.csv'
    pd.DataFrame({'faultNumber': predictions}).to_csv(filename, index=True)
    print(f"  >>> [성공] 저장 완료: {filename}")

print("모든 과정 완료.")