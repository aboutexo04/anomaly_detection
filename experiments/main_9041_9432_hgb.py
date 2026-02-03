import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier # [교체 완료]
from pyod.models.ecod import ECOD
import warnings
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 1. 데이터 로드 (메모리 최적화)
# ============================================================
print("1. [Final Hybrid] 데이터 로드 및 피처 생성...")
DATA_PATH = "./data/"
train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")

id_cols = ['faultNumber', 'simulationRun', 'sample']
feature_cols = [col for col in train.columns if col not in id_cols]

def reduce_mem_usage(df):
    for col in feature_cols:
        df[col] = df[col].astype(np.float32)
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# ============================================================
# [필살기] Skew + Kurt + Diff (모든 무기 장착)
# ============================================================
def add_stats_features(df):
    # 즉시 float32로 변환하여 메모리 절약
    df['row_mean'] = df[feature_cols].mean(axis=1).astype(np.float32)
    df['row_std'] = df[feature_cols].std(axis=1).astype(np.float32)
    df['row_max'] = df[feature_cols].max(axis=1).astype(np.float32)
    df['row_min'] = df[feature_cols].min(axis=1).astype(np.float32)
    
    df['row_skew'] = df[feature_cols].skew(axis=1).astype(np.float32)
    df['row_kurt'] = df[feature_cols].kurt(axis=1).astype(np.float32)
    
    q1 = df[feature_cols].quantile(0.25, axis=1)
    q3 = df[feature_cols].quantile(0.75, axis=1)
    df['row_iqr'] = (q3 - q1).astype(np.float32)

    df_diff = df[feature_cols].diff(axis=1).abs()
    df['row_diff_mean'] = df_diff.mean(axis=1).astype(np.float32)
    df['row_diff_max'] = df_diff.max(axis=1).astype(np.float32)
    
    del df_diff, q1, q3
    gc.collect()
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

# RF
rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=4, random_state=42)
rf.fit(X_train_scaled, y_train)
score_rf = rf.predict_proba(X_test_scaled)[:, 1]

# ECOD (샘플링으로 메모리 보호)
ecod = ECOD(n_jobs=4)
idx_sample = np.random.choice(len(X_train_scaled)//2, int(len(X_train_scaled)*0.3), replace=False)
ecod.fit(X_train_scaled[idx_sample]) 
score_ecod = ecod.decision_function(X_test_scaled)

def normalize(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
final_score_1st = 0.6 * normalize(score_rf) + 0.4 * normalize(score_ecod)

del rf, ecod
gc.collect()

# ============================================================
# 4. Pseudo-Labeling (0.9041 달성 기준)
# ============================================================
print("4. Pseudo-Labeling (Threshold 74)...")
test_temp = test[['simulationRun']].copy()
test_temp['score'] = final_score_1st
run_stats = test_temp.groupby('simulationRun')['score'].agg(['mean', 'max']).reset_index()
run_stats['run_score'] = 0.5 * normalize(run_stats['mean']) + 0.5 * normalize(run_stats['max'])

THRESHOLD_PCT = 74 
th_val = np.percentile(run_stats['run_score'], THRESHOLD_PCT)
anomaly_run_ids = run_stats[run_stats['run_score'] > th_val]['simulationRun'].values

X_pseudo_anomaly = test[test['simulationRun'].isin(anomaly_run_ids)][new_feature_cols]
X_pseudo_scaled = scaler.transform(X_pseudo_anomaly)
y_pseudo = np.ones(len(X_pseudo_anomaly))

X_train_final = np.vstack([X_train_scaled, X_pseudo_scaled])
y_train_final = np.concatenate([y_train, y_pseudo])

del X_train_scaled, X_pseudo_scaled, y_train, y_pseudo
gc.collect()

# ============================================================
# 5. [Final] 하이브리드 학습 (XGB 대체 -> HistGradientBoosting)
# ============================================================
print("5. 하이브리드 엔진 가동 (RF + HistGradientBoosting)...")

# [Step A] Random Forest
print("  - (1/2) Random Forest 학습 중...")
rf_boost = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=4, random_state=42)
rf_boost.fit(X_train_final, y_train_final)
pred_rf = rf_boost.predict_proba(X_test_scaled)[:, 1]

# [중요] RF 모델 삭제하여 메모리 확보
del rf_boost
gc.collect()

# [Step B] HistGradientBoosting (XGBoost와 동급, 설치 불필요)
print("  - (2/2) HistGradientBoosting 학습 중 (XGB 대체용)...")
# XGBoost의 'tree_method=hist'와 동일한 알고리즘입니다.
hgb = HistGradientBoostingClassifier(
    max_iter=300,        # XGB의 n_estimators 대응
    learning_rate=0.05,  # 학습률 동일
    max_depth=6,         # 깊이 동일
    random_state=42
)
hgb.fit(X_train_final, y_train_final)
pred_hgb = hgb.predict_proba(X_test_scaled)[:, 1]

# 메모리 정리
del hgb, X_train_final
gc.collect()

# [Step C] Blending
print("  - 최종 점수 Blending...")
final_prob = 0.5 * normalize(pred_rf) + 0.5 * normalize(pred_hgb)
final_score_hybrid = 0.9 * final_prob + 0.1 * normalize(score_ecod)

# 결과 집계
test_temp['score_hybrid'] = final_score_hybrid
run_stats_boost = test_temp.groupby('simulationRun')['score_hybrid'].agg(['mean', 'max']).reset_index()
run_stats_boost['run_score'] = 0.5 * normalize(run_stats_boost['mean']) + 0.5 * normalize(run_stats_boost['max'])

print("6. 저장: 74 & 74.5")
save_thresholds = [74, 74.5] 

for pct in save_thresholds:
    th_val = np.percentile(run_stats_boost['run_score'], pct)
    run_stats_boost['is_anomaly'] = (run_stats_boost['run_score'] > th_val).astype(int)
    
    final_map = test[['simulationRun']].merge(run_stats_boost[['simulationRun', 'is_anomaly']], on='simulationRun', how='left')
    predictions = final_map['is_anomaly'].values
    
    filename = f'submission_hybrid_hgb_{pct}.csv'
    pd.DataFrame({'faultNumber': predictions}).to_csv(filename, index=True)
    print(f"  >>> 저장 완료: {filename}")

print("-" * 60)
print("완료! 'submission_hybrid_hgb_74.csv' 제출하세요.")
print("XGBoost 에러 없이 바로 돌아갑니다!")
print("-" * 60)