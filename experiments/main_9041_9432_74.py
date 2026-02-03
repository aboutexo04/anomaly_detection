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
# 1. 데이터 로드
# ============================================================
print("1. [90점 돌파용] 데이터 로드 및 피처 엔지니어링...")
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
# [최종 비기] 고급 통계(Skew/Kurt) + 변화량(Diff) 통합
# ============================================================
def add_stats_features(df):
    # 1. 기본 통계
    df['row_mean'] = df[feature_cols].mean(axis=1).astype(np.float32)
    df['row_std'] = df[feature_cols].std(axis=1).astype(np.float32)
    df['row_max'] = df[feature_cols].max(axis=1).astype(np.float32)
    df['row_min'] = df[feature_cols].min(axis=1).astype(np.float32)
    
    # 2. 고급 통계 (Skew/Kurt) - 베스트 점수 만든 핵심
    print("  - 고급 통계 (Skew, Kurt, IQR) 계산 중...")
    df['row_skew'] = df[feature_cols].skew(axis=1).astype(np.float32)
    df['row_kurt'] = df[feature_cols].kurt(axis=1).astype(np.float32)
    
    q1 = df[feature_cols].quantile(0.25, axis=1)
    q3 = df[feature_cols].quantile(0.75, axis=1)
    df['row_iqr'] = (q3 - q1).astype(np.float32)

    # 3. [NEW] 변화량(Diff) - 90점 돌파를 위한 마지막 퍼즐
    # 센서값끼리의 급격한 차이를 잡아냄
    print("  - 변화량 (Diff) 계산 중...")
    df_diff = df[feature_cols].diff(axis=1).abs()
    df['row_diff_mean'] = df_diff.mean(axis=1).astype(np.float32)
    df['row_diff_max'] = df_diff.max(axis=1).astype(np.float32)
    
    # 메모리 폭발 방지: 다 쓴 임시 변수 즉시 삭제
    del df_diff, q1, q3
    gc.collect()
    
    return df

train = add_stats_features(train)
test = add_stats_features(test)

new_feature_cols = [col for col in train.columns if col not in id_cols]
print(f"  - 최종 피처 개수: {len(new_feature_cols)}개 (기본+Skew+Diff)")

# ============================================================
# 2. 가상 고장 주입 (Synthetic)
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

# Random Forest (메모리 보호: n_jobs=4)
rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=4, random_state=42)
rf.fit(X_train_scaled, y_train)
score_rf = rf.predict_proba(X_test_scaled)[:, 1]

# ECOD (샘플링 학습)
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
# 4. Pseudo-Labeling (베스트 74점 기준)
# ============================================================
print("4. Pseudo-Labeling 데이터 병합...")
test_temp = test[['simulationRun']].copy()
test_temp['score'] = final_score_1st
run_stats = test_temp.groupby('simulationRun')['score'].agg(['mean', 'max']).reset_index()
run_stats['run_score'] = 0.5 * normalize(run_stats['mean']) + 0.5 * normalize(run_stats['max'])

# 0.899점을 만든 '74%' 기준 유지
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

del X_train_scaled, X_pseudo_scaled, y_train, y_pseudo
gc.collect()

# ============================================================
# 5. 2차 재학습 (Full Power)
# ============================================================
print("5. 2차 재학습 (Skew/Kurt + Diff 풀파워)...")

# 나무 150그루
rf_boost = RandomForestClassifier(n_estimators=150, max_depth=18, n_jobs=4, random_state=42)
rf_boost.fit(X_train_final, y_train_final)

score_rf_boost = rf_boost.predict_proba(X_test_scaled)[:, 1]
final_score_boost = 0.6 * normalize(score_rf_boost) + 0.4 * normalize(score_ecod)

test_temp['score_boost'] = final_score_boost
run_stats_boost = test_temp.groupby('simulationRun')['score_boost'].agg(['mean', 'max']).reset_index()
run_stats_boost['run_score'] = 0.5 * normalize(run_stats_boost['mean']) + 0.5 * normalize(run_stats_boost['max'])

print("6. 파일 저장 (74 & 74.5)")
# 74가 베스트였으니 74와 약간 위인 74.5 두 개만 딱 저장
save_thresholds = [74, 74.5] 

for pct in save_thresholds:
    th_val = np.percentile(run_stats_boost['run_score'], pct)
    run_stats_boost['is_anomaly'] = (run_stats_boost['run_score'] > th_val).astype(int)
    
    final_map = test[['simulationRun']].merge(run_stats_boost[['simulationRun', 'is_anomaly']], on='simulationRun', how='left')
    predictions = final_map['is_anomaly'].values
    
    filename = f'submission_full_stats_{pct}.csv'
    pd.DataFrame({'faultNumber': predictions}).to_csv(filename, index=True)
    print(f"  >>> 저장 완료: {filename}")

print("-" * 60)
print("완료! 'submission_full_stats_74.csv' 제출하세요.")
print("이게 진정한 최종본입니다.")
print("-" * 60)