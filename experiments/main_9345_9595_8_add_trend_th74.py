import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.ecod import ECOD
import warnings
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 베스트(2of7) 기반 + 새로운 독립 지표 추가
# 기존 7개 + 새 4개 = 11개 지표로 투표 다양화
# ============================================================

print("1. 데이터 로드...")
DATA_PATH = "./data/"
train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")

id_cols = ['faultNumber', 'simulationRun', 'sample']
feature_cols = [col for col in train.columns if col not in id_cols]

for col in feature_cols:
    train[col] = train[col].astype(np.float32)
    test[col] = test[col].astype(np.float32)

def normalize(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)

# ============================================================
# 기존 7개 지표 (동일)
# ============================================================
print("2. Z-Score 지표...")
sensor_mean = train[feature_cols].mean()
sensor_std = train[feature_cols].std()
test_z = ((test[feature_cols] - sensor_mean) / (sensor_std + 1e-8)).astype(np.float32)

test_temp = test[['simulationRun', 'sample']].copy()
test_temp['z_count3'] = (test_z.abs() > 3).sum(axis=1).astype(np.float32)
test_temp['z_count2'] = (test_z.abs() > 2).sum(axis=1).astype(np.float32)
test_temp['z_sum_sq'] = (test_z ** 2).sum(axis=1).astype(np.float32)

# [NEW] 추가 Z-score 지표
test_temp['z_count2_5'] = (test_z.abs() > 2.5).sum(axis=1).astype(np.float32)  # 2.5시그마
test_temp['z_count4'] = (test_z.abs() > 4).sum(axis=1).astype(np.float32)      # 4시그마 (극단)

# [NEW] 범위 이탈: train의 min/max를 벗어난 센서 수
print("  - 범위 이탈 계산...")
sensor_min = train[feature_cols].min()
sensor_max = train[feature_cols].max()
below_min = (test[feature_cols] < sensor_min).sum(axis=1)
above_max = (test[feature_cols] > sensor_max).sum(axis=1)
test_temp['range_violation'] = (below_min + above_max).astype(np.float32)

del test_z
gc.collect()

print("3. PCA T²/SPE...")
scaler_pca = StandardScaler()
X_tr = scaler_pca.fit_transform(train[feature_cols].values)
X_te = scaler_pca.transform(test[feature_cols].values)

pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_tr)
X_t = pca.transform(X_te)
X_r = pca.inverse_transform(X_t)

test_temp['t2'] = np.sum((X_t / np.sqrt(pca.explained_variance_ + 1e-8)) ** 2, axis=1).astype(np.float32)
test_temp['spe'] = np.sum((X_te - X_r) ** 2, axis=1).astype(np.float32)

del X_tr, X_te, X_t, X_r
gc.collect()

print("4. ECOD...")
scaler_e = StandardScaler()
X_tr_e = scaler_e.fit_transform(train[feature_cols].values)
X_te_e = scaler_e.transform(test[feature_cols].values)
ecod = ECOD(n_jobs=4)
ecod.fit(X_tr_e)
test_temp['ecod'] = ecod.decision_function(X_te_e).astype(np.float32)
del X_tr_e, X_te_e, ecod
gc.collect()

print("5. 센서 상관 변화...")
train_corr = train[feature_cols].corr().values
triu_idx = np.triu_indices(len(feature_cols), k=1)
normal_corr_vec = train_corr[triu_idx]

corr_dist_list = []
for run_id, grp in test.groupby('simulationRun'):
    if len(grp) < 10:
        corr_dist_list.append({'simulationRun': run_id, 'corr_dist': 0.0})
        continue
    run_corr = grp[feature_cols].corr().values
    run_corr_vec = run_corr[triu_idx]
    dist = np.sqrt(np.nansum((run_corr_vec - normal_corr_vec) ** 2))
    corr_dist_list.append({'simulationRun': run_id, 'corr_dist': dist})

corr_df = pd.DataFrame(corr_dist_list)

# [NEW] 시간 트렌드: 후반부 이상 점수 - 전반부 이상 점수
print("6. 시간 트렌드 계산...")
trend_list = []
for run_id, grp in test_temp.groupby('simulationRun'):
    grp_sorted = grp.sort_values('sample')
    n = len(grp_sorted)
    first_z3 = grp_sorted['z_count3'].iloc[:n//2].mean()
    second_z3 = grp_sorted['z_count3'].iloc[n//2:].mean()
    trend_list.append({'simulationRun': run_id, 'trend_z3': second_z3 - first_z3})

trend_df = pd.DataFrame(trend_list)

# ============================================================
# Run-Level 집계
# ============================================================
print("7. Run-Level 집계...")

score_cols = ['z_count3', 'z_count2', 'z_sum_sq', 'z_count2_5', 'z_count4',
              'range_violation', 't2', 'spe', 'ecod']
run_agg = test_temp.groupby('simulationRun')[score_cols].agg(['mean']).reset_index()
run_agg.columns = ['simulationRun'] + [f'{c}_mean' for c in score_cols]

# 상관 거리 + 트렌드 추가
run_agg = run_agg.merge(corr_df, on='simulationRun', how='left')
run_agg = run_agg.merge(trend_df, on='simulationRun', how='left')
run_agg['corr_dist'] = run_agg['corr_dist'].fillna(0)
run_agg['trend_z3'] = run_agg['trend_z3'].fillna(0)

# 정규화
feat_cols = [c for c in run_agg.columns if c != 'simulationRun']
for col in feat_cols:
    run_agg[col] = normalize(run_agg[col].values)

# ============================================================
# 투표 조합 탐색
# ============================================================
print("\n8. 투표 조합 탐색...")

# 기존 7개 (베스트 0.9339)
base_7 = ['z_count3_mean', 'z_count2_mean', 'z_sum_sq_mean',
           't2_mean', 'spe_mean', 'ecod_mean', 'corr_dist']

# 새 지표 4개
new_metrics = ['z_count2_5_mean', 'z_count4_mean', 'range_violation_mean', 'trend_z3']

# 전체 11개
all_11 = base_7 + new_metrics

# 조합별 투표
configs = {
    '7base': base_7,                          # 기존 베스트
    '8_add_range': base_7 + ['range_violation_mean'],
    '8_add_z4': base_7 + ['z_count4_mean'],
    '8_add_trend': base_7 + ['trend_z3'],
    '8_add_z25': base_7 + ['z_count2_5_mean'],
    '9_add_range_z4': base_7 + ['range_violation_mean', 'z_count4_mean'],
    '9_add_range_trend': base_7 + ['range_violation_mean', 'trend_z3'],
    '11_all': all_11,
}

for pct_th in [73, 74]:
    print(f"\n  --- threshold {pct_th}% ---")
    for cfg_name, cols in configs.items():
        for col in cols:
            th = np.percentile(run_agg[col], pct_th)
            run_agg[f'{col}_v{pct_th}'] = (run_agg[col] > th).astype(int)

        vote_names = [f'{c}_v{pct_th}' for c in cols]
        run_agg['votes'] = run_agg[vote_names].sum(axis=1)

        # 2of N
        run_agg['is_anomaly'] = (run_agg['votes'] >= 2).astype(int)
        n_anom = run_agg['is_anomaly'].sum()
        pct_anom = 100 * n_anom / len(run_agg)

        final_map = test[['simulationRun']].merge(
            run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
        )
        fn = f'submission_2of_{cfg_name}_th{pct_th}.csv'
        pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
        print(f"  >>> {fn} (이상 {n_anom}개, {pct_anom:.1f}%)")

print("-" * 60)
print("완료! 이상 run 수를 비교하세요.")
print("베스트(2of7 th74) = 210개, 28.4%")
print("이것과 비슷하면서 약간 다른 조합이 개선 후보!")
print("-" * 60)
