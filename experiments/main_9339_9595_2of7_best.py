import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from pyod.models.ecod import ECOD
import warnings
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# [다중 지표 투표 + GMM 자동 경계]
# 핵심: 여러 독립적 이상 지표가 "이 run은 이상이다"에 동의하는가?
# + GMM으로 percentile 대신 자연스러운 경계 탐색
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
# 8가지 독립적 이상 지표 계산
# ============================================================

# [1] Z-Score 기반
print("2. Z-Score 지표...")
sensor_mean = train[feature_cols].mean()
sensor_std = train[feature_cols].std()
test_z = ((test[feature_cols] - sensor_mean) / (sensor_std + 1e-8)).astype(np.float32)

test_temp = test[['simulationRun']].copy()
test_temp['z_count3'] = (test_z.abs() > 3).sum(axis=1).astype(np.float32)
test_temp['z_count2'] = (test_z.abs() > 2).sum(axis=1).astype(np.float32)
test_temp['z_sum_sq'] = (test_z ** 2).sum(axis=1).astype(np.float32)

del test_z
gc.collect()

# [2] PCA T²/SPE
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

# [3] ECOD
print("4. ECOD...")
scaler_e = StandardScaler()
X_tr_e = scaler_e.fit_transform(train[feature_cols].values)
X_te_e = scaler_e.transform(test[feature_cols].values)
ecod = ECOD(n_jobs=4)
ecod.fit(X_tr_e)
test_temp['ecod'] = ecod.decision_function(X_te_e).astype(np.float32)
del X_tr_e, X_te_e, ecod
gc.collect()

# [4] 센서 상관관계 변화 (새로운 정보!)
print("5. 센서 상관 변화 계산...")
# 정상 데이터의 센서 간 상관행렬
train_corr = train[feature_cols].corr().values
# 상삼각 원소만 추출
triu_idx = np.triu_indices(len(feature_cols), k=1)
normal_corr_vec = train_corr[triu_idx]

# 각 run별로 상관행렬 계산 → 정상과의 거리
print("  - run별 상관행렬 거리 계산 중...")
corr_dist_list = []
for run_id, grp in test.groupby('simulationRun'):
    if len(grp) < 10:
        corr_dist_list.append({'simulationRun': run_id, 'corr_dist': 0.0})
        continue
    run_corr = grp[feature_cols].corr().values
    run_corr_vec = run_corr[triu_idx]
    # 상관행렬 간 유클리드 거리
    dist = np.sqrt(np.nansum((run_corr_vec - normal_corr_vec) ** 2))
    corr_dist_list.append({'simulationRun': run_id, 'corr_dist': dist})

corr_df = pd.DataFrame(corr_dist_list)
print(f"  - 완료 ({len(corr_df)}개 run)")

# ============================================================
# Run-Level 집계
# ============================================================
print("6. Run-Level 집계...")

score_cols = ['z_count3', 'z_count2', 'z_sum_sq', 't2', 'spe', 'ecod']
run_agg = test_temp.groupby('simulationRun')[score_cols].agg(['mean', 'max']).reset_index()
run_agg.columns = ['simulationRun'] + [f'{c}_{s}' for c in score_cols for s in ['mean', 'max']]

# 상관 거리 추가
run_agg = run_agg.merge(corr_df, on='simulationRun', how='left')
run_agg['corr_dist'] = run_agg['corr_dist'].fillna(0)

# 정규화
all_score_cols = [c for c in run_agg.columns if c != 'simulationRun']
for col in all_score_cols:
    run_agg[col] = normalize(run_agg[col].values)

# ============================================================
# [전략1] 다중 지표 투표 (Majority Vote)
# 각 지표가 독립적으로 74% threshold → 과반수 동의하면 이상
# ============================================================
print("\n7. Majority Vote...")

# === [A] 원래 7개 지표 투표 (2of7 포함!) ===
vote_cols_7 = ['z_count3_mean', 'z_count2_mean', 'z_sum_sq_mean',
               't2_mean', 'spe_mean', 'ecod_mean', 'corr_dist']

for col in vote_cols_7:
    th = np.percentile(run_agg[col], 74)
    run_agg[f'{col}_vote'] = (run_agg[col] > th).astype(int)

vote_names_7 = [f'{c}_vote' for c in vote_cols_7]
run_agg['votes_7'] = run_agg[vote_names_7].sum(axis=1)

print("  [7개 지표 투표]")
for min_votes in [1, 2, 3, 4, 5]:
    run_agg['is_anomaly'] = (run_agg['votes_7'] >= min_votes).astype(int)
    n_anom = run_agg['is_anomaly'].sum()
    pct_anom = 100 * n_anom / len(run_agg)
    final_map = test[['simulationRun']].merge(
        run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
    )
    fn = f'submission_vote_{min_votes}of7.csv'
    pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
    print(f"  >>> {fn} (이상 {n_anom}개, {pct_anom:.1f}%)")

# === [B] 13개 지표 투표 ===
vote_cols_13 = vote_cols_7 + ['z_count3_max', 'z_count2_max', 'z_sum_sq_max',
                               't2_max', 'spe_max', 'ecod_max']

for col in vote_cols_13:
    if f'{col}_vote' not in run_agg.columns:
        th = np.percentile(run_agg[col], 74)
        run_agg[f'{col}_vote'] = (run_agg[col] > th).astype(int)

vote_names_13 = [f'{c}_vote' for c in vote_cols_13]
run_agg['votes_13'] = run_agg[vote_names_13].sum(axis=1)

print("\n  [13개 지표 투표]")
for min_votes in [3, 4, 5, 6, 7, 8]:
    run_agg['is_anomaly'] = (run_agg['votes_13'] >= min_votes).astype(int)
    n_anom = run_agg['is_anomaly'].sum()
    pct_anom = 100 * n_anom / len(run_agg)
    final_map = test[['simulationRun']].merge(
        run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
    )
    fn = f'submission_vote_{min_votes}of13.csv'
    pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
    print(f"  >>> {fn} (이상 {n_anom}개, {pct_anom:.1f}%)")

# ============================================================
# [전략2] GMM 자동 경계 (percentile 대신)
# 점수 분포에서 정상/이상 2개 클러스터를 자동으로 찾음
# ============================================================
print("\n8. GMM 자동 경계...")

# 가장 강한 지표들의 통합 점수
run_agg['combined'] = (
    0.2 * run_agg['z_count3_mean'] +
    0.15 * run_agg['t2_mean'] +
    0.15 * run_agg['spe_mean'] +
    0.15 * run_agg['ecod_mean'] +
    0.15 * run_agg['corr_dist'] +
    0.1 * run_agg['z_count3_max'] +
    0.1 * run_agg['t2_max']
)

# GMM 2-component
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(run_agg['combined'].values.reshape(-1, 1))
labels = gmm.predict(run_agg['combined'].values.reshape(-1, 1))

# 높은 점수 그룹 = 이상
group_means = [run_agg.loc[labels == i, 'combined'].mean() for i in range(2)]
anomaly_label = np.argmax(group_means)
run_agg['is_anomaly'] = (labels == anomaly_label).astype(int)

n_anom = run_agg['is_anomaly'].sum()
pct_anom = 100 * n_anom / len(run_agg)
print(f"  - GMM 이상 run: {n_anom}개 ({pct_anom:.1f}%)")

final_map = test[['simulationRun']].merge(
    run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
)
fn = 'submission_GMM_auto.csv'
pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
print(f"  >>> {fn}")

# corr_dist 단독 (상관 변화만으로)
for pct in [74]:
    th = np.percentile(run_agg['corr_dist'], pct)
    run_agg['is_anomaly'] = (run_agg['corr_dist'] > th).astype(int)
    final_map = test[['simulationRun']].merge(
        run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
    )
    fn = f'submission_corr_dist_{pct}.csv'
    pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
    print(f"  >>> {fn}")

print("-" * 60)
print("완료! 제출 우선순위:")
print("1. submission_GMM_auto.csv (자동 경계 - percentile 대신)")
print("2. submission_vote_4of7.csv (7개 지표 중 4개 동의)")
print("3. submission_corr_dist_74.csv (센서 상관 변화 - 새 정보)")
print("-" * 60)
