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
# 1등 돌파 v3: 완전히 새로운 피처 (CUSUM, Mahalanobis, 센서패턴)
# + 확실/불확실 분리 전략
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
# 기존 7개 지표 (동일 코드)
# ============================================================
print("2. Z-Score 지표...")
sensor_mean = train[feature_cols].mean()
sensor_std = train[feature_cols].std()
test_z = ((test[feature_cols] - sensor_mean) / (sensor_std + 1e-8)).astype(np.float32)

test_temp = test[['simulationRun', 'sample']].copy()
test_temp['z_count3'] = (test_z.abs() > 3).sum(axis=1).astype(np.float32)
test_temp['z_count2'] = (test_z.abs() > 2).sum(axis=1).astype(np.float32)
test_temp['z_sum_sq'] = (test_z ** 2).sum(axis=1).astype(np.float32)

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

# ============================================================
# [NEW 1] CUSUM (누적합 관리도) — 센서별 지속적 이탈 감지
# Z-score는 "지금 이 순간" 이탈, CUSUM은 "누적 이탈" 감지
# ============================================================
print("6. CUSUM 계산...")

cusum_results = []
# 센서별 정상 평균/표준편차
s_mean = sensor_mean.values
s_std = sensor_std.values

for run_id, grp in test.groupby('simulationRun'):
    grp_sorted = grp.sort_values('sample')
    vals = grp_sorted[feature_cols].values  # (n_samples, n_sensors)

    # 각 센서별 CUSUM 계산
    n_samples = len(vals)
    # 정규화된 편차
    deviations = (vals - s_mean) / (s_std + 1e-8)

    # CUSUM: 양의 누적합 (지속적 상승 감지)
    cusum_pos = np.zeros_like(deviations)
    cusum_neg = np.zeros_like(deviations)
    k = 0.5  # slack parameter

    for t in range(1, n_samples):
        cusum_pos[t] = np.maximum(0, cusum_pos[t-1] + deviations[t] - k)
        cusum_neg[t] = np.maximum(0, cusum_neg[t-1] - deviations[t] - k)

    # Run-level CUSUM 특성
    max_cusum = np.max(np.maximum(cusum_pos, cusum_neg), axis=0)  # 센서별 최대 CUSUM
    row = {
        'simulationRun': run_id,
        'cusum_max_mean': np.mean(max_cusum),        # 평균 최대 CUSUM
        'cusum_max_max': np.max(max_cusum),           # 최대 최대 CUSUM
        'cusum_sensors_over5': np.sum(max_cusum > 5), # CUSUM > 5인 센서 수
        'cusum_sensors_over10': np.sum(max_cusum > 10),
    }
    cusum_results.append(row)

cusum_df = pd.DataFrame(cusum_results)
print(f"  - CUSUM 완료 ({len(cusum_df)}개 run)")

# ============================================================
# [NEW 2] Mahalanobis 거리 — 센서 상관까지 고려
# ============================================================
print("7. Mahalanobis 거리...")

# 정상 데이터의 공분산 행렬 (축소 추정)
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
lw.fit(train[feature_cols].values)
cov_inv = np.linalg.inv(lw.covariance_ + 1e-6 * np.eye(len(feature_cols)))
train_mean = train[feature_cols].mean().values

mahal_results = []
for run_id, grp in test.groupby('simulationRun'):
    vals = grp[feature_cols].values
    # 각 샘플의 Mahalanobis 거리
    diff = vals - train_mean
    mahal_dists = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    mahal_results.append({
        'simulationRun': run_id,
        'mahal_mean': np.mean(mahal_dists),
        'mahal_max': np.max(mahal_dists),
        'mahal_q90': np.percentile(mahal_dists, 90),
    })

mahal_df = pd.DataFrame(mahal_results)
print(f"  - Mahalanobis 완료")

# ============================================================
# [NEW 3] 센서 이탈 패턴 — 어떤 센서가 얼마나
# ============================================================
print("8. 센서 이탈 패턴...")

# 각 run에서 각 센서의 z-score 통계
sensor_pattern_results = []
for run_id, grp in test.groupby('simulationRun'):
    vals = grp[feature_cols].values
    z_vals = (vals - s_mean) / (s_std + 1e-8)

    # 센서별 평균 |z| 계산
    sensor_abs_z = np.mean(np.abs(z_vals), axis=0)  # (52,)

    # Top-5 이탈 센서의 평균 z
    top5_idx = np.argsort(sensor_abs_z)[-5:]
    top10_idx = np.argsort(sensor_abs_z)[-10:]

    sensor_pattern_results.append({
        'simulationRun': run_id,
        'top5_sensor_z': np.mean(sensor_abs_z[top5_idx]),
        'top10_sensor_z': np.mean(sensor_abs_z[top10_idx]),
        'sensor_z_std': np.std(sensor_abs_z),  # 센서 간 이탈 불균형 (특정 센서만 심하면 높음)
        'sensor_z_max': np.max(sensor_abs_z),
    })

sensor_df = pd.DataFrame(sensor_pattern_results)
print(f"  - 센서 패턴 완료")

# ============================================================
# SPE 트렌드 계산 (기존 베스트)
# ============================================================
print("9. SPE 트렌드...")
trend_results = []
for run_id, grp in test_temp.groupby('simulationRun'):
    grp_sorted = grp.sort_values('sample')
    n = len(grp_sorted)
    vals = grp_sorted['spe'].values
    first_half = vals[:n//2].mean()
    second_half = vals[n//2:].mean()
    trend_results.append({
        'simulationRun': run_id,
        'trend_half_spe': second_half - first_half,
    })
trend_df = pd.DataFrame(trend_results)

# z-score 트렌드도 저장 (delete는 안함)
del test_z
gc.collect()

# ============================================================
# Run-Level 집계 + 모든 새 피처 병합
# ============================================================
print("10. 전체 병합...")

score_cols = ['z_count3', 'z_count2', 'z_sum_sq', 't2', 'spe', 'ecod']
run_agg = test_temp.groupby('simulationRun')[score_cols].agg(['mean']).reset_index()
run_agg.columns = ['simulationRun'] + [f'{c}_mean' for c in score_cols]

# 모든 새 피처 병합
run_agg = run_agg.merge(corr_df, on='simulationRun', how='left')
run_agg = run_agg.merge(trend_df, on='simulationRun', how='left')
run_agg = run_agg.merge(cusum_df, on='simulationRun', how='left')
run_agg = run_agg.merge(mahal_df, on='simulationRun', how='left')
run_agg = run_agg.merge(sensor_df, on='simulationRun', how='left')
run_agg = run_agg.fillna(0)

# 정규화
feat_cols = [c for c in run_agg.columns if c != 'simulationRun']
for col in feat_cols:
    run_agg[col] = normalize(run_agg[col].values)

print(f"  - 총 {len(feat_cols)}개 지표")

# ============================================================
# 기존 2of8 투표 실행 (기준선)
# ============================================================
base_8 = ['z_count3_mean', 'z_count2_mean', 'z_sum_sq_mean',
          't2_mean', 'spe_mean', 'ecod_mean', 'corr_dist',
          'trend_half_spe']

for col in base_8:
    th = np.percentile(run_agg[col], 74)
    run_agg[f'{col}_v'] = (run_agg[col] > th).astype(int)

vote_names = [f'{c}_v' for c in base_8]
run_agg['votes_8'] = run_agg[vote_names].sum(axis=1)
base_anomaly = (run_agg['votes_8'] >= 2)

print(f"\n기준선: 2of8 = {base_anomaly.sum()}개")

# ============================================================
# [전략 1] 새 피처를 8번째로 투입
# CUSUM, Mahalanobis, 센서패턴은 기존과 근본적으로 다른 신호!
# ============================================================
print("\n" + "="*60)
print("전략 1: 새 피처를 8번째 지표로 (base7 + new)")
print("="*60)

base_7 = ['z_count3_mean', 'z_count2_mean', 'z_sum_sq_mean',
          't2_mean', 'spe_mean', 'ecod_mean', 'corr_dist']

new_metrics = [
    'cusum_max_mean', 'cusum_max_max', 'cusum_sensors_over5', 'cusum_sensors_over10',
    'mahal_mean', 'mahal_max', 'mahal_q90',
    'top5_sensor_z', 'top10_sensor_z', 'sensor_z_std', 'sensor_z_max',
]

for new_col in new_metrics:
    if new_col not in run_agg.columns:
        continue
    test_8 = base_7 + [new_col]
    for col in test_8:
        th = np.percentile(run_agg[col], 74)
        run_agg[f'{col}_v'] = (run_agg[col] > th).astype(int)

    v_names = [f'{c}_v' for c in test_8]
    run_agg['votes'] = run_agg[v_names].sum(axis=1)
    run_agg['is_anomaly'] = (run_agg['votes'] >= 2).astype(int)
    n_anom = run_agg['is_anomaly'].sum()

    final_map = test[['simulationRun']].merge(
        run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
    )
    fn = f'submission_new8_{new_col}.csv'
    pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
    print(f"  +{new_col}: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

# ============================================================
# [전략 2] 9개 (base7 + SPE trend + 새 피처) → 2of9
# ============================================================
print("\n" + "="*60)
print("전략 2: 2of9 (base7 + SPE_trend + 새 피처)")
print("="*60)

for new_col in new_metrics:
    if new_col not in run_agg.columns:
        continue
    test_9 = base_7 + ['trend_half_spe', new_col]
    for col in test_9:
        th = np.percentile(run_agg[col], 74)
        run_agg[f'{col}_v'] = (run_agg[col] > th).astype(int)

    v_names = [f'{c}_v' for c in test_9]
    run_agg['votes'] = run_agg[v_names].sum(axis=1)

    for min_v in [2]:
        run_agg['is_anomaly'] = (run_agg['votes'] >= min_v).astype(int)
        n_anom = run_agg['is_anomaly'].sum()

        final_map = test[['simulationRun']].merge(
            run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
        )
        fn = f'submission_2of9_spe_{new_col}.csv'
        pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
        print(f"  2of9 +{new_col}: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

# ============================================================
# [전략 3] 확실/불확실 분리 → 불확실 run만 새 방법으로 재판정
# ============================================================
print("\n" + "="*60)
print("전략 3: 경계선 run만 새 피처로 재판정")
print("="*60)

# 확실한 이상 (3+ votes), 확실한 정상 (0 votes)
certain_anomaly = (run_agg['votes_8'] >= 3)
certain_normal = (run_agg['votes_8'] == 0)
borderline = ~certain_anomaly & ~certain_normal  # votes 1 or 2

n_certain_anom = certain_anomaly.sum()
n_certain_norm = certain_normal.sum()
n_border = borderline.sum()
print(f"  확실한 이상: {n_certain_anom}개, 확실한 정상: {n_certain_norm}개, 경계선: {n_border}개")

# 경계선 run에 대해 새 피처 기반 판정
new_score_cols = ['cusum_max_mean', 'mahal_mean', 'top5_sensor_z', 'sensor_z_std']

for score_col in new_score_cols:
    if score_col not in run_agg.columns:
        continue

    # 경계선 run의 새 피처 점수
    border_scores = run_agg.loc[borderline, score_col]

    # 다양한 threshold로 경계선 run 판정
    for border_pct in [50, 60, 70, 74, 80]:
        # 경계선 중 상위 N%를 이상으로
        border_th = np.percentile(border_scores, border_pct)

        run_agg['is_anomaly'] = 0
        run_agg.loc[certain_anomaly, 'is_anomaly'] = 1
        run_agg.loc[borderline & (run_agg[score_col] > border_th), 'is_anomaly'] = 1
        n_anom = run_agg['is_anomaly'].sum()

        if 210 <= n_anom <= 225:  # 유의미한 범위만 저장
            final_map = test[['simulationRun']].merge(
                run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
            )
            fn = f'submission_border_{score_col}_p{border_pct}.csv'
            pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
            print(f"  {score_col} border>{border_pct}%: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

# ============================================================
# [전략 4] 새 피처 단독 앙상블 (기존 7개 대신)
# 완전히 새로운 관점
# ============================================================
print("\n" + "="*60)
print("전략 4: 새 피처 기반 투표 (CUSUM + Mahal + Sensor)")
print("="*60)

new_vote_sets = {
    'cusum3': ['cusum_max_mean', 'cusum_sensors_over5', 'cusum_sensors_over10'],
    'mahal3': ['mahal_mean', 'mahal_max', 'mahal_q90'],
    'sensor3': ['top5_sensor_z', 'sensor_z_std', 'sensor_z_max'],
    'mixed6': ['cusum_max_mean', 'cusum_sensors_over5',
               'mahal_mean', 'mahal_q90',
               'top5_sensor_z', 'sensor_z_std'],
    'all_new': ['cusum_max_mean', 'cusum_max_max', 'cusum_sensors_over5',
                'mahal_mean', 'mahal_max',
                'top5_sensor_z', 'sensor_z_std'],
}

for vname, vcols in new_vote_sets.items():
    valid = [c for c in vcols if c in run_agg.columns]
    if not valid:
        continue
    for col in valid:
        th = np.percentile(run_agg[col], 74)
        run_agg[f'{col}_v'] = (run_agg[col] > th).astype(int)

    v_names = [f'{c}_v' for c in valid]
    run_agg['votes_new'] = run_agg[v_names].sum(axis=1)

    for min_v in [1, 2]:
        run_agg['is_anomaly'] = (run_agg['votes_new'] >= min_v).astype(int)
        n_anom = run_agg['is_anomaly'].sum()

        if n_anom > 0:
            final_map = test[['simulationRun']].merge(
                run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
            )
            fn = f'submission_newvote_{vname}_{min_v}of{len(valid)}.csv'
            pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
            print(f"  {vname} {min_v}of{len(valid)}: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

# ============================================================
# [전략 5] 기존+새 통합 대규모 투표
# 기존 8 + 새 7 = 15개 지표, 다양한 vote threshold
# ============================================================
print("\n" + "="*60)
print("전략 5: 통합 15개 지표 투표")
print("="*60)

all_15 = base_8 + ['cusum_max_mean', 'cusum_sensors_over5',
                    'mahal_mean', 'mahal_q90',
                    'top5_sensor_z', 'sensor_z_std', 'sensor_z_max']

valid_15 = [c for c in all_15 if c in run_agg.columns]
for col in valid_15:
    th = np.percentile(run_agg[col], 74)
    run_agg[f'{col}_v'] = (run_agg[col] > th).astype(int)

v_names_15 = [f'{c}_v' for c in valid_15]
run_agg['votes_15'] = run_agg[v_names_15].sum(axis=1)

for min_v in [2, 3, 4, 5, 6]:
    run_agg['is_anomaly'] = (run_agg['votes_15'] >= min_v).astype(int)
    n_anom = run_agg['is_anomaly'].sum()

    final_map = test[['simulationRun']].merge(
        run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
    )
    fn = f'submission_{min_v}of{len(valid_15)}.csv'
    pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
    print(f"  {min_v}of{len(valid_15)}: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

print("\n" + "="*60)
print("완료!")
print("핵심 후보: 이상 수 ~216 근처 + 기존과 다른 run 구성")
print("="*60)
