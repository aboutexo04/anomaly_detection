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
# 주파수 도메인 접근: FFT + 스펙트럼 특성 + 자기상관
# 기존 모든 방법은 진폭/크기 기반 → 동일 랭킹
# FFT는 진동 패턴 → 완전히 다른 정보!
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
# [NEW] 주파수 도메인 특성 계산
# ============================================================
print("2. 주파수 도메인 특성 계산...")

# 정상 데이터의 센서별 스펙트럼 기준 (run별로 FFT → 평균)
print("  - 정상 데이터 스펙트럼 기준 계산...")

# train에서 run별 센서 스펙트럼 계산
normal_spectra = {}  # sensor -> average power spectrum

# train 데이터는 faultNumber==0인 정상 run
# 정상 run들의 평균 스펙트럼을 기준으로 삼음
train_runs = train.groupby('simulationRun')

# 각 센서의 정상 스펙트럼을 수집
n_sensors = len(feature_cols)
spectra_collection = {col: [] for col in feature_cols}

for run_id, grp in train_runs:
    grp_sorted = grp.sort_values('sample')
    for col in feature_cols:
        vals = grp_sorted[col].values
        if len(vals) < 10:
            continue
        # FFT
        fft_vals = np.abs(np.fft.rfft(vals - vals.mean()))
        spectra_collection[col].append(fft_vals[:min(50, len(fft_vals))])

# 각 센서의 평균 정상 스펙트럼
normal_spectrum_per_sensor = {}
for col in feature_cols:
    if spectra_collection[col]:
        min_len = min(len(s) for s in spectra_collection[col])
        trimmed = [s[:min_len] for s in spectra_collection[col]]
        normal_spectrum_per_sensor[col] = np.mean(trimmed, axis=0)

del spectra_collection
gc.collect()
print(f"  - 정상 스펙트럼 기준: {len(normal_spectrum_per_sensor)}개 센서")

# ============================================================
# Test run별 주파수 특성 계산
# ============================================================
print("  - Test run별 주파수 특성 계산...")

freq_results = []
for run_id, grp in test.groupby('simulationRun'):
    grp_sorted = grp.sort_values('sample')
    n = len(grp_sorted)

    row = {'simulationRun': run_id}

    # 센서별 FFT 특성
    spectral_dists = []       # 정상 스펙트럼과의 거리
    spectral_entropies = []   # 스펙트럼 엔트로피
    high_freq_ratios = []     # 고주파 비율
    dominant_freqs = []       # 지배 주파수
    autocorr_vals = []        # 자기상관 (lag=1)

    for col in feature_cols:
        vals = grp_sorted[col].values

        if n < 10:
            continue

        # FFT
        centered = vals - vals.mean()
        fft_vals = np.abs(np.fft.rfft(centered))
        power = fft_vals ** 2
        power_norm = power / (power.sum() + 1e-8)

        # (1) 스펙트럼 엔트로피 — 주파수 분포의 균등성
        # 높으면 = 노이즈처럼 고른 스펙트럼 (이상)
        # 낮으면 = 특정 주파수에 집중 (정상)
        entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
        spectral_entropies.append(entropy)

        # (2) 고주파 비율 — 전체 에너지 중 고주파 비율
        n_freq = len(power)
        high_freq_energy = power[n_freq//2:].sum()
        total_energy = power.sum() + 1e-8
        high_freq_ratios.append(high_freq_energy / total_energy)

        # (3) 지배 주파수 인덱스
        dominant_freqs.append(np.argmax(power[1:]) + 1)

        # (4) 정상 스펙트럼과의 거리
        if col in normal_spectrum_per_sensor:
            ref = normal_spectrum_per_sensor[col]
            min_len = min(len(fft_vals), len(ref))
            dist = np.sqrt(np.sum((fft_vals[:min_len] - ref[:min_len]) ** 2))
            spectral_dists.append(dist)

        # (5) 자기상관 (lag=1) — 시계열의 smoothness
        if n > 1:
            ac = np.corrcoef(vals[:-1], vals[1:])[0, 1]
            autocorr_vals.append(ac if not np.isnan(ac) else 0)

    # Run-level 집계
    if spectral_entropies:
        row['spectral_entropy_mean'] = np.mean(spectral_entropies)
        row['spectral_entropy_max'] = np.max(spectral_entropies)
        row['spectral_entropy_std'] = np.std(spectral_entropies)

    if high_freq_ratios:
        row['high_freq_ratio_mean'] = np.mean(high_freq_ratios)
        row['high_freq_ratio_max'] = np.max(high_freq_ratios)

    if spectral_dists:
        row['spectral_dist_mean'] = np.mean(spectral_dists)
        row['spectral_dist_max'] = np.max(spectral_dists)
        row['spectral_dist_std'] = np.std(spectral_dists)
        # 스펙트럼 거리가 큰 센서 수 (상위 10% 넘는 센서)
        dist_arr = np.array(spectral_dists)
        row['spectral_outlier_sensors'] = np.sum(dist_arr > np.percentile(dist_arr, 90))

    if dominant_freqs:
        row['dominant_freq_std'] = np.std(dominant_freqs)  # 센서 간 지배주파수 불일치

    if autocorr_vals:
        row['autocorr_mean'] = np.mean(autocorr_vals)
        row['autocorr_min'] = np.min(autocorr_vals)  # 가장 smoothness 낮은 센서
        row['autocorr_std'] = np.std(autocorr_vals)

    freq_results.append(row)

freq_df = pd.DataFrame(freq_results)
print(f"  - 주파수 특성: {len(freq_df.columns) - 1}개")

# ============================================================
# 기존 지표 계산 (동일)
# ============================================================
print("\n3. 기존 지표 계산...")
sensor_mean = train[feature_cols].mean()
sensor_std = train[feature_cols].std()
test_z = ((test[feature_cols] - sensor_mean) / (sensor_std + 1e-8)).astype(np.float32)

test_temp = test[['simulationRun', 'sample']].copy()
test_temp['z_count3'] = (test_z.abs() > 3).sum(axis=1).astype(np.float32)
test_temp['z_count2'] = (test_z.abs() > 2).sum(axis=1).astype(np.float32)
test_temp['z_sum_sq'] = (test_z ** 2).sum(axis=1).astype(np.float32)

del test_z
gc.collect()

print("  - PCA T²/SPE...")
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

print("  - ECOD...")
scaler_e = StandardScaler()
X_tr_e = scaler_e.fit_transform(train[feature_cols].values)
X_te_e = scaler_e.transform(test[feature_cols].values)
ecod = ECOD(n_jobs=4)
ecod.fit(X_tr_e)
test_temp['ecod'] = ecod.decision_function(X_te_e).astype(np.float32)
del X_tr_e, X_te_e, ecod
gc.collect()

print("  - 상관 거리...")
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

print("  - SPE 트렌드...")
trend_results = []
for run_id, grp in test_temp.groupby('simulationRun'):
    grp_sorted = grp.sort_values('sample')
    n = len(grp_sorted)
    vals = grp_sorted['spe'].values
    trend_results.append({
        'simulationRun': run_id,
        'trend_half_spe': vals[n//2:].mean() - vals[:n//2].mean(),
    })
trend_df = pd.DataFrame(trend_results)

# ============================================================
# Run-Level 집계
# ============================================================
print("\n4. Run-Level 집계...")

score_cols = ['z_count3', 'z_count2', 'z_sum_sq', 't2', 'spe', 'ecod']
run_agg = test_temp.groupby('simulationRun')[score_cols].agg('mean').reset_index()
run_agg.columns = ['simulationRun'] + [f'{c}_mean' for c in score_cols]

run_agg = run_agg.merge(corr_df, on='simulationRun', how='left')
run_agg = run_agg.merge(trend_df, on='simulationRun', how='left')
run_agg = run_agg.merge(freq_df, on='simulationRun', how='left')
run_agg = run_agg.fillna(0)

# 정규화
feat_cols = [c for c in run_agg.columns if c != 'simulationRun']
for col in feat_cols:
    run_agg[col] = normalize(run_agg[col].values)

print(f"  - 총 {len(feat_cols)}개 지표")

# ============================================================
# 기존 2of8 투표 기준선
# ============================================================
base_8 = ['z_count3_mean', 'z_count2_mean', 'z_sum_sq_mean',
          't2_mean', 'spe_mean', 'ecod_mean', 'corr_dist',
          'trend_half_spe']

for col in base_8:
    th = np.percentile(run_agg[col], 74)
    run_agg[f'{col}_v'] = (run_agg[col] > th).astype(int)

vote_names = [f'{c}_v' for c in base_8]
run_agg['votes_8'] = run_agg[vote_names].sum(axis=1)
print(f"\n기준선 2of8: {(run_agg['votes_8'] >= 2).sum()}개")

base_7 = ['z_count3_mean', 'z_count2_mean', 'z_sum_sq_mean',
          't2_mean', 'spe_mean', 'ecod_mean', 'corr_dist']

# ============================================================
# [전략 1] 주파수 특성을 8번째 지표로
# ============================================================
print("\n" + "="*60)
print("전략 1: 주파수 특성 8번째 (base7 + freq)")
print("="*60)

freq_metrics = [c for c in freq_df.columns if c != 'simulationRun' and c in run_agg.columns]

for new_col in freq_metrics:
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
    fn = f'submission_freq8_{new_col}.csv'
    pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
    print(f"  +{new_col}: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

# ============================================================
# [전략 2] 주파수 특성을 9번째 (base8 + freq)
# ============================================================
print("\n" + "="*60)
print("전략 2: 주파수 9번째 (base8_spe + freq)")
print("="*60)

for new_col in freq_metrics:
    test_9 = base_8 + [new_col]
    for col in test_9:
        th = np.percentile(run_agg[col], 74)
        run_agg[f'{col}_v'] = (run_agg[col] > th).astype(int)

    v_names = [f'{c}_v' for c in test_9]
    run_agg['votes'] = run_agg[v_names].sum(axis=1)

    for min_v in [2, 3]:
        run_agg['is_anomaly'] = (run_agg['votes'] >= min_v).astype(int)
        n_anom = run_agg['is_anomaly'].sum()

        final_map = test[['simulationRun']].merge(
            run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
        )
        fn = f'submission_{min_v}of9_freq_{new_col}.csv'
        pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
        print(f"  {min_v}of9 +{new_col}: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

# ============================================================
# [전략 3] 주파수 특성만으로 투표
# ============================================================
print("\n" + "="*60)
print("전략 3: 주파수 특성만으로 투표")
print("="*60)

if len(freq_metrics) >= 3:
    for col in freq_metrics:
        th = np.percentile(run_agg[col], 74)
        run_agg[f'{col}_v'] = (run_agg[col] > th).astype(int)

    v_names = [f'{c}_v' for c in freq_metrics]
    run_agg['votes_freq'] = run_agg[v_names].sum(axis=1)

    for min_v in [1, 2, 3, 4, 5]:
        run_agg['is_anomaly'] = (run_agg['votes_freq'] >= min_v).astype(int)
        n_anom = run_agg['is_anomaly'].sum()
        if 150 < n_anom < 400:
            final_map = test[['simulationRun']].merge(
                run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
            )
            fn = f'submission_freqonly_{min_v}of{len(freq_metrics)}.csv'
            pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
            print(f"  {min_v}of{len(freq_metrics)}: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

# ============================================================
# [전략 4] 주파수 연속 점수
# ============================================================
print("\n" + "="*60)
print("전략 4: 주파수 연속 점수")
print("="*60)

# 주요 주파수 특성 합산
key_freq = ['spectral_dist_mean', 'spectral_entropy_mean', 'high_freq_ratio_mean', 'autocorr_mean']
valid_freq = [c for c in key_freq if c in run_agg.columns]

if valid_freq:
    # autocorr은 낮을수록 이상이므로 반전 (1 - autocorr)
    combined_freq = np.zeros(len(run_agg))
    for c in valid_freq:
        if 'autocorr' in c:
            combined_freq += (1 - run_agg[c].values)  # 반전
        else:
            combined_freq += run_agg[c].values
    combined_freq /= len(valid_freq)

    for pct in [72, 73, 74, 75, 76]:
        th = np.percentile(combined_freq, pct)
        run_agg['is_anomaly'] = (combined_freq > th).astype(int)
        n_anom = run_agg['is_anomaly'].sum()

        final_map = test[['simulationRun']].merge(
            run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
        )
        fn = f'submission_freq_cont_th{pct}.csv'
        pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
        print(f"  freq_combined th{pct}: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

# ============================================================
# [전략 5] 기존 + 주파수 통합 투표
# ============================================================
print("\n" + "="*60)
print("전략 5: 기존8 + 주파수 통합 투표")
print("="*60)

# 주파수 중 가장 독립적인 것 선택
freq_best = ['spectral_dist_mean', 'spectral_entropy_mean', 'high_freq_ratio_mean']
valid_fb = [c for c in freq_best if c in run_agg.columns]

if valid_fb:
    all_metrics = base_8 + valid_fb
    for col in all_metrics:
        th = np.percentile(run_agg[col], 74)
        run_agg[f'{col}_v'] = (run_agg[col] > th).astype(int)

    v_names = [f'{c}_v' for c in all_metrics]
    run_agg['votes_all'] = run_agg[v_names].sum(axis=1)

    for min_v in [2, 3, 4]:
        run_agg['is_anomaly'] = (run_agg['votes_all'] >= min_v).astype(int)
        n_anom = run_agg['is_anomaly'].sum()

        final_map = test[['simulationRun']].merge(
            run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
        )
        fn = f'submission_{min_v}of{len(all_metrics)}_withfreq.csv'
        pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
        print(f"  {min_v}of{len(all_metrics)}: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

# ============================================================
# [전략 6] 경계선 재판정: 확실한 것 고정 + 주파수로 경계선 판정
# ============================================================
print("\n" + "="*60)
print("전략 6: 경계선 run을 주파수로 재판정")
print("="*60)

certain_anomaly = (run_agg['votes_8'] >= 3)
borderline = (run_agg['votes_8'] == 1) | (run_agg['votes_8'] == 2)

n_certain = certain_anomaly.sum()
n_border = borderline.sum()
print(f"  확실한 이상: {n_certain}개, 경계선: {n_border}개")

for freq_col in valid_freq:
    border_scores = run_agg.loc[borderline, freq_col]

    for border_pct in [40, 50, 60, 70, 74]:
        border_th = np.percentile(border_scores, border_pct)

        run_agg['is_anomaly'] = 0
        run_agg.loc[certain_anomaly, 'is_anomaly'] = 1
        run_agg.loc[borderline & (run_agg[freq_col] > border_th), 'is_anomaly'] = 1
        n_anom = run_agg['is_anomaly'].sum()

        if 210 <= n_anom <= 225:
            final_map = test[['simulationRun']].merge(
                run_agg[['simulationRun', 'is_anomaly']], on='simulationRun', how='left'
            )
            fn = f'submission_freqborder_{freq_col}_p{border_pct}.csv'
            pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)
            print(f"  {freq_col} border>{border_pct}%: {n_anom}개 ({100*n_anom/len(run_agg):.1f}%)")

print("\n" + "="*60)
print("완료!")
print("주파수 도메인이 기존과 다른 run을 포착했다면 → 점수 변화 기대!")
print("="*60)
