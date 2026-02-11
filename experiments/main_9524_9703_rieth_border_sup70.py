import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from pyod.models.ecod import ECOD
import pyreadr
import warnings
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

def normalize(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)

# ============================================================
# Phase 1: 대회 데이터 기반 baseline 구축
# ============================================================
print("=" * 60)
print("Phase 1: 대회 데이터 + baseline 구축")
print("=" * 60)

DATA_PATH = "./data/"
RIETH_PATH = "./data/rieth_tep/"

train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")

id_cols = ['faultNumber', 'simulationRun', 'sample']
feature_cols = [col for col in train.columns if col not in id_cols]

for col in feature_cols:
    train[col] = train[col].astype(np.float32)
    test[col] = test[col].astype(np.float32)

# --- Z-score baselines ---
sensor_mean = train[feature_cols].mean()
sensor_std = train[feature_cols].std()
s_mean_np = sensor_mean.values
s_std_np = sensor_std.values

# --- PCA baselines ---
scaler_pca = StandardScaler()
X_tr = scaler_pca.fit_transform(train[feature_cols].values)
pca = PCA(n_components=0.95, random_state=42)
pca.fit(X_tr)
print(f"  PCA components: {pca.n_components_}")

# --- ECOD baseline ---
scaler_e = StandardScaler()
X_tr_e = scaler_e.fit_transform(train[feature_cols].values)
ecod_model = ECOD(n_jobs=4)
ecod_model.fit(X_tr_e)

# --- Correlation baseline ---
train_corr = train[feature_cols].corr().values
triu_idx = np.triu_indices(len(feature_cols), k=1)
normal_corr_vec = train_corr[triu_idx]

del X_tr, X_tr_e
gc.collect()

# ============================================================
# Helper: compute run-level metrics
# ============================================================
def compute_run_metrics(df, label="data", compute_ecod=True):
    """대회 train 기반 baseline으로 run-level 8개 메트릭 계산"""
    runs = sorted(df['simulationRun'].unique())
    n_runs = len(runs)

    # --- Per-sample metrics ---
    feat_vals = df[feature_cols].values.astype(np.float32)

    # Z-scores
    z = (feat_vals - s_mean_np) / (s_std_np + 1e-8)
    z_count3 = (np.abs(z) > 3).sum(axis=1).astype(np.float32)
    z_count2 = (np.abs(z) > 2).sum(axis=1).astype(np.float32)
    z_sum_sq = (z ** 2).sum(axis=1).astype(np.float32)
    del z

    # PCA T²/SPE
    X_scaled = scaler_pca.transform(feat_vals)
    X_t = pca.transform(X_scaled)
    X_r = pca.inverse_transform(X_t)
    t2 = np.sum((X_t / np.sqrt(pca.explained_variance_ + 1e-8)) ** 2, axis=1).astype(np.float32)
    spe = np.sum((X_scaled - X_r) ** 2, axis=1).astype(np.float32)
    del X_t, X_r, X_scaled

    # ECOD
    ecod_scores = None
    if compute_ecod:
        X_e = scaler_e.transform(feat_vals)
        ecod_scores = ecod_model.decision_function(X_e).astype(np.float32)
        del X_e

    del feat_vals
    gc.collect()

    # --- Build temp DataFrame ---
    sim_runs = df['simulationRun'].values
    if 'sample' in df.columns:
        samples = df['sample'].values
    else:
        samples = np.zeros(len(df), dtype=np.int32)
        for run_id in runs:
            mask = sim_runs == run_id
            samples[mask] = np.arange(mask.sum()) + 1

    # --- Run-level aggregation ---
    run_data = []
    for run_id in runs:
        mask = sim_runs == run_id
        idx = np.where(mask)[0]

        row = {
            'simulationRun': run_id,
            'z_count3_mean': z_count3[idx].mean(),
            'z_count2_mean': z_count2[idx].mean(),
            'z_sum_sq_mean': z_sum_sq[idx].mean(),
            't2_mean': t2[idx].mean(),
            'spe_mean': spe[idx].mean(),
        }

        if compute_ecod and ecod_scores is not None:
            row['ecod_mean'] = ecod_scores[idx].mean()

        # SPE trend (2nd half - 1st half)
        spe_run = spe[idx]
        sample_run = samples[idx]
        order = np.argsort(sample_run)
        spe_ordered = spe_run[order]
        n = len(spe_ordered)
        row['trend_half_spe'] = spe_ordered[n//2:].mean() - spe_ordered[:n//2].mean()

        run_data.append(row)

    del z_count3, z_count2, z_sum_sq, t2, spe, ecod_scores
    gc.collect()

    run_agg = pd.DataFrame(run_data)

    # --- Correlation distance (per run) ---
    print(f"    {label}: corr_dist 계산 중 ({n_runs} runs)...")
    corr_dists = []
    for run_id in runs:
        grp = df[df['simulationRun'] == run_id]
        if len(grp) < 10:
            corr_dists.append(0.0)
            continue
        run_corr = grp[feature_cols].corr().values
        run_corr_vec = run_corr[triu_idx]
        dist = np.sqrt(np.nansum((run_corr_vec - normal_corr_vec) ** 2))
        corr_dists.append(dist)

    run_agg['corr_dist'] = corr_dists

    return run_agg

# ============================================================
# Phase 2: 대회 test 메트릭 계산
# ============================================================
print("\nPhase 2: 대회 test 메트릭 계산...")
test_metrics = compute_run_metrics(test, label="competition_test", compute_ecod=True)
print(f"  → {len(test_metrics)} test runs")

# --- Base8 투표 (기준선) ---
metric_cols = ['z_count3_mean', 'z_count2_mean', 'z_sum_sq_mean',
               't2_mean', 'spe_mean', 'ecod_mean', 'corr_dist', 'trend_half_spe']

test_norm = test_metrics.copy()
for col in metric_cols:
    test_norm[col] = normalize(test_norm[col].values)

for col in metric_cols:
    th = np.percentile(test_norm[col], 74)
    test_norm[f'{col}_v'] = (test_norm[col] > th).astype(int)

vote_names = [f'{c}_v' for c in metric_cols]
test_norm['votes_8'] = test_norm[vote_names].sum(axis=1)
base8_anomaly = set(test_norm[test_norm['votes_8'] >= 2]['simulationRun'].values)
base8_pred = np.isin(test_metrics['simulationRun'].values, list(base8_anomaly)).astype(int)
print(f"  Base8 참조: {len(base8_anomaly)} anomaly runs")

# ============================================================
# Phase 3: Rieth 대규모 데이터 메트릭 계산
# ============================================================
print("\n" + "=" * 60)
print("Phase 3: Rieth 500runs/fault 데이터 메트릭 계산")
print("=" * 60)

# Rieth Testing 데이터 사용 (960 samples/run = 대회와 동일)
print("  Loading Rieth FaultFree Testing...")
rieth_normal = pyreadr.read_r(f"{RIETH_PATH}TEP_FaultFree_Testing.RData")['fault_free_testing']
for col in feature_cols:
    rieth_normal[col] = rieth_normal[col].astype(np.float32)
print(f"    → {len(rieth_normal)} rows, {rieth_normal['simulationRun'].nunique()} runs")

print("  Loading Rieth Faulty Testing...")
rieth_faulty = pyreadr.read_r(f"{RIETH_PATH}TEP_Faulty_Testing.RData")['faulty_testing']
for col in feature_cols:
    rieth_faulty[col] = rieth_faulty[col].astype(np.float32)
print(f"    → {len(rieth_faulty)} rows, faults: {sorted(rieth_faulty['faultNumber'].unique())}")

# --- Normal runs ---
print("\n  Processing normal runs...")
normal_metrics = compute_run_metrics(rieth_normal, label="rieth_normal", compute_ecod=True)
normal_metrics['label'] = 0
normal_metrics['fault_type'] = 0
all_rieth = [normal_metrics]
print(f"    → {len(normal_metrics)} normal runs")

del rieth_normal
gc.collect()

# --- Faulty runs (one fault type at a time to save memory) ---
for fault_num in sorted(rieth_faulty['faultNumber'].unique()):
    fault_num = int(fault_num)
    print(f"\n  Processing fault {fault_num}/20...")
    chunk = rieth_faulty[rieth_faulty['faultNumber'] == fault_num].copy()

    # Make simulationRun unique
    chunk['simulationRun'] = chunk['simulationRun'] + fault_num * 1000

    metrics = compute_run_metrics(chunk, label=f"fault_{fault_num}", compute_ecod=True)
    metrics['label'] = 1
    metrics['fault_type'] = fault_num
    all_rieth.append(metrics)

    del chunk
    gc.collect()
    print(f"    → {len(metrics)} faulty runs")

del rieth_faulty
gc.collect()

rieth_df = pd.concat(all_rieth, ignore_index=True)
n_normal = len(rieth_df[rieth_df['label'] == 0])
n_faulty = len(rieth_df[rieth_df['label'] == 1])
print(f"\n  Total Rieth: {len(rieth_df)} runs (normal={n_normal}, faulty={n_faulty})")

# ============================================================
# Phase 4: 지도학습 모델 학습
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: 지도학습 모델 학습 (10,500 labeled runs)")
print("=" * 60)

train_features = metric_cols  # 8개 메트릭
X_rieth = rieth_df[train_features].values
y_rieth = rieth_df['label'].values
X_comp_test = test_metrics[train_features].values

print(f"  Features: {train_features}")
print(f"  Train shape: {X_rieth.shape}, Test shape: {X_comp_test.shape}")
print(f"  Class distribution: normal={n_normal}, faulty={n_faulty}")

# --- RandomForest ---
print("  Training RandomForest...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=20,
    class_weight='balanced', n_jobs=4, random_state=42
)
rf.fit(X_rieth, y_rieth)
rf_proba = rf.predict_proba(X_comp_test)[:, 1]
print(f"    RF feature importance: {dict(zip(train_features, rf.feature_importances_.round(3)))}")

# --- HistGradientBoosting ---
print("  Training HistGradientBoosting...")
sample_weights = compute_sample_weight('balanced', y_rieth)
hgb = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.05, max_depth=6, random_state=42
)
hgb.fit(X_rieth, y_rieth, sample_weight=sample_weights)
hgb_proba = hgb.predict_proba(X_comp_test)[:, 1]

# --- Ensemble ---
ensemble_proba = 0.5 * rf_proba + 0.5 * hgb_proba

# ============================================================
# Phase 5: 제출 파일 생성
# ============================================================
print("\n" + "=" * 60)
print("Phase 5: 제출 파일 생성")
print("=" * 60)

test_metrics['rf_proba'] = rf_proba
test_metrics['hgb_proba'] = hgb_proba
test_metrics['ensemble_proba'] = ensemble_proba

strategies = {}

# [1] Supervised standalone - 다양한 threshold
for pct in [70, 72, 74, 76]:
    th = np.percentile(ensemble_proba, pct)
    pred = (ensemble_proba > th).astype(int)
    strategies[f'sup_p{pct}'] = pred

# [2] Top-N by ensemble probability
for n_top in [210, 213, 216, 219, 222]:
    sorted_proba = np.sort(ensemble_proba)[::-1]
    if n_top <= len(sorted_proba):
        th = sorted_proba[n_top - 1]
        pred = (ensemble_proba >= th).astype(int)
        strategies[f'sup_top{n_top}'] = pred

# [3] Supervised as 9th voter in base8+1
sup_vote = (ensemble_proba > np.percentile(ensemble_proba, 74)).astype(int)
test_norm['sup_vote'] = sup_vote
test_norm['votes_9'] = test_norm['votes_8'] + test_norm['sup_vote']
strategies['2of9_sup'] = (test_norm['votes_9'].values >= 2).astype(int)
strategies['3of9_sup'] = (test_norm['votes_9'].values >= 3).astype(int)

# [4] Hybrid: base8 AND supervised agree
# 4a: Intersection (both agree = anomaly)
sup_binary = (ensemble_proba > np.percentile(ensemble_proba, 74)).astype(int)
strategies['hybrid_intersect'] = (base8_pred * sup_binary)

# 4b: Use supervised to re-classify base8 borderline cases
# Base8에서 votes=1 (경계선) → supervised가 높으면 anomaly로
borderline_mask = (test_norm['votes_8'].values == 1)
certain_anom = (test_norm['votes_8'].values >= 2)
for sup_pct in [50, 60, 70]:
    sup_th = np.percentile(ensemble_proba, sup_pct)
    pred = certain_anom.astype(int).copy()
    pred[borderline_mask & (ensemble_proba > sup_th)] = 1
    strategies[f'base8_border_sup{sup_pct}'] = pred

# [5] Base8에서 votes>=2인 것 중 supervised 확률이 낮은 것만 제거
for removal_pct in [10, 20, 30]:
    base8_runs = test_metrics.loc[base8_pred == 1, 'simulationRun'].values
    base8_probas = test_metrics.loc[base8_pred == 1, 'ensemble_proba'].values
    # 하위 removal_pct% 제거
    th = np.percentile(base8_probas, removal_pct)
    keep_mask = base8_probas >= th
    keep_runs = set(base8_runs[keep_mask])
    pred = np.isin(test_metrics['simulationRun'].values, list(keep_runs)).astype(int)
    strategies[f'base8_prune{removal_pct}'] = pred

# --- 결과 출력 및 저장 ---
print(f"\n{'Strategy':<30} {'Runs':>5} {'Overlap':>7} {'Added':>6} {'Removed':>7}")
print("-" * 60)

for name, pred in sorted(strategies.items()):
    n_anom = pred.sum()
    sup_runs = set(test_metrics.loc[pred == 1, 'simulationRun'].values)
    overlap = len(sup_runs & base8_anomaly)
    added = len(sup_runs - base8_anomaly)
    removed = len(base8_anomaly - sup_runs)

    final_map = test[['simulationRun']].merge(
        pd.DataFrame({'simulationRun': test_metrics['simulationRun'], 'is_anomaly': pred}),
        on='simulationRun', how='left'
    )
    fn = f'submission_rieth_{name}.csv'
    pd.DataFrame({'faultNumber': final_map['is_anomaly'].values}).to_csv(fn, index=True)

    marker = " ◀" if n_anom == 216 else ""
    print(f"  {name:<28} {n_anom:>5} {overlap:>7} {'+'+str(added):>6} {'-'+str(removed):>7}{marker}")

# --- Per-fault analysis ---
print("\n" + "=" * 60)
print("Per-fault RF probability (얼마나 잘 구분하는지)")
print("=" * 60)
for fault_num in sorted(rieth_df['fault_type'].unique()):
    mask = rieth_df['fault_type'] == fault_num
    label = "Normal" if fault_num == 0 else f"Fault {fault_num}"
    proba_vals = rf.predict_proba(rieth_df.loc[mask, train_features].values)[:, 1]
    print(f"  {label:<10}: mean_proba={proba_vals.mean():.3f}, detected={np.mean(proba_vals > 0.5)*100:.1f}%")

print("\n" + "=" * 60)
print("완료! 핵심 후보:")
print("  1. submission_rieth_sup_top216.csv (supervised top 216)")
print("  2. submission_rieth_2of9_sup.csv (base8 + supervised 투표)")
print("  3. submission_rieth_hybrid_intersect.csv (교집합)")
print("=" * 60)
