"""
화학 공정 이상 탐지 - Run 단위 판정 전략
핵심 인사이트: 같은 simulationRun은 모두 정상이거나 모두 이상!

전략:
1. 샘플별로 anomaly score 계산
2. Run별로 score 집계 (mean, max 등)
3. Run 단위로 정상/이상 판정
4. 해당 Run의 모든 샘플에 같은 라벨 부여
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

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

if 'Unnamed: 0' in train.columns:
    train = train.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in test.columns:
    test = test.drop(columns=['Unnamed: 0'])

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")
print(f"Train simulationRun 수: {train['simulationRun'].nunique()}")
print(f"Test simulationRun 수: {test['simulationRun'].nunique()}")

# ============================================================
# 2. 피처 준비
# ============================================================
print("\n" + "=" * 60)
print("2. 피처 준비")
print("=" * 60)

id_cols = ['faultNumber', 'simulationRun', 'sample']
feature_cols = [col for col in train.columns if col not in id_cols]

# Run별 통계 피처 추가
def add_run_features(df, feature_cols):
    """Run 내 위치 및 변화량 피처"""
    result = df.copy()
    
    # 기본 통계
    result['row_mean'] = df[feature_cols].mean(axis=1)
    result['row_std'] = df[feature_cols].std(axis=1)
    result['row_max'] = df[feature_cols].max(axis=1)
    result['row_min'] = df[feature_cols].min(axis=1)
    
    # 시계열 변화량 (주요 센서)
    for col in feature_cols[:10]:
        result[f'{col}_diff'] = df.groupby('simulationRun')[col].diff().fillna(0)
        result[f'{col}_diff2'] = df.groupby('simulationRun')[col].diff(2).fillna(0)
    
    return result

train_fe = add_run_features(train, feature_cols)
test_fe = add_run_features(test, feature_cols)

all_feature_cols = [col for col in train_fe.columns if col not in id_cols]
print(f"피처 수: {len(all_feature_cols)}")

X_train = train_fe[all_feature_cols].values
X_test = test_fe[all_feature_cols].values

# 스케일링
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 3. 모델 학습 및 샘플별 스코어 계산
# ============================================================
print("\n" + "=" * 60)
print("3. 모델 학습")
print("=" * 60)

# 여러 모델 앙상블
print("ECOD 학습...")
ecod = ECOD(contamination=0.1)
ecod.fit(X_train_scaled)
score_ecod = ecod.decision_function(X_test_scaled)

print("COPOD 학습...")
copod = COPOD(contamination=0.1)
copod.fit(X_train_scaled)
score_copod = copod.decision_function(X_test_scaled)

print("IForest 학습...")
iforest = IForest(n_estimators=500, contamination=0.1, random_state=42)
iforest.fit(X_train_scaled)
score_if = iforest.decision_function(X_test_scaled)

# 스코어 정규화 및 앙상블
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

ensemble_score = (
    0.4 * normalize(score_ecod) +
    0.3 * normalize(score_copod) +
    0.3 * normalize(score_if)
)

print(f"샘플별 앙상블 스코어: mean={ensemble_score.mean():.4f}, std={ensemble_score.std():.4f}")

# ============================================================
# 4. Run 단위 집계 및 판정 (핵심!)
# ============================================================
print("\n" + "=" * 60)
print("4. Run 단위 판정")
print("=" * 60)

# 테스트 데이터에 스코어 붙이기
test_with_score = test[['simulationRun', 'sample']].copy()
test_with_score['score'] = ensemble_score

# Run별 스코어 집계
run_stats = test_with_score.groupby('simulationRun').agg({
    'score': ['mean', 'max', 'std', 'median', 
              lambda x: np.percentile(x, 75),  # 75th percentile
              lambda x: np.percentile(x, 90)]  # 90th percentile
}).reset_index()

run_stats.columns = ['simulationRun', 'score_mean', 'score_max', 'score_std', 
                     'score_median', 'score_p75', 'score_p90']

# Run별 종합 스코어 (여러 집계 방식 조합)
run_stats['run_score'] = (
    0.3 * normalize(run_stats['score_mean'].values) +
    0.3 * normalize(run_stats['score_max'].values) +
    0.2 * normalize(run_stats['score_p90'].values) +
    0.2 * normalize(run_stats['score_p75'].values)
)

print(f"Run 수: {len(run_stats)}")
print(f"Run 스코어: mean={run_stats['run_score'].mean():.4f}, std={run_stats['run_score'].std():.4f}")

# 다양한 threshold 확인
print("\nRun 단위 threshold별 이상 Run 비율:")
for pct in [50, 55, 60, 65, 70, 75, 80]:
    th = np.percentile(run_stats['run_score'], pct)
    n_anomaly_runs = (run_stats['run_score'] > th).sum()
    print(f"  {pct}%ile: 이상 Run = {n_anomaly_runs} / {len(run_stats)}")

# ============================================================
# 5. 최종 예측 (Run 단위 → 샘플 단위)
# ============================================================
print("\n" + "=" * 60)
print("5. 최종 예측")
print("=" * 60)

# Threshold 선택 (조절!)
RUN_THRESHOLD_PCT = 65  # 50~80 사이에서 조절

run_threshold = np.percentile(run_stats['run_score'], RUN_THRESHOLD_PCT)
run_stats['is_anomaly'] = (run_stats['run_score'] > run_threshold).astype(int)

print(f"Threshold: {RUN_THRESHOLD_PCT}%ile = {run_threshold:.4f}")
print(f"이상 Run 수: {run_stats['is_anomaly'].sum()} / {len(run_stats)}")

# Run 라벨을 샘플에 매핑
test_result = test[['simulationRun']].merge(
    run_stats[['simulationRun', 'is_anomaly']], 
    on='simulationRun', 
    how='left'
)

predictions = test_result['is_anomaly'].values
print(f"이상 샘플 수: {predictions.sum():,} / {len(predictions):,}")
print(f"이상 비율: {predictions.mean():.4f}")

# ============================================================
# 6. 제출 파일 생성
# ============================================================
print("\n" + "=" * 60)
print("6. 제출 파일 생성")
print("=" * 60)

output = sample_submission.copy()
target_col = sample_submission.columns[-1]
output[target_col] = predictions
output.to_csv('output.csv', index=True)

print("저장 완료: output.csv")
print(output.head())

# 다양한 threshold로 여러 파일 생성
print("\n다양한 threshold 파일 생성:")
for pct in [50, 55, 60, 65, 70, 75]:
    th = np.percentile(run_stats['run_score'], pct)
    run_stats['pred_temp'] = (run_stats['run_score'] > th).astype(int)
    
    temp_result = test[['simulationRun']].merge(
        run_stats[['simulationRun', 'pred_temp']], 
        on='simulationRun', 
        how='left'
    )
    
    temp_output = sample_submission.copy()
    temp_output[target_col] = temp_result['pred_temp'].values
    temp_output.to_csv(f'output_run_{pct}.csv', index=True)
    
    anomaly_ratio = temp_result['pred_temp'].mean()
    print(f"  {pct}%ile: 이상비율={anomaly_ratio:.4f} → output_run_{pct}.csv")

print("\n" + "=" * 60)
print("""
완료! 

output_run_50.csv ~ output_run_75.csv 중에서 
하나씩 제출해보고 F1 가장 높은 거 찾으세요!

보통 TEP 데이터는 이상 비율 10~30% 정도예요.
""")