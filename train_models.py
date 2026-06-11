"""
train_models.py
---------------
Run this once to train and save the 3 missing model files:
  - models/isolation_forest.pkl
  - models/xgboost.pkl
  - models/shap_explainer.pkl

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
DATA_PATH = "data/bank_transactions_data_2.csv"
logger.info(f"Loading dataset from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
logger.info(f"Dataset shape: {df.shape}")
logger.info(f"Columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 2. Feature engineering — matches exactly what app.py expects
# ---------------------------------------------------------------------------
logger.info("Engineering features...")

# Handle date columns if present
date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
for col in date_cols:
    try:
        df[col] = pd.to_datetime(df[col])
    except Exception:
        pass

# Rename columns to match app.py feature names if needed
col_map = {}
for col in df.columns:
    low = col.lower().replace(' ', '').replace('_', '')
    if low in ['transactionamount', 'amount']:
        col_map[col] = 'TransactionAmount'
    elif low in ['transactionduration', 'duration']:
        col_map[col] = 'TransactionDuration'
    elif low in ['loginattempts']:
        col_map[col] = 'LoginAttempts'
    elif low in ['accountbalance', 'balance']:
        col_map[col] = 'AccountBalance'
    elif low in ['transactiontype', 'type']:
        col_map[col] = 'TransactionType'
    elif low in ['location']:
        col_map[col] = 'Location'
    elif low in ['deviceid', 'device']:
        col_map[col] = 'DeviceID'
    elif low in ['merchantid', 'merchant']:
        col_map[col] = 'MerchantID'
    elif low in ['channel']:
        col_map[col] = 'Channel'
    elif low in ['customeroccupation', 'occupation']:
        col_map[col] = 'CustomerOccupation'
    elif low in ['isfraud', 'fraud', 'fraudflag', 'label']:
        col_map[col] = 'isFraud'

if col_map:
    df.rename(columns=col_map, inplace=True)
    logger.info(f"Renamed columns: {col_map}")

# Ensure required numeric columns exist with defaults if missing
if 'TransactionDuration' not in df.columns:
    df['TransactionDuration'] = np.random.randint(30, 300, len(df))
if 'LoginAttempts' not in df.columns:
    df['LoginAttempts'] = 1
if 'AccountBalance' not in df.columns:
    df['AccountBalance'] = df.get('TransactionAmount', pd.Series(np.zeros(len(df)))) * 10

# Customer-level aggregates
logger.info("Computing customer-level aggregates...")
account_col = next((c for c in df.columns if 'account' in c.lower()), None)
if account_col:
    agg = df.groupby(account_col)['TransactionAmount'].agg(
        AvgAmount='mean', StdAmount='std', MaxAmount='max'
    ).fillna(0).reset_index()
    df = df.merge(agg, on=account_col, how='left')
else:
    df['AvgAmount'] = df['TransactionAmount'].mean()
    df['StdAmount'] = df['TransactionAmount'].std()
    df['MaxAmount'] = df['TransactionAmount'].max()

df['StdAmount']   = df['StdAmount'].fillna(1).clip(lower=0.001)
df['AvgDuration'] = df.get('TransactionDuration', pd.Series(np.full(len(df), 120.0))).mean()
df['UniqueLocations'] = 3  # default; replace with real calc if Location col exists

df['TransactionSpeed'] = df['TransactionAmount'] / df['TransactionDuration'].clip(lower=0.001)
df['AmountDeviation']  = (df['TransactionAmount'] - df['AvgAmount']) / df['StdAmount']
df['DurationDeviation'] = (df['TransactionDuration'] - df['AvgDuration']) / df['AvgDuration'].clip(lower=0.001)
df['DaysSinceLastTransaction'] = 7  # default

# Encode categoricals
for col, mapping in [
    ('TransactionType',    {'Debit': 0, 'Credit': 1}),
    ('Channel',            {'ATM': 0, 'Online': 1, 'Branch': 2}),
    ('CustomerOccupation', {'Student': 0, 'Doctor': 1, 'Engineer': 2, 'Retired': 3}),
]:
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(0).astype(int)
    else:
        df[col] = 0

for col in ['Location', 'DeviceID', 'MerchantID']:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: hash(str(x)) % 100)
    else:
        df[col] = 0

# ---------------------------------------------------------------------------
# 3. Define feature matrix and target
# ---------------------------------------------------------------------------
feature_cols = [
    'TransactionAmount', 'TransactionDuration', 'LoginAttempts',
    'AccountBalance', 'DaysSinceLastTransaction', 'TransactionSpeed',
    'AvgAmount', 'StdAmount', 'MaxAmount', 'AvgDuration', 'UniqueLocations',
    'AmountDeviation', 'DurationDeviation', 'TransactionType',
    'Location', 'DeviceID', 'MerchantID', 'Channel', 'CustomerOccupation'
]

X = df[feature_cols].fillna(0)

# Target column
target_col = 'isFraud'
if target_col not in df.columns:
    # Try to find fraud column
    fraud_candidates = [c for c in df.columns if 'fraud' in c.lower() or 'label' in c.lower()]
    if fraud_candidates:
        target_col = fraud_candidates[0]
        logger.info(f"Using '{target_col}' as target column")
    else:
        logger.warning("No fraud/label column found — creating synthetic labels for demo")
        df[target_col] = (df['AmountDeviation'] > 2.5).astype(int)

y = df[target_col].fillna(0).astype(int)
logger.info(f"Class distribution:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------------------------------------------------------
# 4. Train Isolation Forest
# ---------------------------------------------------------------------------
logger.info("Training Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=max(y.mean(), 0.01),  # use actual fraud rate
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train)
iso_scores = -iso_forest.decision_function(X_test)
logger.info(f"Isolation Forest — AUC: {roc_auc_score(y_test, iso_scores):.4f}")

# ---------------------------------------------------------------------------
# 5. Train XGBoost
# ---------------------------------------------------------------------------
logger.info("Training XGBoost...")
scale_pos_weight = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1)
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,   # handles class imbalance
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
logger.info(f"XGBoost — AUC: {roc_auc_score(y_test, xgb_probs):.4f}")
logger.info(f"\n{classification_report(y_test, xgb_model.predict(X_test))}")

# ---------------------------------------------------------------------------
# 6. Build SHAP explainer
# ---------------------------------------------------------------------------
logger.info("Building SHAP explainer (this may take a minute)...")
shap_explainer = shap.TreeExplainer(xgb_model)
# Quick sanity check
sample = X_test.iloc[:5]
sv = shap_explainer.shap_values(sample)
if isinstance(sv, list):
    sv = sv[1]
logger.info(f"SHAP values shape: {np.array(sv).shape} — OK")

# ---------------------------------------------------------------------------
# 7. Save all models
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(iso_forest,     'models/isolation_forest.pkl')
joblib.dump(xgb_model,      'models/xgboost.pkl')
joblib.dump(shap_explainer, 'models/shap_explainer.pkl')

logger.info("Saved:")
logger.info("  models/isolation_forest.pkl")
logger.info("  models/xgboost.pkl")
logger.info("  models/shap_explainer.pkl")
logger.info("Done! Now run: python app.py")