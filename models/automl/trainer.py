import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib
import shap
import mlflow
from datetime import datetime
import time
import os
import logging
import numpy as np

class AutoMLTrainer:
    def __init__(self, data_path="data/bank_transactions_data_2.csv", experiment_name="fraud_detection"):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)

        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        self._init_mlflow()

    def _init_mlflow(self):
        try:
            mlflow.set_tracking_uri("http://localhost:5001")
            if not mlflow.get_experiment_by_name(self.experiment_name):
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            self.logger.warning(f"MLflow initialization failed, using local tracking: {e}")
            mlflow.set_tracking_uri("file:./mlruns")
            try:
                mlflow.set_experiment(self.experiment_name)
            except Exception:
                pass

    def _generate_fraud_labels(self, df):
        fraud = np.zeros(len(df))

        if 'TransactionAmount' in df.columns and 'AccountBalance' in df.columns:
            ratio = df['TransactionAmount'] / df['AccountBalance'].clip(lower=1)
            fraud[ratio > 0.8] = 1

        if 'LoginAttempts' in df.columns:
            fraud[df['LoginAttempts'] > 2] = 1

        if 'TransactionDuration' in df.columns:
            fraud[df['TransactionDuration'] < 10] = 1

        if 'TransactionAmount' in df.columns:
            threshold = df['TransactionAmount'].quantile(0.97)
            fraud[df['TransactionAmount'] > threshold] = 1

        if fraud.mean() < 0.01:
            n = max(int(len(df) * 0.03), 5)
            idx = df['TransactionAmount'].nlargest(n).index
            fraud[idx] = 1

        self.logger.info(f"Synthetic fraud rate: {fraud.mean():.2%} ({int(fraud.sum())} / {len(df)})")
        return fraud

    def _prepare_features(self, df):
        for col in ['TransactionDate', 'PreviousTransactionDate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        if 'TransactionDate' in df.columns and 'PreviousTransactionDate' in df.columns:
            df['DaysSinceLastTransaction'] = (
                df['TransactionDate'] - df['PreviousTransactionDate']
            ).dt.days.fillna(7)
        else:
            df['DaysSinceLastTransaction'] = 7

        df['TransactionDuration'] = df.get(
            'TransactionDuration', pd.Series(np.full(len(df), 60))
        ).clip(lower=0.001)
        df['TransactionSpeed'] = df['TransactionAmount'] / df['TransactionDuration']

        if 'AccountID' in df.columns:
            agg = df.groupby('AccountID')['TransactionAmount'].agg(
                AvgAmount='mean', StdAmount='std', MaxAmount='max'
            ).fillna(0).reset_index()
            df = df.merge(agg, on='AccountID', how='left')
        else:
            df['AvgAmount'] = df['TransactionAmount'].mean()
            df['StdAmount'] = df['TransactionAmount'].std()
            df['MaxAmount'] = df['TransactionAmount'].max()

        df['StdAmount']   = df['StdAmount'].fillna(1).clip(lower=0.001)
        df['AvgDuration'] = df['TransactionDuration'].mean()
        df['UniqueLocations'] = 3

        df['AmountDeviation']   = (df['TransactionAmount'] - df['AvgAmount']) / df['StdAmount']
        df['DurationDeviation'] = (df['TransactionDuration'] - df['AvgDuration']) / df['AvgDuration']

        df['TransactionType'] = df.get('TransactionType', pd.Series(['Debit'] * len(df))).map(
            {'Debit': 0, 'Credit': 1}).fillna(0).astype(int)
        df['Channel'] = df.get('Channel', pd.Series(['ATM'] * len(df))).map(
            {'ATM': 0, 'Online': 1, 'Branch': 2}).fillna(0).astype(int)
        df['CustomerOccupation'] = df.get('CustomerOccupation', pd.Series(['Student'] * len(df))).map(
            {'Student': 0, 'Doctor': 1, 'Engineer': 2, 'Retired': 3}).fillna(0).astype(int)

        for col in ['Location', 'DeviceID', 'MerchantID']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: hash(str(x)) % 100)
            else:
                df[col] = 0

        feature_cols = [
            'TransactionAmount', 'TransactionDuration', 'LoginAttempts',
            'AccountBalance', 'DaysSinceLastTransaction', 'TransactionSpeed',
            'AvgAmount', 'StdAmount', 'MaxAmount', 'AvgDuration', 'UniqueLocations',
            'AmountDeviation', 'DurationDeviation', 'TransactionType',
            'Location', 'DeviceID', 'MerchantID', 'Channel', 'CustomerOccupation'
        ]

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        return df[feature_cols].fillna(0)

    def preprocess_data(self):
        df = pd.read_csv(self.data_path)
        self.logger.info(f"Loaded {len(df)} rows from {self.data_path}")

        X = self._prepare_features(df)

        fraud_col = next(
            (c for c in df.columns if c.lower() in ['isfraud', 'is_fraud', 'fraud', 'label']),
            None
        )
        if fraud_col:
            y = df[fraud_col].fillna(0).astype(int)
            self.logger.info(f"Using real fraud column: '{fraud_col}'")
        else:
            y = pd.Series(self._generate_fraud_labels(df).astype(int), index=df.index)

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train_models(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        fraud_rate = y_train.mean()
        # FIX: cap scale_pos_weight so it doesn't explode on very low fraud rates
        scale_pos = min(max((1 - fraud_rate) / max(fraud_rate, 0.001), 1), 100)

        # --- Train IsolationForest (always saved — app.py requires it) ---
        self.logger.info("Training IsolationForest...")
        iso = IsolationForest(
            n_estimators=100,
            contamination=max(fraud_rate, 0.01),
            random_state=42,
            n_jobs=-1
        )
        iso.fit(X_train)
        iso_pred  = -iso.decision_function(X_test)
        try:
            iso_auc = roc_auc_score(y_test, iso_pred)
        except Exception:
            iso_auc = float('nan')
        self.logger.info(f"IsolationForest AUC: {iso_auc:.4f}" if not np.isnan(iso_auc) else "IsolationForest AUC: n/a")
        joblib.dump(iso, "models/isolation_forest.pkl")
        self.logger.info("Saved: models/isolation_forest.pkl")

        # --- Train XGBoost (always saved — app.py requires it) ---
        self.logger.info("Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict_proba(X_test)[:, 1]
        try:
            xgb_auc = roc_auc_score(y_test, xgb_pred)
        except Exception:
            xgb_auc = float('nan')
        self.logger.info(f"XGBoost AUC: {xgb_auc:.4f}" if not np.isnan(xgb_auc) else "XGBoost AUC: n/a")
        joblib.dump(xgb, "models/xgboost.pkl")
        self.logger.info("Saved: models/xgboost.pkl")

        # --- Train RandomForest (saved for completeness / automl comparison) ---
        self.logger.info("Training RandomForest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict_proba(X_test)[:, 1]
        try:
            rf_auc = roc_auc_score(y_test, rf_pred)
        except Exception:
            rf_auc = float('nan')
        self.logger.info(f"RandomForest AUC: {rf_auc:.4f}" if not np.isnan(rf_auc) else "RandomForest AUC: n/a")
        joblib.dump(rf, "models/random_forest.pkl")
        self.logger.info("Saved: models/random_forest.pkl")

        # --- Build and save SHAP explainer (always saved — app.py requires it) ---
        self.logger.info("Building SHAP explainer...")
        try:
            explainer = shap.TreeExplainer(xgb)
            # Smoke-test on a small sample
            sample = X_train.iloc[:min(5, len(X_train))]
            _ = explainer.shap_values(sample)
            joblib.dump(explainer, "models/shap_explainer.pkl")
            self.logger.info("Saved: models/shap_explainer.pkl")
        except Exception as e:
            self.logger.error(f"SHAP explainer failed: {e}")
            raise

        # --- MLflow logging ---
        try:
            with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_metric("iso_auc",  iso_auc if not np.isnan(iso_auc) else 0)
                mlflow.log_metric("xgb_auc",  xgb_auc if not np.isnan(xgb_auc) else 0)
                mlflow.log_metric("rf_auc",   rf_auc  if not np.isnan(rf_auc)  else 0)
        except Exception:
            pass  # MLflow logging failure should never block training

        # --- Return best supervised model ---
        scores = {"xgboost": xgb_auc, "random_forest": rf_auc}
        best_name  = max(scores, key=lambda k: scores[k] if not np.isnan(scores[k]) else -1)
        best_model = xgb if best_name == "xgboost" else rf
        best_score = scores[best_name]

        self.logger.info(f"Best model: {best_name} (AUC {best_score:.4f})")
        return best_model, best_score if not np.isnan(best_score) else 0.0

    def retrain_schedule(self, interval_days=7):
        while True:
            try:
                self.logger.info("Starting scheduled retraining...")
                self.train_models()
            except Exception as e:
                self.logger.error(f"Retraining failed: {e}")
            finally:
                time.sleep(interval_days * 24 * 60 * 60)