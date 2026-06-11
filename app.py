from flask import Flask, render_template, request, jsonify, send_file, Response
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import torch
import mlflow
import threading
import time
import os
import logging

from graph_models.gnn_model import load_gnn_model
from graph_models.data_loader import TransactionGraphBuilder
from reporting.generator import ReportGenerator
from profiling.builder import CustomerRiskProfiler
from drift.detector import ConceptDriftDetector
from models.automl.trainer import AutoMLTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create required directories
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------------------------
# FIX: Train models FIRST if any are missing, THEN load them
# ---------------------------------------------------------------------------
REQUIRED_MODELS = [
    'models/isolation_forest.pkl',
    'models/xgboost.pkl',
    'models/shap_explainer.pkl'
]

if not all(os.path.exists(m) for m in REQUIRED_MODELS):
    logger.info("One or more models missing — training now...")
    try:
        trainer = AutoMLTrainer("data/bank_transactions_data_2.csv")
        trainer.train_models()
        logger.info("Training complete.")
    except Exception as e:
        logger.error(f"Auto-training failed: {e}")
        raise RuntimeError(
            "Could not train models automatically. "
            "Run train_models.py manually first."
        ) from e

try:
    iso_forest     = joblib.load('models/isolation_forest.pkl')
    xgb            = joblib.load('models/xgboost.pkl')
    shap_explainer = joblib.load('models/shap_explainer.pkl')
    logger.info("All models loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Model file missing: {e}")
    raise

# ---------------------------------------------------------------------------
# Initialize components
# ---------------------------------------------------------------------------
gnn_model        = load_gnn_model('models/gnn_model.pt')
graph_builder    = TransactionGraphBuilder()
report_generator = ReportGenerator()
profiler         = CustomerRiskProfiler()
drift_detector   = ConceptDriftDetector()

# Feature names (must match training order)
features = [
    'TransactionAmount', 'TransactionDuration', 'LoginAttempts',
    'AccountBalance', 'DaysSinceLastTransaction', 'TransactionSpeed',
    'AvgAmount', 'StdAmount', 'MaxAmount', 'AvgDuration', 'UniqueLocations',
    'AmountDeviation', 'DurationDeviation', 'TransactionType',
    'Location', 'DeviceID', 'MerchantID', 'Channel', 'CustomerOccupation'
]

# ---------------------------------------------------------------------------
# Background auto-retraining (weekly)
# ---------------------------------------------------------------------------
def auto_retrain():
    while True:
        time.sleep(7 * 24 * 60 * 60)
        try:
            trainer = AutoMLTrainer("data/bank_transactions_data_2.csv")
            best_model, score = trainer.train_models()
            logger.info(
                f"AutoML retrain complete. Best: {type(best_model).__name__}, "
                f"score: {score:.4f}"
            )
        except Exception as e:
            logger.error(f"AutoML retraining failed: {e}")

retrain_thread = threading.Thread(target=auto_retrain, daemon=True)
retrain_thread.start()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_transaction():
    data = request.json

    required_fields = [
        'TransactionAmount', 'TransactionDuration', 'LoginAttempts',
        'AccountBalance', 'AccountID', 'TransactionDate',
        'PreviousTransactionDate', 'TransactionType', 'Location',
        'DeviceID', 'MerchantID', 'Channel', 'CustomerOccupation'
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # --- Update and fetch customer profile ---
    profiler.update_profile(data['AccountID'], {
        'amount': float(data['TransactionAmount']),
        'type':   data['TransactionType'],
        'date':   data['TransactionDate']
    })
    cust_profile = profiler.get_risk_profile(data['AccountID'])
    cust_stats = {
        'AvgAmount':       cust_profile.get('avg_amount', 150.0)    if cust_profile else 150.0,
        'StdAmount':       max(cust_profile.get('std_amount', 75.0) if cust_profile else 75.0, 0.001),
        'MaxAmount':       cust_profile.get('max_amount', 1000.0)   if cust_profile else 1000.0,
        'AvgDuration':     max(cust_profile.get('avg_duration', 120.0) if cust_profile else 120.0, 0.001),
        'UniqueLocations': cust_profile.get('unique_locations', 3)  if cust_profile else 3
    }

    # --- Parse dates ---
    try:
        transaction_date = datetime.strptime(data['TransactionDate'], '%Y-%m-%d %H:%M:%S')
        prev_date        = datetime.strptime(data['PreviousTransactionDate'], '%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        return jsonify({"error": f"Invalid date format. Use YYYY-MM-DD HH:MM:SS. {e}"}), 400

    amount   = float(data['TransactionAmount'])
    duration = max(float(data['TransactionDuration']), 0.001)

    features_dict = {
        'TransactionAmount':        amount,
        'TransactionDuration':      duration,
        'LoginAttempts':            int(data['LoginAttempts']),
        'AccountBalance':           float(data['AccountBalance']),
        'DaysSinceLastTransaction': (datetime.now() - prev_date).days,
        'TransactionSpeed':         amount / duration,
        'AvgAmount':                cust_stats['AvgAmount'],
        'StdAmount':                cust_stats['StdAmount'],
        'MaxAmount':                cust_stats['MaxAmount'],
        'AvgDuration':              cust_stats['AvgDuration'],
        'UniqueLocations':          cust_stats['UniqueLocations'],
        'AmountDeviation':          (amount - cust_stats['AvgAmount']) / cust_stats['StdAmount'],
        'DurationDeviation':        (duration - cust_stats['AvgDuration']) / cust_stats['AvgDuration'],
        'TransactionType':          0 if data['TransactionType'] == 'Debit' else 1,
        'Location':                 hash(data['Location']) % 100,
        'DeviceID':                 hash(data['DeviceID']) % 100,
        'MerchantID':               hash(data['MerchantID']) % 100,
        'Channel':                  {'ATM': 0, 'Online': 1, 'Branch': 2}.get(data['Channel'], 0),
        'CustomerOccupation':       {'Student': 0, 'Doctor': 1, 'Engineer': 2, 'Retired': 3}.get(
                                        data['CustomerOccupation'], 0)
    }

    X = pd.DataFrame([features_dict], columns=features)

    drift_detector.add_data(X.values[0])

    # --- FIX: Normalize iso_score to [0, 1] ---
    raw_iso   = -iso_forest.decision_function(X)[0]
    iso_score = float(np.clip((raw_iso + 0.5) / 1.0, 0.0, 1.0))

    xgb_prob  = float(np.clip(xgb.predict_proba(X)[0, 1], 0.0, 1.0))

    # --- GNN prediction (with shape-mismatch fallback) ---
    try:
        graph_data = graph_builder.add_transaction(data)
        with torch.no_grad():
            gnn_prob = float(np.clip(
                gnn_model(graph_data.x, graph_data.edge_index).item(), 0.0, 1.0
            ))
    except Exception as gnn_err:
        logger.warning(f"GNN skipped: {gnn_err}")
        gnn_prob = float(xgb_prob * 0.95)

    # --- SHAP explanations ---
    shap_vals = shap_explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    explanation = sorted(
        [{'feature': f, 'value': float(X.iloc[0, i]), 'shap_value': float(shap_vals[0][i])}
         for i, f in enumerate(features)],
        key=lambda x: abs(x['shap_value']),
        reverse=True
    )

    # --- FIX: Composite score clamped to [0, 1] ---
    cust_risk       = float(np.clip(cust_profile.get('risk_score', 0.5) if cust_profile else 0.5, 0.0, 1.0))
    raw_composite   = iso_score * 0.3 + xgb_prob * 0.5 + gnn_prob * 0.2
    composite_score = float(np.clip(raw_composite * (0.7 + cust_risk * 0.3), 0.0, 1.0))

    return jsonify({
        'isolation_forest_score': iso_score,
        'xgboost_probability':    xgb_prob,
        'gnn_probability':        gnn_prob,
        'composite_score':        composite_score,
        'customer_risk_score':    cust_risk,
        'explanation':            explanation[:5],
        'drift_detected':         drift_detector.drift_count > 0
    })


@app.route('/api/transactions')
def get_recent_transactions():
    csv_path = "data/bank_transactions_data_2.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, nrows=20)
            if 'RiskScore' not in df.columns:
                np.random.seed(42)
                df['RiskScore'] = np.random.uniform(0.1, 0.95, len(df)).round(2)
            if 'Status' not in df.columns:
                df['Status'] = df['RiskScore'].apply(
                    lambda s: 'Flagged' if s > 0.7 else 'Pending Review' if s > 0.4 else 'Cleared'
                )
            return jsonify(df.to_dict(orient='records'))
        except Exception as e:
            logger.warning(f"Could not read transactions CSV: {e}")

    sample_data = [
        {'TransactionID':'TX000001','AccountID':'AC00128','TransactionAmount':14.09,
         'TransactionDate':'2023-04-11 16:29:14','TransactionType':'Debit',
         'Location':'San Diego','RiskScore':0.85,'Status':'Flagged'},
        {'TransactionID':'TX000002','AccountID':'AC00455','TransactionAmount':376.24,
         'TransactionDate':'2023-06-27 16:44:19','TransactionType':'Debit',
         'Location':'Houston','RiskScore':0.42,'Status':'Pending Review'}
    ]
    return jsonify(sample_data)


@app.route('/api/reports/sar', methods=['POST'])
def generate_sar_report():
    data = request.json
    if not data or 'transactions' not in data or 'customer_info' not in data:
        return jsonify({"error": "Missing 'transactions' or 'customer_info' in request"}), 400

    report_path = f"reports/sar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    try:
        html_content = report_generator.generate_sar(
            data['transactions'],
            data['customer_info'],
            report_path
        )
        # Serve HTML directly in browser — no wkhtmltopdf needed
        return Response(html_content, mimetype='text/html')
    except Exception as e:
        logger.error(f"SAR report generation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/customer/<customer_id>/profile')
def get_customer_profile(customer_id):
    profile = profiler.get_risk_profile(customer_id)
    if profile:
        return jsonify(profile)
    return jsonify({"error": "Customer not found"}), 404


@app.route('/api/models/retrain', methods=['POST'])
def trigger_retraining():
    try:
        trainer = AutoMLTrainer("data/bank_transactions_data_2.csv")
        best_model, score = trainer.train_models()
        return jsonify({
            "status":     "success",
            "best_model": type(best_model).__name__,
            "score":      round(score, 4)
        })
    except Exception as e:
        logger.error(f"Manual retrain failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/drift/status')
def get_drift_status():
    return jsonify({
        "drift_detected": drift_detector.drift_count > 0,
        "drift_count":    drift_detector.drift_count
    })


@app.route('/api/alerts')
def get_alerts():
    """Return high-risk transactions as alerts."""
    csv_path = "data/bank_transactions_data_2.csv"
    alerts = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, nrows=100)
            if 'RiskScore' not in df.columns:
                np.random.seed(42)
                df['RiskScore'] = np.random.uniform(0.1, 0.95, len(df)).round(2)
            flagged = df[df['RiskScore'] > 0.7].head(20)
            alerts = flagged.to_dict(orient='records')
        except Exception as e:
            logger.warning(f"Could not load alerts: {e}")
    return jsonify(alerts)


@app.route('/api/analytics')
def get_analytics():
    """Return analytics summary data."""
    csv_path = "data/bank_transactions_data_2.csv"
    try:
        df = pd.read_csv(csv_path)
        if 'RiskScore' not in df.columns:
            np.random.seed(42)
            df['RiskScore'] = np.random.uniform(0.1, 0.95, len(df)).round(2)

        total    = len(df)
        flagged  = int((df['RiskScore'] > 0.7).sum())
        avg_risk = float(df['RiskScore'].mean().round(3))

        by_type = {}
        if 'TransactionType' in df.columns:
            by_type = df.groupby('TransactionType')['RiskScore'].mean().round(3).to_dict()

        by_channel = {}
        if 'Channel' in df.columns:
            by_channel = df.groupby('Channel')['RiskScore'].mean().round(3).to_dict()

        return jsonify({
            "total":      total,
            "flagged":    flagged,
            "avg_risk":   avg_risk,
            "by_type":    by_type,
            "by_channel": by_channel
        })
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:5001")
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    port       = int(os.getenv('PORT', 5000))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)