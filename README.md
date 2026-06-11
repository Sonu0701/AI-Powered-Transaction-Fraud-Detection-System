# 🛡️ AI-Powered Transaction Fraud Detection System

A real-time intelligent fraud detection system built with Flask, XGBoost, Isolation Forest, Graph Neural Networks (GNN), and SHAP explainability. Features a fully interactive single-page dashboard for monitoring, analyzing, and reporting suspicious financial transactions.

---

## 📸 Dashboard Preview

> Real-time monitoring dashboard with live risk scoring, transaction network graph, and SHAP explanations.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **3-Model Ensemble** | XGBoost + Isolation Forest + GNN combined for higher accuracy |
| 🕸️ **Graph Neural Network** | Detects fraud rings via transaction relationship graphs |
| 📊 **SHAP Explainability** | Shows exactly which features drove each fraud decision |
| 📈 **Live Dashboard** | Real-time metrics, risk charts, transaction table |
| 🔔 **Alerts Page** | All flagged transactions with Review / Clear / Escalate actions |
| 👤 **Customer Profiles** | Per-account risk history and behavioral scoring |
| 📄 **Reports** | SAR report, CSV export, model performance summary |
| 📉 **Analytics** | Fraud by type, channel, monthly trend, model comparison radar chart |
| ⚙️ **Settings** | Adjustable risk thresholds, model weights, manual retrain trigger |
| 🔄 **Concept Drift Detection** | Kolmogorov-Smirnov + Mahalanobis distance monitoring |
| 🔁 **Auto-Retraining** | Weekly background retraining with AutoML |
| 🐳 **Docker Support** | docker-compose.yml included for containerized deployment |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **Backend** | Flask, Pandas, NumPy, Joblib, Waitress |
| **ML Models** | XGBoost, Isolation Forest, scikit-learn, SHAP |
| **GNN** | PyTorch, PyTorch Geometric |
| **Frontend** | Bootstrap 5, Chart.js, Vis-Network, Bootstrap Icons |
| **Experiment Tracking** | MLflow (local fallback if server not running) |
| **Drift Detection** | SciPy (KS test), scikit-learn (MinCovDet) |
| **Reporting** | Jinja2, HTML/CSS reports |
| **Deployment** | Docker, docker-compose |

---

## 📁 Project Structure

```
AI-Powered-Transaction-Fraud-Detection-System/
├── app.py                        # Main Flask application
├── train_models.py               # Standalone model training script
├── requirements.txt              # All Python dependencies
├── docker-compose.yml            # Docker deployment config
├── README.md
│
├── data/
│   └── bank_transactions_data_2.csv   # Transaction dataset
│
├── models/
│   ├── isolation_forest.pkl      # Trained Isolation Forest
│   ├── xgboost.pkl               # Trained XGBoost classifier
│   ├── shap_explainer.pkl        # SHAP TreeExplainer
│   ├── gnn_model.pt              # Trained GNN model
│   └── automl/
│       └── trainer.py            # AutoML retraining pipeline
│
├── templates/
│   └── dashboard.html            # Single-page app (all 6 sections)
│
├── graph_models/
│   ├── gnn_model.py              # FraudGNN architecture (GCN)
│   ├── data_loader.py            # Transaction graph builder (32 features)
│   └── train_gnn.py              # GNN training script
│
├── drift/
│   └── detector.py               # Concept drift detector (KS + Mahalanobis)
│
├── reporting/
│   └── generator.py              # SAR & summary report generator
│
└── profiling/
    └── builder.py                # Customer risk profiler
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Powered-Transaction-Fraud-Detection-System.git
cd AI-Powered-Transaction-Fraud-Detection-System
```

### 2. Create and activate virtual environment

```bash
# Using uv (recommended — much faster)
pip install uv
uv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
uv pip install shap           # Required for explainability
```

### 4. Train the models

```bash
python train_models.py
```

Expected output:
```
Saved: models/isolation_forest.pkl
Saved: models/xgboost.pkl
Saved: models/shap_explainer.pkl
Done! Now run: python app.py
```

### 5. Run the application

```bash
python app.py
```

Visit: **http://127.0.0.1:5000**

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

Visit: **http://localhost:5000**

---

## 🔌 API Reference

### POST `/api/analyze`
Analyze a transaction and return fraud risk score with SHAP explanation.

**Request Body:**
```json
{
  "AccountID": "AC00128",
  "TransactionAmount": 9500,
  "TransactionType": "Debit",
  "Channel": "Online",
  "Location": "Moscow",
  "CustomerOccupation": "Engineer",
  "TransactionDate": "2026-06-11 14:30:00",
  "PreviousTransactionDate": "2026-06-09 10:00:00",
  "TransactionDuration": 45,
  "LoginAttempts": 2,
  "AccountBalance": 3500.00,
  "DeviceID": "D042",
  "MerchantID": "M17"
}
```

**Response:**
```json
{
  "composite_score": 0.59,
  "xgboost_probability": 0.82,
  "isolation_forest_score": 0.04,
  "gnn_probability": 0.78,
  "customer_risk_score": 0.67,
  "drift_detected": false,
  "explanation": [
    {"feature": "AmountDeviation", "value": 1264.67, "shap_value": 0.42},
    {"feature": "TransactionAmount", "value": 9500.0, "shap_value": 0.38}
  ]
}
```

### GET `/api/transactions`
Returns recent transactions with risk scores and statuses.

### GET `/api/drift/status`
Returns current concept drift detection status.

### POST `/api/models/retrain`
Manually triggers model retraining on the latest data.

### GET `/api/customer/<customer_id>/profile`
Returns behavioral risk profile for a specific customer.

---

## 🧪 Test Transactions

**High Risk (should score > 0.7):**
| Field | Value |
|-------|-------|
| Account ID | AC00128 |
| Amount | 95000 |
| Type | Debit |
| Channel | Online |
| Location | Moscow |
| Occupation | Student |

**Low Risk (should score < 0.3):**
| Field | Value |
|-------|-------|
| Account ID | AC00245 |
| Amount | 45 |
| Type | Debit |
| Channel | ATM |
| Location | New York |
| Occupation | Doctor |

---

## 🤖 ML Architecture

```
Transaction Input
       │
       ▼
┌─────────────────────────────────────────┐
│           Feature Engineering            │
│  19 features: amount deviation, speed,  │
│  location hash, customer aggregates...  │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
XGBoost   Isolation   GNN
 (0.4w)   Forest      (0.2w)
          (0.4w)
    └─────────┼─────────┘
              ▼
    Composite Score (0–1)
    × Customer Risk Factor
              │
              ▼
    SHAP Explanation
    (Top 5 features)
```

---

## 📊 Dashboard Pages

| Page | Features |
|------|----------|
| **Dashboard** | Metric cards, transactions table, risk donut chart, trend line, analysis form, network graph |
| **Alerts** | Flagged transactions, filter by risk level, Review/Clear/Escalate actions |
| **Customers** | Per-account risk profiles, search and filter, total volume |
| **Reports** | SAR report, CSV export, model performance summary |
| **Analytics** | 4 charts: fraud by type, model radar comparison, by channel, monthly trend |
| **Settings** | Risk threshold sliders, model weight controls, manual retrain button |

---

## ⚠️ Known Limitations

- Dataset (`bank_transactions_data_2.csv`) has no real fraud labels — synthetic labels are used for demo. For production, use a labeled dataset like [PaySim on Kaggle](https://www.kaggle.com/datasets/ntnu-testimon/paysim1).
- PDF report generation requires `wkhtmltopdf` installed separately. HTML reports work without it.
- MLflow tracking requires a separate MLflow server on port 5001. Falls back to local file tracking automatically.

---

## 🗺️ Roadmap

- [ ] Real labeled fraud dataset integration
- [ ] User authentication (Flask-Login)
- [ ] Real-time Kafka streaming pipeline
- [ ] Neo4j graph database for fraud ring visualization
- [ ] REST API with FastAPI for production
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Email/SMS alerts on high-risk detection

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

Built and maintained by **Sonu Kumar**

---

⭐ If this project helped you, please give it a star on GitHub!