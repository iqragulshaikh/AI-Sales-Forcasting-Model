# AI Sales Forecasting Model for Restaurant Chain

An end-to-end sales forecasting prototype built for multi-branch restaurant operations. The system accepts a branch ID and forecast horizon, validates input, predicts future sales, displays interactive visualization plots, and provides natural-language AI explanations grounded in numeric predictions.

## 🛠️ Architecture & Pipeline Overview
1. **Data Validation (`src/data_validation.py`):** Schema validation, edge-case error checks, and bounds verification.
2. **Preprocessing & Feature Engineering (`src/features.py`):** Calendar features, branch encodings, and leakage-safe lag/rolling variables.
3. **Forecasting Model (`src/forecasting.py`):** Numeric multi-period time-series forecasting model.
4. **Model Evaluation (`src/evaluation.py`):** Chronological evaluation calculating MAE, RMSE, and sMAPE.
5. **RAG & LLM Explainer (`src/llm_explainer.py`):** LangChain and vector retrieval interface with built-in API fallback.

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt