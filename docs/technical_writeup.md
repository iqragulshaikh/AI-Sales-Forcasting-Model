# Technical Write-up: AI Sales Forecasting Model for Restaurant Chain

## 1. Executive Summary
This project delivers an end-to-end sales forecasting pipeline designed for multi-branch restaurant operations. The system processes historical daily sales, enforces schema validation, performs leakage-safe feature engineering, computes numeric sales forecasts, and evaluates model performance. An integrated natural-language explainer provides context-aware summaries of forecasts and performance metrics, complete with an API key fallback for offline reliability.

## 2. Dataset & Data Validation
* **Dataset Structure:** Contains daily sales records across branches (`date`, `branch_id`, `sales`, `orders`, `promotion`, `holiday`, `store_open`).
* **Input Validation (`src/data_validation.py`):** Ensures required columns exist, checks for negative sales values, parses dates, validates branch presence, and enforces constraints on forecast horizons.
* **Error Handling:** Robustly catches unknown branch IDs, non-numeric inputs, and out-of-range forecast horizons, returning descriptive error messages before execution.

## 3. Preprocessing, Features & Baseline
* **Feature Engineering (`src/features.py`):** Generates calendar features (day of week, month, weekend flags), branch encodings, lag features, and rolling averages.
* **Data Leakage Prevention:** All lag and rolling calculations strictly utilize past observations shifted relative to the target prediction date.
* **Baseline Benchmark (`src/baseline.py`):** Established using a previous-value / rolling historical average model to benchmark primary model improvements.

## 4. Modeling & Evaluation
* **Primary Forecasting Model (`src/forecasting.py`):** Leverages feature-based numeric forecasting on chronologically split historical data.
* **Performance Metrics (`src/evaluation.py`):** Evaluated over a chronological test period:
  * **MAE:** 1898.24
  * **RMSE:** 2736.17
  * **SMAPE:** 1.55%
* **Hugging Face Transformers Experiment:** Evaluated time-series transformer architectures for forecasting feasibility. Primary reliance remains on the reliable Python feature model to ensure deterministic execution.

## 5. RAG & LLM Explanation Layer
* **Knowledge Retrieval (`src/retrieval.py`):** Uses vector stores to index project documentation, technical metrics, and data dictionaries.
* **Grounded Prompt Design (`src/llm_explainer.py`):** Prompt constraints enforce strict numeric grounding on the output forecast table to prevent hallucinations.
* **API Fallback Strategy:** If API keys are missing or unconfigured, the system automatically bypasses live generation and returns the structured numeric forecast table directly without raising errors.

## 6. Limitations & Future Work
* **Limitations:** Forecast accuracy depends on historical pattern consistency; extreme outer-bound events or unannounced promotions may impact variance.
* **Future Work:** Incorporate external weather API features, test advanced transformer architectures on larger multi-year datasets, and integrate real-time point-of-sale data streams.
*