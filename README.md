# Alphafold

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/AlphaFold?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/AlphaFold?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Alphafold — an intelligent, interactive ML application built for real-world prediction and analysis.

---

**Topics:** `protein-structure-prediction` · `deep-learning` · `computational-biology` · `bioinformatics` · `alphafold` · `structural-biology` · `ai-for-science` · `evolutionary-scale-modeling` · `molecular-biology` · `equivariant-transformer`

## Overview

Alphafold is a machine learning web application built on Streamlit that provides a clean, accessible interface for a trained predictive model. The application takes structured user input, preprocesses it through a scikit-learn Pipeline, and returns model predictions with confidence scores and explanatory visualisations.

The project follows a full ML lifecycle: data collection and EDA, feature engineering, model training and cross-validation, serialisation, and deployment as an interactive web application. All preprocessing steps are encapsulated in a Pipeline object to guarantee consistency between training and inference environments.

An embedded model evaluation module allows users to inspect the trained model's performance on the held-out test set, view the confusion matrix or regression residual plots, and compare multiple candidate models before selecting the deployment model.

---

## Motivation

This project was built to bridge the gap between a trained ML model and a non-technical end user. A model that lives only in a Jupyter notebook creates no value. Wrapping it in a clean, deployable Streamlit application makes it accessible, shareable, and immediately useful.

---

## Architecture

```
User Input (form / sliders)
        │
  sklearn Pipeline (scaler + model)
        │
  Prediction + Confidence
        │
  SHAP / Feature Importance Visualisation
```

---

## Features

### Input Interface
Validated form widgets covering all model features with sensible defaults and range constraints derived from the training data distribution.

### Model Inference
Trained scikit-learn pipeline loaded from a serialised joblib file at startup; inference is performed on every form submission.

### Confidence Display
Prediction probability or regression confidence interval displayed alongside the point prediction for transparent uncertainty communication.

### SHAP Explainability
Per-prediction SHAP waterfall chart showing individual feature contributions to the model output.

### Model Evaluation Panel
Confusion matrix (classification) or residual plot (regression) with accuracy, F1, AUC, MAE, RMSE metrics on the test split.

### Dataset Explorer
Interactive table and chart view of the training dataset with filtering, sorting, and distribution visualisations.

### Batch CSV Prediction
Upload a CSV file for bulk inference; results are returned as a downloadable table with predictions and confidence scores.

### Responsive Layout
Mobile-compatible Streamlit layout with sidebar controls and main panel visualisations.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **Streamlit** | Web application framework | Python-native UI with widget state management |
| **scikit-learn** | ML pipeline | Preprocessing, model, cross-validation, metrics |
| **pandas** | Data handling | CSV I/O, feature engineering, batch processing |
| **Plotly** | Visualisation | Interactive charts, confusion matrix, residual plots |
| **SHAP** | Explainability | Feature importance per prediction |
| **joblib** | Model persistence | Serialise and load pipeline |
| **NumPy** | Numerical ops | Array operations and statistical utilities |

> **Key packages detected in this repo:** `streamlit` · `google-generativeai` · `pandas` · `numpy` · `plotly` · `statsmodels`

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JS projects)
- `pip` or `npm` package manager
- Relevant API keys (see Configuration section)

### Installation

```bash
git clone https://github.com/Devanik21/AlphaFold.git
cd AlphaFold
python -m venv venv && source venv/bin/activate
pip install streamlit scikit-learn shap pandas plotly joblib numpy
streamlit run app.py
```

---

## Usage

```bash
streamlit run app.py

# Batch prediction
python batch_predict.py --input data.csv --output results.csv

# Retrain
python train.py --data dataset.csv
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model.pkl` | Path to serialised model pipeline |
| `SCALER_PATH` | `scaler.pkl` | Path to fitted scaler |
| `SHAP_ENABLED` | `True` | Enable SHAP explanations |
| `TOP_K` | `3` | Top-K results to display |

> Copy `.env.example` to `.env` and populate all required values before running.

---

## Project Structure

```
AlphaFold/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Continuous learning pipeline with user feedback loop
- [ ] A/B testing framework for model comparison in production
- [ ] REST API wrapper (FastAPI) for programmatic access
- [ ] Docker container for reproducible deployment
- [ ] Monitoring dashboard for prediction drift detection

---

## Contributing

Contributions, issues, and feature requests are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please follow conventional commit messages and ensure any new code is documented.

---

## Notes

All model performance metrics are reported on the held-out test set. Production performance may vary with distribution shift. Retrain periodically with fresh data.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Crafted with curiosity, precision, and a belief that good software is worth building well.*
