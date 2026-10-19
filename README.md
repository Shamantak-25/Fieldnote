# 🌾 Fieldnote Edge: Multi-Modal Neuro-Symbolic & Federated Agriculture Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-OpenCLIP-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-FedAvg-FF7043)](https://flower.ai)
[![Opacus](https://img.shields.io/badge/Opacus-DP--SGD-0081CB)](https://opacus.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An offline-first, privacy-preserving multi-modal AI platform designed to deliver clinical-grade agronomic advisory directly to farmers at the network edge, without cloud latency or hallucinated predictions.

---

## 🌟 Key Architectural Pillars

### 1. 👁️ Zero-Shot Leaf Pathology (OpenCLIP ViT-B-32)
- Evaluates foliage images against pre-indexed text prompt embeddings across **13 crop disease classes**.
- Requires zero heavy local model fine-tuning; operates offline using cosine similarity spectra.

### 2. 🌱 In-Context Tabular Soil & Crop Yield Inference (TabPFN)
- Evaluates soil macronutrients (Nitrogen, Phosphorus, Potassium), pH, and microclimate (temperature, humidity, rainfall).
- Trained on **2,200 agronomic instances** achieving **92.73% classification accuracy** and **0.0788 t/ha MAE** in yield forecasting.

### 3. 🧠 First-Order Neuro-Symbolic Biological Guardrails
- Audits deep learning predictions against deterministic domain knowledge:
  - **Fungal Sporulation Guardrail**: If humidity $< 25\%$, fungal predictions (e.g. Late Blight) are overridden as false positives because spore germination is biologically unviable in arid air.
  - **Soil Acidity Guardrail**: If soil $\text{pH} < 4.5$, crops suffer nutrient lock; yield forecasts are scaled down by $35\%$.

### 4. 🌐 Privacy-Preserving Decentralized Federated Learning
- Implements **Flower (flwr) FedAvg** combined with **Opacus DP-SGD** (Gaussian gradient perturbation).
- Farm edge nodes train locally on proprietary soil telemetry. Raw records never leave edge devices, achieving strict differential privacy guarantees ($\varepsilon \le 2.0, \delta = 10^{-5}$).

### 5. 💻 Modern Interactive React.js UI (No Streamlit)
- **Interactive Leaf Pathology Tool**: 2.8x hover magnifying loupe, pathogen heatmap overlay, and click-to-pin lesion markers.
- **Real-Time Soil Telemetry**: Dynamic N-P-K balance bars, pH acidity meters, and microclimate risk gauges.
- **Actionable Remediation Checklist**: Interactive treatment progress bar ($0\% \to 100\%$).
- **Crop Yield 'What-If' Sandbox**: Real-time interactive yield delta calculator.
- **Multilingual Voice Synthesis**: Native speech readouts in English, हिन्दी (Hindi), and தமிழ் (Tamil) with animated soundwave visualizers.
- **Sunlight / Field Mode**: High-contrast outdoor view for mobile/tablet field inspections.

---

## 📂 Project Structure

```
├── artifacts/                           # Pre-trained model weights & tensors
│   ├── crop_tabular_models.joblib       # In-context crop classifier & yield regressor
│   ├── crop_model_metadata.json         # Dataset schemas and evaluation metrics
│   ├── federated_training_summary.json  # 3-round FedAvg + Opacus DP-SGD summary
│   └── vision_prompt_embeddings.pt      # 13x512 OpenCLIP pre-indexed tensor
├── data/
│   ├── tabular/Crop_recommendation.csv  # 2,200-row agronomic training dataset
│   └── vision/                          # Standard leaf benchmark samples
├── src/
│   ├── federated_sync.py                # Flower FedAvg + Opacus DP-SGD engine
│   ├── symbolic_engine.py               # First-order neuro-symbolic logic filter
│   ├── tabular_engine.py                # TabPFN / Random Forest surrogate engine
│   └── vision_engine.py                 # OpenCLIP ViT-B-32 zero-shot classifier
├── static/                              # Modern React.js frontend
│   ├── app.jsx                          # Main React 18 component tree
│   ├── index.html                       # Entry point (offline vendor imports)
│   ├── style.css                        # Organic agricultural UI design system
│   └── vendor/                          # Local React, ReactDOM, Babel standalone
├── tests/                               # Automated unit & integration test suite
│   ├── test_api.py                      # FastAPI endpoint tests
│   ├── test_engines.py                  # Vision, Tabular, Symbolic, & FedAvg tests
│   └── test_training_setup.py           # Dataset schema & training pipeline tests
├── config.py                            # Central agronomic constants & multilingual texts
├── requirements.txt                     # Production Python dependencies
└── server.py                            # High-performance FastAPI backend server
```

---

## ⚡ Quickstart

### 1. Prerequisites
- Python 3.10+ installed

### 2. Install Dependencies
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests -v
```

### 4. Launch the Platform
```bash
python server.py
```
Open your browser at **[http://localhost:8000](http://localhost:8000)**.

---
