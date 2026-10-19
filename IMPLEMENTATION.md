# Fieldnote Agriculture Platform: Neuro-Symbolic & Federated Multi-Modal AI

**Course:** BCSE209L | **Faculty:** Dr. S. Sridevi | **Category 3:** Community Service-Oriented Project

---

## 🏗️ Architecture & Technology Stack

- **Frontend:** Pure HTML5, CSS3, JavaScript (ES6+), and React.js (`static/index.html`, `static/style.css`, `static/app.jsx`). Streamlit is completely removed from the frontend runtime.
- **Backend Server:** FastAPI & Uvicorn (`server.py`) serving high-performance REST APIs and offline static React assets.
- **Offline-First Vendor Cache:** Local React 18, ReactDOM, and Babel standalone packages stored in `static/vendor/` to allow completely offline execution on edge devices.
- **Vision Engine:** Zero-shot OpenCLIP `ViT-B-32` (`src/vision_engine.py`) utilizing pre-indexed tensor embeddings (`artifacts/vision_prompt_embeddings.pt`).
- **Tabular Engine:** In-context tabular learning for crop recommendation and yield regression (`src/tabular_engine.py`) loaded from `artifacts/crop_tabular_models.joblib` (92.73% test accuracy).
- **Neuro-Symbolic Filter:** First-Order Logic verification (`src/symbolic_engine.py`) enforcing biological plausibility (humidity thresholds & pH limits).
- **Federated Edge Privacy:** Multi-node simulation using Flower (`flwr`) `FedAvg` and Opacus Differential Privacy (`src/federated_sync.py`).

---

## 🚀 Running the Platform

To launch the FastAPI backend and React.js frontend:

```powershell
python server.py
```

Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 🧪 Verification & Automated Testing

Run the complete test suite with `pytest`:

```powershell
python -m pytest tests/
```

- `tests/test_engines.py`: Tests the Vision classifier, Tabular predictor, First-Order Logic rules, and Federated sync simulation.
- `tests/test_api.py`: Validates FastAPI REST endpoints (`/api/status`, `/api/presets`, `/api/diagnose`, `/api/federated`).
- `tests/test_training_setup.py`: Verifies training pipelines and dataset ingestion requirements.