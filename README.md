# Coffee Disease AI API

FastAPI service for coffee leaf disease prediction.

## Local Development

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Open:

- API: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Render Deployment

This repo includes `render.yaml`, so you can create the service from Render Blueprint.

Manual Render settings:

- Runtime: Python
- Build command: `pip install --upgrade pip && pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Python version: `3.11.9`

Environment variables:

- `COFFEE_CORS_ORIGINS`: comma-separated allowed origins, or `*` for local/testing.
- `MODEL_PATH`: optional custom model path.
- `LABELS_PATH`: optional custom labels path.
- `TF_CPP_MIN_LOG_LEVEL`: set to `2` to reduce TensorFlow logs.

Make sure these files are deployed with the service:

- `models/coffee_disease_model.keras`
- `models/labels.txt`
