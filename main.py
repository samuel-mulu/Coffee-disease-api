import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Coffee Disease AI API")


def get_cors_origins():
    raw = os.getenv("COFFEE_CORS_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "models", "coffee_disease_model.keras"),
)
LABELS_PATH = os.getenv(
    "LABELS_PATH",
    os.path.join(BASE_DIR, "models", "labels.txt"),
)

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.85

model = None
class_names = []


@app.on_event("startup")
def load_assets():
    global model, class_names

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(LABELS_PATH):
        raise RuntimeError(f"Labels not found: {LABELS_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    if len(class_names) == 0:
        raise RuntimeError("No labels found in labels.txt")


@app.get("/")
def home():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "classes": class_names,
        "num_classes": len(class_names),
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "labels_loaded": len(class_names) > 0
    }


def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32")

    # IMPORTANT:
    # Use this only if your saved model does NOT contain Lambda(preprocess_input)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(class_names) == 0:
        raise HTTPException(status_code=503, detail="No labels are configured")

    image_bytes = await file.read()

    try:
        image_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )

    predictions = model.predict(image_array, verbose=0)[0]

    index = int(np.argmax(predictions))
    confidence = float(predictions[index])
    predicted_class = class_names[index]

    all_predictions = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "unknown",
            "raw_prediction": predicted_class,
            "confidence": confidence,
            "message": "Low confidence. Please retake image or ask expert.",
            "all_predictions": all_predictions
        }

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "message": "Prediction completed successfully.",
        "all_predictions": all_predictions
    }