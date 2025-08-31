import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib, pandas as pd, numpy as np
from datetime import datetime
import shap, cv2
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# 1. Load model + scalers
# ======================================================
MODEL_PATH = "ai_nose_xgb_model.joblib"
SCALER_X_PATH = "scaler_X.pkl"

model = joblib.load(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
explainer = shap.Explainer(model)

# ======================================================
# 2. NLP intent detection
# ======================================================
INTENTS = {
    "housing": ["Should I build a house here?", "Is this location good for living?", "residential zone safe?", "Can I live here?"],
    "asthma": ["Is this safe for asthma patients?", "Can asthma patients live here?", "health risk for sensitive groups"],
    "factory": ["Would a factory be good here?", "Can I set up industry?", "suitable for industry?", "factory location safe?"],
    "outdoor": ["Can kids play outside tomorrow?", "Is it safe to exercise outdoors?", "Can I jog outside?", "Is outdoor activity safe today?"],
    "general": ["How is the air quality today?", "Tell me about pollution now", "What’s the condition here?", "give me forecast"]
}

intent_texts, intent_labels = [], []
for label, examples in INTENTS.items():
    for ex in examples:
        intent_texts.append(ex)
        intent_labels.append(label)

vectorizer = TfidfVectorizer().fit(intent_texts)
intent_vectors = vectorizer.transform(intent_texts)

def detect_intent(question: str):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, intent_vectors)[0]
    best_idx = sims.argmax()
    return intent_labels[best_idx]

# ======================================================
# 3. Image analysis (YOLO + OpenCV)
# ======================================================
yolo = YOLO("yolov8n.pt")  # lightweight pretrained YOLOv8

def analyze_image(img_bytes):
    # Save temporary file
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = yolo.predict(img, verbose=False)
    objects = results[0].boxes.cls.cpu().numpy()

    # Vehicle classes in COCO dataset
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    vehicle_count = sum([obj in vehicle_classes for obj in objects])

    # Greenery ratio
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85)
    greenery_ratio = green_mask.sum() / (img.shape[0]*img.shape[1])

    # Haze score (contrast proxy)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haze_score = gray.std()

    return {
        "vehicle_count": int(vehicle_count),
        "greenery_ratio": round(float(greenery_ratio), 3),
        "haze_score": round(float(haze_score), 2)
    }

# ======================================================
# 4. Advice logic
# ======================================================
def give_advice(pm25, intent, img_features=None):
    if intent == "housing":
        return "Yes 🏠 — Suitable for housing." if pm25 <= 100 else "No ❌ — Not suitable for housing."
    elif intent == "asthma":
        if pm25 <= 50: return "Safe ✅ — Asthma patients can live here."
        elif pm25 <= 100: return "Caution ⚠️ — Possible flare-ups."
        else: return "Unsafe ❌ — Relocation advised."
    elif intent == "factory":
        return "Possible 🏭 — Allowed with compliance." if pm25 <= 100 else "Yes but ⚠️ — Pollution risk, need controls."
    elif intent == "outdoor":
        if pm25 <= 50: return "✅ Safe for outdoor activities."
        elif pm25 <= 100: return "⚠️ Limited outdoor activity recommended."
        else: return "❌ Not safe for outdoor activity."
    else:
        if pm25 <= 50: return "✅ Good air quality. Safe for everyone."
        elif pm25 <= 100: return "⚠️ Moderate. Sensitive groups should limit exposure."
        elif pm25 <= 150: return "❌ Unhealthy for sensitive groups."
        elif pm25 <= 200: return "❌ Unhealthy for all."
        else: return "🚨 Very hazardous."

# ======================================================
# 5. Forecast function (mock for demo)
# ======================================================
def forecast_future(lat, lon, horizon=3):
    # In real system, fetch from your data pipeline
    # Here we mock predictions
    times = pd.date_range(datetime.now(), periods=horizon, freq="H")
    forecasts = []
    for t in times:
        pm25 = float(np.random.uniform(20,120))  # mock PM2.5
        forecasts.append({"time": str(t), "predicted": pm25, "narrative": "Drivers: PM10, CO, SO₂"})
    return forecasts

# ======================================================
# 6. FastAPI App
# ======================================================
app = FastAPI(title="AI Nose 3.0 API", description="Multimodal Air Pollution Intelligence API", version="1.0")

class Query(BaseModel):
    lat: float
    lon: float
    horizon: int = 3
    question: str = None

@app.post("/chatbot")
def chatbot_endpoint(query: Query):
    try:
        intent = detect_intent(query.question) if query.question else "general"
        forecasts = forecast_future(query.lat, query.lon, horizon=query.horizon)

        results = []
        for f in forecasts:
            pm25 = f["predicted"]
            advice = give_advice(pm25, intent)
            results.append({
                "time": f["time"],
                "pm25": pm25,
                "narrative": f["narrative"],
                "advice": advice
            })

        return {"status": "ok", "intent": intent, "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chatbot_with_image")
async def chatbot_with_image(
    lat: float = Form(...),
    lon: float = Form(...),
    horizon: int = Form(3),
    question: str = Form(""),
    file: UploadFile = File(...)
):
    try:
        # Extract features from image
        img_bytes = await file.read()
        img_features = analyze_image(img_bytes)

        # Run normal forecast
        intent = detect_intent(question) if question else "general"
        forecasts = forecast_future(lat, lon, horizon=horizon)

        results = []
        for f in forecasts:
            pm25 = f["predicted"]
            advice = give_advice(pm25, intent, img_features)

            narrative = f["narrative"] + f" | 📷 Image analysis: {img_features}"
            results.append({
                "time": f["time"],
                "pm25": pm25,
                "narrative": narrative,
                "advice": advice
            })

        return {"status": "ok", "intent": intent, "img_features": img_features, "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ======================================================
# 7. Run server
# ======================================================
if __name__ == "__main__":
    uvicorn.run("ai_nose_api:app", host="0.0.0.0", port=8000, reload=True)
