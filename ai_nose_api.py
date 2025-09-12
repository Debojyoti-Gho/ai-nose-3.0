import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import joblib, pandas as pd, numpy as np
from datetime import datetime, timedelta
import shap, cv2, io, base64, requests
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# 1. Load model + scalers
# ======================================================
MODEL_PATH = "models/ai_nose_xgb_model.joblib"
SCALER_X_PATH = "models/scaler_X.pkl"

model = joblib.load(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
explainer = shap.Explainer(model)

# Use same feature order as training (X_cols from Colab)
FEATURES = [
    "pm10","co","no2","so2","o3","temp","rh","wind_speed","pressure",
    "NDVI","LST","NightLights","industrial_density_per_km2","highway_density_km_per_km2"
]


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
yolo = YOLO("yolov8n.pt")

def analyze_image(img_bytes):
    """
    Returns:
      {
        "vehicle_count": int,
        "greenery_ratio": float,
        "haze_score": float,
        "annotated_image_b64": str,
        "detections": [ { "class_id": int, "class_name": str, "xyxy": [x1,y1,x2,y2], "conf": float }, ... ]
      }
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    # YOLO predict (silent)
    results = yolo.predict(img, verbose=False)
    r = results[0]

    # extract boxes / classes / confidences
    boxes = []
    if hasattr(r, "boxes") and len(r.boxes):
        xyxy = r.boxes.xyxy.cpu().numpy()  # N x 4
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # N
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else [None]*len(cls_ids)
        names = yolo.model.names if hasattr(yolo.model, "names") else {i: str(i) for i in range(1000)}
        for i in range(len(cls_ids)):
            boxes.append({
                "class_id": int(cls_ids[i]),
                "class_name": names.get(int(cls_ids[i]), str(int(cls_ids[i]))),
                "xyxy": [float(v) for v in xyxy[i]],
                "conf": float(confs[i]) if confs is not None else None
            })

    # Count vehicles using common COCO class ids
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)
    vehicle_count = sum(1 for b in boxes if b["class_id"] in vehicle_classes)

    # Greenery ratio from HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85)
    greenery_ratio = float(green_mask.sum() / (img.shape[0]*img.shape[1]))

    # haze_score = simple metric: image gray stddev (lower std -> more haze)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haze_score = float(gray.std())

    # Draw annotated image (OpenCV)
    annotated = img.copy()
    h, w = annotated.shape[:2]

    # Draw boxes + labels
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["xyxy"])
        cls_name = b["class_name"]
        conf = b.get("conf", None)
        label = f"{cls_name}" + (f" {conf:.2f}" if conf is not None else "")
        # rectangle and filled label background
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0, 255, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0,255,255), -1)
        cv2.putText(annotated, label, (x1+3, max(4, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # Header with quick stats
    header = f"Vehicles: {vehicle_count}  Greenery: {greenery_ratio:.2f}  HazeScore: {haze_score:.1f}"
    cv2.rectangle(annotated, (0,0), (w, 28), (0,0,0), -1)
    cv2.putText(annotated, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # encode annotated image to base64
    _, buf = cv2.imencode('.png', annotated)
    annotated_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    return {
        "vehicle_count": int(vehicle_count),
        "greenery_ratio": round(float(greenery_ratio), 3),
        "haze_score": round(float(haze_score), 2),
        "annotated_image_b64": annotated_b64,
        "detections": boxes
    }

# --- Helper: create short AI-like summary from image features + detections ---
def image_summary_from_features(img_features, max_items=3):
    parts = []
    vc = img_features.get("vehicle_count", 0)
    gr = img_features.get("greenery_ratio", 0.0)
    hz = img_features.get("haze_score", 0.0)
    if vc > 0:
        parts.append(f"{vc} motor vehicles detected — likely local traffic emissions (NO₂, CO, PM).")
    if gr < 0.2:
        parts.append(f"Low visible greenery ({gr:.2f}) — reduced natural particulate removal.")
    else:
        parts.append(f"Moderate greenery visible ({gr:.2f}) — some natural mitigation.")
    # interpret haze score heuristically
    if hz < 30:
        parts.append("Low contrast/haze level — scene appears clear.")
    elif hz < 70:
        parts.append("Moderate haze — possible fine particles or humidity.")
    else:
        parts.append("High haze/low contrast — visual indicator of particulate pollution or fog.")
    # keep it short + actionable
    summary = " ".join(parts[:max_items])
    suggestion = "Recommendation: reduce local traffic, check industrial sources, and monitor meteorology."
    return f"{summary} {suggestion}"

# --- Helper: map image features to a small numeric adjustment (heuristic) ---
def image_to_adjustment(img_features):
    """
    Produce a small µg/m³ adjustment to add to the model prediction.
    This is a heuristic fallback so image features contribute now.
    Recommended: retrain model to include these features for principled results.
    """
    vc = img_features.get("vehicle_count", 0)
    gr = img_features.get("greenery_ratio", 0.0)
    hz = img_features.get("haze_score", 0.0)

    # Heuristic scales (tunable)
    # - vehicles increase PM2.5 roughly proportionally
    veh_score = vc * 0.9            # each vehicle adds ~0.9 units (tunable)
    # - low greenery increases PM by (0.4 - gr) * scale
    greenery_score = max(0.0, (0.4 - gr)) * 25.0   # baseline 0.4
    # - haze_score: lower std -> more haze, so invert
    haze_penalty = max(0.0, 50.0 - hz) * 0.2

    image_score = veh_score + greenery_score + haze_penalty

    # map to µg/m3; conservative multiplier
    adjustment = image_score * 0.35
    return float(round(adjustment, 2))

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
# 5. Forecast with Open-Meteo API
# ======================================================
def fetch_air_quality(lat, lon):
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&"
        f"hourly=pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,"
        f"ozone,temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl&forecast_days=1"
    )
    try:
        resp = requests.get(url, timeout=20).json()
        h = resp["hourly"]
        return {
            "pm10": h["pm10"][0],
            "co": h["carbon_monoxide"][0],
            "no2": h["nitrogen_dioxide"][0],
            "so2": h["sulphur_dioxide"][0],
            "o3": h.get("ozone",[np.nan])[0],
            "temp": h.get("temperature_2m",[np.nan])[0],
            "rh": h.get("relative_humidity_2m",[np.nan])[0],
            "wind_speed": h.get("wind_speed_10m",[np.nan])[0],
            "pressure": h.get("pressure_msl",[np.nan])[0]
        }
    except Exception:
        # fallback random if API fails
        return {
            "pm10": np.random.uniform(10,100),
            "co": np.random.uniform(0.1,1.0),
            "no2": np.random.uniform(5,50),
            "so2": np.random.uniform(2,20),
            "o3": np.random.uniform(5,50),
            "temp": np.random.uniform(10,35),
            "rh": np.random.uniform(20,90),
            "wind_speed": np.random.uniform(0,10),
            "pressure": np.random.uniform(950,1050)
        }


def forecast_future(lat, lon, horizon=3):
    now = datetime.now()
    forecasts = []

    for i in range(horizon):
        t = now + timedelta(hours=i)
        aq_data = fetch_air_quality(lat, lon)

        features = {
            **aq_data,
            # placeholders for satellite/OSM — replace later with real GEE/Overpass
            "NDVI": 0.4,
            "LST": 28.0,
            "NightLights": 15.0,
            "traffic_proxy": 1.0,
            "industrial_density_per_km2": 0.1,
            "highway_density_km_per_km2": 0.5,
            "src_lat": lat,
            "src_lon": lon,
            "hour": t.hour,
            "dow": t.weekday()
        }

        X = pd.DataFrame([features])[FEATURES]
        X_scaled = scaler_X.transform(X)
        pm25_pred = float(model.predict(X_scaled)[0])

        forecasts.append({
            "time": str(t),
            "predicted": pm25_pred,
            "narrative": f"Drivers: pm10={aq_data['pm10']}, co={aq_data['co']}, no2={aq_data['no2']}, so2={aq_data['so2']}"
        })

    return forecasts

# ======================================================
# 6. Graph + SHAP generation
# ======================================================
def generate_forecast_plot(forecasts):
    times = [f["time"] for f in forecasts]
    pm25 = [f["predicted"] for f in forecasts]

    plt.figure(figsize=(6,3))
    sns.lineplot(x=times, y=pm25, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("PM2.5 Forecast")
    plt.ylabel("µg/m³")
    plt.xlabel("Time")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_shap_plot(X):
    try:
        shap_vals = explainer(X)
        vals = shap_vals.values[0]
        feats = explainer.feature_names if explainer.feature_names else FEATURES

        idx = np.argsort(np.abs(vals))[-5:]
        top_feats = [feats[i] for i in idx]
        top_vals = [vals[i] for i in idx]

        plt.figure(figsize=(5,3))
        sns.barplot(x=top_vals, y=top_feats)
        plt.title("Top Feature Drivers")
        plt.xlabel("SHAP contribution")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None

def generate_shap_explanations(X):
    try:
        shap_vals = explainer(X)
        vals = shap_vals.values[0]
        feats = explainer.feature_names if explainer.feature_names else FEATURES

        idx = np.argsort(np.abs(vals))[-5:]
        explanations = []
        for i in idx[::-1]:
            impact = "increased" if vals[i] > 0 else "reduced"
            explanations.append(f"{feats[i]} {impact} PM2.5 by {abs(vals[i]):.2f}")
        return explanations
    except Exception:
        return ["Explanation unavailable"]

# ======================================================
# 7. FastAPI App
# ======================================================
app = FastAPI(title="AI Nose 3.0 API",
              description="Multimodal Air Pollution Intelligence API with Graphs + SHAP + Open-Meteo",
              version="1.5")

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

        # SHAP on last features
        aq_data = fetch_air_quality(query.lat, query.lon)
        last_features = pd.DataFrame([{
            **aq_data,
            "NDVI": 0.4,
            "LST": 28.0,
            "NightLights": 15.0,
            "industrial_density_per_km2": 0.1,
            "highway_density_km_per_km2": 0.5,
            "src_lat": query.lat,
            "src_lon": query.lon,
            "hour": datetime.now().hour,
            "dow": datetime.now().weekday()
        }])[FEATURES]

        X_scaled = scaler_X.transform(last_features)

        forecast_graph = generate_forecast_plot(forecasts)
        shap_graph = generate_shap_plot(X_scaled)
        shap_explanations = generate_shap_explanations(X_scaled)

        return {"status": "ok", "intent": intent, "results": forecasts,
                "forecast_graph": forecast_graph,
                "shap_graph": shap_graph,
                "shap_explanations": shap_explanations}
    except Exception as e:
        return {"status": "error", "message": str(e)}



@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/chatbot_with_image")
async def chatbot_with_image(
    lat: float = Form(...),
    lon: float = Form(...),
    horizon: int = Form(3),
    question: str = Form(""),
    file: UploadFile = File(...)
):
    try:
        img_bytes = await file.read()
        img_analysis = analyze_image(img_bytes)

        intent = detect_intent(question) if question else "general"
        forecasts = forecast_future(lat, lon, horizon=horizon)

        aq_data = fetch_air_quality(lat, lon)
        last_features = pd.DataFrame([{
            **aq_data,
            "NDVI": 0.4,
            "LST": 28.0,
            "NightLights": 15.0,
            "industrial_density_per_km2": 0.1,
            "highway_density_km_per_km2": 0.5,
            "src_lat": lat,
            "src_lon": lon,
            "hour": datetime.now().hour,
            "dow": datetime.now().weekday()
        }])[FEATURES]

        X_scaled = scaler_X.transform(last_features)
        base_pred = float(model.predict(X_scaled)[0])

        # Convert image -> adjustment and apply to the *latest* prediction
        image_adj = image_to_adjustment(img_analysis)
        adjusted_pred = float(round(base_pred + image_adj, 2))

        # Optionally adjust the first forecast entry (simple approach)
        if len(forecasts) > 0:
            forecasts[0]["predicted"] = float(round(forecasts[0]["predicted"] + image_adj, 2))
            forecasts[0]["note"] = f"Adjusted by image-derived +{image_adj} µg/m³"

        forecast_graph = generate_forecast_plot(forecasts)
        shap_graph = generate_shap_plot(X_scaled)
        shap_explanations = generate_shap_explanations(X_scaled)

        # AI-like summary from image
        image_summary = image_summary_from_features(img_analysis)

        return {
            "status": "ok",
            "intent": intent,
            "img_features": {
                "vehicle_count": img_analysis["vehicle_count"],
                "greenery_ratio": img_analysis["greenery_ratio"],
                "haze_score": img_analysis["haze_score"],
                "detections": img_analysis["detections"]
            },
            "annotated_image": img_analysis["annotated_image_b64"],  # PNG base64 - prefix with data:image/png;base64, to display
            "image_summary": image_summary,
            "base_prediction": base_pred,
            "image_adjustment": image_adj,
            "adjusted_prediction": adjusted_pred,
            "results": forecasts,
            "forecast_graph": forecast_graph,
            "shap_graph": shap_graph,
            "shap_explanations": shap_explanations
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ======================================================
# 8. Run server
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # use Render's $PORT if available
    uvicorn.run("ai_nose_api:app", host="0.0.0.0", port=port)
