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
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = yolo.predict(img, verbose=False)
    objects = results[0].boxes.cls.cpu().numpy()

    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    vehicle_count = sum([obj in vehicle_classes for obj in objects])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85)
    greenery_ratio = green_mask.sum() / (img.shape[0]*img.shape[1])

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
        sns.barplot(x=top_vals, y=top_feats, palette="coolwarm")
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
        img_features = analyze_image(img_bytes)

        intent = detect_intent(question) if question else "general"
        forecasts = forecast_future(lat, lon, horizon=horizon)

        aq_data = fetch_air_quality(lat, lon)
        last_features = pd.DataFrame([{
            **aq_data,
            "src_lat": lat,
            "src_lon": lon,
            "hour": datetime.now().hour,
            "dow": datetime.now().weekday()
        }])[FEATURES]
        X_scaled = scaler_X.transform(last_features)

        forecast_graph = generate_forecast_plot(forecasts)
        shap_graph = generate_shap_plot(X_scaled)
        shap_explanations = generate_shap_explanations(X_scaled)

        return {"status": "ok", "intent": intent, "img_features": img_features,
                "results": forecasts,
                "forecast_graph": forecast_graph,
                "shap_graph": shap_graph,
                "shap_explanations": shap_explanations}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ======================================================
# 8. Run server
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # use Render's $PORT if available
    uvicorn.run("ai_nose_api:app", host="0.0.0.0", port=port)
