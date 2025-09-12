# ai_nose_api.py
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
import warnings

# NEW imports for improved NLP
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SENTE = True
except Exception:
    SENTE = False

try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False

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
# 2. NLP intent detection (upgraded)
# ======================================================
INTENTS = {
    "housing": ["Should I build a house here?", "Is this location good for living?", "residential zone safe?", "Can I live here?"],
    "asthma": ["Is this safe for asthma patients?", "Can asthma patients live here?", "health risk for sensitive groups"],
    "factory": ["Would a factory be good here?", "Can I set up industry?", "suitable for industry?", "factory location safe?"],
    "outdoor": ["Can kids play outside tomorrow?", "Is it safe to exercise outdoors?", "Can I jog outside?", "Is outdoor activity safe today?"],
    "general": ["How is the air quality today?", "Tell me about pollution now", "What’s the condition here?", "give me forecast"]
}

# Build flattened lists of examples + labels (same as before)
intent_texts, intent_labels = [], []
for label, examples in INTENTS.items():
    for ex in examples:
        intent_texts.append(ex)
        intent_labels.append(label)

# Keep original TF-IDF vectorizer as fallback and for explainability
vectorizer = TfidfVectorizer().fit(intent_texts)
intent_vectors_tfidf = vectorizer.transform(intent_texts)

# If sentence-transformers available, prepare embedding-based centroids + example embeddings
if SENTE:
    try:
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        # compute embeddings for each example text
        intent_example_embeddings = st_model.encode(intent_texts, convert_to_tensor=True, show_progress_bar=False)
        # compute centroid embedding for each intent
        intent_centroids = {}
        for label in INTENTS:
            idxs = [i for i, lab in enumerate(intent_labels) if lab == label]
            if idxs:
                emb = intent_example_embeddings[idxs]
                centroid = st_util.pytorch_cos_sim(emb.mean(dim=0), emb.mean(dim=0))  # dummy to ensure shape
                intent_centroids[label] = emb.mean(dim=0)
        SENTE_OK = True
    except Exception as e:
        warnings.warn(f"SentenceTransformer init failed: {e}. Falling back to TF-IDF.")
        SENTE_OK = False
else:
    SENTE_OK = False

def detect_intent(question: str, top_k: int = 3):
    """
    Improved intent detection:
      - Prefer semantic similarity using sentence-transformers (if available)
      - Fallback to TF-IDF + cosine similarity if not
    Returns:
      (label, confidence (0..1), matches_list)
      matches_list: list of (example_text, example_label, similarity_score)
    """
    question = (question or "").strip()
    if question == "":
        return "general", 1.0, []

    # Use sentence-transformers semantic similarity if available
    if SENTE_OK:
        try:
            q_emb = st_model.encode(question, convert_to_tensor=True, show_progress_bar=False)
            # similarity to intent centroids
            cent_sims = {}
            for lab, cent in intent_centroids.items():
                sim = float(st_util.pytorch_cos_sim(q_emb, cent).item())
                cent_sims[lab] = sim
            # best label from centroids
            best_label = max(cent_sims, key=cent_sims.get)
            best_conf = cent_sims[best_label]
            # also compute top-k example-level matches for transparency
            example_sims = st_util.pytorch_cos_sim(q_emb, intent_example_embeddings)[0].cpu().numpy()
            top_idx = list(np.argsort(example_sims)[::-1][:top_k])
            matches = []
            for i in top_idx:
                matches.append((intent_texts[i], intent_labels[i], float(example_sims[i])))
            # If confidence low, fallback to general
            if best_conf < 0.45:
                # low confidence -> mark as general but still provide matches
                return "general", float(best_conf), matches
            else:
                return best_label, float(best_conf), matches
        except Exception as e:
            warnings.warn(f"Sentence-transformer detect failed: {e} - falling back to TF-IDF.")
            # fall through to tfidf fallback

    # TF-IDF fallback (original approach)
    try:
        q_vec = vectorizer.transform([question])
        sims = cosine_similarity(q_vec, intent_vectors_tfidf)[0]
        best_idx = int(sims.argmax())
        best_label = intent_labels[best_idx]
        best_conf = float(sims[best_idx])  # similarity between 0..1 (depending on text)
        # prepare matches
        top_idx = list(np.argsort(sims)[::-1][:top_k])
        matches = []
        for i in top_idx:
            matches.append((intent_texts[i], intent_labels[i], float(sims[i])))
        # if low TF-IDF similarity, mark as general
        if best_conf < 0.18:
            return "general", best_conf, matches
        return best_label, best_conf, matches
    except Exception as e:
        # should not happen, but safe fallback
        warnings.warn(f"Intent detection fallback failed: {e}")
        return "general", 0.0, []

# ======================================================
# 3. Image analysis (YOLO + OpenCV) - expanded
# (UNCHANGED from user code — kept intact)
# ======================================================
yolo = YOLO("yolov8n.pt")

# Heuristic helpers for fire/smoke/bright flashes/dust
def detect_fire_color_regions(img, min_area=150, saturation_thr=120, value_thr=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, saturation_thr, value_thr])
    upper1 = np.array([30, 255, 255])
    lower2 = np.array([160, saturation_thr, value_thr])
    upper2 = np.array([179, 255, 255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in contours if cv2.contourArea(c) >= min_area]
    mask_frac = mask.sum() / (img.shape[0] * img.shape[1] * 255.0)
    return len(blobs), float(mask_frac), mask

def detect_smoke_regions(img, min_area=400):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    _, s, v = cv2.split(hsv)
    sat_mask = (s < 50)
    v_norm = (v / 255.0)
    bright_mask = (v_norm > 0.12)
    candidate = (sat_mask & bright_mask).astype('uint8') * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in contours if cv2.contourArea(c) >= min_area]
    mask_frac = candidate.sum() / (img.shape[0] * img.shape[1] * 255.0)
    return len(blobs), float(mask_frac), candidate

def detect_bright_flashes(img, small_area=(8, 300), intensity_thresh=245):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, intensity_thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smalls = [c for c in contours if small_area[0] <= cv2.contourArea(c) <= small_area[1]]
    return len(smalls), th

def detect_dust_regions(img, min_area=500):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    candidate = ((b > 150) & (l > 80)).astype('uint8') * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in contours if cv2.contourArea(c) >= min_area]
    mask_frac = candidate.sum() / (img.shape[0] * img.shape[1] * 255.0)
    return len(blobs), float(mask_frac), candidate

def analyze_image(img_bytes):
    """
    Returns a comprehensive analysis dictionary including:
      - detections (YOLO boxes)
      - detections_by_label (counts)
      - per-class counts for key classes (vehicle/truck/bus/motorbike)
      - heuristic masks: smoke_frac, fire_frac, dust_frac, bright_flashes_count
      - cooking indicators, brick_kiln, smokestack counts (if detected by YOLO)
      - annotated_image_b64
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    # YOLO predict
    results = yolo.predict(img, verbose=False)
    r = results[0]

    # extract boxes / classes / confidences
    boxes = []
    names = {i: str(i) for i in range(1000)}
    try:
        if hasattr(yolo, "model") and hasattr(yolo.model, "names"):
            names = yolo.model.names
    except Exception:
        pass

    if hasattr(r, "boxes") and len(r.boxes):
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = None
        if hasattr(r.boxes, "conf"):
            try:
                confs = r.boxes.conf.cpu().numpy()
            except Exception:
                confs = None
        for i in range(len(cls_ids)):
            boxes.append({
                "class_id": int(cls_ids[i]),
                "class_name": names.get(int(cls_ids[i]), str(int(cls_ids[i]))),
                "xyxy": [float(v) for v in xyxy[i]],
                "conf": float(confs[i]) if confs is not None else None
            })

    # build detections_by_label
    detections_by_label = {}
    for b in boxes:
        nm = b["class_name"]
        detections_by_label[nm] = detections_by_label.get(nm, 0) + 1

    # per-class counts (legacy + extended)
    vehicle_class_ids = [2,3,5,7]  # car, motorbike, bus, truck (COCO)
    vehicle_count = sum(1 for b in boxes if b["class_id"] in vehicle_class_ids)
    car_count = sum(1 for b in boxes if b["class_id"] == 2)
    motorbike_count = sum(1 for b in boxes if b["class_id"] == 3)
    bus_count = sum(1 for b in boxes if b["class_id"] == 5)
    truck_count = sum(1 for b in boxes if b["class_id"] == 7)

    # detect presence of domain-specific labels by name substrings (for custom YOLOs)
    def count_name_contains(subs):
        c = 0
        for nm in detections_by_label:
            nml = nm.lower()
            for s in subs:
                if s in nml:
                    c += detections_by_label[nm]
                    break
        return c

    brick_kiln_count = count_name_contains(["kiln", "brick"])
    smokestack_count = count_name_contains(["smokestack", "smoke", "stack", "chimney", "flare"])
    generator_count = count_name_contains(["generator", "genset", "genny"])
    cooking_count = count_name_contains(["stove", "pan", "pot", "tandoor", "grill", "barbecue", "kitchen", "cooking"])
    construction_machinery_count = count_name_contains(["excavator", "bulldozer", "loader", "backhoe", "dump", "crane"])

    # greenery ratio
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85)
    greenery_ratio = float(green_mask.sum() / (img.shape[0]*img.shape[1]))

    # haze_score (gray std)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haze_score = float(gray.std())

    # heuristic masks
    fire_blobs, fire_mask_frac, fire_mask = detect_fire_color_regions(img)
    smoke_blobs, smoke_mask_frac, smoke_mask = detect_smoke_regions(img)
    bright_count, bright_mask = detect_bright_flashes(img)
    dust_blobs, dust_mask_frac, dust_mask = detect_dust_regions(img)

    fire_detected = (fire_blobs > 0) or (fire_mask_frac > 0.003)
    smoke_detected = (smoke_blobs > 0) or (smoke_mask_frac > 0.006)
    dust_detected = (dust_blobs > 0) or (dust_mask_frac > 0.004)
    firecracker_like = (bright_count >= 1) and smoke_detected

    # open burn heuristic: fire + smoke + not inside cooking
    open_burn_detected = bool(fire_detected and smoke_detected and (cooking_count == 0))

    # Draw annotated image
    annotated = img.copy()
    h, w = annotated.shape[:2]

    # overlay masks lightly (optional): smoke in gray, fire in orange, dust in tan
    try:
        # convert single channel masks to 3-channel for overlay
        if smoke_mask is not None:
            sm_rgb = cv2.cvtColor(smoke_mask, cv2.COLOR_GRAY2BGR)
            annotated = cv2.addWeighted(annotated, 1.0, (sm_rgb // 2), 0.25, 0)
        if fire_mask is not None:
            fm = (fire_mask > 0).astype('uint8') * 255
            fm_rgb = cv2.cvtColor(fm, cv2.COLOR_GRAY2BGR)
            orange = np.zeros_like(annotated)
            orange[:,:,2] = fm  # red channel
            orange[:,:,1] = (fm * 0.6).astype('uint8')  # green
            annotated = cv2.addWeighted(annotated, 1.0, orange, 0.25, 0)
        if dust_mask is not None:
            dm = (dust_mask > 0).astype('uint8') * 255
            brown = np.zeros_like(annotated)
            brown[:,:,2] = (dm * 0.9).astype('uint8')  # hint of red/yellow
            brown[:,:,1] = (dm * 0.7).astype('uint8')
            annotated = cv2.addWeighted(annotated, 1.0, brown, 0.12, 0)
    except Exception:
        # overlay is optional; proceed if fails
        pass

    # Draw boxes + labels
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["xyxy"])
        cls_name = b["class_name"]
        conf = b.get("conf", None)
        label = f"{cls_name}" + (f" {conf:.2f}" if conf is not None else "")
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0, 255, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0,255,255), -1)
        cv2.putText(annotated, label, (x1+3, max(4, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

    # Header with many quick stats
    header = (f"V:{vehicle_count} C:{car_count} T:{truck_count} M:{motorbike_count} "
              f"Smoke:{smoke_mask_frac:.3f} Fire:{fire_mask_frac:.3f} Dust:{dust_mask_frac:.3f}")
    cv2.rectangle(annotated, (0,0), (w, 36), (0,0,0), -1)
    cv2.putText(annotated, header, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    # encode annotated image to base64
    _, buf = cv2.imencode('.png', annotated)
    annotated_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    return {
        "detections": boxes,
        "detections_by_label": detections_by_label,
        "vehicle_count": int(vehicle_count),
        "car_count": int(car_count),
        "truck_count": int(truck_count),
        "bus_count": int(bus_count),
        "motorbike_count": int(motorbike_count),
        "brick_kiln_count": int(brick_kiln_count),
        "smokestack_count": int(smokestack_count),
        "generator_count": int(generator_count),
        "cooking_count": int(cooking_count),
        "construction_machinery_count": int(construction_machinery_count),
        "greenery_ratio": round(float(greenery_ratio), 3),
        "haze_score": round(float(haze_score), 2),
        "fire_blobs": int(fire_blobs),
        "fire_mask_frac": round(float(fire_mask_frac), 6),
        "smoke_blobs": int(smoke_blobs),
        "smoke_mask_frac": round(float(smoke_mask_frac), 6),
        "dust_blobs": int(dust_blobs),
        "dust_mask_frac": round(float(dust_mask_frac), 6),
        "bright_flash_count": int(bright_count),
        "fire_detected": bool(fire_detected),
        "smoke_detected": bool(smoke_detected),
        "dust_detected": bool(dust_detected),
        "firecracker_like": bool(firecracker_like),
        "open_burn_detected": bool(open_burn_detected),
        "annotated_image_b64": annotated_b64
    }

# --- Helper: image summary considering many detections ---
def image_summary_from_features(img_features, max_items=4):
    parts = []
    # strong signals first
    if img_features.get("open_burn_detected"):
        parts.append("Open burning detected — strong local source of particulate emissions.")
    if img_features.get("smoke_detected"):
        parts.append(f"Smoke plume visible (frac={img_features.get('smoke_mask_frac'):.4f}).")
    if img_features.get("fire_detected"):
        parts.append("Flame-like regions detected — active combustion present.")
    if img_features.get("firecracker_like"):
        parts.append("Bright small flashes + smoke — fireworks or firecrackers likely.")
    # vehicles & industrial
    vc = img_features.get("vehicle_count", 0)
    if vc > 0:
        parts.append(f"{vc} vehicle(s) visible — traffic emissions likely.")
    if img_features.get("truck_count", 0) > 0:
        parts.append(f"{img_features.get('truck_count')} heavy vehicle(s) (trucks) — diesel source.")
    if img_features.get("brick_kiln_count", 0) > 0 or img_features.get("smokestack_count", 0) > 0:
        parts.append("Industrial chimney/brick-kiln features detected — point-source emissions probable.")
    if img_features.get("construction_machinery_count", 0) > 0 or img_features.get("dust_detected"):
        parts.append("Construction/dust activity detected — coarse particulate contribution.")
    if img_features.get("cooking_count", 0) > 0:
        parts.append("Cooking/food stalls visible — localized PM and VOCs possible.")
    # greenery
    gr = img_features.get("greenery_ratio", 0.0)
    if gr < 0.15:
        parts.append(f"Low greenery ({gr:.2f}) — limited natural mitigation.")
    else:
        parts.append(f"Greenery present ({gr:.2f}) — some mitigation.")
    suggestion = "Recommendation: verify with sensor readings; reduce open burning and heavy traffic if possible."
    summary = " ".join(parts[:max_items])
    return f"{summary} {suggestion}"

# --- Helper: map a rich set of image features to a numeric µg/m³ adjustment ---
def image_to_adjustment(img_features):
    """
    Conservative heuristic mapping many visual signals -> µg/m³ adjustment.
    Tunable; recommended retraining with these features for principled integration.
    """
    # Extract features with sensible defaults
    vc = float(img_features.get("vehicle_count", 0))
    truck = float(img_features.get("truck_count", 0))
    bus = float(img_features.get("bus_count", 0))
    motor = float(img_features.get("motorbike_count", 0))
    smoke_frac = float(img_features.get("smoke_mask_frac", 0.0))
    fire_frac = float(img_features.get("fire_mask_frac", 0.0))
    dust_frac = float(img_features.get("dust_mask_frac", 0.0))
    open_burn = 1.0 if img_features.get("open_burn_detected", False) else 0.0
    brick = float(img_features.get("brick_kiln_count", 0))
    chimney = float(img_features.get("smokestack_count", 0))
    gen = float(img_features.get("generator_count", 0))
    cooking = float(img_features.get("cooking_count", 0))
    bright = float(img_features.get("bright_flash_count", 0))
    greenery = float(img_features.get("greenery_ratio", 0.0))

    # Normalize counts to avoid large raw values
    vc_norm = vc / 10.0           # per 10 vehicles
    truck_norm = truck / 2.0      # each 2 trucks ~ 1 unit
    bus_norm = bus / 2.0
    motor_norm = motor / 10.0
    brick_norm = brick / 1.0
    chimney_norm = chimney / 1.0
    gen_norm = gen / 1.0
    cooking_norm = cooking / 2.0
    bright_norm = bright / 1.0

    # Weights (conservative). These can be tuned with validation data.
    W = {
        "vc": 0.6,
        "truck": 1.2,
        "bus": 1.0,
        "motor": 0.35,
        "smoke_frac": 55.0,
        "fire_frac": 65.0,
        "dust_frac": 25.0,
        "open_burn": 30.0,
        "brick": 40.0,
        "chimney": 25.0,
        "gen": 8.0,
        "cooking": 6.0,
        "bright": 10.0,
        "greenery": -18.0
    }

    score = 0.0
    score += W["vc"] * vc_norm
    score += W["truck"] * truck_norm
    score += W["bus"] * bus_norm
    score += W["motor"] * motor_norm
    score += W["smoke_frac"] * smoke_frac
    score += W["fire_frac"] * fire_frac
    score += W["dust_frac"] * dust_frac
    score += W["open_burn"] * open_burn
    score += W["brick"] * brick_norm
    score += W["chimney"] * chimney_norm
    score += W["gen"] * gen_norm
    score += W["cooking"] * cooking_norm
    score += W["bright"] * bright_norm
    score += W["greenery"] * greenery

    # Convert score -> µg/m3 adjustment using conservative multiplier
    adjustment = score * 0.28

    # Clip to safe bounds to avoid huge swings
    adjustment = float(np.clip(adjustment, -25.0, 120.0))
    return round(adjustment, 2)

# ======================================================
# 4. Advice logic (UNCHANGED)
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
# 5. Forecast with Open-Meteo API (UNCHANGED)
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
# 6. Graph + SHAP generation (UNCHANGED)
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
# 7. FastAPI App (unchanged endpoints but include intent metadata)
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
        intent_label, intent_conf, intent_matches = detect_intent(query.question) if query.question else ("general", 1.0, [])
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

        # add a best-effort natural language reply combining intent and forecast
        pm25_latest = float(model.predict(X_scaled)[0])
        advice = give_advice(pm25_latest, intent_label)

        # optional: extract GPE/location entities for transparency (spaCy if available)
        entities = []
        if SPACY_OK and query.question:
            doc = nlp_spacy(query.question)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

        return {
            "status": "ok",
            "intent": intent_label,
            "intent_confidence": intent_conf,
            "intent_matches": [
                {"example": ex, "label": lab, "sim": sim} for (ex, lab, sim) in intent_matches
            ],
            "entities": entities,
            "advice": advice,
            "results": forecasts,
            "forecast_graph": forecast_graph,
            "shap_graph": shap_graph,
            "shap_explanations": shap_explanations,
            "pm25_estimate": pm25_latest
        }
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

        intent_label, intent_conf, intent_matches = detect_intent(question) if question else ("general", 1.0, [])

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

        # optional: spaCy entities
        entities = []
        if SPACY_OK and question:
            doc = nlp_spacy(question)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

        return {
            "status": "ok",
            "intent": intent_label,
            "intent_confidence": intent_conf,
            "intent_matches": [
                {"example": ex, "label": lab, "sim": sim} for (ex, lab, sim) in intent_matches
            ],
            "entities": entities,
            "img_features": {
                "detections_by_label": img_analysis.get("detections_by_label"),
                "vehicle_count": img_analysis.get("vehicle_count"),
                "car_count": img_analysis.get("car_count"),
                "truck_count": img_analysis.get("truck_count"),
                "bus_count": img_analysis.get("bus_count"),
                "motorbike_count": img_analysis.get("motorbike_count"),
                "brick_kiln_count": img_analysis.get("brick_kiln_count"),
                "smokestack_count": img_analysis.get("smokestack_count"),
                "generator_count": img_analysis.get("generator_count"),
                "cooking_count": img_analysis.get("cooking_count"),
                "construction_machinery_count": img_analysis.get("construction_machinery_count"),
                "greenery_ratio": img_analysis.get("greenery_ratio"),
                "haze_score": img_analysis.get("haze_score"),
                "smoke_mask_frac": img_analysis.get("smoke_mask_frac"),
                "fire_mask_frac": img_analysis.get("fire_mask_frac"),
                "dust_mask_frac": img_analysis.get("dust_mask_frac"),
                "bright_flash_count": img_analysis.get("bright_flash_count"),
                "fire_detected": img_analysis.get("fire_detected"),
                "smoke_detected": img_analysis.get("smoke_detected"),
                "dust_detected": img_analysis.get("dust_detected"),
                "open_burn_detected": img_analysis.get("open_burn_detected"),
                "firecracker_like": img_analysis.get("firecracker_like"),
                "detections": img_analysis.get("detections")
            },
            "annotated_image": img_analysis.get("annotated_image_b64"),  # PNG base64 - prefix with data:image/png;base64, to display
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
