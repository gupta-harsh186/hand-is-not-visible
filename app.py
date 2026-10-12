from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
import os
import json
import sqlite3
import time
import re
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from datetime import datetime

# Optional TensorFlow import for Hybrid Architecture
tf = None
try:
    import tensorflow as tf
except Exception as e:
    print("[INFO] TensorFlow not found or could not be loaded:", e)

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ── GEMINI AI SETUP (HYBRID ADVISORY) ───
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
gemini_model = None
gemini_available = False

def init_gemini(api_key=None):
    global GEMINI_API_KEY, gemini_model, gemini_available
    if api_key is not None:
        GEMINI_API_KEY = api_key.strip()
    else:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Prioritize ultra-fast lightweight models for sub-second responses
            candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest"]
            active_model = None
            last_err = None
            
            for m_name in candidate_models:
                try:
                    test_m = genai.GenerativeModel(m_name)
                    test_m.generate_content("ok", generation_config={"max_output_tokens": 5, "temperature": 0.1})
                    active_model = test_m
                    print(f"[OK] Google Gemini AI connected with ultra-fast model: {m_name}")
                    break
                except Exception as ex:
                    last_err = ex
                    continue

            if active_model is not None:
                gemini_model = active_model
                gemini_available = True
                return True, "Connected to Gemini AI successfully!"
            else:
                gemini_available = False
                gemini_model = None
                print("[WARN] Could not find an active Gemini model:", last_err)
                return False, str(last_err)
        except Exception as e:
            gemini_available = False
            gemini_model = None
            print("[WARN] Gemini AI initialization error:", e)
            return False, str(e)
    else:
        gemini_available = False
        gemini_model = None
        print("[INFO] GEMINI_API_KEY not configured. Running in local TensorFlow mode.")
        return False, "GEMINI_API_KEY is not set or placeholder."

init_gemini()

# ── SQLITE DATABASE SETUP ──────────
DB_PATH = "plant_disease.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                plant TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                treatment TEXT,
                image_url TEXT
            )
        """)
        conn.commit()
        conn.close()
        print("[OK] SQLite database initialized successfully: " + DB_PATH)
    except Exception as e:
        print("[ERROR] Database initialization failed:", e)

init_db()

def save_scan_record(plant, disease, confidence, severity, treatment, image_bytes):
    try:
        image_url = None
        if image_bytes:
            filename = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000}.jpg"
            upload_dir = os.path.join("static", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            filepath = os.path.join(upload_dir, filename)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            image_url = f"/static/uploads/{filename}"

        formatted_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scan_history (timestamp, plant, disease, confidence, severity, treatment, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (formatted_time, plant, disease, float(confidence), severity, treatment, image_url))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        return record_id, image_url, formatted_time
    except Exception as e:
        print("[ERROR] Failed to save scan to database:", e)
        return None, None, None

# ── LOAD MODEL (WITH TENSORFLOW OR GEMINI FALLBACK) ───
model = None
if tf is not None:
    try:
        if os.path.exists("model/plant_model.keras"):
            model = tf.keras.models.load_model("model/plant_model.keras")
            print("[OK] Loaded model/plant_model.keras")
        elif os.path.exists("model/plant_model.h5"):
            model = tf.keras.models.load_model("model/plant_model.h5")
            print("[OK] Loaded model/plant_model.h5")
        else:
            print("[WARN] Neither plant_model.keras nor plant_model.h5 found in model/ directory.")
    except Exception as e:
        print("[WARN] Failed to load local TensorFlow model:", e)
        model = None
else:
    print("[INFO] TensorFlow not present. PlantGuard AI will use Gemini Multimodal Vision.")

# ── LOAD CLASSES (from training-time saved JSON) ───────────────────
CLASS_NAMES_PATH = "model/class_names.json"
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        class_map = json.load(f)
    # class_map is {"0": "Apple___Apple_scab", "1": ...}
    class_names = [class_map[str(i)] for i in range(len(class_map))]
    print("[OK] Loaded class names from class_names.json")
else:
    # Fallback
    class_names = sorted(os.listdir("dataset")) if os.path.exists("dataset") else []
    print("[WARN] class_names.json not found, using fallback")

print("Total classes loaded:", len(class_names))
if model is not None:
    print("Model output shape:", model.output_shape)

# ── TREATMENT DATABASE ─────────────
treatments = {
    "Apple___Apple_scab": {
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis, producing olive-green to black lesions on leaves and fruit.",
        "treatment": "Apply fungicides (captan, myclobutanil) during spring. Remove fallen leaves.",
        "severity": "Moderate",
        "prevention": "Plant resistant varieties, ensure good air circulation, rake fallen leaves."
    },
    "Apple___Black_rot": {
        "description": "Black rot is caused by the fungus Botryosphaeria obtusa, leading to leaf spots, fruit rot, and cankers.",
        "treatment": "Prune infected branches, apply captan or thiophanate-methyl fungicides.",
        "severity": "High",
        "prevention": "Remove mummified fruits, prune dead wood, maintain tree vigor."
    },
    "Apple___Cedar_apple_rust": {
        "description": "Cedar apple rust is caused by Gymnosporangium juniperi-virginianae. Orange spots appear on leaves.",
        "treatment": "Apply fungicides (myclobutanil) in spring. Remove nearby juniper hosts.",
        "severity": "Moderate",
        "prevention": "Remove cedar/juniper trees within 2 miles, plant resistant apple varieties."
    },
    "Apple___healthy": {
        "description": "The apple plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required. Maintain current care.",
        "severity": "None",
        "prevention": "Continue regular pruning, watering, and monitoring."
    },
    "Blueberry___healthy": {
        "description": "The blueberry plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required.",
        "severity": "None",
        "prevention": "Maintain acidic soil pH, proper watering, and mulching."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "Powdery mildew is a fungal disease causing white powdery coating on leaves and shoots.",
        "treatment": "Apply sulfur-based or potassium bicarbonate fungicides.",
        "severity": "Moderate",
        "prevention": "Ensure good air circulation, avoid overhead watering, prune dense canopy."
    },
    "Cherry_(including_sour)___healthy": {
        "description": "The cherry plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required.",
        "severity": "None",
        "prevention": "Regular pruning and proper watering schedule."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Gray leaf spot is caused by Cercospora zeae-maydis, creating rectangular gray lesions on leaves.",
        "treatment": "Apply foliar fungicides (strobilurins or triazoles). Use resistant hybrids.",
        "severity": "High",
        "prevention": "Crop rotation, tillage to reduce residue, plant resistant varieties."
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Common rust is caused by Puccinia sorghi, producing small reddish-brown pustules on leaves.",
        "treatment": "Apply foliar fungicides if severe. Most hybrids have partial resistance.",
        "severity": "Moderate",
        "prevention": "Plant resistant hybrids, monitor fields regularly."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Northern leaf blight is caused by Exserohilum turcicum, creating long cigar-shaped gray-green lesions.",
        "treatment": "Apply fungicides at early tassel stage. Use resistant hybrids.",
        "severity": "High",
        "prevention": "Crop rotation, residue management, resistant varieties."
    },
    "Corn_(maize)___healthy": {
        "description": "The corn plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required.",
        "severity": "None",
        "prevention": "Maintain proper fertilization and irrigation."
    },
    "Grape___Black_rot": {
        "description": "Black rot is caused by Guignardia bidwellii, creating brown circular lesions on leaves and shriveled black fruit.",
        "treatment": "Apply fungicides (myclobutanil, mancozeb) from bud break to veraison.",
        "severity": "High",
        "prevention": "Remove mummified berries, prune for air circulation, sanitation."
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Esca (Black Measles) is a complex trunk disease causing tiger-stripe patterns on leaves.",
        "treatment": "No effective chemical cure. Remove severely affected vines.",
        "severity": "Critical",
        "prevention": "Protect pruning wounds, avoid large pruning cuts, use clean tools."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Leaf blight caused by Isariopsis causes dark brown spots with yellow halos on leaves.",
        "treatment": "Apply copper-based fungicides or mancozeb.",
        "severity": "Moderate",
        "prevention": "Good canopy management, remove infected leaves, proper spacing."
    },
    "Grape___healthy": {
        "description": "The grape plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required.",
        "severity": "None",
        "prevention": "Continue regular pruning and canopy management."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Huanglongbing (Citrus Greening) is a devastating bacterial disease spread by Asian citrus psyllid.",
        "treatment": "No cure exists. Remove infected trees to prevent spread.",
        "severity": "Critical",
        "prevention": "Control psyllid vectors, use disease-free nursery stock, regular scouting."
    },
    "Peach___Bacterial_spot": {
        "description": "Bacterial spot is caused by Xanthomonas arboricola, creating water-soaked lesions on leaves and fruit.",
        "treatment": "Apply copper sprays or oxytetracycline during bloom.",
        "severity": "High",
        "prevention": "Plant resistant varieties, avoid overhead irrigation, proper spacing."
    },
    "Peach___healthy": {
        "description": "The peach plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required.",
        "severity": "None",
        "prevention": "Regular pruning and proper fertilization."
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "Bacterial spot is a devastating disease of peppers caused by Xanthomonas campestris.",
        "treatment": "Copper-based fungicides. Remove infected plant parts.",
        "severity": "High",
        "prevention": "Use disease-free seed, crop rotation, avoid overhead watering."
    },
    "Pepper,_bell___healthy": {
        "description": "The pepper plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required. Maintain current care.",
        "severity": "None",
        "prevention": "Continue standard watering and fertilization."
    },
    "Potato___Early_blight": {
        "description": "Fungal disease causing brown spots with concentric rings on lower leaves.",
        "treatment": "Apply fungicides like Mancozeb or Chlorothalonil.",
        "severity": "Moderate",
        "prevention": "Crop rotation, remove plant debris, proper spacing."
    },
    "Potato___Late_blight": {
        "description": "Severe disease causing rapidly spreading water-soaked spots, destroying crops quickly.",
        "treatment": "Apply systemic fungicides immediately. Destroy heavily infected plants.",
        "severity": "Critical",
        "prevention": "Plant resistant varieties, avoid overhead irrigation."
    },
    "Potato___healthy": {
        "description": "The potato plant is healthy.",
        "treatment": "No treatment required.",
        "severity": "None",
        "prevention": "Ensure good soil drainage."
    },
    "Raspberry___healthy": {
        "description": "The raspberry plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required.",
        "severity": "None",
        "prevention": "Proper pruning, good drainage, and adequate spacing."
    },
    "Soybean___healthy": {
        "description": "The soybean plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required.",
        "severity": "None",
        "prevention": "Crop rotation and proper soil management."
    },
    "Squash___Powdery_mildew": {
        "description": "Powdery mildew is a common fungal disease creating white powdery patches on squash leaves.",
        "treatment": "Apply neem oil, sulfur, or potassium bicarbonate fungicides.",
        "severity": "Moderate",
        "prevention": "Good air circulation, avoid overhead watering, plant resistant varieties."
    },
    "Strawberry___Leaf_scorch": {
        "description": "Leaf scorch is caused by Diplocarpon earlianum, producing dark purple spots that merge into scorched areas.",
        "treatment": "Apply fungicides (captan or thiram). Remove severely infected leaves.",
        "severity": "Moderate",
        "prevention": "Use resistant varieties, proper spacing, renovate beds after harvest."
    },
    "Strawberry___healthy": {
        "description": "The strawberry plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required.",
        "severity": "None",
        "prevention": "Mulching, proper watering, and regular monitoring."
    },
    "Tomato___Bacterial_spot": {
        "description": "Bacterial disease causing small, dark, water-soaked spots on leaves.",
        "treatment": "Copper fungicides can slow the spread.",
        "severity": "High",
        "prevention": "Keep foliage dry, avoid working with wet plants."
    },
    "Tomato___Early_blight": {
        "description": "Fungal infection creating 'bullseye' spots on older leaves.",
        "treatment": "Use fungicidal sprays (e.g., copper or chlorothalonil). Remove affected leaves.",
        "severity": "Moderate",
        "prevention": "Mulch to prevent soil splash, stake plants."
    },
    "Tomato___Late_blight": {
        "description": "Aggressive fungal disease causing large dark brown blotches on leaves and stems.",
        "treatment": "Fungicide application. Pull and destroy infected plants if severe.",
        "severity": "Critical",
        "prevention": "Keep leaves dry, ensure good air circulation."
    },
    "Tomato___Leaf_Mold": {
        "description": "Fungal disease common in humid environments, causing yellow spots and moldy undersides.",
        "treatment": "Improve ventilation, apply appropriate fungicides.",
        "severity": "Moderate",
        "prevention": "Space plants well, prune lower leaves."
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Destructive fungal disease causing numerous small circular spots on lower leaves.",
        "treatment": "Apply organic or chemical fungicides. Remove diseased leaves.",
        "severity": "High",
        "prevention": "Do not compost infected plants, use crop rotation."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Tiny pests that suck sap, causing yellow stippling and fine webbing on leaves.",
        "treatment": "Use neem oil, insecticidal soap, or horticultural oils.",
        "severity": "Moderate",
        "prevention": "Keep plants well-watered (mites thrive in dry conditions)."
    },
    "Tomato___Target_Spot": {
        "description": "Fungal disease causing dark brown lesions with concentric rings.",
        "treatment": "Fungicide sprays (chlorothalonil or mancozeb).",
        "severity": "Moderate",
        "prevention": "Good airflow, avoid overhead watering."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Viral disease transmitted by whiteflies, causing stunted growth and curled, yellowing leaves.",
        "treatment": "No cure for the virus. Remove infected plants immediately.",
        "severity": "Critical",
        "prevention": "Control whitefly populations, use reflective mulches."
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Highly contagious virus causing mottled light and dark green leaves and stunted growth.",
        "treatment": "No cure. Destroy infected plants and wash hands/tools thoroughly.",
        "severity": "Critical",
        "prevention": "Disinfect tools, do not smoke near plants, use resistant seeds."
    },
    "Tomato___healthy": {
        "description": "The tomato plant is healthy with no visible signs of disease.",
        "treatment": "No treatment required. Plant is thriving.",
        "severity": "None",
        "prevention": "Continue standard care and monitoring."
    }
}

# ── LEAF & PLANT VALIDATION ────────
FACE_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH) if os.path.exists(FACE_CASCADE_PATH) else None

def validate_leaf_image(img_bgr):
    """
    Validates whether the uploaded image contains a real plant leaf.
    Rejects non-leaf images (human faces, animals, cars, blank screens, electronics, etc.).
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, "Could not decode image. Please upload a valid JPG, PNG, or WEBP file."

    h, w = img_bgr.shape[:2]
    if h < 50 or w < 50:
        return False, "Image resolution is too low. Please upload a clearer photo of a plant leaf."

    # Blank / solid color / low detail check
    std_dev = float(np.std(img_bgr))
    if std_dev < 15.0:
        return False, "The image appears blank or lacks detail. Please upload a clear photo of a plant leaf."

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Foliage and vegetation color analysis in HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Healthy green foliage (Hue 25 to 92, Sat >= 25, Val >= 25)
    green_mask = (hue >= 25) & (hue <= 92) & (sat >= 25) & (val >= 25)

    # Yellowing / chlorotic / rust / scorch foliage (Hue 12 to 25, Sat >= 30, Val >= 30)
    yellow_mask = (hue >= 12) & (hue < 25) & (sat >= 30) & (val >= 30)

    # Necrotic brown leaf spots (Hue 0 to 12 or 168 to 180, Sat >= 25, Val between 20 and 160)
    brown_mask = ((hue <= 12) | (hue >= 168)) & (sat >= 25) & (val >= 20) & (val <= 160)

    plant_mask = green_mask | yellow_mask | brown_mask
    total_pixels = h * w
    plant_ratio = float(np.sum(plant_mask) / total_pixels)

    # Excess Green Index: ExG = 2*G - R - B
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_ratio = float(np.sum(exg > 10) / total_pixels)

    # Check for human face (if plant coverage is not dominating)
    if face_cascade and plant_ratio < 0.35:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(60, 60))
        if len(faces) > 0:
            return False, "Human face detected! Please upload a photo of a plant leaf, not a person."

    # Minimum vegetation / plant material threshold
    if plant_ratio < 0.10 and exg_ratio < 0.08:
        return False, "No plant leaf detected! Please upload a clear photo of a plant leaf."

    # Texture / edge complexity check (rejects solid green wallpapers, walls, or synthetic screens)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 15.0:
        return False, "Image lacks natural leaf texture. Please upload a real, focused leaf photo."

    return True, None


# ── PREDICTION FUNCTION ────────────
def predict_image(file_bytes):
    img_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, "Invalid image data. Please upload a valid image file.", 0.0

    # 1. Validate that the image contains a plant leaf
    is_leaf, error_msg = validate_leaf_image(img_bgr)
    if not is_leaf:
        return None, error_msg, 0.0

    # If local TensorFlow model is not loaded, defer to Gemini Vision
    if model is None or tf is None:
        return None, "LOCAL_MODEL_UNAVAILABLE", 0.0

    # 2. Preprocess for MobileNetV2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized.astype(np.float32))
    img_batch = np.reshape(img_preprocessed, (1, 224, 224, 3))

    pred = model.predict(img_batch, verbose=0)
    index = int(np.argmax(pred))
    confidence = float(np.max(pred))

    print(f"Predicted index: {index}, confidence: {confidence:.4f}")

    # Safety checks
    if index >= len(class_names):
        return None, "Unrecognized plant leaf variety.", confidence

    if confidence < 0.45:
        return None, f"Confidence too low ({confidence*100:.1f}%). Please upload a clearer, closer photo of the leaf.", confidence

    return class_names[index], None, confidence


# ── GEMINI AI HYBRID ADVISORY FUNCTIONS ──
def clean_json_markdown(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def get_gemini_advisory(plant, disease, confidence, fallback_info):
    """
    Enriches the diagnosis with tailored, in-depth agricultural advisory from Gemini AI.
    Includes Organic remedies, Chemical treatments, and long-term prevention.
    """
    if not gemini_available or not gemini_model:
        return fallback_info

    prompt = f"""You are an expert plant pathologist and agronomist.
A diagnostic scan was conducted on a plant leaf:
- Plant Species: {plant}
- Detected Condition/Disease: {disease}
- AI Confidence: {confidence*100:.1f}%

Provide a comprehensive, practical advisory formatted strictly as JSON with the following keys:
{{
  "description": "Clear explanation of the disease, pathogen cause, and visual symptoms (2-3 sentences).",
  "treatment": "Practical, step-by-step treatment recommendation combining immediate action and curative steps.",
  "organic_treatment": "Natural/organic remedies (e.g., neem oil, bio-fungicides, baking soda spray, pruning infected leaves).",
  "chemical_treatment": "Recommended chemical fungicides or pesticides with safe usage guidance.",
  "prevention": "Preventive measures (watering technique, crop spacing, soil hygiene, resistant varieties).",
  "severity": "{fallback_info.get('severity', 'Moderate')}"
}}

Respond ONLY with valid JSON, nothing else."""

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 280, "temperature": 0.2}
        )
        raw_text = clean_json_markdown(response.text)
        parsed = json.loads(raw_text)
        return {
            "description": parsed.get("description", fallback_info.get("description", "")),
            "treatment": parsed.get("treatment", fallback_info.get("treatment", "")),
            "organic_treatment": parsed.get("organic_treatment", ""),
            "chemical_treatment": parsed.get("chemical_treatment", ""),
            "prevention": parsed.get("prevention", fallback_info.get("prevention", "")),
            "severity": parsed.get("severity", fallback_info.get("severity", "Moderate")),
            "powered_by_gemini": True
        }
    except Exception as e:
        print("[WARN] Gemini advisory generation failed, using local database fallback:", e)
        return fallback_info


def gemini_vision_diagnose(file_bytes):
    """
    Multimodal Vision fallback using Gemini when local model confidence is low or uncertain.
    """
    if not gemini_available or not gemini_model:
        return None

    try:
        pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
        # Resize image to 384x384 max for blazing fast upload & inference
        pil_img.thumbnail((384, 384), Image.Resampling.LANCZOS)
        
        prompt = """Analyze this plant leaf photo carefully as an agricultural botanist.
Identify:
1. Plant species (e.g. Tomato, Apple, Corn, Potato, Grape, Pepper, Rice, etc.)
2. Disease/Condition (e.g. Early Blight, Healthy, Powdery Mildew, etc.)
3. Confidence score (0.70 to 0.99)
4. Severity level (None, Low, Moderate, High, Critical)
5. Concise advice.

Respond ONLY with valid JSON:
{
  "plant": "Plant Name",
  "disease": "Disease Name or Healthy",
  "confidence": 0.92,
  "severity": "Moderate",
  "description": "Short explanation (2 sentences).",
  "treatment": "Direct action plan.",
  "organic_treatment": "Natural remedies.",
  "chemical_treatment": "Recommended spray/fungicide.",
  "prevention": "Key prevention step."
}"""
        response = gemini_model.generate_content(
            [prompt, pil_img],
            generation_config={"max_output_tokens": 280, "temperature": 0.2}
        )
        raw_text = clean_json_markdown(response.text)
        data = json.loads(raw_text)
        data["powered_by_gemini_vision"] = True
        return data
    except Exception as e:
        print("[WARN] Gemini Vision diagnosis failed:", e)
        return None


# ── ROUTES ─────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    model_name = gemini_model.model_name if gemini_model else None
    return jsonify({
        "gemini_active": gemini_available,
        "gemini_model": model_name,
        "local_model": "Loaded" if model else "Not Loaded"
    })


@app.route("/api/config/gemini-key", methods=["GET", "POST"])
def config_gemini_key():
    if request.method == "POST":
        data = request.get_json() or {}
        new_key = data.get("api_key", "").strip()
        if not new_key:
            return jsonify({"success": False, "error": "API Key cannot be empty"}), 400

        success, msg = init_gemini(new_key)
        if success:
            # Update .env file
            try:
                env_lines = []
                key_written = False
                if os.path.exists(".env"):
                    with open(".env", "r") as f:
                        for line in f:
                            if line.startswith("GEMINI_API_KEY="):
                                env_lines.append(f"GEMINI_API_KEY={new_key}\n")
                                key_written = True
                            else:
                                env_lines.append(line)
                if not key_written:
                    env_lines.append(f"GEMINI_API_KEY={new_key}\n")
                with open(".env", "w") as f:
                    f.writelines(env_lines)
            except Exception as e:
                print("[WARN] Could not write key to .env:", e)

            return jsonify({"success": True, "message": "Gemini API connected successfully!"})
        else:
            return jsonify({"success": False, "error": f"Failed to connect: {msg}"}), 400

    # GET
    masked_key = ""
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        masked_key = GEMINI_API_KEY[:4] + "..." + GEMINI_API_KEY[-4:] if len(GEMINI_API_KEY) > 8 else "***"
    return jsonify({
        "configured": gemini_available,
        "masked_key": masked_key
    })


@app.route("/api/chat", methods=["POST"])
def plant_doctor_chat():
    """
    AI Plant Doctor chat assistant powered by Gemini AI.
    Optimized for high-speed sub-second responses.
    """
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    context_plant = data.get("plant", "")
    context_disease = data.get("disease", "")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    if not gemini_available or not gemini_model:
        return jsonify({
            "reply": "Gemini API Key is not configured yet. Please add your GEMINI_API_KEY in the .env file or click on the '🔑 Gemini API' button to enable the AI Plant Doctor!",
            "configured": False
        })

    try:
        chat_prompt = f"""You are PlantGuard AI Doctor, an expert agronomist.
Context: {context_plant or 'Plant'} - {context_disease or 'General care'}.
User Question: {user_message}

Instructions:
1. Answer directly and practically in the user's language (Hindi / Hinglish / English).
2. Give exactly 2-3 short, actionable bullet points (max 60 words total).
3. Be fast, direct, and helpful with dosages, spray timing, or organic remedy. No long introductions."""

        response = gemini_model.generate_content(
            chat_prompt,
            generation_config={"max_output_tokens": 160, "temperature": 0.3}
        )
        return jsonify({
            "reply": response.text.strip(),
            "configured": True
        })
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "quota" in err_msg.lower():
            friendly_reply = "⏳ **API Quota Limit Reached:** आपके Gemini API की फ़्री लिमिट पूरी हो गई है। कृपया 1 मिनट बाद दुबारा सवाल पूछें, या Google AI Studio से एक नई Free API Key जोड़ें।"
        else:
            friendly_reply = f"Sorry, could not process your question right now: {err_msg}"
        return jsonify({
            "reply": friendly_reply,
            "configured": True
        }), 200


@app.route("/analyze", methods=["POST"])
@app.route("/predict", methods=["POST"])
def analyze():
    target_file = request.files.get("file") or request.files.get("image")
    if not target_file:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = target_file.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400

    label, error_msg, confidence = predict_image(file_bytes)

    # ── HYBRID FALLBACK WITH GEMINI VISION ──
    # If local model had low confidence or was uncertain, use Gemini Vision!
    gemini_vision_used = False
    if (error_msg or not label) and gemini_available:
        print("[HYBRID] Local model uncertain or rejected leaf. Invoking Gemini Vision fallback...")
        vision_result = gemini_vision_diagnose(file_bytes)
        if vision_result and vision_result.get("plant"):
            plant = vision_result.get("plant", "Plant")
            disease = vision_result.get("disease", "Unknown")
            confidence = float(vision_result.get("confidence", 0.90))
            info = {
                "description": vision_result.get("description", "Diagnosed by Gemini Multimodal Vision."),
                "treatment": vision_result.get("treatment", "Consult local agricultural extension."),
                "organic_treatment": vision_result.get("organic_treatment", ""),
                "chemical_treatment": vision_result.get("chemical_treatment", ""),
                "prevention": vision_result.get("prevention", "Maintain proper ventilation and plant hygiene."),
                "severity": vision_result.get("severity", "Moderate"),
                "powered_by_gemini": True,
                "powered_by_gemini_vision": True
            }
            label = f"{plant}___{disease}"
            error_msg = None
            gemini_vision_used = True

    if error_msg or not label:
        return jsonify({"error": error_msg or "No plant leaf detected!"}), 400

    print("Predicted label:", label)

    # ── SAFE LABEL PARSE ───────────
    if not gemini_vision_used:
        # Split on ___ (triple underscore) which separates plant from disease
        if "___" in label:
            parts = label.split("___", 1)
            plant = parts[0].replace("_", " ").replace(",", ",")
            disease = parts[1].replace("_", " ").strip()
        else:
            plant = label.replace("_", " ")
            disease = "Unknown"

        # Clean up plant names
        plant = re.sub(r'\s+', ' ', plant).strip()
        disease = re.sub(r'\s+', ' ', disease).strip()

        if disease.lower() == "healthy":
            disease = "Healthy (No Disease)"

        # ── TREATMENT BASELINE ─────────
        info = treatments.get(label, {
            "description": "Information not available for this condition.",
            "treatment": "Consult a local agricultural extension specialist.",
            "severity": "Low",
            "prevention": "Maintain proper plant care and sanitary practices."
        }).copy()

        # ── HYBRID ENRICHMENT: GEMINI ADVISORY ──
        if gemini_available:
            print(f"[HYBRID] Requesting rich Gemini AI advisory for {plant} - {disease}...")
            info = get_gemini_advisory(plant, disease, confidence, info)

    # ── SAVE TO SQLITE DATABASE ────
    record_id, saved_image_url, saved_time = save_scan_record(
        plant, disease, confidence, info.get("severity", "Moderate"), info.get("treatment", ""), file_bytes
    )

    return jsonify({
        "id": record_id,
        "plant": plant,
        "disease": disease,
        "confidence": confidence,
        "description": info.get("description", ""),
        "treatment": info.get("treatment", ""),
        "organic_treatment": info.get("organic_treatment", ""),
        "chemical_treatment": info.get("chemical_treatment", ""),
        "severity": info.get("severity", "Moderate"),
        "prevention": info.get("prevention", ""),
        "prediction": disease,
        "class": label,
        "image_url": saved_image_url,
        "timestamp": saved_time,
        "gemini_enhanced": bool(info.get("powered_by_gemini") or info.get("powered_by_gemini_vision"))
    })


# ── HISTORY API ROUTES ─────────────
@app.route("/api/history", methods=["GET"])
def get_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scan_history ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        history = [dict(row) for row in rows]
        conn.close()
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scan_history")
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "History cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── PDF ────────────────────────────
@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    data = request.get_json()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = styles['Title']
    title_style.textColor = colors.HexColor("#14532d")
    heading_style = styles['Heading2']
    heading_style.textColor = colors.HexColor("#16a34a")
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.leading = 14

    content = []

    # Header
    content.append(Paragraph("PlantGuard AI - Diagnostic Report", title_style))
    content.append(Spacer(1, 10))
    
    # Date & Report ID
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content.append(Paragraph(f"<b>Date:</b> {now}", normal_style))
    content.append(Spacer(1, 20))

    # Basic Info Table
    info_data = [
        ["Plant Species:", data.get('plant', 'Unknown'), "Severity:", data.get('severity', 'Unknown')],
        ["Detected Disease:", data.get('disease', 'Unknown'), "AI Confidence:", f"{float(data.get('confidence', 0))*100:.1f}%"]
    ]
    t = Table(info_data, colWidths=[100, 160, 70, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#1f2937")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#86efac"))
    ]))
    content.append(t)
    content.append(Spacer(1, 20))

    # Detailed Diagnosis
    content.append(Paragraph("Diagnosis Details", heading_style))
    content.append(Spacer(1, 5))
    content.append(Paragraph(data.get('description', 'No description available.'), normal_style))
    content.append(Spacer(1, 15))

    # Treatment
    content.append(Paragraph("Prescribed Treatment", heading_style))
    content.append(Spacer(1, 5))
    content.append(Paragraph(data.get('treatment', 'Consult expert.'), normal_style))
    content.append(Spacer(1, 15))

    # Organic Treatment if available
    if data.get('organic_treatment'):
        content.append(Paragraph("Organic & Biological Remedies", heading_style))
        content.append(Spacer(1, 5))
        content.append(Paragraph(data.get('organic_treatment'), normal_style))
        content.append(Spacer(1, 15))

    # Chemical Treatment if available
    if data.get('chemical_treatment'):
        content.append(Paragraph("Chemical Controls & Fungicides", heading_style))
        content.append(Spacer(1, 5))
        content.append(Paragraph(data.get('chemical_treatment'), normal_style))
        content.append(Spacer(1, 15))

    # Prevention
    content.append(Paragraph("Prevention Strategies", heading_style))
    content.append(Spacer(1, 5))
    content.append(Paragraph(data.get('prevention', 'General care.'), normal_style))
    content.append(Spacer(1, 40))

    # Consultants / Signatures (Blood Report style)
    content.append(Paragraph("<b>Consultant Agronomists & AI Specialists:</b>", normal_style))
    content.append(Spacer(1, 10))
    
    # Signatures Table
    sig_data = [
        ["Deepika Singh", "Hritik Mishra", "Kartik Gupta", "Rakshit Kumar", "Saurav Kumar"],
        ["AI Specialist", "AI Specialist", "AI Specialist", "AI Specialist", "AI Specialist"]
    ]
    t_sig = Table(sig_data, colWidths=[103, 103, 103, 103, 103])
    t_sig.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, 1), 8),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Oblique'),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.gray),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    content.append(t_sig)

    doc.build(content)
    buffer.seek(0)

    return send_file(buffer, mimetype="application/pdf",
                     as_attachment=True,
                     download_name="PlantGuard_Diagnostic_Report.pdf")


# ── RUN ────────────────────────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)