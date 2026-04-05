from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

YOLO_MODEL_PATH = APP_DIR / "models" / "best.pt"
CLS_MODEL_PATH = APP_DIR / "models" / "bestcromodel_merged.h5"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

YOLO_SIG_MIN_CONF = 0.50
FINAL_COMBINED_THRESHOLD = 0.75

SIGNATURE_CLASS_NAMES = {"signature", "sign"}
STAMP_CLASS_NAMES = {"stamp", "seal"}

API_KEY = os.getenv("CERT_API_KEY", "change-this-local-key")
