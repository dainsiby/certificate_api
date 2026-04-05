from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from .inference import verifier
from .utils import save_upload_file
from .config import API_KEY

app = FastAPI(title="Certificate Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    verifier.load_models()

def check_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/verify-certificate")
async def verify_certificate(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    check_api_key(x_api_key)

    saved_path = None
    try:
        saved_path = await save_upload_file(file)
        result = verifier.verify_certificate(str(saved_path))
        result["uploaded_file"] = Path(saved_path).name
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if saved_path and Path(saved_path).exists():
            Path(saved_path).unlink(missing_ok=True)
