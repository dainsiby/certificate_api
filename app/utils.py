import uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException
from .config import ALLOWED_EXTENSIONS, UPLOAD_DIR

def validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    return ext

def generate_upload_path(filename: str) -> Path:
    ext = validate_extension(filename)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    return UPLOAD_DIR / unique_name

async def save_upload_file(upload_file: UploadFile) -> Path:
    file_path = generate_upload_path(upload_file.filename)
    content = await upload_file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path
