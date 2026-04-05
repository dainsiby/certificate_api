import cv2
import numpy as np

def expand_bbox(x1, y1, x2, y2, img_w, img_h, expand_ratio=0.15):
    bw = x2 - x1
    bh = y2 - y1

    ex = int(bw * expand_ratio)
    ey = int(bh * expand_ratio)

    nx1 = max(0, x1 - ex)
    ny1 = max(0, y1 - ey)
    nx2 = min(img_w, x2 + ex)
    ny2 = min(img_h, y2 + ey)

    return nx1, ny1, nx2, ny2

def preprocess_signature_crop(crop_bgr, target_size=(224, 224)):
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty crop received for preprocessing.")

    # mild blur / denoise
    denoised = cv2.GaussianBlur(crop_bgr, (3, 3), 0)

    # CLAHE on luminance
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    enhanced_lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # mild sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(enhanced, -1, kernel)

    resized = cv2.resize(sharpened, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    return input_tensor, resized
