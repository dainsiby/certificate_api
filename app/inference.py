import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

from .config import (
    YOLO_MODEL_PATH,
    CLS_MODEL_PATH,
    YOLO_SIG_MIN_CONF,
    FINAL_COMBINED_THRESHOLD,
    SIGNATURE_CLASS_NAMES,
    STAMP_CLASS_NAMES,
)
from .preprocessing import expand_bbox, preprocess_signature_crop


class CertificateVerifier:
    def __init__(self):
        self.yolo_model = None
        self.cls_model = None
        self.class_name_map = {}

    def load_models(self):
        self.yolo_model = YOLO(str(YOLO_MODEL_PATH))
        self.cls_model = load_model(str(CLS_MODEL_PATH))
        self.class_name_map = self.yolo_model.names

        print("YOLO loaded")
        print("Classifier loaded")
        print("YOLO classes:", self.class_name_map)

    def _predict_signature_score(self, crop_bgr):
        input_tensor, _ = preprocess_signature_crop(crop_bgr)

        pred = self.cls_model.predict(input_tensor, verbose=0)
        pred = np.array(pred).squeeze()

        # supports sigmoid or softmax-style output
        if pred.ndim == 0:
            cls_score = float(pred)
        elif np.isscalar(pred):
            cls_score = float(pred)
        elif pred.shape == (1,):
            cls_score = float(pred[0])
        else:
            # assume last index is genuine probability
            cls_score = float(pred[-1])

        return max(0.0, min(1.0, cls_score))

    def verify_certificate(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read uploaded image.")

        img_h, img_w = image.shape[:2]

        results = self.yolo_model(image_path, verbose=False)
        r = results[0]

        signature_candidates = []
        stamp_present = False
        stamp_conf = 0.0

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                class_name = str(self.class_name_map.get(cls_id, cls_id)).lower()

                if class_name in STAMP_CLASS_NAMES:
                    stamp_present = True
                    stamp_conf = max(stamp_conf, conf)

                if class_name in SIGNATURE_CLASS_NAMES and conf > YOLO_SIG_MIN_CONF:
                    ex1, ey1, ex2, ey2 = expand_bbox(
                        x1, y1, x2, y2, img_w, img_h, expand_ratio=0.15
                    )

                    crop = image[ey1:ey2, ex1:ex2]
                    if crop is None or crop.size == 0:
                        continue

                    try:
                        cls_score = self._predict_signature_score(crop)
                    except Exception:
                        continue

                    combined_score = 0.4 * conf + 0.6 * cls_score

                    signature_candidates.append({
                        "bbox": [ex1, ey1, ex2, ey2],
                        "yolo_conf": round(conf, 4),
                        "classifier_score": round(cls_score, 4),
                        "combined_score": round(combined_score, 4),
                    })

        if not signature_candidates:
            return {
                "result": "FAKE",
                "reason": "No valid signature detected above YOLO confidence threshold.",
                "yolo_conf": 0.0,
                "classifier_score": 0.0,
                "combined_score": 0.0,
                "stamp_present": stamp_present,
                "stamp_conf": round(stamp_conf, 4),
                "all_signatures": [],
            }

        best = max(signature_candidates, key=lambda x: x["combined_score"])

        if best["combined_score"] > FINAL_COMBINED_THRESHOLD and stamp_present:
            result = "GENUINE"
            reason = "Best combined score passed threshold and stamp is present."
        elif best["combined_score"] > FINAL_COMBINED_THRESHOLD and not stamp_present:
            result = "SUSPICIOUS"
            reason = "Best combined score passed threshold but stamp is missing."
        else:
            result = "FAKE"
            reason = "Best combined score is below threshold."

        return {
            "result": result,
            "reason": reason,
            "yolo_conf": best["yolo_conf"],
            "classifier_score": best["classifier_score"],
            "combined_score": best["combined_score"],
            "stamp_present": stamp_present,
            "stamp_conf": round(stamp_conf, 4),
            "all_signatures": signature_candidates,
        }


verifier = CertificateVerifier()
