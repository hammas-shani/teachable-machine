"""
trainer.py — Training pipeline with face-aware feature extraction and augmentation.

Key improvements:
  - Face detection + crop via ml.mobilenet (handled transparently)
  - Augmentation: horizontal flip + brightness variation for richer centroids
  - Per-class validation to catch empty-class edge cases early
"""

import sys
import numpy as np
import cv2
import time
import os

# Fix Windows console encoding so non-ASCII characters in log messages
# do NOT crash the training process (cp1252 cannot encode many Unicode chars).
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from ml.mobilenet import extract_features
from core.state import load_state, set_trained

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

job_status = {
    "status": "idle",
    "currentEpoch": 0,
    "totalEpochs": 10,
    "error": None,
    "classes": []
}


def resolve_path(img_path):
    """Convert relative dataset paths to absolute."""
    if os.path.isabs(img_path):
        return img_path
    return os.path.join(BACKEND_DIR, img_path)


def load_image_rgb(img_path) -> np.ndarray:
    """Load an image as uint8 RGB (224×224). Raises on failure."""
    abs_path = resolve_path(img_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image not found: {abs_path}")
    img = cv2.imread(abs_path)
    if img is None:
        raise ValueError(f"OpenCV could not decode: {abs_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img  # uint8, shape (224,224,3)


def augment_image(img: np.ndarray) -> list:
    """
    Return a list of augmented variants of img (all uint8 RGB 224×224).
    Augmentations: original + horizontal flip + 2 brightness variants.
    """
    variants = [img]

    # Horizontal flip
    variants.append(cv2.flip(img, 1))

    # Brightness ±30
    for delta in (30, -30):
        bright = img.astype(np.int16) + delta
        bright = np.clip(bright, 0, 255).astype(np.uint8)
        variants.append(bright)

    return variants


def train_model(epochs: int = 10):
    global job_status
    job_status["status"] = "training"
    job_status["currentEpoch"] = 0
    job_status["totalEpochs"] = epochs
    job_status["error"] = None
    job_status["classes"] = []

    try:
        state = load_state()

        # ── 1. Clean stale / missing image paths ──────────────────────────────
        cleaned_classes = {}
        for cls, images in state["classes"].items():
            valid = [p for p in images if os.path.exists(resolve_path(p))]
            if valid:
                cleaned_classes[cls] = valid
        state["classes"] = cleaned_classes

        if len(cleaned_classes) < 2:
            job_status["status"] = "error"
            job_status["error"] = (
                f"Need at least 2 classes with images. "
                f"Found {len(cleaned_classes)}: {list(cleaned_classes.keys())}"
            )
            return

        # ── 2. Extract features (with face crop + augmentation) ───────────────
        X_feats = []
        y_labels = []
        class_counts = {}
        skipped = 0

        for class_name, images in state["classes"].items():
            class_feat_count = 0
            for img_path in images:
                try:
                    img = load_image_rgb(img_path)
                    # Augment each training image for a richer centroid
                    for aug_img in augment_image(img):
                        feat = extract_features(aug_img)  # face crop happens inside
                        X_feats.append(feat)
                        y_labels.append(class_name)
                        class_feat_count += 1
                except Exception as err:
                    print(f"[trainer] Skipping {img_path}: {err}")
                    skipped += 1

            class_counts[class_name] = class_feat_count
            print(f"[trainer] Class '{class_name}': {class_feat_count} feature vectors "
                  f"(from {len(images)} images × 4 augmentations)")

        print(f"[trainer] Total feature vectors: {len(X_feats)}, skipped: {skipped}")

        # ── 3. Validate ───────────────────────────────────────────────────────
        if len(X_feats) < 2:
            job_status["status"] = "error"
            job_status["error"] = f"Too few valid images after loading. Got {len(X_feats)}."
            return

        missing_classes = [c for c, cnt in class_counts.items() if cnt == 0]
        if missing_classes:
            job_status["status"] = "error"
            job_status["error"] = f"No valid images for classes: {missing_classes}"
            return

        # ── 4. Fit centroid classifier ─────────────────────────────────────────
        X = np.array(X_feats, dtype=np.float32)
        y = np.array(y_labels)

        import ml.model
        ml.model.model.fit(X, y)
        ml.model.is_trained = True
        set_trained(True)

        job_status["classes"] = list(state["classes"].keys())

        # Tick epoch counter for UI progress
        for i in range(1, epochs + 1):
            time.sleep(0.01)
            job_status["currentEpoch"] = i

        job_status["status"] = "completed"
        print(f"[trainer] Training complete! Classes: {job_status['classes']}")

    except Exception as e:
        import traceback
        job_status["status"] = "error"
        job_status["error"] = str(e)
        # Encode to ASCII-safe string so Windows cp1252 console never crashes
        # even if the traceback itself contains non-ASCII characters.
        tb_text = traceback.format_exc().encode('ascii', errors='replace').decode('ascii')
        print(f"[trainer] FATAL:\n{tb_text}")

