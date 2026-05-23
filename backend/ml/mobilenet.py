"""
mobilenet.py — MobileNetV2 feature extractor with automatic face detection + crop.

Pipeline for each image:
  1. Detect face with OpenCV Haar cascade
  2. Crop face with padding (or use full image if no face detected)
  3. Pass crop through MobileNetV2 conv layers + AdaptiveAvgPool
  4. Return 1280-dim float32 feature vector

For prediction: use extract_features_with_detection() which also returns
face_found (False = no person in frame → caller returns Unknown).
For training:   use extract_features() which ignores the face_found flag.
"""

import sys
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from ml.face_utils import detect_and_crop_face

# Fix Windows console encoding (cp1252 cannot handle many Unicode chars)
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

# ── Model ──────────────────────────────────────────────────────────────────────
_mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
_mobilenet.eval()

# Conv features + global avg pool → (1, 1280, 1, 1)
feature_extractor = torch.nn.Sequential(
    _mobilenet.features,
    torch.nn.AdaptiveAvgPool2d((1, 1))
)
feature_extractor.eval()

# ── Transform ──────────────────────────────────────────────────────────────────
# Expects uint8 numpy (H, W, 3) RGB
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure the array is uint8 RGB. Handles float32 [0-255] or [0-1] inputs."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() > 1.0:          # [0-255] float → uint8
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:                          # [0-1] float → uint8
            image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def extract_features(image) -> np.ndarray:
    """
    Extract a 1280-dim feature vector from an image, focusing on the face.
    Used by the TRAINER — face_found flag is ignored (always extracts features).

    Args:
        image: numpy ndarray (H, W, 3) in RGB or PIL Image.
               Accepts uint8 [0-255] OR float32 in any range.
    Returns:
        numpy float32 array of shape (1280,)
    """
    face_crop, _ = _extract_face_crop(image)
    return _run_mobilenet(face_crop)


def extract_features_with_detection(image):
    """
    Extract features AND return whether a face was detected.
    Used by the PREDICTOR — face_found=False means return Unknown immediately.

    Args:
        image: numpy ndarray (H, W, 3) RGB or PIL Image.
    Returns:
        (features: np.ndarray shape (1280,), face_found: bool)
    """
    face_crop, face_found = _extract_face_crop(image)
    features = _run_mobilenet(face_crop)
    return features, face_found


def _extract_face_crop(image) -> tuple:
    """Normalise input and run face detection. Returns (crop, face_found)."""
    # ── 1. Normalise to uint8 RGB ──────────────────────────────────────────────
    if isinstance(image, np.ndarray):
        image = _to_uint8_rgb(image)
    elif isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ── 2. Detect & crop the face (or use full image if no face found) ─────────
    face_crop, face_found = detect_and_crop_face(image)
    if not face_found:
        print("[mobilenet] No face detected - will return Unknown")

    return face_crop, face_found


def _run_mobilenet(face_crop: np.ndarray) -> np.ndarray:
    """Run the crop through MobileNetV2 and return a 1280-dim feature vector."""
    # ── 3. Apply ImageNet transform ────────────────────────────────────────────
    tensor = _transform(face_crop).unsqueeze(0)  # (1, 3, 224, 224)

    # ── 4. Forward pass through MobileNetV2 feature extractor ─────────────────
    with torch.no_grad():
        features = feature_extractor(tensor)       # (1, 1280, 1, 1)

    return features.flatten().numpy()              # (1280,) float32