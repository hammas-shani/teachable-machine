"""
face_utils.py — Face detection and crop using OpenCV.

Detects the largest frontal face in an image and returns a padded crop.
Returns (crop, face_found) where face_found=False means no person in frame.

NO edge mask is applied — raw pixel data is preserved so MobileNetV2
sees the same features during training and prediction.
"""

import cv2
import numpy as np

# OpenCV's built-in Haar cascades — no download needed
_frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
_alt_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)


def detect_and_crop_face(img_rgb: np.ndarray, padding: float = 0.25):
    """
    Detect the largest face in img_rgb (uint8 RGB HxWx3) and return a padded crop.

    Args:
        img_rgb:  uint8 numpy array in RGB colour space, shape (H, W, 3)
        padding:  fractional padding added around the detected face box (0.25 = 25%)

    Returns:
        (cropped_rgb, face_found)
        - cropped_rgb: the face crop, or the original image if no face detected
        - face_found:  bool — False means no person in frame → caller returns Unknown
    """
    if img_rgb is None or img_rgb.size == 0:
        return img_rgb, False

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)  # improves detection in poor lighting

    faces = []

    # Try primary frontal cascade
    detected = _frontal_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    if len(detected) > 0:
        faces = detected

    # If no face found, try the alt cascade (detects tilted / partially visible faces)
    if len(faces) == 0:
        detected2 = _alt_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(detected2) > 0:
            faces = detected2

    if len(faces) == 0:
        # No face detected at all → caller should return Unknown
        return img_rgb, False

    # Pick the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Apply padding around the face box
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    H, W = img_rgb.shape[:2]

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    face_crop = img_rgb[y1:y2, x1:x2]

    if face_crop.size == 0:
        return img_rgb, False

    return face_crop, True
