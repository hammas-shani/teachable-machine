"""
model.py -- Centroid-based classifier using L2 distance.

Works for faces AND general objects (like Google Teachable Machine).

face_mode behaviour in predict():
  - face_mode=True  → if Haar cascade finds NO face, return Unknown immediately
  - face_mode=False → classify the full image even if no face is detected
                      (correct for general objects: cups, chairs, cats, etc.)
"""

import sys
import numpy as np

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

TEMPERATURE = 5.0   # higher = more decisive softmax distribution
CONFIDENCE_THRESHOLD = 0.75  # minimum probability required to not be marked Unknown


class CentroidClassifier:
    def __init__(self, threshold=1.5):
        """
        threshold: max L2 distance to the nearest centroid before marking 'uncertain'.
                   Lower = stricter (more Unknowns). Good range: 0.8 - 1.5.
        """
        self.centroids = {}
        self.classes_ = []
        self.threshold = threshold

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.centroids = {}

        for cls in self.classes_:
            class_vecs = X[y == cls].astype(np.float32)
            # Normalize each embedding before averaging → unit-hypersphere centroid
            normed = np.array([self._normalize(v) for v in class_vecs])
            centroid = np.mean(normed, axis=0)
            centroid = self._normalize(centroid)   # keep on unit sphere
            self.centroids[cls] = centroid

        print(f"[model] Trained {len(self.classes_)} classes: {list(self.classes_)}")
        print(f"[model] Threshold: {self.threshold}")

    def _l2_distances(self, x: np.ndarray) -> list:
        """L2 distance from normalised x to each class centroid."""
        x_n = self._normalize(x.astype(np.float32))
        dists = []
        for cls in self.classes_:
            d = float(np.linalg.norm(x_n - self.centroids[cls]))
            dists.append(d)
        return dists

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probabilities from temperature-scaled softmax over negative L2 distances.
        Smaller L2 distance = higher probability.
        """
        result = []
        for x in X:
            dists = np.array(self._l2_distances(x), dtype=np.float32)
            logits = -dists * TEMPERATURE
            logits -= logits.max()   # numerical stability
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum()
            result.append(probs)
        return np.array(result)

    def predict_distances(self, X: np.ndarray) -> np.ndarray:
        """Raw L2 distances to each centroid."""
        return np.array([self._l2_distances(x) for x in X])

    @property
    def is_fitted(self):
        return len(self.centroids) > 0 and len(self.classes_) > 0


model = CentroidClassifier(threshold=1.5)
is_trained = False

# Global mode flag — set to True if training data contains faces
# When False (default): behaves like Google Teachable Machine for any object
face_mode = False


def predict(features, face_found: bool = True):
    """
    Predict the class of a feature vector.

    Args:
        features:   1280-dim float32 feature vector from MobileNetV2
        face_found: result of Haar cascade detection.
                    - face_mode=True  → face_found=False returns Unknown (empty frame)
                    - face_mode=False → face_found is IGNORED (generalised object mode)

    Returns dict:
      - predicted_class: class name or "Unknown"
      - confidence: 1.0 for a match, 0.0 for Unknown
      - all_probs: {class: probability} — winner = 1.0, rest = 0.0
      - is_uncertain: bool
      - l2_distance: distance to nearest centroid
    """
    if not is_trained or not model.is_fitted:
        return {"error": "Model not trained yet. Please train first."}

    classes = model.classes_

    # In face_mode, immediately return Unknown if no face detected
    # In general-object mode, always classify (like Google Teachable Machine)
    if face_mode and not face_found:
        return {
            "predicted_class": "Unknown",
            "confidence": 0.0,
            "all_probs": {str(cls): 0.0 for cls in classes},
            "is_uncertain": True,
            "l2_distance": -1.0
        }

    features = np.array(features, dtype=np.float32)

    # Handle edge case: Blocked camera or empty frames yielding pure black/near-zero features
    if np.linalg.norm(features) < 1e-4:
        return {
            "predicted_class": "Unknown",
            "confidence": 0.0,
            "all_probs": {str(cls): 0.0 for cls in classes},
            "is_uncertain": True,
            "l2_distance": -1.0
        }

    probs = model.predict_proba([features])[0]
    dists = model.predict_distances([features])[0]

    best_idx = int(np.argmax(probs))
    best_dist = float(dists[best_idx])
    best_prob = float(probs[best_idx])

    # Threshold check: too far from all centroids or confidence below threshold → Unknown
    is_uncertain = (best_dist > model.threshold) or (best_prob < CONFIDENCE_THRESHOLD)

    if is_uncertain:
        display_probs = {str(cls): 0.0 for cls in classes}
        display_confidence = 0.0
    else:
        # Winner gets 100%, all others 0%
        display_probs = {str(cls): 0.0 for cls in classes}
        display_probs[str(classes[best_idx])] = 1.0
        display_confidence = 1.0

    return {
        "predicted_class": "Unknown" if is_uncertain else str(classes[best_idx]),
        "confidence": display_confidence,
        "all_probs": display_probs,
        "is_uncertain": is_uncertain,
        "l2_distance": round(best_dist, 4)
    }