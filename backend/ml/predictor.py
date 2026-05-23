import joblib
import numpy as np
from PIL import Image

from ml.mobilenet import extract_features

MODEL_PATH = "model/classifier.pkl"

def predict_image(image):

    model = joblib.load(MODEL_PATH)

    features = extract_features(image)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    class_names = model.classes_

    return {
        "prediction": prediction,
        "confidence": float(max(probabilities)),
        "probabilities": dict(zip(class_names, probabilities.tolist()))
    }