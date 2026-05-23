import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File

from ml.model import predict
from ml.mobilenet import extract_features_with_detection

router = APIRouter()


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    features, face_found = extract_features_from_array(img)

    # face_found is passed through — model.py decides whether to use it
    # based on the current face_mode setting (True=face-only, False=any object)
    result = predict(features, face_found=face_found)

    return result


def extract_features_from_array(img):
    # 1. Convert BGR (OpenCV default) -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Resize to 224x224
    img = cv2.resize(img, (224, 224))

    # 3. Extract features + face detection flag using PyTorch MobileNetV2
    features, face_found = extract_features_with_detection(img)
    return features, face_found