import cv2
import numpy as np

def extract_features(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (64, 64))
    img = img.flatten()

    return img