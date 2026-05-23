import numpy as np
from ml.mobilenet import extract_features as _extract_features_torch


def extract_features(img):
    """
    img: numpy array (H, W, 3) in RGB uint8 or float32
    returns: 1D feature vector (numpy)
    """
    return _extract_features_torch(img)


def batch_features(images):
    return np.array([extract_features(img) for img in images])