import numpy as np
import cv2

def augment_image(img):
    aug_images = []

    img = cv2.resize(img, (224, 224))

    for i in range(40):  # 30-50 augmentations
        augmented = img.copy()

        # flip
        if np.random.rand() > 0.5:
            augmented = cv2.flip(augmented, 1)

        # brightness
        value = np.random.randint(-30, 30)
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        aug_images.append(augmented)

    return aug_images


def should_augment(images):
    return len(images) < 5