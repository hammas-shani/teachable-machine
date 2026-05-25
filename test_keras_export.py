import tensorflow as tf
import numpy as np

# Mock classes and centroids
classes_ = ["cat", "dog"]
centroids = {"cat": np.random.randn(1280), "dog": np.random.randn(1280)}
for c in centroids:
    centroids[c] = centroids[c] / np.linalg.norm(centroids[c])

inputs = tf.keras.Input(shape=(224, 224, 3))
# Dummy base
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None,
    pooling="avg"
)
x = base(inputs)
x = tf.math.l2_normalize(x, axis=1)

num_classes = len(classes_)
dense = tf.keras.layers.Dense(num_classes, use_bias=False, name="dot_product")
x = dense(x)

# Math ops
x = tf.subtract(2.0, tf.multiply(2.0, x))
x = tf.maximum(x, 0.0)
x = tf.sqrt(x + 1e-8)
logits = tf.multiply(x, -5.0)
probs = tf.keras.layers.Softmax(name="predictions")(logits)

model = tf.keras.Model(inputs=inputs, outputs=probs)

# Set weights
weights = np.zeros((1280, num_classes), dtype=np.float32)
for i, cls in enumerate(classes_):
    weights[:, i] = centroids[cls]
dense.set_weights([weights])

model.save("test_export.keras")
model.save("test_export.h5")
print("Saved successfully!")
