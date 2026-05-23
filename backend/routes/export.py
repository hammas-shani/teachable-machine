"""
export.py — Multi-format model export endpoint.

Supported formats:
  json  — Centroid data as JSON (default, lightweight, portable)
  pkl   — Full Python pickle of CentroidClassifier object
  h5    — MobileNetV2 feature extractor saved in HDF5 format (Keras)
  keras — MobileNetV2 feature extractor saved in native Keras format
"""

import io
import json
import pickle
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
import ml.model as ml_model

router = APIRouter()


def _check_trained():
    """Return error response if model is not trained yet."""
    if not ml_model.is_trained or not ml_model.model.is_fitted:
        return JSONResponse(
            status_code=400,
            content={"error": "Model is not trained yet. Please train first."}
        )
    return None


# ── /api/export/json ────────────────────────────────────────────────────────
@router.get("/export/json")
async def export_json():
    """Export model as JSON — centroids + metadata. Lightweight and portable."""
    err = _check_trained()
    if err:
        return err

    export_data = {
        "model_type": "CentroidDistanceClassifier",
        "backbone": "MobileNetV2-ImageNet",
        "feature_dim": 1280,
        "temperature": 5.0,
        "threshold": float(ml_model.model.threshold),
        "face_mode": ml_model.face_mode,
        "classes": [str(c) for c in ml_model.model.classes_],
        "centroids": {
            str(cls): centroid.tolist()
            for cls, centroid in ml_model.model.centroids.items()
        }
    }

    return Response(
        content=json.dumps(export_data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="teachable_model.json"'}
    )


# ── /api/export/pkl ─────────────────────────────────────────────────────────
@router.get("/export/pkl")
async def export_pkl():
    """Export the full CentroidClassifier object as a Python pickle file."""
    err = _check_trained()
    if err:
        return err

    buf = io.BytesIO()
    pickle.dump(ml_model.model, buf, protocol=4)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="teachable_model.pkl"'}
    )


# ── /api/export/h5 ──────────────────────────────────────────────────────────
@router.get("/export/h5")
async def export_h5():
    """
    Export full end-to-end MobileNetV2 + Centroid Classifier as HDF5 (.h5).
    """
    err = _check_trained()
    if err:
        return err

    try:
        import tempfile
        import os
        import tensorflow as tf

        classes_ = ml_model.model.classes_
        num_classes = len(classes_)
        temperature = ml_model.TEMPERATURE

        inputs = tf.keras.Input(shape=(224, 224, 3))
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        x = base(inputs)
        x = tf.math.l2_normalize(x, axis=1)

        dense = tf.keras.layers.Dense(num_classes, use_bias=False, name="dot_product")
        x = dense(x)

        x = tf.subtract(2.0, tf.multiply(2.0, x))
        x = tf.maximum(x, 0.0)
        x = tf.sqrt(x + 1e-8)
        logits = tf.multiply(x, -float(temperature))
        probs = tf.keras.layers.Softmax(name="predictions")(logits)

        model = tf.keras.Model(inputs=inputs, outputs=probs)

        weights = np.zeros((1280, num_classes), dtype=np.float32)
        for i, cls in enumerate(classes_):
            # Centroids are already normalized
            weights[:, i] = ml_model.model.centroids[cls]
        dense.set_weights([weights])

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            model.save(tmp_path)
            with open(tmp_path, "rb") as f:
                content = f.read()
        finally:
            os.unlink(tmp_path)

        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="teachable_model.h5"'}
        )

    except ImportError as e:
        return JSONResponse(
            status_code=501,
            content={"error": f"TensorFlow import error: {str(e)}. Make sure you started the server with the venv python."}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── /api/export/keras ────────────────────────────────────────────────────────
@router.get("/export/keras")
async def export_keras():
    """
    Export full end-to-end MobileNetV2 + Centroid Classifier in native Keras (.keras) format.
    """
    err = _check_trained()
    if err:
        return err

    try:
        import tempfile
        import os
        import tensorflow as tf

        classes_ = ml_model.model.classes_
        num_classes = len(classes_)
        temperature = ml_model.TEMPERATURE

        inputs = tf.keras.Input(shape=(224, 224, 3))
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        x = base(inputs)
        x = tf.math.l2_normalize(x, axis=1)

        dense = tf.keras.layers.Dense(num_classes, use_bias=False, name="dot_product")
        x = dense(x)

        x = tf.subtract(2.0, tf.multiply(2.0, x))
        x = tf.maximum(x, 0.0)
        x = tf.sqrt(x + 1e-8)
        logits = tf.multiply(x, -float(temperature))
        probs = tf.keras.layers.Softmax(name="predictions")(logits)

        model = tf.keras.Model(inputs=inputs, outputs=probs)

        weights = np.zeros((1280, num_classes), dtype=np.float32)
        for i, cls in enumerate(classes_):
            # Centroids are already normalized
            weights[:, i] = ml_model.model.centroids[cls]
        dense.set_weights([weights])

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            model.save(tmp_path)
            with open(tmp_path, "rb") as f:
                content = f.read()
        finally:
            os.unlink(tmp_path)

        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="teachable_model.keras"'}
        )

    except ImportError as e:
        return JSONResponse(
            status_code=501,
            content={"error": f"TensorFlow import error: {str(e)}. Make sure you started the server with the venv python."}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Legacy route — keep backward compatibility ───────────────────────────────
@router.get("/export")
async def export_model_legacy():
    """Backward-compatible alias → same as /api/export/json"""
    return await export_json()
