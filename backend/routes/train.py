from fastapi import APIRouter, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import numpy as np
import cv2
from core.state import add_samples, load_state
from ml.trainer import train_model, job_status

router = APIRouter()

class TrainParams(BaseModel):
    epochs: int = 10
    face_mode: bool = False
    confidence_threshold: float = 0.75

@router.post("/train")
def train(background_tasks: BackgroundTasks, params: TrainParams = TrainParams()):
    import ml.model
    ml.model.face_mode = params.face_mode
    ml.model.confidence_threshold = params.confidence_threshold
    background_tasks.add_task(train_model, params.epochs)
    return {"status": "training_started"}

@router.get("/training_status")
def get_training_status():
    return job_status