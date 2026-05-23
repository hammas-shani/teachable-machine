import os
import shutil
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from core.state import save_state, load_state
import ml.model

router = APIRouter()

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BACKEND_DIR, "dataset")

@router.get("/reset")
async def reset_project():
    """Wipes the dataset folder and project state, restoring a fresh start."""
    try:
        # 1. Delete dataset folder
        if os.path.exists(DATASET_PATH):
            shutil.rmtree(DATASET_PATH)
        os.makedirs(DATASET_PATH, exist_ok=True)
        
        # 2. Reset project state
        state = load_state()
        state["classes"] = {}
        state["isTrained"] = False
        save_state(state)
        
        # 3. Reset ML Model in memory
        ml.model.is_trained = False
        from ml.model import CentroidClassifier
        ml.model.model = CentroidClassifier(threshold=1.0)  # fresh model
        ml.model.model.classes_ = []
        
        return JSONResponse(status_code=200, content={"message": "Project reset successfully."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
