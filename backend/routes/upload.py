import os
import shutil
import uuid

from typing import List, Annotated

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form
)
from core.state import add_samples, set_trained, remove_class
import ml.model

router = APIRouter()

# Always resolve dataset path relative to THIS file's directory (backend root)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BACKEND_DIR, "dataset")


@router.post("/upload")
async def upload(
    class_name: Annotated[str, Form(...)],
    files: Annotated[List[UploadFile], File(...)]
):
    class_folder = os.path.join(DATASET_PATH, class_name)
    os.makedirs(class_folder, exist_ok=True)
    saved_files = []

    for file in files:
        ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "jpg"
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        # Use absolute path — so state JSON is always portable & resolvable
        file_path = os.path.join(class_folder, unique_name)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        add_samples(class_name, file_path)  # Store absolute path in state
        saved_files.append(unique_name)

    # Invalidate current model since new data was added
    set_trained(False)
    ml.model.is_trained = False

    return {
        "message": "uploaded successfully",
        "class_name": class_name,
        "files": saved_files
    }

@router.delete("/class/{class_name}")
async def delete_class(class_name: str):
    # Remove from state
    remove_class(class_name)
    
    # Invalidate current model since data was removed
    set_trained(False)
    ml.model.is_trained = False
    
    # Delete dataset folder
    class_folder = os.path.join(DATASET_PATH, class_name)
    if os.path.exists(class_folder):
        shutil.rmtree(class_folder)
        
    return {"message": f"Class {class_name} deleted successfully"}
