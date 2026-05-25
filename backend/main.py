from fastapi import FastAPI
from routes.upload import router as upload_router
from routes.train import router as train_router
from routes.predict import router as predict_router
from routes.export import router as export_router
from routes.reset import router as reset_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ✅ NOTE: We do NOT wipe dataset/storage on startup anymore.
# Data is only cleared when the user explicitly calls GET /api/reset.
# Wiping on startup caused all uploaded images to be lost on every server restart.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api")
app.include_router(train_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(export_router, prefix="/api")
app.include_router(reset_router, prefix="/api")

@app.get("/api")
def api_home():
    return {"message": "Teachable Machine Backend API Running"}

# Mount frontend at the root (must be placed AFTER API routes to avoid capturing /api requests)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
