from fastapi import FastAPI
from routes.upload import router as upload_router
from routes.train import router as train_router
from routes.predict import router as predict_router
from routes.export import router as export_router
from routes.reset import router as reset_router
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

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


@app.get("/")
def home():
    return {"message": "Teachable Machine Backend Running"}
