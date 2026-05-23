"""
Run this file instead of `python -m uvicorn main:app --reload`
It starts uvicorn programmatically and EXCLUDES the dataset/ and storage/
directories from the file watcher so that saving images or state JSON
does NOT trigger a server restart mid-training.

Usage:
    python run.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["routes", "core"],          # ONLY watch route/core changes
        reload_excludes=["dataset", "storage", "ml"],  # NEVER reload on ml/ or data writes
    )
