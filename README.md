# Google Teachable (Backend)

Backend FastAPI app for image upload, training, and prediction.

Quick start

1. Create and activate a virtualenv, then install requirements:

```bash
python -m venv venv
venv\Scripts\Activate.ps1  # PowerShell on Windows
pip install -r backend/requirements.txt
```

2. Run the API from the `backend/` folder:

```bash
cd backend
uvicorn main:app --reload
```

3. Open the interactive docs at http://127.0.0.1:8000/docs

Testing

Run tests with:

```bash
pytest -q
```
