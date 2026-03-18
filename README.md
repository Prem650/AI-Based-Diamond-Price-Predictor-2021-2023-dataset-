# Diamond Price Analyser

## Features
- ML-powered diamond price prediction (RandomForest, R² 0.976)
- FastAPI backend
- Responsive frontend with video BG

## Local Run
```bash
uv sync
uv run python -m uvicorn app_api:app --reload
```
Frontend at http://localhost:8000 (static server needed or serve files)

## Deploy
1. GitHub repo from this dir
2. Railway: New Project → GitHub → auto-deploy with Procfile/requirements.txt
3. Set frontend API_URL to Railway backend URL

## Test
Fill form at enter.html, submit → result.html
