"""Minimal development FastAPI app for quick smoke tests.

Use this with:
  .\.venv\Scripts\python.exe -m uvicorn agi_chatbot.dev_api:app --host 127.0.0.1 --port 8000

It avoids the heavy startup logic in the main `api_server.py`.
"""

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

app = FastAPI(title="AGI Chatbot API (dev)", default_response_class=ORJSONResponse)


@app.get("/health")
async def health_check():
    return {"status": "ok", "dev_mode": True}


@app.get("/ready")
async def ready_check():
    return {"ready": True, "dev_mode": True}
