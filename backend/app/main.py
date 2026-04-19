import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import classify, compose, speak

load_dotenv()

app = FastAPI(title="ASL Translator API", version="0.1.0")

frontend_origins = [
    o.strip()
    for o in os.getenv("FRONTEND_ORIGIN", "http://localhost:5173").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify.router, prefix="/api", tags=["classify"])
app.include_router(compose.router, prefix="/api", tags=["compose"])
app.include_router(speak.router, prefix="/api", tags=["speak"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
