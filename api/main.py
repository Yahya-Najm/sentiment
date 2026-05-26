import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_ID    = "yahyasafdari/setiment_prediction"
HF_API_URL  = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN    = os.getenv("HF_TOKEN", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")]

app = FastAPI(title="Sentiment Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label:      str
    confidence: float
    neg_prob:   float
    pos_prob:   float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ReviewRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text field must not be empty")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text},
        )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"HF Inference API error: {response.text}")

    # HF returns [[{"label": "pos", "score": 0.97}, {"label": "neg", "score": 0.03}]]
    scores = {item["label"]: item["score"] for item in response.json()[0]}
    label  = max(scores, key=scores.get)

    return PredictionResponse(
        label=label,
        confidence=scores[label],
        neg_prob=scores.get("neg", 0.0),
        pos_prob=scores.get("pos", 0.0),
    )
