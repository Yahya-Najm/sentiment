import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID        = "yahyasafdari/setiment_prediction"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    app.state.model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    app.state.model.eval()
    yield


app = FastAPI(title="Sentiment Prediction API", lifespan=lifespan)

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
def predict(request: ReviewRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text field must not be empty")

    inputs = app.state.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logits = app.state.model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()

    id2label = app.state.model.config.id2label
    scores   = {id2label[i]: float(p) for i, p in enumerate(probs)}
    label    = max(scores, key=scores.get)

    return PredictionResponse(
        label=label,
        confidence=scores[label],
        neg_prob=scores.get("neg", 0.0),
        pos_prob=scores.get("pos", 0.0),
    )
