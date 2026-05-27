---
title: Sentiment API
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Sentiment Prediction API

FastAPI service serving the [`yahyasafdari/setiment_prediction`](https://huggingface.co/yahyasafdari/setiment_prediction) DistilBERT model in-process.

## Endpoints

- `GET /health` — service info
- `POST /predict` — body `{"text": "..."}` → `{label, confidence, neg_prob, pos_prob}`

## Configuration

`ALLOWED_ORIGINS` — comma-separated list of CORS origins. Set in Space **Settings → Variables and secrets**. Defaults to `http://localhost:3000`.
