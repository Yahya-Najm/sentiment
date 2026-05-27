FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; m='yahyasafdari/setiment_prediction'; AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m)"

COPY --chown=user main.py .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
