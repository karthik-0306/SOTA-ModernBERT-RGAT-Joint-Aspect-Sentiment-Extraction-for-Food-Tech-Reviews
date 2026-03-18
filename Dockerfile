FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# HF Spaces uses port 7860
EXPOSE 7860

# Environment variables (override in docker-compose or at runtime)
ENV PYTHONUNBUFFERED=1
ENV HF_MODEL_REPO="karthik0306/ModernBERT-RGAT-ABSA"
ENV HF_TOKEN=""

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "7860"]
