FROM python:3.11-slim

# System deps for OCR + PDF text extraction + image support
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-swe poppler-utils \
    libjpeg62-turbo-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent dirs (Render disk)
ENV UPLOAD_DIR=/data/uploads \
    GENERATED_DIR=/data/generated \
    CORS_ALLOW_ORIGINS=https://YOUR-VERCEL-DOMAIN.vercel.app \
    OCR_MODE=auto

RUN mkdir -p $UPLOAD_DIR $GENERATED_DIR

EXPOSE 8000
CMD uvicorn docs_master:app --host 0.0.0.0 --port ${PORT:-8000}

