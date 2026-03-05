FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ghostscript \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-frk \
    unpaper \
    poppler-utils \
    jbig2dec \
    pngquant \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.7.0/de_core_news_lg-3.7.0-py3-none-any.whl

# Copy app files
COPY . .

# Create working directories
RUN mkdir -p uploads output logs

EXPOSE 8000

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.ocr:app", "--host", "0.0.0.0", "--port", "8000"]