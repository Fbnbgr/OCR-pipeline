# OCR Pipeline with WebUI/CLI

Anwendung, die mitthilfe von [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF)/Tesseract PDF input verarbeitet.

## Schnellstart
git clone https://github.com/Fbnbgr/OCR-pipeline
docker compose build
docker compose up -d

Die App ist verfügbar unter: **http://localhost:8000**

branch cli mit:
docker compose run --rm ocr-pipeline
-> hier werden alle PDFs aus dem Ordner input verarbeitet

## Techstack
FastAPI
uvicorn
asyncio

## Unsterstütze Sprachen
- deu, eng, chi_sim, chi_tra