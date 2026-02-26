import ocrmypdf
import uuid
import os
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging

# Evaluation
import pdfplumber
from pathlib import Path
import unicodedata
import re
from rapidfuzz import fuzz, process
import spacy
import json
from spellchecker import SpellChecker
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
(BASE_DIR / "dict").mkdir(exist_ok=True)

app = FastAPI(title="OCRmyPDF Web Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store  { job_id: { status, filename, output_path, error, options } }
jobs: dict[str, dict] = {}


def run_ocr(job_id: str, input_path: Path, output_path: Path, options: dict):
    jobs[job_id]["status"] = "processing"
    try:
        kwargs = {
            "deskew": options.get("deskew", True),
            "language": options.get("language", "deu+eng"),
            "optimize": int(options.get("optimize", 1)),
            "jobs": os.cpu_count(),
        }

        mode = options.get("mode", "normal")
        if mode == "force":
            kwargs["force_ocr"] = True
        elif mode == "skip":
            kwargs["skip_text"] = True
        elif mode == "redo":
            kwargs["redo_ocr"] = True

        if options.get("rotate_pages"):
            kwargs["rotate_pages"] = True
        # if options.get("remove_background"):
        #     kwargs["remove_background"] = True
        if options.get("clean"):
            kwargs["clean"] = True

        pages = options.get("pages", "").strip()
        if pages:
            kwargs["pages"] = pages

        ocrmypdf.ocr(str(input_path), str(output_path), **kwargs)
        jobs[job_id]["status"] = "done"
        logger.info(f"Job {job_id} completed: {output_path.name}")
        logger.info(f"kwargs {kwargs}")
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"Job {job_id} failed: {e}")
    finally:
        if input_path.exists():
            input_path.unlink()

        # evaluation nach Abschluss der OCR durchführen und loggen
        if jobs[job_id]["status"] == "done":
            eval_result = evaluate_pdf(output_path, vocab, known_names)
            jobs[job_id]["eval"] = eval_result
            logger.info(
                f"Job {job_id} eval: "
                f"recognition={eval_result['overall_recognition_rate']:.1%} "
                f"exact={eval_result['vocabulary']['exact_hits']} "
                f"fuzzy={eval_result['vocabulary']['fuzzy_hits']} "
                f"tokens={eval_result['token_count']} "
                f"char_count={eval_result['char_count']} "
                f"misses={eval_result['vocabulary']['misses']} "
                f"suspicious_names={len(eval_result['proper_nouns']['suspicious'])} "
                f"top_misses={eval_result['vocabulary']['top_misses']}"
            )
            logger.info(f"Compound cache: {is_valid_compound.cache_info()}")

### Evaluation functions ###

def extract_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )

def load_wordlist(path: Path) -> set[str]:
    with open(path) as f:
        return {line.strip().lower() for line in f if line.strip()}
    
def build_vocab() -> set[str]:
    vocab = set()
    
    # 1. Deutsche Grundwortliste (z.B. aus wordfreq oder hunspell)
    from wordfreq import top_n_list
    vocab.update(str(w) for w in top_n_list("de", 50000))
    vocab.update(str(w) for w in top_n_list("en", 20000))
    
    # 2. Domain-spezifische Wörter (z.B. aus bekannten Dokumenten)
    vocab.update(load_wordlist(BASE_DIR / "dict" / "domain_vocab.txt"))
    
    return vocab

def split_proper_nouns(text: str) -> tuple[list[str], list[str]]:
    # Trennstriche am Zeilenende zusammenführen + Normalisierung
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"-\n", "", text)
    
    doc = nlp(text[:100_000])
    
    # Eigennamen sammeln
    proper_nouns = set()
    for ent in doc.ents:
        if ent.label_ in ("PER", "ORG", "LOC", "GPE"):
            for token in ent.text.split():
                proper_nouns.add(token.lower())
    
    # Tokens mit Lemmatisierung – ein Durchlauf für alles
    regular = []
    names = []
    for token in doc:
        if not token.is_alpha:       # Zahlen, Sonderzeichen raus
            continue
        if token.is_stop:            # Stoppwörter raus (der, die, das, ...)
            continue
        if len(token.text) < 2:      # Einzelbuchstaben raus
            continue
        
        lemma = token.lemma_.lower()
        
        if lemma in proper_nouns:
            names.append(lemma)
        else:
            regular.append(lemma)
    
    return regular, names

def evaluate_against_vocab(
    tokens: list[str],
    vocab: set[str],
    fuzzy_threshold: int = 85
) -> dict:
    exact_hits = 0
    fuzzy_hits = 0
    misses = []

    for token in tokens:
        if (token in vocab or is_valid_compound(token)):
            exact_hits += 1
        else:
            # Fuzzy-Match für OCR-typische Fehler (0→O, l→1, etc.)
            match, score, _ = process.extractOne(
                token, vocab, scorer=fuzz.ratio
            )
            if score >= fuzzy_threshold:
                fuzzy_hits += 1
            else:
                misses.append((token, str(match), score))

    total = len(tokens)
    return {
        "total_tokens": total,
        "exact_hits": exact_hits,
        "fuzzy_hits": fuzzy_hits,
        "misses": len(misses),
        "exact_rate": exact_hits / total if total else 0,
        "recognition_rate": (exact_hits + fuzzy_hits) / total if total else 0,
        "top_misses": sorted(misses, key=lambda x: x[2])[:20],
    }

def evaluate_proper_nouns(
    name_tokens: list[str],
    known_names: set[str]
) -> dict:  
    exact_hits = 0
    plausible = []
    suspicious = []
    
    for name in name_tokens:
        if name in known_names:
            exact_hits += 1
        # OCR-Artefakte in Namen erkennen
        if re.search(r"[0-9|\\/{}<>]", name):
            suspicious.append(name)
        elif len(name) < 2:
            suspicious.append(name)
        else:
            plausible.append(name)
    
    return {
        "exact_hits": exact_hits,
        "total_names": len(name_tokens),
        "plausible": len(plausible),
        "suspicious": suspicious,
        "plausibility_rate": len(plausible) / len(name_tokens) if name_tokens else 0
    }

def evaluate_pdf(pdf_path: Path, vocab: set[str], known_names: set[str] = None) -> dict:
    # Laden der Datei
    text = extract_text(pdf_path)
    # Aufteilung in reguläre Tokens (evtl. bekannte Worte) und Eigennamen
    regular_tokens, name_tokens = split_proper_nouns(text)
    
    # Prüfung der regulären Tokens gegen das Vokabular
    vocab_result = evaluate_against_vocab(regular_tokens, vocab)
    name_result = evaluate_proper_nouns(name_tokens, known_names or set())
    
    # Gesamt-Score: Namen bekommen z.B. 20% Gewicht
    overall = (
        vocab_result["recognition_rate"] * 0.8 +
        name_result["plausibility_rate"] * 0.2
    )
    
    return {
        "file": pdf_path.name,
        "overall_recognition_rate": round(overall, 4),
        "vocabulary": vocab_result,
        "proper_nouns": name_result,
        "char_count": len(text),
        "token_count": len(regular_tokens) + len(name_tokens),
    }

# Cache für zusammengesetzte Wörter, um wiederholte Checks zu beschleunigen
@lru_cache(maxsize=10000)
# Hilfsfunktion, um zusammengesetzte Wörter zu erkennen (z.B. "Donaudampfschifffahrtsgesellschaft")
def is_valid_compound(token: str) -> bool:
    if len(token) < 6:  # kurze Wörter nicht als Komposita behandeln
        return False
    # Wenn unbekannt, versuche das Wort in Teile zu zerlegen
    if token in spell:
        return True
    # Brute-force: prüfe alle möglichen Splits
    for i in range(3, len(token) - 2):
        if token[:i] not in spell.unknown([token[:i]]) and \
           token[i:] not in spell.unknown([token[i:]]):
            return True
    return False
    

### Evaluation setup ###
logger.info("Building vocabulary and loading known names...")
vocab = build_vocab()
known_names = load_wordlist(BASE_DIR / "dict" / "names.txt")
nlp = spacy.load("de_core_news_lg")
spell = SpellChecker(language="de")

### API Endpoints ###
@app.post("/api/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("normal"),
    pages: str = Form(""),
    language: str = Form("deu+eng"),
    deskew: str = Form("true"),
    rotate_pages: str = Form("false"),
    # remove_background: str = Form("false"),
    clean: str = Form("false"),
    optimize: str = Form("1"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    job_id = str(uuid.uuid4())
    safe_name = Path(file.filename).stem[:60]
    input_path = UPLOAD_DIR / f"{job_id}_{safe_name}.pdf"
    output_path = OUTPUT_DIR / f"{safe_name}_ocr.pdf"

    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 200 MB limit.")

    with open(input_path, "wb") as f:
        f.write(content)

    options = {
        "mode": mode,
        "pages": pages,
        "language": language,
        "deskew": deskew.lower() == "true",
        "rotate_pages": rotate_pages.lower() == "true",
        # "remove_background": remove_background.lower() == "true",
        "clean": clean.lower() == "true",
        "optimize": optimize,
    }

    jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "output_name": output_path.name,
        "output_path": str(output_path),
        "error": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_ocr, job_id, input_path, output_path, options)

    return JSONResponse({"job_id": job_id, "filename": file.filename})


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "error": job.get("error"),
        "eval": job.get("eval"),
    }


@app.get("/api/download/{job_id}")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not completed yet.")
    output_path = Path(job["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(
        path=str(output_path),
        media_type="application/pdf",
        filename=job["output_name"],
        headers={"Content-Disposition": f'attachment; filename="{job["output_name"]}"'},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")



