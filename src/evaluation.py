import pdfplumber
from pathlib import Path
import unicodedata
import re
from rapidfuzz import fuzz, process
import spacy
import json
from spellchecker import SpellChecker
from functools import lru_cache
import ollama
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DICT_DIR = BASE_DIR / "dict"
DICT_DIR.mkdir(exist_ok=True)

### Evaluation functions ###
# Vergleich der Ergebnisse vor und nach LLM-Korrektur
def compare_eval_rounds(result: dict) -> dict:
    v1 = result["vocabulary"]
    v2 = result["vocabulary_final"]
    n1 = result["proper_nouns"]
    n2 = result["proper_nouns_final"]

    # Token-Unterschiede
    hits1 = set(v1["hits_all"])
    hits2 = set(v2["hits_all"])
    fuzzy1 = set(v1["fuzzy_hits_all"])
    fuzzy2 = set(v2["fuzzy_hits_all"])
    misses1 = set(v1["misses_all"])
    misses2 = set(v2["misses_all"])

    # Vergleich
    newly_correct   = misses1 & hits2        
    newly_fuzzy     = misses1 & fuzzy2      
    newly_broken    = (hits1 | fuzzy1) & misses2 
    still_missing   = misses1 & misses2 

    # Score-Delta
    score_delta = round(result["final_score"] - result["initial_score"], 4)
    rate_delta  = round(v2["recognition_rate"] - v1["recognition_rate"], 4)

    comparison = {
        "score_initial":   result["initial_score"],
        "score_final":     result["final_score"],
        "score_delta":     score_delta,
        "improved":        score_delta > 0,

        "recognition_rate_initial": round(v1["recognition_rate"], 4),
        "recognition_rate_final":   round(v2["recognition_rate"], 4),
        "recognition_rate_delta":   rate_delta,

        "tokens": {
            "newly_correct":  sorted(newly_correct),   # LLM hat OCR-Fehler gefixt
            "newly_fuzzy":    sorted(newly_fuzzy),     # LLM hat teilweise gefixt
            "newly_broken":   sorted(newly_broken),    # LLM hat Wörter kaputt gemacht
            "still_missing":  sorted(still_missing),   # unverändert schlecht
        },
        "counts": {
            "newly_correct":  len(newly_correct),
            "newly_fuzzy":    len(newly_fuzzy),
            "newly_broken":   len(newly_broken),
            "still_missing":  len(still_missing),
        },
        "proper_nouns": {
            "suspicious_initial": n1["suspicious"],
            "suspicious_final":   n2["suspicious"],
            "newly_clean": list(set(n1["suspicious"]) - set(n2["suspicious"])),
        }
    }

    # Logging
    logger.info(f"Score-Delta: {score_delta:+.4f} ({'besser' if score_delta > 0 else 'schlechter'})")
    logger.info(f"Neu korrekt: {len(newly_correct)} | Neu fuzzy: {len(newly_fuzzy)} | Verschlimmbessert: {len(newly_broken)}")
    if newly_broken:
        logger.warning(f"LLM hat diese Tokens verschlimmbessert: {sorted(newly_broken)}")

    return comparison

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
    vocab.update(load_wordlist(DICT_DIR / "domain_vocab.txt"))
    logger.info(f"Länge des gegengeprüften Vokabulars: {len(vocab)}")
    
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
        if re.search(r"[^a-zA-ZäöüÄÖÜß]", token.text):  # Sonderzeichen raus
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
    exact_hits = []
    fuzzy_hits = []
    misses = []

    for token in tokens:
        if not token.isalpha():
            continue
        if (token in vocab or is_valid_compound(token)):
            exact_hits.append(token)
        else:
            # Fuzzy-Match für OCR-typische Fehler (0→O, l→1, etc.)
            match, score, _ = process.extractOne(
                token, vocab, scorer=fuzz.ratio
            )
            if score >= fuzzy_threshold:
                fuzzy_hits.append((token, str(match), score))
            else:
                misses.append((token, str(match), score))

    total = len(tokens)

    # Misses: niedrigster Score zuerst = schlimmste Fehler
    misses = list(set(
        m[0] for m in sorted(misses, key=lambda x: x[2])
    ))

    # Fuzzy: niedrigster Score zuerst = unsicherste Treffer
    fuzzy_hits = list(set(
        m[0] for m in sorted(fuzzy_hits, key=lambda x: x[2])
    ))
    logger.info(f"exact_hits: {exact_hits}")
    logger.info(f"fuzzy_hits: {fuzzy_hits}")

    return {
        "total_tokens": total,
        "exact_hits": len(exact_hits),
        "fuzzy_hits": len(fuzzy_hits),
        "misses": len(misses),
        "exact_rate": len(exact_hits) / total if total else 0,
        "recognition_rate": (len(exact_hits) + len(fuzzy_hits)) / total if total else 0,
        # "top_misses": sorted(misses, key=lambda x: x[2])[:20],
        "misses_all": misses,
        "hits_all": exact_hits,
        "fuzzy_hits_all": fuzzy_hits,
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

def evaluate_pdf(pdf_path: Path) -> dict:
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

    initial_score = round(overall, 4)
    logger.info(f"Ausgangsscore wurde ermittelt mit {initial_score}")

    logger.info(f"Starte LLM-Korrektur")
    text_final = correct_with_llm(text)

    # 2. Eva-Runde -> Evaluierung der LLM Ergebnisse
    logger.info(f"Starte 2. Eva-Runde")
    regular_tokens, name_tokens = split_proper_nouns(text_final)
    vocab_result_final = evaluate_against_vocab(regular_tokens, vocab)
    name_result_final = evaluate_proper_nouns(name_tokens, known_names or set())
    overall = (
        vocab_result_final["recognition_rate"] * 0.8 +
        name_result_final["plausibility_rate"] * 0.2
    )
    final_score = round(overall, 4)
    logger.info(f"2. Eva-Score wurde ermittelt mit {final_score}")
 
    eval_result = {
        "file": pdf_path.name,
        "initial_score": initial_score,
        "final_score": final_score,
        "vocabulary": vocab_result,
        "vocabulary_final": vocab_result_final,
        "proper_nouns": name_result,
        "proper_nouns_final": name_result_final,
        "char_count": len(text),
        "char_count_final": len(text_final),
        "token_count": len(regular_tokens) + len(name_tokens),
        "llm_corrections": text_final
    }
    
    # Vergleich der Ergebnisse
    eval_result["comparison"] = compare_eval_rounds(eval_result)

    return eval_result, text_final


# Cache für zusammengesetzte Wörter, um wiederholte Checks zu beschleunigen
@lru_cache(maxsize=10000)
# Hilfsfunktion, um zusammengesetzte Wörter zu erkennen
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

# Korrektur der OCR-Fehler mit LLM, basierend auf den erkannten Fehlern
async def correct_with_llm_async(text: str, max_concurrent: int = 5) -> str:
    chunk_size = 3000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    logger.info(f"LLM Korrektur: {len(chunks)} Chunks, max {max_concurrent} parallel")

    async_client = ollama.AsyncClient(host="http://host.docker.internal:11434")
    semaphore = asyncio.Semaphore(max_concurrent)  # begrenzt parallele Requests

    async def limited_correct(chunk, index):
        async with semaphore:
            return await correct_chunk(async_client, chunk, chunk_size, index)

    tasks = [limited_correct(chunk, i) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    # Reihenfolge sicherstellen
    results_sorted = sorted(results, key=lambda x: x[0])
    return "".join(text for _, text in results_sorted)
   
    
async def correct_chunk(async_client, chunk: str, chunk_size: int, index: int) -> tuple[int, str]:
    prompt = f"""Du bist ein OCR-Korrektur-Assistent.
    Analysiere den Text und korrigiere NUR offensichtliche OCR-Fehler.
    Erfinde KEINE Wörter. Behalte Eigennamen und Fremdwörter unverändert.
    Deine Antwort besteht NUR aus dem korrigierten Text.
    Text: {chunk}"""

    logger.info(f"Chunk {index + 1} gestartet")
    response = await async_client.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": chunk_size + 200}
    )
    return index, response.message.content.strip()

def correct_with_llm(text: str) -> str:
    return asyncio.run(correct_with_llm_async(text))

### Evaluation setup ###
logger.info("Building vocabulary and loading known names...")
try:
    vocab = build_vocab()
    known_names = load_wordlist(DICT_DIR / "names.txt")
    nlp = spacy.load("de_core_news_lg")
    spell = SpellChecker(language="de")
    client = ollama.Client(host="http://host.docker.internal:11434")
    logger.info("Setup complete.")
except Exception as e:
    import traceback
    logger.error(f"Startup failed: {e}")
    logger.error(traceback.format_exc())
    raise
