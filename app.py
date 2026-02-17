import ocrmypdf
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="OCRmyPDF Web Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
            "jobs": 1,
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
        if options.get("remove_background"):
            kwargs["remove_background"] = True
        if options.get("clean"):
            kwargs["clean"] = True

        pages = options.get("pages", "").strip()
        if pages:
            kwargs["pages"] = pages

        ocrmypdf.ocr(str(input_path), str(output_path), **kwargs)
        jobs[job_id]["status"] = "done"
        logger.info(f"Job {job_id} completed: {output_path.name}")
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Job {job_id} failed: {e}")
    finally:
        if input_path.exists():
            input_path.unlink()


@app.post("/api/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("normal"),
    pages: str = Form(""),
    language: str = Form("deu+eng"),
    deskew: str = Form("true"),
    rotate_pages: str = Form("false"),
    remove_background: str = Form("false"),
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
        "remove_background": remove_background.lower() == "true",
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