import os
import re
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "app_data"
UPLOAD_DIR = DATA_DIR / "uploads"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
STATIC_DIR = BASE_DIR / "static"
TRANSCRIBE_SCRIPT = BASE_DIR / "transcribe.py"

ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large"}
ALLOWED_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".m4a",
    ".mpeg",
    ".mpga",
    ".wav",
    ".webm",
    ".mov",
    ".aac",
    ".flac",
    ".ogg",
}

app = FastAPI(title="Meeting Transcript Generator")
executor = ThreadPoolExecutor(max_workers=int(os.environ.get("TRANSCRIBE_WORKERS", "1")))
jobs = {}
jobs_lock = threading.Lock()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename):
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", filename).strip(" .")
    return cleaned or "recording"


def public_job(job):
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "speakers": job["speakers"],
        "model": job["model"],
        "language": job["language"],
        "error": job.get("error"),
        "download_url": f"/api/jobs/{job['id']}/download" if job["status"] == "completed" else None,
        "text_url": f"/api/jobs/{job['id']}/text" if job["status"] == "completed" else None,
    }


def set_job(job_id, **updates):
    with jobs_lock:
        job = jobs[job_id]
        job.update(updates)
        job["updated_at"] = now_iso()


def run_transcription(job_id):
    with jobs_lock:
        job = jobs[job_id]
        input_path = Path(job["input_path"])
        output_path = Path(job["output_path"])

    set_job(job_id, status="running", error=None)

    command = [
        sys.executable,
        "-u",
        str(TRANSCRIBE_SCRIPT),
        str(input_path),
        "--output",
        str(output_path),
        "--overwrite",
        "--no-beep",
        "--quiet",
        "--model",
        job["model"],
    ]

    if job["language"]:
        command.extend(["--language", job["language"]])

    if job["speakers"]:
        command.append("--speakers")
        if job["speaker_names"]:
            command.extend(["--speaker-names", job["speaker_names"]])

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".cache" / "matplotlib"))
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    try:
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("TRANSCRIBE_TIMEOUT_SECONDS", "7200")),
        )
    except Exception as error:
        set_job(job_id, status="failed", error=str(error))
        return

    log = "\n".join(part for part in [completed.stdout, completed.stderr] if part.strip())

    if completed.returncode != 0:
        set_job(job_id, status="failed", error=log or f"Command failed with exit code {completed.returncode}")
        return

    if not output_path.is_file():
        set_job(job_id, status="failed", error="Transcription finished but no transcript file was created.")
        return

    set_job(job_id, status="completed", error=None, log=log)


@app.on_event("startup")
def startup():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    speakers: bool = Form(False),
    speaker_names: str = Form(""),
    model: str = Form("base"),
    language: str = Form("en"),
):
    if model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported Whisper model.")

    original_name = safe_filename(file.filename or "recording")
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    job_id = uuid.uuid4().hex
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_upload_dir / original_name
    output_path = TRANSCRIPT_DIR / f"{job_id}.txt"

    with input_path.open("wb") as target:
        while chunk := await file.read(1024 * 1024):
            target.write(chunk)

    job = {
        "id": job_id,
        "filename": original_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "queued",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "speakers": speakers,
        "speaker_names": speaker_names.strip(),
        "model": model,
        "language": language.strip() or None,
        "error": None,
        "log": "",
    }

    with jobs_lock:
        jobs[job_id] = job

    executor.submit(run_transcription, job_id)
    return public_job(job)


@app.get("/api/jobs")
def list_jobs():
    with jobs_lock:
        ordered = sorted(jobs.values(), key=lambda item: item["created_at"], reverse=True)
        return [public_job(job) for job in ordered]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return public_job(job)


@app.get("/api/jobs/{job_id}/text")
def get_transcript_text(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Transcript is not ready yet.")

    output_path = Path(job["output_path"])
    return {"text": output_path.read_text(encoding="utf-8")}


@app.get("/api/jobs/{job_id}/download")
def download_transcript(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Transcript is not ready yet.")

    output_path = Path(job["output_path"])
    download_name = f"{Path(job['filename']).stem}-transcript.txt"
    return FileResponse(output_path, media_type="text/plain", filename=download_name)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
