# Sledat OCR API
#
# POST /api/extract         submit image, returns { job_id }
# GET  /api/jobs/{job_id}   poll result, returns { status, ...result }
# GET  /health              { status, jobs_pending }
# GET  /metrics             Prometheus metrics
#
# Auth: X-API-Key header
#
# Per-request pipeline:
#   1. save original to storage_path/{job_id}_original.jpg
#   2. detect document outline and deskew
#   3. save result to storage_path/{job_id}_obdelana.jpg
#   4. run OCR (mode: bic / cmr / auto)
#   5. append row to telemetry.csv
#
# Config keys (config.json):
#   datalab_key    - DataLabs API key
#   keys           - { "api-key": { "rate_limit_per_minute": 60 } }
#   storage_path   - where to save images (default: ./images)
#   max_image_mb   - upload size limit in MB (default: 25)

import asyncio
import base64
import csv
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.routing import Mount

import func

# Prometheus metrics

_jobs_total = Counter(
    "sledat_jobs_total", "Total jobs processed", ["mode", "success"]
)
_processing_duration = Histogram(
    "sledat_processing_duration_seconds", "Job processing duration", ["mode"],
    buckets=[5, 10, 20, 30, 45, 60, 90, 120, 180],
)
_cmr_fields_filled = Histogram(
    "sledat_cmr_fields_filled", "CMR fields filled per job",
    buckets=[0, 4, 8, 12, 16, 20, 24],
)
_bic_confidence = Counter(
    "sledat_bic_confidence_total", "BIC extractions by confidence level", ["confidence"]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_FILE   = Path(__file__).parent / "config.json"
TELEMETRY_CSV = Path(__file__).parent / "telemetry.csv"

_IMAGE_MAGIC = [b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"GIF8", b"\x42\x4d"]
_csv_lock = Lock()


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception as e:
        log.error(f"Failed to load config: {e}")
        raise


def _validate_config(cfg: dict) -> None:
    for key in ("datalab_key", "keys"):
        if key not in cfg:
            raise RuntimeError(f"config.json missing key: {key!r}")
    if not cfg["datalab_key"]:
        raise RuntimeError("config.json: datalab_key is empty")
    if not cfg["keys"]:
        raise RuntimeError("config.json: no API keys defined")


def _storage_path() -> Path:
    cfg = _load_config()
    p = Path(cfg.get("storage_path", Path(__file__).parent / "images"))
    p.mkdir(parents=True, exist_ok=True)
    return p


# Telemetry CSV

_TELEMETRY_FIELDS = [
    "job_id", "timestamp_start", "timestamp_end",
    "duration_s", "mode", "bic", "fields_filled", "success",
]


def _log_telemetry(job_id: str, ts_start: float, ts_end: float,
                   mode: str, bic: str, fields_filled: int, success: bool) -> None:
    row = {
        "job_id":          job_id,
        "timestamp_start": datetime.fromtimestamp(ts_start).strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_end":   datetime.fromtimestamp(ts_end).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s":      round(ts_end - ts_start, 1),
        "mode":            mode,
        "bic":             bic,
        "fields_filled":   fields_filled,
        "success":         1 if success else 0,
    }
    write_header = not TELEMETRY_CSV.exists()
    with _csv_lock:
        with open(TELEMETRY_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_TELEMETRY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# Rate limiting (sliding window per API key)

_rate_buckets: dict[str, deque] = {}
_rate_lock = Lock()


def _allow_request(key: str, limit: int) -> bool:
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets.setdefault(key, deque())
        while bucket and bucket[0] < now - 60:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


def verify_key(x_api_key: str = Header(...)) -> str:
    cfg = _load_config()
    keys = cfg.get("keys", {})
    if x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    limit = keys[x_api_key].get("rate_limit_per_minute", 60)
    if not _allow_request(x_api_key, limit):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({limit} req/min)")
    return x_api_key


def verify_key_readonly(x_api_key: str = Header(...)) -> str:
    cfg = _load_config()
    if x_api_key not in cfg.get("keys", {}):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# In-memory job store, TTL 10 minutes

_jobs: dict[str, dict | None] = {}
_job_times: dict[str, float] = {}
_JOB_TTL = 600
_sem = asyncio.Semaphore(1)


def _store_job(job_id: str, result: dict | None) -> None:
    _jobs[job_id] = result
    _job_times[job_id] = time.time()


async def _cleanup_jobs() -> None:
    while True:
        await asyncio.sleep(120)
        now = time.time()
        expired = [jid for jid, t in list(_job_times.items()) if now - t > _JOB_TTL]
        for jid in expired:
            _jobs.pop(jid, None)
            _job_times.pop(jid, None)
        if expired:
            log.info(f"Evicted {len(expired)} expired job(s)")


app = FastAPI(
    title="Sledat OCR API",
    docs_url=None,
    redoc_url=None,
    routes=[Mount("/metrics", app=make_asgi_app())],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    ms = round((time.time() - t0) * 1000)
    log.info(f"{request.method} {request.url.path} -> {response.status_code} ({ms}ms)")
    return response


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.on_event("startup")
async def _startup():
    cfg = _load_config()
    _validate_config(cfg)
    func.DATALAB_API_KEY = cfg["datalab_key"]
    func.ANTHROPIC_API_KEY = cfg.get("anthropic_key", "")
    asyncio.create_task(_cleanup_jobs())
    log.info("Sledat API started")


@app.get("/health", include_in_schema=False)
def health():
    return {
        "status": "ok",
        "jobs_pending": sum(1 for v in _jobs.values() if v is None),
    }


async def _run_extract(job_id: str, image_bytes: bytes, mode: str, reject_items: list[str] | None = None) -> None:
    cfg = _load_config()
    func.DATALAB_API_KEY = cfg["datalab_key"]
    func.ANTHROPIC_API_KEY = cfg.get("anthropic_key", "")
    storage = _storage_path()
    ts_start = time.time()

    (storage / f"{job_id}_original.jpg").write_bytes(image_bytes)

    # Content filter — runs before OCR, cheap fast check
    if reject_items:
        try:
            rejected, reason = await asyncio.to_thread(func.check_image_content, image_bytes, reject_items)
            if rejected:
                log.info(f"[{job_id}] rejected: {reason}")
                _store_job(job_id, {"status": "rejected", "reason": reason})
                _log_telemetry(job_id, ts_start, time.time(), mode, "", 0, False)
                return
        except Exception as e:
            log.warning(f"[{job_id}] content check failed ({e}), proceeding anyway")

    # Crop independently — does not affect OCR input
    try:
        cropped = func.crop_document(image_bytes)
    except Exception as e:
        log.warning(f"[{job_id}] crop failed ({e}), skipping")
        cropped = None

    if cropped and cropped != image_bytes:
        (storage / f"{job_id}_obdelana.jpg").write_bytes(cropped)
        cropped_b64 = base64.b64encode(cropped).decode()
    else:
        cropped_b64 = None

    try:
        async with _sem:
            result = await asyncio.to_thread(func.run_pipeline, image_bytes, mode, job_id)
        if cropped_b64:
            result["cropped_image"] = cropped_b64
        _store_job(job_id, result)

        ts_end = time.time()
        actual_mode = result.get("mode", mode)
        bic = result.get("bic") or ""
        fields_filled = sum(1 for v in (result.get("fields") or {}).values() if v)
        confidence = result.get("confidence", "")

        _jobs_total.labels(mode=actual_mode, success="1").inc()
        _processing_duration.labels(mode=actual_mode).observe(ts_end - ts_start)
        if actual_mode == "cmr":
            _cmr_fields_filled.observe(fields_filled)
        if actual_mode == "bic" and confidence:
            _bic_confidence.labels(confidence=confidence).inc()

        _log_telemetry(job_id, ts_start, ts_end, actual_mode, bic, fields_filled, True)
        log.info(f"[{job_id}] done: mode={actual_mode!r} bic={bic!r}")

    except Exception as e:
        log.error(f"[{job_id}] error: {e}", exc_info=True)
        _store_job(job_id, {"error": str(e)})
        _jobs_total.labels(mode=mode, success="0").inc()
        _log_telemetry(job_id, ts_start, time.time(), mode, "", 0, False)


@app.post("/api/extract")
async def post_extract(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    reject_if: str = Form(""),
    _key: str = Depends(verify_key),
):
    """Submit an image. Returns { job_id }. Poll /api/jobs/{job_id} for result.

    reject_if: optional comma-separated list of objects that should not appear
               in the image (e.g. 'cigarette,pen'). If found, job returns
               { status: rejected, reason: '...' } without running OCR.
    """
    if mode not in ("auto", "bic", "cmr"):
        raise HTTPException(status_code=400, detail="mode must be auto, bic, or cmr")

    cfg = _load_config()
    max_mb = cfg.get("max_image_mb", 25)
    image_bytes = await file.read()

    if len(image_bytes) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {max_mb} MB")
    if not any(image_bytes.startswith(sig) for sig in _IMAGE_MAGIC):
        raise HTTPException(status_code=415, detail="File does not appear to be a valid image")

    reject_items = [x.strip() for x in reject_if.split(",") if x.strip()] if reject_if else []

    job_id = str(uuid.uuid4())
    _store_job(job_id, None)
    asyncio.create_task(_run_extract(job_id, image_bytes, mode, reject_items))
    log.info(f"[{job_id}] queued mode={mode!r} reject_if={reject_items}")
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, _key: str = Depends(verify_key_readonly)):
    """Returns { status: pending } or { status: done, ...result }."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    result = _jobs[job_id]
    if result is None:
        return {"status": "pending"}
    del _jobs[job_id]
    _job_times.pop(job_id, None)
    if result.get("status") == "rejected":
        return result
    return {"status": "done", **result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
