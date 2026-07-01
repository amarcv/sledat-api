# Sledat OCR API

OCR backend for reading shipping container BIC codes and CMR transport documents from phone camera images.

Uses [DataLabs](https://datalab.to) for text extraction. Includes document deskewing, image archiving, CSV telemetry, and a Prometheus/Grafana monitoring stack.

---

## How it works

Each request goes through the following steps:

1. Original image saved to disk as `{job_id}_original.jpg`
2. Document outline detected and image deskewed
3. Deskewed image saved as `{job_id}_obdelana.jpg`
4. OCR run via DataLabs API
5. Result stored and telemetry row appended to `telemetry.csv`

Processing is async. Submit a job, get a `job_id`, poll until done.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Edit `config.json`:

```json
{
  "datalab_key": "your-datalab-api-key",
  "storage_path": "/path/to/image/storage",
  "max_image_mb": 25,
  "keys": {
    "your-api-key": {
      "label": "miha",
      "rate_limit_per_minute": 60
    }
  }
}
```

Start the server:

```bash
python app.py
```

Runs on port 8000.

---

## API

All requests require `X-API-Key` header.

### POST /api/extract

Submit an image for processing.

**Form fields:**
- `file` - image file (JPEG or PNG)
- `mode` - `auto` (default), `bic` (container only), or `cmr` (document only)

**Response:**
```json
{ "job_id": "550e8400-e29b-41d4-a716-446655440000" }
```

### GET /api/jobs/{job_id}

Poll for result.

**Pending:**
```json
{ "status": "pending" }
```

**BIC result:**
```json
{
  "status": "done",
  "mode": "bic",
  "bic": "MSCU7263541",
  "confidence": "exact",
  "edit_distance": 0,
  "ocr_raw": "MSCU 726354 1"
}
```

**CMR result:**
```json
{
  "status": "done",
  "mode": "cmr",
  "fields": {
    "box1_sender": "...",
    "box2_consignee": "...",
    "box3_place_of_delivery": "...",
    ...
  }
}
```

Jobs expire after 10 minutes.

### GET /health

```json
{ "status": "ok", "jobs_pending": 2 }
```

---

## Modes

| Mode | What it does | DataLabs tier | Approx. time |
|------|-------------|---------------|--------------|
| `bic` | Reads container code | fast | 10-30s |
| `cmr` | Reads transport document | balanced | 30-90s |
| `auto` | Tries BIC first, falls back to CMR if not found | both | 20-90s |

---

## Telemetry

Every completed job appends a row to `telemetry.csv`:

| Column | Description |
|--------|-------------|
| `job_id` | Unique job identifier |
| `timestamp_start` | When processing started |
| `timestamp_end` | When processing finished |
| `duration_s` | Total processing time in seconds |
| `mode` | `bic` or `cmr` |
| `bic` | Extracted BIC code (empty for CMR jobs) |
| `fields_filled` | Number of CMR fields extracted (out of 24) |
| `success` | 1 = ok, 0 = error |

---

## Monitoring

Prometheus and Grafana run via Docker Compose:

```bash
cd monitoring
docker compose up -d
```

- Grafana: http://localhost:3000 (admin / sledat)
- Prometheus: http://localhost:9090

The Grafana dashboard loads automatically and shows job counts, processing times, CMR field fill rates, and BIC confidence breakdown.

Prometheus scrapes `/metrics` on the API every 15 seconds.
