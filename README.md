# StrikeNet API

This FastAPI service accepts user-uploaded wildlife photos, submits them to an OpenAI vision-capable model, and reports whether the detected species is invasive in South Florida.

## Getting Started

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables** (prefix `STRIKENET_`):
   - `STRIKENET_OPENAI_API_KEY` *(required)* – your OpenAI API key with access to vision models.
   - `STRIKENET_OPENAI_MODEL` – defaults to `gpt-4o-mini`.
   - `STRIKENET_OPENAI_TEMPERATURE` – defaults to `0.0` for deterministic output.
   - `STRIKENET_OPENAI_MAX_OUTPUT_TOKENS` – defaults to `600`.
   - `STRIKENET_CLASSIFICATION_CONFIDENCE_THRESHOLD` – defaults to `0.6`.
   - `STRIKENET_TOP_K` – defaults to `5` predictions.

   Example:
   ```bash
   export STRIKENET_OPENAI_API_KEY="sk-..."
   export STRIKENET_OPENAI_MODEL="gpt-4o-mini"
   ```
4. **Run the API**:
   ```bash
   uvicorn app.main:app --reload
   ```

## API

### `POST /api/classify`
Uploads an image and returns model predictions annotated with invasive-species metadata.

- **Body**: `multipart/form-data` with `image` field (JPEG/PNG/etc.).
- **Response** (`200 OK`):
  ```json
  {
    "decision": "auto-flagged-invasive",
    "invasive": true,
    "threshold": 0.6,
    "top_prediction": {
      "label": "red lionfish",
      "score": 0.92,
      "species": {
        "common_name": "Red Lionfish",
        "scientific_name": "Pterois volitans",
        "is_invasive": true,
        "notes": "Aggressive invasive predator known to disrupt reef ecosystems."
      }
    },
    "predictions": [
      {
        "label": "red lionfish",
        "score": 0.92,
        "species": {
          "common_name": "Red Lionfish",
          "scientific_name": "Pterois volitans",
          "is_invasive": true,
          "notes": "Aggressive invasive predator known to disrupt reef ecosystems."
        }
      }
    ]
  }
  ```

### Curl Example
```bash
curl -X POST "http://localhost:8000/api/classify" \
  -H "accept: application/json" \
  -F "image=@/path/to/photo.jpg"
```

## Species Metadata

Species metadata lives in `app/data/species.py`. Expand this file with additional invasive and native species as you refine the model’s label vocabulary and alias mappings.

## Next Steps

- Tune the OpenAI prompt/temperature to better match your desired confidence scoring.
- Persist classifications and user feedback to a database for auditing and retraining.
- Add a manual review queue for low-confidence predictions.
