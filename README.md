# StrikeNet API

This project exposes a FastAPI service that accepts user-uploaded wildlife photos, sends them to a configurable image-classification model (e.g. a Hugging Face inference endpoint), and reports whether the detected species is invasive in South Florida.

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
3. **Set configuration** via environment variables (prefix `STRIKENET_`):
   - `STRIKENET_HUGGINGFACE_API_TOKEN` (optional) – token for private or rate-limited models.
   - `STRIKENET_HUGGINGFACE_MODEL_ID` – defaults to `microsoft/resnet-50`; replace with your marine model.
   - `STRIKENET_CLASSIFICATION_CONFIDENCE_THRESHOLD` – defaults to `0.6`.

   Example:
   ```bash
   export STRIKENET_HUGGINGFACE_MODEL_ID="your-org/your-marine-model"
   export STRIKENET_HUGGINGFACE_API_TOKEN="hf_..."
   ```
4. **Run the API**:
   ```bash
   uvicorn app.main:app --reload
   ```

## API

### `POST /api/classify`
Uploads an image and returns model predictions annotated with invasive-species metadata.

- **Body**: `multipart/form-data` with `image` field.
- **Response** (`200 OK`):
  ```json
  {
    "decision": "auto-flagged-invasive",
    "invasive": true,
    "threshold": 0.6,
    "top_prediction": {
      "label": "lionfish",
      "score": 0.94,
      "species": {
        "common_name": "Red Lionfish",
        "scientific_name": "Pterois volitans",
        "is_invasive": true,
        "notes": "Aggressive invasive predator known to disrupt reef ecosystems."
      }
    },
    "predictions": [
      {
        "label": "lionfish",
        "score": 0.94,
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

Species metadata lives in `app/data/species.py`. Expand this file with additional invasive and native species as you refine the model’s label vocabulary.

## Next Steps

- Swap in a marine life classifier (e.g. a fine-tuned iNaturalist checkpoint) and update alias mappings.
- Persist classifications and user feedback to a database for auditing and retraining.
- Add a manual review queue for low-confidence predictions.
