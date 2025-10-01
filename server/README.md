# Docs Service (FastAPI) – Offert / Order / Invoice + Images + Voice + OCR + Autofill

## Run locally
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel && pip install -r requirements.txt
uvicorn docs_master:app --host 0.0.0.0 --port 8000 --reload

## Env (copy .env.example → .env)
OPENAI_API_KEY=...
CORS_ALLOW_ORIGINS=http://localhost:3000

## Render deploy (Docker)
- New Web Service → Use Docker → Root Directory: `server/`
- Add persistent disk mounted at `/data`
- Env: OPENAI_API_KEY, CORS_ALLOW_ORIGINS, UPLOAD_DIR, GENERATED_DIR, OCR_MODE
- Health check path: `/health`

## Key endpoints
- /flow/start            (voice or text) → draft
- /flow/create-offer/:id ({offer, layout, images}) → Offert PDF
- /flow/send-offer/:id   → mark "Sent"
- /flow/generate-order/:id → Order JSON + PDF (autofill)
- /flow/generate-invoice/:id → Invoice JSON + PDF + pricing
- /flow/confirm/:id      → Order + Invoice (both)
- /flow/:id              → current status/pricing/PDF paths
- /assets/upload         → save file and serve as /assets/*
- /ocr/analyze           → file → text
- /ocr/offer-draft       → file/text → offer JSON
- /ocr/receipt-draft     → file/text → bookkeeping draft
- /flow/start-ocr        → file → draft flow

