# AayoLens - Google-Lens-style document scanner (Aadhaar, PAN, PDF, any text)

Pipeline: image/PDF -> OpenCV cleanup (upscale, denoise, deskew, threshold) -> Tesseract OCR (eng+hin+mar)
(digital PDFs use exact embedded text, scanned PDFs are OCR'd) -> regex with OCR-confusion fixes ->
Groq LLM corrects OCR errors and returns structured JSON -> hard validation (Aadhaar Verhoeff checksum,
PAN format + holder-type letter, DOB, PIN).

## Setup
1. Install Tesseract OCR (Windows: https://github.com/UB-Mannheim/tesseract/wiki - tick Hindi + Marathi languages).
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and put your Groq key (https://console.groq.com/keys).
4. `uvicorn main:app --reload` then open http://localhost:8000
   (Phone camera works when you open the page on your phone via your PC's LAN IP.)

## Notes
- Keep the Groq key only in `.env` (server side); never put it in frontend JS.
- Aadhaar/PAN are sensitive personal data: don't log or store uploads; get user consent; mask Aadhaar (show last 4) when displaying to others.
- A passing checksum proves the number is well-formed, not that it belongs to the person. For real KYC use official verification (e.g. DigiLocker / UIDAI / NSDL APIs).

## Saving to MongoDB Atlas (Node + Mongoose server)
Architecture: Browser -> Node server (port 5000: login, Mongoose, history) -> Python OCR service (port 8000).
1. Terminal A (OCR): `uvicorn main:app --port 8000`
2. Terminal B: `cd server && npm install`, copy `.env.example` to `.env`, fill MONGO_URI / JWT_SECRET / ENC_KEY, then `npm start`
3. Open http://localhost:5000 -> register -> scan. Every scan is saved; see "My documents" for history, edit, delete.
Atlas: Network Access must allow your IP; Aadhaar, PAN and raw OCR text are AES-256-GCM encrypted in the DB (masked copies are used in the list).
