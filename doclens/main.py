"""AayoLens - scan Aadhaar / PAN / any document text: OCR -> Groq correction -> validation."""
from datetime import date
import io
import json
import os
from pathlib import Path
import re

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import pymupdf as fitz
import pytesseract

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
elif os.name == "nt" and os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
REQUESTED_LANGS = os.getenv("OCR_LANGS", "eng+hin+mar")
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(15 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "15"))


def resolve_ocr_langs(requested: str) -> tuple[str, bool]:
    """Return installed Tesseract languages matching requested list and whether OSD is available."""
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception:
        return requested, False
    available = [lang for lang in requested.split("+") if lang and lang in installed]
    langs = "+".join(available) if available else ("eng" if "eng" in installed else requested)
    return langs, ("osd" in installed)


LANGS, HAS_OSD = resolve_ocr_langs(REQUESTED_LANGS)
client = None
if GROQ_KEY:
    from groq import Groq
    client = Groq(api_key=GROQ_KEY, timeout=25.0, max_retries=2)

app = FastAPI(title="AayoLens")

# ---------------- Verhoeff (Aadhaar checksum) ----------------
_d = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
      [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
      [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
_p = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
      [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def verhoeff_ok(num: str) -> bool:
    if not num or not num.isdigit():
        return False
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _d[c][_p[i % 8][int(ch)]]
    return c == 0


# ---------------- Image decoding, orientation & preprocessing ----------------
def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """Decode image bytes honoring EXIF orientation, falling back to OpenCV."""
    try:
        with Image.open(io.BytesIO(data)) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError):
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def normalize_size(img: np.ndarray) -> np.ndarray:
    """Keep longest edge in [1600, 1900] px for fast, sharp OCR on CPU-constrained hosts."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest < 1600:
        s = 1600 / longest
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    if longest > 1900:
        s = 1900 / longest
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def auto_rotate_osd(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Rotate 90/180/270 degrees if Tesseract OSD detects a rotated page; returns (img, changed)."""
    if not HAS_OSD:
        return gray, False
    h, w = gray.shape[:2]
    probe = gray
    if max(h, w) > 1200:
        s = 1200 / max(h, w)
        probe = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    try:
        osd = pytesseract.image_to_osd(probe, output_type=pytesseract.Output.DICT)
        rot = int(osd.get("rotate", 0))
        conf = float(osd.get("orientation_conf", 0))
    except Exception:
        return gray, False
    if conf < 2.0 or rot == 0:
        return gray, False
    if rot == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), True
    if rot == 180:
        return cv2.rotate(gray, cv2.ROTATE_180), True
    if rot == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE), True
    return gray, False


def deskew(g: np.ndarray) -> np.ndarray:
    """Correct small camera tilt (-12 to +12 deg) using text-mask coordinates."""
    edges = cv2.Canny(g, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    ys, xs = np.where(dilated > 0)
    if len(xs) < 400:
        return g
    pts = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    if rect[1][0] < rect[1][1]:
        angle = angle - 90
    if angle < -45:
        angle += 90
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.5 or abs(angle) > 12:
        return g
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _build_variants(gray: np.ndarray) -> list[np.ndarray]:
    """Produce fast complementary grayscale/threshold variants from a deskewed grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return [clahe, otsu, adapt]


def _data_to_text_and_score(data: dict) -> tuple[str, float, float, int]:
    """Reconstruct formatted text and compute weighted score in one pass without calling Tesseract twice."""
    confs = []
    alnum_chars = 0
    blocks: dict[tuple[int, int, int], list[str]] = {}

    texts = data.get("text", [])
    raw_confs = data.get("conf", [])
    b_nums = data.get("block_num", [0] * len(texts))
    p_nums = data.get("par_num", [0] * len(texts))
    l_nums = data.get("line_num", [0] * len(texts))

    for txt, c, b, p, l in zip(texts, raw_confs, b_nums, p_nums, l_nums):
        word = str(txt).strip()
        if not word:
            continue
        try:
            cf = float(c)
        except (TypeError, ValueError):
            cf = -1.0
        blocks.setdefault((int(b), int(p), int(l)), []).append(word)
        cleaned = re.sub(r"[^A-Za-z0-9\u0900-\u097F]", "", word)
        if cf >= 0 and cleaned:
            confs.append(cf)
            alnum_chars += len(cleaned)

    lines_out = []
    prev_block = None
    for (b, _p, _l), words in blocks.items():
        if prev_block is not None and b != prev_block:
            lines_out.append("")
        lines_out.append(" ".join(words))
        prev_block = b

    full_text = "\n".join(lines_out).strip()
    if not confs:
        return full_text, -1.0, 0.0, 0
    mean_conf = sum(confs) / len(confs)
    yield_factor = min(alnum_chars, 80) / 80.0
    score = mean_conf * (0.45 + 0.55 * yield_factor)
    return full_text, score, mean_conf, alnum_chars


def _run_ocr_pass(img_gray: np.ndarray, psm: int) -> tuple[str, float, float, int]:
    data = pytesseract.image_to_data(
        img_gray, lang=LANGS, config=f"--oem 3 --psm {psm}", output_type=pytesseract.Output.DICT
    )
    return _data_to_text_and_score(data)


def ocr_image(img: np.ndarray) -> tuple[str, float]:
    scaled = normalize_size(img)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    gray = deskew(gray)
    variants = _build_variants(gray)

    # Fast primary pass (single Tesseract call covers clean scans/screenshots/clear cards)
    best_text, best_score, best_conf, best_chars = _run_ocr_pass(variants[0], 3)

    # Lazy OSD check only when primary pass looks rotated/garbled
    if best_conf < 58 or best_chars < 15:
        rotated, changed = auto_rotate_osd(gray)
        if changed:
            gray = deskew(rotated)
            variants = _build_variants(gray)
            t, s, c, ch = _run_ocr_pass(variants[0], 3)
            if s > best_score:
                best_text, best_score, best_conf, best_chars = t, s, c, ch

    # Only try additional thresholding / PSM passes if primary pass wasn't already strong
    if best_conf < 85 or best_chars < 25:
        for v in variants[1:]:
            t, s, c, ch = _run_ocr_pass(v, 3)
            if s > best_score:
                best_text, best_score, best_conf, best_chars = t, s, c, ch
            if best_conf >= 86 and best_chars >= 30:
                break

    if best_conf < 65 or best_score < 45:
        for psm in (6, 11):
            t, s, c, ch = _run_ocr_pass(variants[0], psm)
            if s > best_score:
                best_text, best_score, best_conf, best_chars = t, s, c, ch

    return best_text, round(best_conf, 1)


def read_pdf(data: bytes) -> tuple[str, float]:
    doc = fitz.open(stream=data, filetype="pdf")
    if doc.page_count == 0:
        return "", 0.0
    if doc.page_count > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds maximum supported page limit ({MAX_PDF_PAGES} pages)")
    texts, confs = [], []
    for page in doc:
        t = page.get_text().strip()
        if len(t) >= 30:
            texts.append(t)
            confs.append(100.0)
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            img = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n), cv2.COLOR_RGB2BGR)
            t, c = ocr_image(img)
            texts.append(t)
            confs.append(c)
    return "\n\n".join(texts).strip(), round(sum(confs) / max(len(confs), 1), 1)


# ---------------- Rule-based extraction & validation ----------------
TO_DIGIT = str.maketrans("OoIlSBZGQD", "0011586200")
TO_ALPHA = str.maketrans("01586", "OISBG")
PAN_HOLDER_TYPES = set("PCHFATBLJG")


def valid_calendar_date(dd: int, mm: int, yyyy: int) -> bool:
    if not (1900 <= yyyy <= date.today().year):
        return False
    try:
        date(yyyy, mm, dd)
        return True
    except ValueError:
        return False


def fix_pan(tok: str) -> tuple[str, int] | None:
    """Attempt OCR-confusion normalization on a 10-char PAN candidate; return (pan, edit_count)."""
    raw = re.sub(r"\s+", "", tok).upper()
    if len(raw) != 10 or raw.isalpha():
        return None
    # Require at least 2 real digits in the 4-digit middle block to avoid English words
    if sum(ch.isdigit() for ch in raw[5:9]) < 2:
        return None
    fixed = raw[:5].translate(TO_ALPHA) + raw[5:9].translate(TO_DIGIT) + raw[9].translate(TO_ALPHA)
    if not re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", fixed):
        return None
    edits = sum(a != b for a, b in zip(raw, fixed))
    if edits > 3:
        return None
    return fixed, edits


def extract_aadhaar_candidates(flat: str) -> list[str]:
    pattern = r"(?<!\d)(?<!\d )(?<!\d-)[\dOoIlSBZG][\dOoIlSBZG]{3}[\s\-]{0,3}[\dOoIlSBZG]{4}[\s\-]{0,3}[\dOoIlSBZG]{4}(?![\s\-]?\d)"
    verified, fallback = [], []
    for m in re.finditer(pattern, flat):
        n = re.sub(r"[\s\-]", "", m.group()).translate(TO_DIGIT)
        if len(n) == 12 and n.isdigit() and n[0] not in "01":
            if verhoeff_ok(n):
                verified.append(n)
            else:
                fallback.append(n)
    return verified + fallback


def extract_pan_candidates(flat: str) -> list[str]:
    candidates = []
    # Match contiguous or lightly space-split 10-char alphanumeric PAN tokens
    for m in re.finditer(r"(?<![A-Za-z0-9])[A-Za-z0-9]{5}\s?[A-Za-z0-9]{4}\s?[A-Za-z0-9](?![A-Za-z0-9])", flat):
        res = fix_pan(m.group())
        if not res:
            continue
        pan, edits = res
        holder_ok = pan[3] in PAN_HOLDER_TYPES
        # Avoid false positives like 10-char words unless holder letter is valid and edits are minimal
        if not holder_ok and edits > 0:
            continue
        candidates.append((0 if holder_ok else 1, edits, pan))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [c[2] for c in candidates]


def extract_dob(flat: str) -> str | None:
    labeled = re.search(
        r"(?:DOB|D\.O\.B|DOR|Date\s*of\s*Birth|Birth\s*Date|जन्म\s*तिथि|जन्म\s*तारीख)\s*[:\-/]?\s*(\d{2})\s*[/\-.]?\s*(\d{2})\s*[/\-.]?\s*(\d{4})\b",
        flat,
        re.I,
    )
    if labeled:
        d, m, y = int(labeled[1]), int(labeled[2]), int(labeled[3])
        if valid_calendar_date(d, m, y):
            return f"{d:02d}/{m:02d}/{y:04d}"

    yob = re.search(r"(?:Year\s*of\s*Birth|YOB|जन्म\s*वर्ष)\s*[:\-]?\s*(\d{4})\b", flat, re.I)
    if yob and 1900 <= int(yob[1]) <= date.today().year:
        return yob[1]

    for m in re.finditer(r"\b(\d{2})\s*[/\-.]\s*(\d{2})\s*[/\-.]\s*(\d{4})\b", flat):
        # Skip dates explicitly labeled as issue/download/print dates
        prefix = flat[max(0, m.start() - 25):m.start()].lower()
        if any(w in prefix for w in ("issue", "download", "print", "valid", "expiry", "exp")):
            continue
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
        if valid_calendar_date(d, mo, y):
            return f"{d:02d}/{mo:02d}/{y:04d}"
    return None


def rule_extract(text: str) -> dict:
    out = {}
    flat = re.sub(r"[ \t]+", " ", text)

    aadhaars = extract_aadhaar_candidates(flat)
    if aadhaars:
        out["aadhaar_number"] = aadhaars[0]

    pans = extract_pan_candidates(flat)
    if pans:
        out["pan_number"] = pans[0]

    dob = extract_dob(flat)
    if dob:
        out["dob"] = dob

    g = re.search(r"\b(MALE|FEMALE|TRANSGENDER)\b", flat, re.I)
    if g:
        out["gender"] = g.group(1).capitalize()

    # 6-digit Indian PIN code (avoid matching inside 12-digit numbers)
    for pm in re.finditer(r"(?<!\d)([1-9]\d{5})(?!\d)", flat):
        # Prefer PIN codes near address/PIN labels or at line boundaries
        out["pincode"] = pm.group(1)
        ctx = flat[max(0, pm.start() - 20):pm.start()].lower()
        if any(k in ctx for k in ("pin", "india", "maharashtra", "delhi", "karnataka", "gujarat", "tamil", "uttar", "-")):
            break

    return out


# ---------------- Groq correction & deterministic merge ----------------
SYSTEM = """You are a strict, high-precision document-data extractor for Indian identity documents
(Aadhaar, PAN, Voter ID, Driving Licence, Passport) and general documents.
Input is noisy OCR text (English/Hindi/Marathi) plus deterministic rule-based hints.
Rules:
1. Fix obvious OCR mistakes (0/O, 1/I, 5/S, 8/B, broken spacing, broken words) ONLY when supported by context.
2. NEVER invent or hallucinate fields. If a field is not present, return null.
3. Distinguish Date of Birth (dob) from Issue Date / Print Date / Download Date.
4. For PAN, return 10 uppercase characters (AAAAA9999A). For Aadhaar, return 12 digits with no spaces.
5. Return ONLY valid JSON with keys:
doc_type, name, father_or_guardian_name, dob (DD/MM/YYYY or YYYY if only YOB), gender, address, pincode,
aadhaar_number, pan_number, other_ids (object), summary (1 concise line), corrections (array of short strings), confidence (0-100)."""


def groq_clean(text: str, hints: dict) -> dict | None:
    if not client or not text.strip():
        return None
    r = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"RULE HINTS:\n{json.dumps(hints)}\n\nOCR TEXT:\n{text[:12000]}"},
        ],
    )
    content = (r.choices[0].message.content or "").strip()
    if not content:
        return None
    return json.loads(content)


def merge_and_normalize(hints: dict, ai: dict | None) -> dict:
    fields = dict(hints)
    if ai:
        for k, v in ai.items():
            if v not in (None, "", {}, []):
                fields[k] = v

    # Normalize & checksum-protect Aadhaar
    raw_a = re.sub(r"\D", "", str(fields.get("aadhaar_number") or ""))
    hint_a = re.sub(r"\D", "", str(hints.get("aadhaar_number") or ""))
    if raw_a and len(raw_a) == 12 and verhoeff_ok(raw_a):
        fields["aadhaar_number"] = raw_a
    elif hint_a and len(hint_a) == 12 and verhoeff_ok(hint_a):
        fields["aadhaar_number"] = hint_a
    elif raw_a:
        fields["aadhaar_number"] = raw_a
    else:
        fields.pop("aadhaar_number", None)

    # Normalize & holder-type-protect PAN
    raw_p = fix_pan(str(fields.get("pan_number") or ""))
    hint_p = fix_pan(str(hints.get("pan_number") or ""))
    if raw_p and raw_p[0][3] in PAN_HOLDER_TYPES:
        fields["pan_number"] = raw_p[0]
    elif hint_p and hint_p[0][3] in PAN_HOLDER_TYPES:
        fields["pan_number"] = hint_p[0]
    elif raw_p:
        fields["pan_number"] = raw_p[0]
    elif not fields.get("pan_number"):
        fields.pop("pan_number", None)

    # Normalize PIN code
    pin = re.sub(r"\D", "", str(fields.get("pincode") or ""))
    if re.fullmatch(r"[1-9]\d{5}", pin):
        fields["pincode"] = pin
    elif hints.get("pincode"):
        fields["pincode"] = hints["pincode"]
    else:
        fields.pop("pincode", None)

    return fields


def validate(f: dict) -> dict:
    v = {}
    a = re.sub(r"\D", "", str(f.get("aadhaar_number") or ""))
    if a:
        v["aadhaar"] = {"ok": len(a) == 12 and verhoeff_ok(a), "note": "12 digits + Verhoeff checksum"}

    p = str(f.get("pan_number") or "").strip().upper()
    if p:
        ok = bool(re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", p))
        v["pan"] = {"ok": ok and p[3] in PAN_HOLDER_TYPES, "note": "format AAAAA9999A + valid holder-type (4th letter)"}

    d = str(f.get("dob") or "").strip()
    if d:
        m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", d)
        if m:
            ok = valid_calendar_date(int(m[1]), int(m[2]), int(m[3]))
            v["dob"] = {"ok": ok, "note": "valid calendar DD/MM/YYYY"}
        elif re.fullmatch(r"\d{4}", d):
            v["dob"] = {"ok": 1900 <= int(d) <= date.today().year, "note": "valid Year of Birth (YYYY)"}
        else:
            v["dob"] = {"ok": False, "note": "expected DD/MM/YYYY"}

    pin = str(f.get("pincode") or "").strip()
    if pin:
        v["pincode"] = {"ok": bool(re.fullmatch(r"[1-9]\d{5}", pin)), "note": "6-digit PIN"}

    return v


@app.post("/api/scan")
async def scan(file: UploadFile = File(...), x_ocr_token: str = Header(None)):
    if os.getenv("OCR_TOKEN") and x_ocr_token != os.getenv("OCR_TOKEN"):
        raise HTTPException(status_code=401, detail="Bad token")
    data = await file.read()
    if not data:
        return {"error": "Empty file uploaded"}
    if len(data) > MAX_FILE_BYTES:
        return {"error": f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit"}

    filename = (file.filename or "").lower()
    try:
        if filename.endswith(".pdf") or file.content_type == "application/pdf":
            text, conf = read_pdf(data)
        else:
            img = decode_image_bytes(data)
            if img is None:
                return {"error": "Could not read image"}
            text, conf = ocr_image(img)
    except pytesseract.TesseractNotFoundError:
        return {"error": "Tesseract OCR is not installed / not on PATH. See README."}
    except pytesseract.TesseractError as e:
        return {"error": f"Tesseract OCR error: {e}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Document processing error: {e}"}

    hints = rule_extract(text)
    fields, ai_used, ai_error = dict(hints), False, None
    try:
        ai = groq_clean(text, hints)
        if ai:
            ai_used = True
            fields = merge_and_normalize(hints, ai)
    except Exception as e:
        ai_error = str(e)

    return {
        "ocr_confidence": conf,
        "ai_used": ai_used,
        "ai_error": ai_error,
        "fields": fields,
        "validation": validate(fields),
        "raw_text": text,
    }


@app.get("/health")
def health():
    return {"status": "ok", "ocr_langs": LANGS, "osd_enabled": HAS_OSD, "ai_enabled": bool(client)}


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
