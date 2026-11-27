require('dotenv').config();
const express = require('express'), mongoose = require('mongoose'), multer = require('multer');
const jwt = require('jsonwebtoken'), bcrypt = require('bcryptjs'), crypto = require('crypto'), path = require('path');

const { MONGO_URI, JWT_SECRET, ENC_KEY, OCR_URL = 'http://localhost:8000/api/scan', PORT = 5000 } = process.env;
if (!MONGO_URI || !JWT_SECRET) { console.error('Set MONGO_URI and JWT_SECRET in .env / environment'); process.exit(1); }
// ENC_KEY is OPTIONAL. If not set, data is stored as plain text in the database.
const KEY = ENC_KEY && ENC_KEY.length === 64 ? Buffer.from(ENC_KEY, 'hex') : null;

/* ---------- encryption at rest (AES-256-GCM) for Aadhaar / PAN / raw text ---------- */
const enc = (t) => { if (!t) return null; if (!KEY) return String(t); const iv = crypto.randomBytes(12);
  const c = crypto.createCipheriv('aes-256-gcm', KEY, iv);
  const ct = Buffer.concat([c.update(String(t), 'utf8'), c.final()]);
  return 'enc:' + [iv, c.getAuthTag(), ct].map(b => b.toString('base64')).join(':'); };
const dec = (s) => { if (!s) return null; if (!s.startsWith('enc:')) return s; if (!KEY) return null;
  const [iv, tag, ct] = s.slice(4).split(':').map(x => Buffer.from(x, 'base64'));
  const d = crypto.createDecipheriv('aes-256-gcm', KEY, iv); d.setAuthTag(tag);
  return Buffer.concat([d.update(ct), d.final()]).toString('utf8'); };
const maskAadhaar = (a) => a ? 'XXXX XXXX ' + a.slice(-4) : null;
const maskPan = (p) => p ? p.slice(0, 2) + 'XXXXX' + p.slice(-3) : null;

/* ---------- validation (same rules as the OCR service) ---------- */
const D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]];
const P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]];
const verhoeff = (n) => { let c = 0; [...n].reverse().forEach((ch, i) => { c = D[c][P[i % 8][+ch]]; }); return c === 0; };
function validCalendarDob(dob) {
  const m = /^(\d{2})\/(\d{2})\/(\d{4})$/.exec(dob);
  if (m) {
    const d = +m[1], mo = +m[2], y = +m[3];
    if (y < 1900 || y > new Date().getFullYear()) return { ok: false, note: 'valid calendar DD/MM/YYYY' };
    const dt = new Date(y, mo - 1, d);
    const ok = dt.getFullYear() === y && dt.getMonth() === mo - 1 && dt.getDate() === d;
    return { ok, note: 'valid calendar DD/MM/YYYY' };
  }
  if (/^\d{4}$/.test(dob)) {
    const y = +dob;
    return { ok: y >= 1900 && y <= new Date().getFullYear(), note: 'valid Year of Birth (YYYY)' };
  }
  return { ok: false, note: 'valid calendar DD/MM/YYYY' };
}
function validate(f, aadhaar, pan) {
  const v = {};
  if (aadhaar) v.aadhaar = { ok: /^\d{12}$/.test(aadhaar) && verhoeff(aadhaar), note: '12 digits + Verhoeff checksum' };
  if (pan) v.pan = { ok: /^[A-Z]{5}\d{4}[A-Z]$/.test(pan) && 'PCHFATBLJG'.includes(pan[3]), note: 'AAAAA9999A + holder-type letter' };
  if (f.dob) v.dob = validCalendarDob(String(f.dob).trim());
  if (f.pincode) v.pincode = { ok: /^[1-9]\d{5}$/.test(f.pincode), note: '6-digit PIN' };
  return v;
}

/* ---------- models ---------- */
const User = mongoose.model('User', new mongoose.Schema({
  name: String, email: { type: String, unique: true, lowercase: true, required: true }, passwordHash: String,
}, { timestamps: true }));

const Doc = mongoose.model('Document', new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', index: true, required: true },
  fileName: String, docType: String,
  fields: { name: String, fatherName: String, dob: String, gender: String, address: String, pincode: String, summary: String },
  aadhaarEnc: String, aadhaarMasked: String, panEnc: String, panMasked: String, rawTextEnc: String,
  ocrConfidence: Number, aiUsed: Boolean, edited: { type: Boolean, default: false }, validation: mongoose.Schema.Types.Mixed,
}, { timestamps: true }));

const dto = (d, full) => ({
  id: d._id, fileName: d.fileName, docType: d.docType, fields: d.fields, validation: d.validation,
  ocrConfidence: d.ocrConfidence, aiUsed: d.aiUsed, edited: d.edited, createdAt: d.createdAt,
  aadhaar: full ? dec(d.aadhaarEnc) : d.aadhaarMasked, pan: full ? dec(d.panEnc) : d.panMasked,
  ...(full ? { rawText: dec(d.rawTextEnc) } : {}),
});

/* ---------- app + auth ---------- */
const app = express();
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } });
const wrap = (fn) => (req, res) => fn(req, res).catch(e => { console.error(e); res.status(500).json({ error: e.message }); });
const sign = (u) => jwt.sign({ id: u._id }, JWT_SECRET, { expiresIn: '7d' });
const auth = (req, res, next) => { try { req.uid = jwt.verify((req.headers.authorization || '').replace('Bearer ', ''), JWT_SECRET).id; next(); }
  catch { res.status(401).json({ error: 'Please log in' }); } };

app.post('/api/auth/register', wrap(async (req, res) => {
  const { name, email, password } = req.body || {};
  if (!email || !password || password.length < 6) return res.status(400).json({ error: 'Email and 6+ char password required' });
  if (await User.findOne({ email: email.toLowerCase() })) return res.status(409).json({ error: 'Email already registered' });
  const u = await User.create({ name, email, passwordHash: await bcrypt.hash(password, 10) });
  res.json({ token: sign(u), name: u.name });
}));
app.post('/api/auth/login', wrap(async (req, res) => {
  const u = await User.findOne({ email: (req.body.email || '').toLowerCase() });
  if (!u || !(await bcrypt.compare(req.body.password || '', u.passwordHash))) return res.status(401).json({ error: 'Wrong email or password' });
  res.json({ token: sign(u), name: u.name });
}));

/* ---------- scan -> OCR service -> save ---------- */
app.post('/api/scan', auth, upload.single('file'), wrap(async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file' });
  const fd = new FormData();
  fd.append('file', new Blob([req.file.buffer], { type: req.file.mimetype }), req.file.originalname);
  let r;
  try {
    const resp = await fetch(OCR_URL, {
      method: 'POST',
      body: fd,
      headers: { 'x-ocr-token': process.env.OCR_TOKEN || '' },
      signal: AbortSignal.timeout(45000),
    });
    if (!resp.ok) {
      const msg = await resp.text();
      return res.status(502).json({ error: `OCR service error (${resp.status}): ${msg.slice(0, 160)}` });
    }
    r = await resp.json();
  } catch (err) {
    return res.status(502).json({ error: err.name === 'TimeoutError' ? 'OCR service timed out' : 'OCR service not running (start the Python service on port 8000)' });
  }
  if (r.error) return res.status(422).json({ error: r.error });
  const f = r.fields || {};
  const aadhaar = f.aadhaar_number ? String(f.aadhaar_number).replace(/\D/g, '') : null;
  const pan = f.pan_number || null;
  const doc = await Doc.create({
    user: req.uid, fileName: req.file.originalname, docType: f.doc_type || 'Unknown',
    fields: { name: f.name, fatherName: f.father_or_guardian_name, dob: f.dob, gender: f.gender, address: f.address, pincode: f.pincode, summary: f.summary },
    aadhaarEnc: enc(aadhaar), aadhaarMasked: maskAadhaar(aadhaar), panEnc: enc(pan), panMasked: maskPan(pan),
    rawTextEnc: enc(r.raw_text), ocrConfidence: r.ocr_confidence, aiUsed: r.ai_used, validation: r.validation,
  });
  res.json({ ...dto(doc, true), aiError: r.ai_error, corrections: f.corrections || [] });
}));

/* ---------- history (only the owner's documents) ---------- */
app.get('/api/documents', auth, wrap(async (req, res) => {
  const q = { user: req.uid };
  if (req.query.q) { const rx = new RegExp(req.query.q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
    q.$or = [{ 'fields.name': rx }, { docType: rx }, { fileName: rx }]; }
  res.json((await Doc.find(q).sort({ createdAt: -1 }).limit(200)).map(d => dto(d, false)));
}));
app.get('/api/documents/:id', auth, wrap(async (req, res) => {
  const d = await Doc.findOne({ _id: req.params.id, user: req.uid });
  d ? res.json(dto(d, true)) : res.status(404).json({ error: 'Not found' });
}));
app.put('/api/documents/:id', auth, wrap(async (req, res) => {
  const d = await Doc.findOne({ _id: req.params.id, user: req.uid });
  if (!d) return res.status(404).json({ error: 'Not found' });
  const b = req.body || {}, F = ['name', 'fatherName', 'dob', 'gender', 'address', 'pincode', 'summary'];
  F.forEach(k => { if (b.fields && k in b.fields) d.fields[k] = b.fields[k]; });
  if ('aadhaar' in b) { const a = String(b.aadhaar || '').replace(/\D/g, '') || null; d.aadhaarEnc = enc(a); d.aadhaarMasked = maskAadhaar(a); }
  if ('pan' in b) { const p = String(b.pan || '').toUpperCase().trim() || null; d.panEnc = enc(p); d.panMasked = maskPan(p); }
  if (b.docType) d.docType = b.docType;
  d.validation = validate(d.fields.toObject(), dec(d.aadhaarEnc), dec(d.panEnc));
  d.edited = true; d.markModified('validation'); await d.save();
  res.json(dto(d, true));
}));
app.delete('/api/documents/:id', auth, wrap(async (req, res) => {
  const r = await Doc.deleteOne({ _id: req.params.id, user: req.uid });
  r.deletedCount ? res.json({ ok: true }) : res.status(404).json({ error: 'Not found' });
}));

mongoose.connect(MONGO_URI).then(() => {
  console.log('MongoDB Atlas connected');
  app.listen(PORT, () => console.log('AayoLens running on http://localhost:' + PORT));
}).catch(e => { console.error('MongoDB connection failed:', e.message); process.exit(1); });
