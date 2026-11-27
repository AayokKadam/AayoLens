"""Accuracy, preprocessing, and validation tests for AayoLens."""
import io
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as doclens  # noqa: E402


def make_verhoeff_aadhaar(prefix11: str) -> str:
    """Compute valid 12th Verhoeff check digit for an 11-digit Aadhaar prefix."""
    c = 0
    for i, ch in enumerate(reversed(prefix11)):
        c = doclens._d[c][doclens._p[(i + 1) % 8][int(ch)]]
    for d in range(10):
        if doclens._d[c][doclens._p[0][d]] == 0:
            return prefix11 + str(d)
    raise AssertionError("No check digit found")


class TestAayoLensPipeline(unittest.TestCase):
    def test_verhoeff_checksum(self):
        valid = make_verhoeff_aadhaar("45678901234")
        self.assertTrue(doclens.verhoeff_ok(valid))
        bad = valid[:-1] + str((int(valid[-1]) + 1) % 10)
        self.assertFalse(doclens.verhoeff_ok(bad))

    def test_aadhaar_extraction_with_ocr_noise_and_vid_exclusion(self):
        valid = make_verhoeff_aadhaar("56781234901")
        noisy = f"{valid[:4]}  {valid[4:8]} - {valid[8:]}".replace("0", "O", 1).replace("1", "I", 1)
        # 16-digit VID should not be truncated into an Aadhaar number
        text = f"VID: 9123 4567 8901 2345\nYour Aadhaar No: {noisy}\nPIN: 400001"
        res = doclens.rule_extract(text)
        self.assertEqual(res["aadhaar_number"], valid)
        self.assertEqual(res["pincode"], "400001")

    def test_pan_extraction_avoids_words_and_fixes_split_ocr(self):
        text = "INCOME TAX DEPARTMENT\nGOVT OF INDIA\nPermanent Account Number\nABCPB 1234F"
        res = doclens.rule_extract(text)
        self.assertEqual(res["pan_number"], "ABCPB1234F")

        # OCR confused digits/letters in PAN
        noisy_pan = "ABCPB I234F"
        res2 = doclens.rule_extract(f"Account No {noisy_pan}")
        self.assertEqual(res2["pan_number"], "ABCPB1234F")

    def test_dob_prioritization_and_calendar_validation(self):
        text = "Download Date: 10/11/2025\nIssue Date: 01/02/2020\nDOB: 15/08/1995\nMALE"
        res = doclens.rule_extract(text)
        self.assertEqual(res["dob"], "15/08/1995")
        self.assertEqual(res["gender"], "Male")

        v = doclens.validate({"dob": "31/02/2000"})
        self.assertFalse(v["dob"]["ok"])

        v_yob = doclens.validate({"dob": "1988"})
        self.assertTrue(v_yob["dob"]["ok"])

    def test_merge_prefers_valid_checksums_and_omits_empty_aadhaar(self):
        valid_aadhaar = make_verhoeff_aadhaar("78901234567")
        hints = {"aadhaar_number": valid_aadhaar, "pan_number": "ABCPB1234F"}
        ai = {"aadhaar_number": "123456789012", "pan_number": "abcpb 1234f", "name": "RAHUL SHARMA"}
        merged = doclens.merge_and_normalize(hints, ai)
        self.assertEqual(merged["aadhaar_number"], valid_aadhaar)
        self.assertEqual(merged["pan_number"], "ABCPB1234F")

        pan_only = doclens.merge_and_normalize({"pan_number": "ABCPB1234F"}, {"doc_type": "PAN"})
        self.assertNotIn("aadhaar_number", pan_only)

    def test_exif_transpose_and_tilted_ocr_detection(self):
        # Create tilted image with EXIF rotation tag=6 (90 deg CW -> must be transposed)
        img = Image.new("RGB", (900, 260), "white")
        draw = ImageDraw.Draw(img)
        draw.text((40, 50), "INCOME TAX DEPARTMENT\nName: ANITA DESAI\nPAN: ABCPB5678K\nDOB: 24/03/1992", fill="black")
        # Tilt slightly (-4 deg) then rotate 270 deg and tag EXIF Orientation=6 (90 CW)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), -4.0, 1.0)
        tilted = cv2.warpAffine(cv_img, M, (w, h), borderValue=(255, 255, 255))
        pil_tilted = Image.fromarray(cv2.cvtColor(tilted, cv2.COLOR_BGR2RGB)).rotate(90, expand=True)
        exif = pil_tilted.getexif()
        exif[0x0112] = 6  # Orientation = Rotate 90 CW
        buf = io.BytesIO()
        pil_tilted.save(buf, format="JPEG", exif=exif.tobytes())

        decoded = doclens.decode_image_bytes(buf.getvalue())
        self.assertIsNotNone(decoded)
        text, conf = doclens.ocr_image(decoded)
        self.assertGreater(conf, 50)
        extracted = doclens.rule_extract(text)
        self.assertEqual(extracted.get("pan_number"), "ABCPB5678K")
        self.assertEqual(extracted.get("dob"), "24/03/1992")

    def test_digital_pdf_extraction(self):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "GOVERNMENT OF INDIA\nName: VIKRAM PATEL\nDOB: 11/12/1990\nPAN: ABCPB4321Z\nPIN: 560001")
        pdf_bytes = doc.tobytes()
        text, conf = doclens.read_pdf(pdf_bytes)
        self.assertEqual(conf, 100.0)
        res = doclens.rule_extract(text)
        self.assertEqual(res.get("pan_number"), "ABCPB4321Z")
        self.assertEqual(res.get("dob"), "11/12/1990")
        self.assertEqual(res.get("pincode"), "560001")


if __name__ == "__main__":
    unittest.main()
