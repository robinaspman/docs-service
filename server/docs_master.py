#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docs + Voice Flow service (Images + OCR + Autofill/status endpoints + File logging/URLs)

Adds:
- Flow status/history & list:  GET /flows, GET /flow/{id}
- File logging + public URLs:  every generated/ uploaded file logged and returned with absolute URLs
- Autofill endpoint:           POST /autofill/offer (text or partial offer; optional flow_id to update)
- Root route:                  GET /  (friendly JSON)
Keeps:
- Offert / Order / Invoice PDF generation (ReportLab), alignment + global spacing controls
- Image support with captions + placement
- Voice-first flow (STT) + text fallback
- OCR endpoints (PDF/Image):   /ocr/analyze, /ocr/offer-draft, /flow/start-ocr, /ocr/receipt-draft
- Stepwise UI endpoints:       /flow/send-offer, /flow/generate-order, /flow/generate-invoice
"""

import io, os, re, json, shutil, base64, mimetypes
from uuid import uuid4
from pathlib import Path
from datetime import date, datetime, timezone
from typing import List, Optional, Literal, Tuple, Any, Dict

# -------- PDF / ReportLab ----------
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# -------- API / Schema -------------
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# -------- Optional OpenAI (managed STT/TTS + Vision OCR) --------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -------- Optional OCR/Parsing libs ----------
HAS_TESS = False
HAS_PYPDF = False
HAS_PDF2IMG = False
try:
    import pytesseract
    from PIL import Image
    HAS_TESS = True
except Exception:
    pass
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    pass
try:
    from pdf2image import convert_from_bytes
    HAS_PDF2IMG = True
except Exception:
    pass

# =========================================================
# File storage (uploads + generated) with public mounts
# =========================================================
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
GENERATED_DIR = os.getenv("GENERATED_DIR", "generated")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# =========================================================
# Global alignment control
# =========================================================
LEFT_ALIGN_PAD = 2 * mm  # set 0 to align with the blue band edge

# =========================================================
# Global spacing controls (points)
# =========================================================
SPC_AFTER_LOGO_COMPACT   = 6
SPC_AFTER_LOGO_RELAXED   = 8
SPC_AFTER_TITLE_COMPACT  = 4
SPC_AFTER_TITLE_RELAXED  = 6
SPC_META_TO_ITEMS_COMPACT = 6
SPC_META_TO_ITEMS_RELAXED = 8
SPC_AFTER_ITEMS_COMPACT  = 6
SPC_AFTER_ITEMS_RELAXED  = 8
SPC_AFTER_VERSION_COMPACT = 6
SPC_AFTER_VERSION_RELAXED = 8
SPC_AFTER_TOTAL_COMPACT   = 4
SPC_AFTER_TOTAL_RELAXED   = 6
SPC_AFTER_PAYMENT_COMPACT = 6
SPC_AFTER_PAYMENT_RELAXED = 8
SPC_AFTER_GDPR_COMPACT    = 5
SPC_AFTER_GDPR_RELAXED    = 6
SPC_AFTER_VALIDITY_COMPACT = 10
SPC_AFTER_VALIDITY_RELAXED = 12

SPC_PANEL_TITLE_AFTER_COMPACT = 4
SPC_PANEL_TITLE_AFTER_RELAXED = 6
SPC_PANEL_INNER_PAD = 5

# =========================================================
# Table padding controls (points)
# =========================================================
TBL_HEAD_TOP_PAD_COMPACT    = 4
TBL_HEAD_TOP_PAD_RELAXED    = 6
TBL_HEAD_BOTTOM_PAD_COMPACT = 4
TBL_HEAD_BOTTOM_PAD_RELAXED = 6
TBL_ROW_TOP_PAD_COMPACT     = 4
TBL_ROW_TOP_PAD_RELAXED     = 6
TBL_ROW_BOTTOM_PAD_COMPACT  = 4
TBL_ROW_BOTTOM_PAD_RELAXED  = 6

ORDINV_HEAD_TOP_PAD_COMPACT    = 4
ORDINV_HEAD_TOP_PAD_RELAXED    = 6
ORDINV_HEAD_BOTTOM_PAD_COMPACT = 4
ORDINV_HEAD_BOTTOM_PAD_RELAXED = 6
ORDINV_ROW_TOP_PAD_COMPACT     = 4
ORDINV_ROW_TOP_PAD_RELAXED     = 6
ORDINV_ROW_BOTTOM_PAD_COMPACT  = 4
ORDINV_ROW_BOTTOM_PAD_RELAXED  = 6

META_CELL_BOTTOM_PAD_COMPACT   = 1
META_CELL_BOTTOM_PAD_RELAXED   = 2

def pick(compact: bool, a: float, b: float) -> float:
    return a if compact else b

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# =========================================================
# OpenAI managed audio/vision configuration (optional)
# =========================================================
OPENAI_MODEL_STT     = os.getenv("OPENAI_MODEL_STT", "gpt-4o-mini-transcribe")
OPENAI_MODEL_TTS     = os.getenv("OPENAI_MODEL_TTS", "gpt-4o-mini-tts")
OPENAI_MODEL_PARSER  = os.getenv("OPENAI_MODEL_PARSER", "gpt-4o-mini")
OPENAI_MODEL_VISION  = os.getenv("OPENAI_MODEL_VISION", "gpt-4.1-mini")
_client = None
if OpenAI and os.getenv("OPENAI_API_KEY"):
    _client = OpenAI()

# =========================================================
# App + CORS + Static files
# =========================================================
app = FastAPI(title="Docs Service (Alignment + Images + Voice + OCR + Autofill + Logs)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Static serving (uploads + generated)
app.mount("/assets", StaticFiles(directory=UPLOAD_DIR), name="assets")
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")

# =========================================================
# Helpers
# =========================================================
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
PDF_EXTS = (".pdf",)

def register_fonts() -> str:
    base = "Helvetica"
    try:
        if os.path.exists("DejaVuSans.ttf"):
            pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
            base = "DejaVu"
    except Exception:
        base = "Helvetica"
    return base

def fmt_currency(n: float) -> str:
    s = f"{n:,.0f}".replace(",", " ").replace(".", " ")
    return f"{s} kr"

def safe_mkdir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

def _list_images_in(dirpath: str) -> List[str]:
    try:
        return [
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.lower().endswith(IMAGE_EXTS)
            and os.path.isfile(os.path.join(dirpath, f))
        ]
    except Exception:
        return []

def _score_logo_candidate(path: str) -> Tuple[int, int, str]:
    name = os.path.basename(path).lower()
    name_score = 1 if "logo" in name else 0
    try:
        size = os.path.getsize(path)
    except Exception:
        size = 0
    return (-name_score, -size, name)

def search_for_logo(explicit_path: Optional[str], explicit_dir: Optional[str]) -> Optional[str]:
    for p in [explicit_path, os.getenv("LOGO_PATH")]:
        if p and os.path.exists(p):
            return p
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = [
        explicit_dir or os.getenv("LOGO_DIR"),
        os.getcwd(),
        script_dir,
        os.path.join(script_dir, "public"),
        os.path.abspath(os.path.join(script_dir, "..", "public")),
        "/mnt/data",
    ]
    candidates, seen = [], set()
    for d in filter(None, dirs):
        for path in _list_images_in(d):
            if path not in seen:
                candidates.append(path); seen.add(path)
    if not candidates:
        return None
    candidates.sort(key=_score_logo_candidate)
    return candidates[0]

def logo_flowable(path: Optional[str], max_w=70 * mm, max_h=45 * mm, align="CENTER"):
    if not path or not os.path.exists(path):
        return None
    try:
        img = RLImage(path)
        iw, ih = img.wrap(0, 0)
        if iw <= 0 or ih <= 0: return None
        scale = min(max_w / iw, max_h / ih)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        img.hAlign = align
        return img
    except Exception as e:
        print(f"[logo] Skipping logo at {path}: {e}")
        return None

def styles_offer(base_font: str, compact: bool = True) -> dict:
    s = getSampleStyleSheet()
    if "OfferTitle" not in s.byName:
        s.add(ParagraphStyle(
            name="OfferTitle", parent=s["Title"], fontName=base_font,
            fontSize=24 if compact else 26, leading=28 if compact else 30,
            alignment=1, spaceAfter=0
        ))
    if "Body" not in s.byName:
        s.add(ParagraphStyle(name="Body", parent=s["Normal"], fontName=base_font,
                             fontSize=10 if compact else 11, leading=13 if compact else 14))
    if "Small" not in s.byName:
        s.add(ParagraphStyle(name="Small", parent=s["Normal"], fontName=base_font,
                             fontSize=8.5 if compact else 9, leading=11 if compact else 12))
    if "BoldBody" not in s.byName:
        s.add(ParagraphStyle(name="BoldBody", parent=s["Normal"], fontName=base_font,
                             fontSize=11 if compact else 12, leading=14 if compact else 15))
    if "ImgCaption" not in s.byName:
        s.add(ParagraphStyle(name="ImgCaption", parent=s["Normal"], fontName=base_font,
                             fontSize=8.5 if compact else 9, leading=11 if compact else 12, alignment=1, textColor=colors.grey))
    return s

def styles_generic(base_font: str, compact: bool = True) -> dict:
    s = getSampleStyleSheet()
    if "H1" not in s.byName:
        s.add(ParagraphStyle(
            name="H1", parent=s["Title"], fontName=base_font,
            fontSize=22 if compact else 24, leading=26 if compact else 28,
            alignment=1, spaceAfter=0
        ))
    if "Body" not in s.byName:
        s.add(ParagraphStyle(name="Body", parent=s["Normal"], fontName=base_font,
                             fontSize=10 if compact else 11, leading=13 if compact else 14))
    if "Small" not in s.byName:
        s.add(ParagraphStyle(name="Small", parent=s["Normal"], fontName=base_font,
                             fontSize=8.5 if compact else 9, leading=11 if compact else 12))
    if "Strong" not in s.byName:
        s.add(ParagraphStyle(name="Strong", parent=s["Normal"], fontName=base_font,
                             fontSize=11 if compact else 12, leading=14 if compact else 15))
    if "ImgCaption" not in s.byName:
        s.add(ParagraphStyle(name="ImgCaption", parent=s["Normal"], fontName=base_font,
                             fontSize=8.5 if compact else 9, leading=11 if compact else 12, alignment=1, textColor=colors.grey))
    return s

# =========================================================
# Layout overrides (frontend-driven)
# =========================================================
def _get_layout_overrides(layout: Dict[str, Any], compact: bool):
    left_align_pad = (layout.get("left_align_pad_mm", (LEFT_ALIGN_PAD / mm)) * mm)

    def pair(key: str, d_compact: float, d_relax: float):
        v = layout.get("spacing", {}).get(key)
        if isinstance(v, (int, float)): return float(v), float(v)
        if isinstance(v, dict): return float(v.get("compact", d_compact)), float(v.get("relaxed", d_relax))
        return d_compact, d_relax

    sp_after_logo       = pair("after_logo",       SPC_AFTER_LOGO_COMPACT,       SPC_AFTER_LOGO_RELAXED)
    sp_after_title      = pair("after_title",      SPC_AFTER_TITLE_COMPACT,      SPC_AFTER_TITLE_RELAXED)
    sp_meta_to_items    = pair("meta_to_items",    SPC_META_TO_ITEMS_COMPACT,    SPC_META_TO_ITEMS_RELAXED)
    sp_after_items      = pair("after_items",      SPC_AFTER_ITEMS_COMPACT,      SPC_AFTER_ITEMS_RELAXED)
    sp_after_version    = pair("after_version",    SPC_AFTER_VERSION_COMPACT,    SPC_AFTER_VERSION_RELAXED)
    sp_after_total      = pair("after_total",      SPC_AFTER_TOTAL_COMPACT,      SPC_AFTER_TOTAL_RELAXED)
    sp_after_payment    = pair("after_payment",    SPC_AFTER_PAYMENT_COMPACT,    SPC_AFTER_PAYMENT_RELAXED)
    sp_after_gdpr       = pair("after_gdpr",       SPC_AFTER_GDPR_COMPACT,       SPC_AFTER_GDPR_RELAXED)
    sp_after_validity   = pair("after_validity",   SPC_AFTER_VALIDITY_COMPACT,   SPC_AFTER_VALIDITY_RELAXED)

    panel = layout.get("panel", {})
    panel_title_after = float(panel.get("panel_title_after",
                             (SPC_PANEL_TITLE_AFTER_COMPACT if compact else SPC_PANEL_TITLE_AFTER_RELAXED)))
    panel_inner_pad   = float(panel.get("inner_pad", SPC_PANEL_INNER_PAD))

    tbl = layout.get("table", {})
    tbl_head_top    = float(tbl.get("head_top_pad",    pick(compact, TBL_HEAD_TOP_PAD_COMPACT,    TBL_HEAD_TOP_PAD_RELAXED)))
    tbl_head_bottom = float(tbl.get("head_bottom_pad", pick(compact, TBL_HEAD_BOTTOM_PAD_COMPACT, TBL_HEAD_BOTTOM_PAD_RELAXED)))
    tbl_row_top     = float(tbl.get("row_top_pad",     pick(compact, TBL_ROW_TOP_PAD_COMPACT,     TBL_ROW_TOP_PAD_RELAXED)))
    tbl_row_bottom  = float(tbl.get("row_bottom_pad",  pick(compact, TBL_ROW_BOTTOM_PAD_COMPACT,  TBL_ROW_BOTTOM_PAD_RELAXED)))

    meta = layout.get("meta", {})
    meta_cell_bottom = float(meta.get("cell_bottom_pad",
                                      pick(compact, META_CELL_BOTTOM_PAD_COMPACT, META_CELL_BOTTOM_PAD_RELAXED)))

    def choose(pairv): return pairv[0] if compact else pairv[1]
    return {
        "left_align_pad": left_align_pad,
        "after_logo":     choose(sp_after_logo),
        "after_title":    choose(sp_after_title),
        "meta_to_items":  choose(sp_meta_to_items),
        "after_items":    choose(sp_after_items),
        "after_version":  choose(sp_after_version),
        "after_total":    choose(sp_after_total),
        "after_payment":  choose(sp_after_payment),
        "after_gdpr":     choose(sp_after_gdpr),
        "after_validity": choose(sp_after_validity),
        "panel_title_after": panel_title_after,
        "panel_inner_pad":   panel_inner_pad,
        "tbl_head_top":    tbl_head_top,
        "tbl_head_bottom": tbl_head_bottom,
        "tbl_row_top":     tbl_row_top,
        "tbl_row_bottom":  tbl_row_bottom,
        "meta_cell_bottom": meta_cell_bottom,
    }

# =========================================================
# Image helpers
# =========================================================
def _resolve_image_src(src: str) -> Optional[str]:
    if not src: return None
    if src.startswith("/assets/"):
        fn = src.split("/assets/", 1)[1]
        p = os.path.join(UPLOAD_DIR, fn)
        return p if os.path.exists(p) else None
    if os.path.isabs(src):
        return src if os.path.exists(src) else None
    p = os.path.join(os.getcwd(), src)
    return p if os.path.exists(p) else None

def _image_flowables(specs: List[Dict[str, Any]], doc_width: float, styles: dict) -> List[Any]:
    flows: List[Any] = []
    for sp in specs:
        src = _resolve_image_src(sp.get("src",""))
        if not src: continue
        caption = sp.get("caption") or ""
        width_mm = float(sp.get("width_mm") or 0.0)
        align = (sp.get("align") or "CENTER").upper()
        target_w = (width_mm * mm) if width_mm > 0 else min(80 * mm, doc_width)
        try:
            img = RLImage(src)
            iw, ih = img.wrap(0, 0)
            if iw <= 0 or ih <= 0: continue
            scale = min(target_w / iw, 1.0)
            img.drawWidth = iw * scale
            img.drawHeight = ih * scale
            img.hAlign = align if align in ("LEFT", "CENTER", "RIGHT") else "CENTER"
            if caption:
                cap = Paragraph(caption, styles["ImgCaption"])
                flows.append(KeepTogether([img, Spacer(1, 2), cap, Spacer(1, 6)]))
            else:
                flows.append(KeepTogether([img, Spacer(1, 6)]))
        except Exception as e:
            print(f"[image] Skipping {src}: {e}")
            continue
    return flows

def _filter_images(images: List[Dict[str, Any]], page: str, section: str) -> List[Dict[str, Any]]:
    out = []
    for sp in images or []:
        p = (sp.get("page") or "offer").lower()
        sec = (sp.get("section") or "after_items").lower()
        if p in (page, "all") and sec == section:
            out.append(sp)
    return out

# =========================================================
# Tables
# =========================================================
def _blue_panel(panel: "SidePanel", base_font: str, compact: bool,
                panel_title_after: float, panel_inner_pad: float) -> Table:
    title_style = ParagraphStyle(
        "PanelTitle", parent=getSampleStyleSheet()["Normal"], fontName=base_font,
        fontSize=10 if compact else 11, leading=13 if compact else 14,
        textColor=colors.HexColor("#0F3D99"), spaceAfter=panel_title_after, alignment=0
    )
    body_style = ParagraphStyle(
        "PanelBody", parent=getSampleStyleSheet()["Normal"], fontName=base_font,
        fontSize=9 if compact else 10, leading=12 if compact else 13,
        textColor=colors.HexColor("#0F3D99"),
    )
    rows = [[Paragraph(panel.title or "Viktigt", title_style)]]
    for ln in panel.lines: rows.append([Paragraph(ln, body_style)])
    t = Table(rows, colWidths=[56 * mm], hAlign="RIGHT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#E9F2FF")),
        ("BOX", (0,0), (-1,-1), 0.6, colors.HexColor("#A8C5FF")),
        ("INNERPADDING", (0,0), (-1,-1), panel_inner_pad),
    ]))
    return t

def _offer_items_table(services_rows, base_font: str, compact: bool, frame_w: float,
                       left_align_pad: float, head_top: float, head_bottom: float,
                       row_top: float, row_bottom: float) -> Table:
    blue_text = colors.HexColor("#0F3D99")
    blue_bg = colors.HexColor("#E9F2FF")
    ss = getSampleStyleSheet()
    head = ParagraphStyle("TblHead", parent=ss["Normal"], fontName=base_font,
                          fontSize=10 if compact else 11, leading=13 if compact else 14,
                          textColor=blue_text)
    cell = ParagraphStyle("TblCell", parent=ss["Normal"], fontName=base_font,
                          fontSize=10 if compact else 11, leading=13 if compact else 14)
    header = [Paragraph("Tjänst", head), Paragraph("Timmar", head),
              Paragraph("Pris/timme", head), Paragraph("Totalsumma", head)]
    body = []
    for title, hours, rate, total in services_rows:
        body.append([Paragraph(title, cell), Paragraph(f"{int(hours):d}", cell),
                     Paragraph(fmt_currency(rate), cell), Paragraph(fmt_currency(total), cell)])
    w1 = 0.56 * frame_w; w2 = 0.14 * frame_w; w3 = 0.15 * frame_w
    w4 = frame_w - (w1 + w2 + w3)
    t = Table([header] + body, colWidths=[w1, w2, w3, w4], repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), base_font, 10 if compact else 11),
        ("BACKGROUND", (0,0), (-1,0), blue_bg),
        ("TEXTCOLOR", (0,0), (-1,0), blue_text),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("LINEBELOW", (0,1), (-1,-1), 0.25, colors.lightgrey),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 2),
        ("LEFTPADDING",  (0,0), (0,-1), left_align_pad),
        ("TOPPADDING",    (0,0), (-1,0),  head_top),
        ("BOTTOMPADDING", (0,0), (-1,0),  head_bottom),
        ("TOPPADDING",    (0,1), (-1,-1), row_top),
        ("BOTTOMPADDING", (0,1), (-1,-1), row_bottom),
    ]))
    return t

# =========================================================
# Data models
# =========================================================
class Customer(BaseModel):
    name: str
    orgnr: Optional[str] = None
    address: Optional[str] = None
    contact: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

class ImageSpec(BaseModel):
    src: str
    caption: Optional[str] = None
    width_mm: Optional[float] = None
    align: Literal["LEFT","CENTER","RIGHT"] = "CENTER"
    page: Literal["offer","order","invoice","all"] = "offer"
    section: Literal["meta","before_items","after_items","end"] = "after_items"

class Item(BaseModel):
    title: str
    hours: float = 0.0
    rate: float = 0.0
    quantity: float = 1.0
    type: Literal["labour", "material", "other"] = "labour"
    def line_total(self) -> float:
        if self.hours and self.rate:
            return round(self.hours * self.rate)
        return round(self.quantity * self.rate)

class Payment(BaseModel):
    terms: str = "30 dagar"
    late_fee: str = "8% enligt räntelagen"
    recipient: Optional[str] = None
    bankgiro: Optional[str] = None
    method: str = "Faktura"

class SidePanel(BaseModel):
    enabled: bool = False
    title: Optional[str] = "Viktigt"
    lines: List[str] = []

class DocCommon(BaseModel):
    customer: Customer
    items: List[Item]
    date: str = Field(default_factory=lambda: str(date.today()))
    updated: Optional[str] = None
    planned_start: Optional[str] = None
    delivery_time: Optional[str] = None
    payment: Payment = Field(default_factory=Payment)
    gdpr: Optional[str] = (
        "Vi hanterar kunduppgifter enligt Dataskyddsförordningen (GDPR). "
        "Personuppgifter används endast för att uppfylla avtal, fakturering och kontakt."
    )
    validity_days: int = 30
    note: Optional[str] = None
    vat_rate: float = 0.25
    rot_applies: bool = False
    rot_percent: float = 0.30
    logo_path: Optional[str] = None
    logo_dir: Optional[str] = None
    compact: bool = True
    side_panel: Optional[SidePanel] = Field(default_factory=SidePanel)
    layout: Optional[Dict[str, Any]] = None
    images: List[ImageSpec] = Field(default_factory=list)

class OfferPayload(DocCommon):
    offer_number: str = "#2025-091"
    version: str = "1.0"

class OrderPayload(DocCommon):
    order_number: str = "#-ORDER-001"
    offer_ref: Optional[str] = None

class InvoicePayload(DocCommon):
    invoice_number: str = "#-FAKTURA-001"
    order_ref: Optional[str] = None
    due_days: int = 30

# =========================================================
# Pricing
# =========================================================
class PricingResult(BaseModel):
    subtotal_labour: float
    subtotal_material: float
    subtotal_other: float
    subtotal_ex_vat: float
    rot_deduction: float
    vat_amount: float
    total_due: float

def compute_pricing(items: List[Item], vat_rate: float, rot_applies: bool, rot_percent: float) -> PricingResult:
    lab = sum(i.line_total() for i in items if i.type == "labour")
    mat = sum(i.line_total() for i in items if i.type == "material")
    oth = sum(i.line_total() for i in items if i.type == "other")
    subtotal = lab + mat + oth
    rot = round(lab * rot_percent) if rot_applies else 0
    taxable = max(subtotal - rot, 0)
    vat = round(taxable * vat_rate)
    total = taxable + vat
    return PricingResult(
        subtotal_labour=lab, subtotal_material=mat, subtotal_other=oth,
        subtotal_ex_vat=subtotal, rot_deduction=rot, vat_amount=vat, total_due=total
    )

# =========================================================
# OFFERT (layout & images)
# =========================================================
def build_offer_template_pdf(data: dict, outpath: str):
    compact = bool(data.get("compact", True))
    base_font = register_fonts()
    st = styles_offer(base_font, compact=compact)

    def compute_totals(services):
        rows, grand = [], 0
        for it in services:
            hours = float(it.get("hours", 0)); rate = float(it.get("rate", 0))
            total = round(hours * rate)
            rows.append((it["title"], hours, rate, total)); grand += total
        return rows, grand

    services_rows, total_sum = compute_totals(data["services"])

    safe_mkdir(outpath)
    doc = SimpleDocTemplate(
        outpath, pagesize=A4,
        leftMargin=25 * mm, rightMargin=25 * mm, topMargin=20 * mm, bottomMargin=18 * mm
    )
    story = []

    L = _get_layout_overrides(data.get("layout") or {}, compact=compact)
    images = data.get("images") or []

    # Logo
    lp = search_for_logo(data.get("logo_path"), data.get("logo_dir"))
    logo = logo_flowable(lp, max_w=70 * mm, max_h=45 * mm, align="CENTER")
    if logo:
        story.append(logo); story.append(Spacer(1, L["after_logo"]))

    # Title
    story.append(Paragraph("OFFERT", st["OfferTitle"]))
    story.append(Spacer(1, L["after_title"]))

    # Header meta + panel
    frame_w = float(doc.width)
    label_w = 36 * mm if compact else 38 * mm
    gap_w = 3 * mm
    label_style = ParagraphStyle("MetaLabel", parent=st["Body"], alignment=0)
    value_style = st["Body"]
    meta_rows = [
        [Paragraph("Kund:", label_style), "", Paragraph(data["customer"]["name"], value_style)],
        [Paragraph("Datum:", label_style), "", Paragraph(data["date"], value_style)],
        [Paragraph("Offertnummer:", label_style), "", Paragraph(data["offer_number"], value_style)],
        [Paragraph("Org.nr:", label_style), "", Paragraph(data["customer"].get("orgnr", "") or "", value_style)],
        [Paragraph("Adress:", label_style), "", Paragraph(data["customer"].get("address", "") or "", value_style)],
        [Paragraph("Kontaktperson:", label_style), "", Paragraph(data["customer"].get("contact", "") or "", value_style)],
        [Paragraph("Telefon:", label_style), "", Paragraph(data["customer"].get("phone", "") or "", value_style)],
        [Paragraph("E-post:", label_style), "", Paragraph(data["customer"].get("email", "") or "", value_style)],
    ]
    sp = data.get("side_panel") or {}
    sp_model = SidePanel(**sp) if isinstance(sp, dict) else sp

    if sp_model.enabled:
        panel_w = 56 * mm
        value_w = frame_w - panel_w - label_w - gap_w
        meta_tbl = Table(meta_rows, colWidths=[label_w, gap_w, value_w])
        meta_tbl.setStyle(TableStyle([
            ("FONT", (0,0), (-1,-1), base_font, 10 if compact else 11),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",  (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("LEFTPADDING",  (0,0), (0,-1), L["left_align_pad"]),
            ("BOTTOMPADDING",(0,0), (-1,-1), L["meta_cell_bottom"]),
        ]))
        right_cell = _blue_panel(sp_model, base_font, compact,
                                 panel_title_after=L["panel_title_after"],
                                 panel_inner_pad=L["panel_inner_pad"])
        header_row = Table([[meta_tbl, right_cell]], colWidths=[frame_w - panel_w, panel_w], hAlign="LEFT")
        header_row.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING",(0,0), (-1,-1), 0),
        ]))
        story.append(header_row)
    else:
        value_w = frame_w - label_w - gap_w
        meta_tbl = Table(meta_rows, colWidths=[label_w, gap_w, value_w])
        meta_tbl.setStyle(TableStyle([
            ("FONT", (0,0), (-1,-1), base_font, 10 if compact else 11),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",  (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("LEFTPADDING",  (0,0), (0,-1), L["left_align_pad"]),
            ("BOTTOMPADDING",(0,0), (-1,-1), L["meta_cell_bottom"]),
        ]))
        story.append(meta_tbl)

    # Images in 'meta'
    meta_imgs = _filter_images(images, "offer", "meta")
    if meta_imgs: story += _image_flowables(meta_imgs, doc.width, st)

    # Meta → Items
    story.append(Spacer(1, L["meta_to_items"]))

    # Images before items
    before_items_imgs = _filter_images(images, "offer", "before_items")
    if before_items_imgs: story += _image_flowables(before_items_imgs, doc.width, st)

    # Items table
    services_rows, total_sum = compute_totals(data["services"])
    tbl = _offer_items_table(
        services_rows, base_font, compact, frame_w,
        L["left_align_pad"], L["tbl_head_top"], L["tbl_head_bottom"], L["tbl_row_top"], L["tbl_row_bottom"]
    )
    story.append(tbl)
    story.append(Spacer(1, L["after_items"]))

    # Images after items
    after_items_imgs = _filter_images(images, "offer", "after_items")
    if after_items_imgs: story += _image_flowables(after_items_imgs, doc.width, st)

    # Version + timings
    version_line = (
        f"Offert V{data.get('version','1.0')} – Uppdaterad: {data.get('updated', data['date'])}<br/>"
        f"Planerad start: {data.get('planned_start','-')} / Leveranstid: {data.get('delivery_time','-')}"
    )
    story.append(Paragraph(version_line, st["Small"]))
    story.append(Spacer(1, L["after_version"]))

    # Total
    story.append(Paragraph(f"<b>Total summa:</b> {fmt_currency(total_sum)}", st["BoldBody"]))
    story.append(Spacer(1, L["after_total"]))

    # Payment
    pt = data["payment"]
    pay_lines = [
        f"Betaltid: {pt['terms']}",
        f"Dröjsmålsränta: {pt['late_fee']}",
        f"Fakturamottagare: {pt.get('recipient','') or ''}",
        f"Bankgiro: {pt.get('bankgiro','') or ''}",
        f"Betalningsmetod: {pt['method']}",
        "<i>*Observera att moms tillkommer*</i>",
    ]
    story.append(Paragraph("<br/>".join(pay_lines), st["Body"]))
    story.append(Spacer(1, L["after_payment"]))

    # GDPR
    if data.get("gdpr"):
        story.append(Paragraph(data["gdpr"], st["Small"]))
        story.append(Spacer(1, L["after_gdpr"]))

    # Validity
    story.append(Paragraph(
        f"Denna offert är giltig i {data['validity_days']} dagar från utskriftsdatum. Priser anges exklusive moms.",
        st["Body"],
    ))
    story.append(Spacer(1, L["after_validity"]))

    # Images end
    end_imgs = _filter_images(images, "offer", "end")
    if end_imgs: story += _image_flowables(end_imgs, doc.width, st)

    doc.build(story)

def offer_payload_to_template_data(p: "OfferPayload") -> dict:
    return {
        "customer": {
            "name": p.customer.name, "orgnr": p.customer.orgnr or "",
            "address": p.customer.address or "", "contact": p.customer.contact or "",
            "phone": p.customer.phone or "", "email": p.customer.email or "",
        },
        "date": p.date,
        "offer_number": p.offer_number,
        "services": [{"title": i.title, "hours": i.hours or i.quantity, "rate": i.rate} for i in p.items],
        "version": p.version,
        "updated": p.updated or p.date,
        "planned_start": p.planned_start or "",
        "delivery_time": p.delivery_time or "",
        "payment": {
            "terms": p.payment.terms, "late_fee": p.payment.late_fee,
            "recipient": p.payment.recipient or "", "bankgiro": p.payment.bankgiro or "",
            "method": p.payment.method,
        },
        "gdpr": p.gdpr or "",
        "validity_days": p.validity_days,
        "sign": {"name": (p.customer.contact or p.customer.name), "email": p.customer.email or ""},
        "logo_path": p.logo_path, "logo_dir": p.logo_dir,
        "compact": p.compact,
        "side_panel": p.side_panel.dict() if p.side_panel else SidePanel().dict(),
        "layout": p.layout or {},
        "images": [img.dict() for img in p.images] if p.images else [],
    }

# =========================================================
# ORDER & INVOICE (image-aware)
# =========================================================
def build_order_pdf(p: "OrderPayload", outpath: str):
    base = register_fonts(); st = styles_generic(base, compact=p.compact); safe_mkdir(outpath)
    doc = SimpleDocTemplate(outpath, pagesize=A4, leftMargin=25*mm, rightMargin=25*mm, topMargin=20*mm, bottomMargin=18*mm)
    story = []
    logo = logo_flowable(search_for_logo(p.logo_path, p.logo_dir), align="CENTER")
    if logo: story.append(logo); story.append(Spacer(1, pick(p.compact, SPC_AFTER_LOGO_COMPACT, SPC_AFTER_LOGO_RELAXED)))
    story.append(Paragraph("ORDERBEKRÄFTELSE", st["H1"]))
    story.append(Spacer(1, pick(p.compact, SPC_AFTER_TITLE_COMPACT, SPC_AFTER_TITLE_RELAXED)))
    frame_w = float(doc.width); label_w = 36*mm; gap_w = 3*mm
    label_style = ParagraphStyle("MetaLabel", parent=st["Body"], alignment=0); value_style = st["Body"]
    meta_rows = [
        [Paragraph("Kund:", label_style), "", Paragraph(p.customer.name, value_style)],
        [Paragraph("Datum:", label_style), "", Paragraph(p.date, value_style)],
        [Paragraph("Ordernr:", label_style), "", Paragraph(p.order_number, value_style)],
        [Paragraph("Ref offert:", label_style), "", Paragraph(p.offer_ref or "-", value_style)],
        [Paragraph("Adress:", label_style), "", Paragraph(p.customer.address or "", value_style)],
        [Paragraph("Kontakt:", label_style), "", Paragraph(p.customer.contact or "", value_style)],
        [Paragraph("Telefon:", label_style), "", Paragraph(p.customer.phone or "", value_style)],
        [Paragraph("E-post:", label_style), "", Paragraph(p.customer.email or "", value_style)],
    ]
    meta_tbl = Table(meta_rows, colWidths=[label_w, gap_w, frame_w - label_w - gap_w])
    meta_tbl.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), base, 10),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("LEFTPADDING",  (0,0), (0,-1), LEFT_ALIGN_PAD),
        ("BOTTOMPADDING",(0,0), (-1,-1), pick(p.compact, META_CELL_BOTTOM_PAD_COMPACT, META_CELL_BOTTOM_PAD_RELAXED)),
    ]))
    story.append(meta_tbl)

    items = [["Tjänst/Artikel", "Typ", "Tid/Antal", "Pris", "Summa"]]
    for i in p.items:
        qty = f"{int(i.hours)} h" if i.hours else f"{i.quantity:g} st"
        items.append([i.title, i.type, qty, fmt_currency(i.rate), fmt_currency(i.line_total())])
    tbl = Table(items, colWidths=[0.48*frame_w, 0.14*frame_w, 0.12*frame_w, 0.12*frame_w, 0.14*frame_w], repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), base, 10),
        ("ALIGN", (2,1), (-1,-1), "RIGHT"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("LINEBELOW", (0,0), (-1,0), 0.75, colors.black),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 2),
        ("LEFTPADDING",  (0,0), (0,-1), LEFT_ALIGN_PAD),
        ("TOPPADDING",    (0,0), (-1,0),  pick(p.compact, ORDINV_HEAD_TOP_PAD_COMPACT,    ORDINV_HEAD_TOP_PAD_RELAXED)),
        ("BOTTOMPADDING", (0,0), (-1,0),  pick(p.compact, ORDINV_HEAD_BOTTOM_PAD_COMPACT, ORDINV_HEAD_BOTTOM_PAD_RELAXED)),
        ("TOPPADDING",    (0,1), (-1,-1), pick(p.compact, ORDINV_ROW_TOP_PAD_COMPACT,     ORDINV_ROW_TOP_PAD_RELAXED)),
        ("BOTTOMPADDING", (0,1), (-1,-1), pick(p.compact, ORDINV_ROW_BOTTOM_PAD_COMPACT,  ORDINV_ROW_BOTTOM_PAD_RELAXED)),
    ]))
    story.append(tbl)
    doc.build(story)

def build_invoice_pdf(p: "InvoicePayload", outpath: str):
    base = register_fonts(); st = styles_generic(base, compact=p.compact); safe_mkdir(outpath)
    pricing = compute_pricing(p.items, p.vat_rate, p.rot_applies, p.rot_percent)
    doc = SimpleDocTemplate(outpath, pagesize=A4, leftMargin=25*mm, rightMargin=25*mm, topMargin=20*mm, bottomMargin=18*mm)
    story = []
    logo = logo_flowable(search_for_logo(p.logo_path, p.logo_dir), align="CENTER")
    if logo: story.append(logo); story.append(Spacer(1, pick(p.compact, SPC_AFTER_LOGO_COMPACT, SPC_AFTER_LOGO_RELAXED)))
    story.append(Paragraph("FAKTURA", st["H1"]))
    story.append(Spacer(1, pick(p.compact, SPC_AFTER_TITLE_COMPACT, SPC_AFTER_TITLE_RELAXED)))

    frame_w = float(doc.width)
    items = [["Tjänst/Artikel", "Typ", "Tid/Antal", "Pris", "Summa"]]
    for i in p.items:
        qty = f"{int(i.hours)} h" if i.hours else f"{i.quantity:g} st"
        items.append([i.title, i.type, qty, fmt_currency(i.rate), fmt_currency(i.line_total())])
    tbl = Table(items, colWidths=[0.48*frame_w, 0.14*frame_w, 0.12*frame_w, 0.12*frame_w, 0.14*frame_w], repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), base, 10),
        ("ALIGN", (2,1), (-1,-1), "RIGHT"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("LINEBELOW", (0,0), (-1,0), 0.75, colors.black),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 2),
        ("LEFTPADDING",  (0,0), (0,-1), LEFT_ALIGN_PAD),
        ("TOPPADDING",    (0,0), (-1,0),  pick(p.compact, ORDINV_HEAD_TOP_PAD_COMPACT,    ORDINV_HEAD_TOP_PAD_RELAXED)),
        ("BOTTOMPADDING", (0,0), (-1,0),  pick(p.compact, ORDINV_HEAD_BOTTOM_PAD_COMPACT, ORDINV_HEAD_BOTTOM_PAD_RELAXED)),
        ("TOPPADDING",    (0,1), (-1,-1), pick(p.compact, ORDINV_ROW_TOP_PAD_COMPACT,     ORDINV_ROW_TOP_PAD_RELAXED)),
        ("BOTTOMPADDING", (0,1), (-1,-1), pick(p.compact, ORDINV_ROW_BOTTOM_PAD_COMPACT,  ORDINV_ROW_BOTTOM_PAD_RELAXED)),
    ]))
    story.append(tbl)

    story.append(Spacer(1, pick(p.compact, SPC_AFTER_ITEMS_COMPACT, SPC_AFTER_ITEMS_RELAXED)))
    story.append(Paragraph(f"Exkl. moms: {fmt_currency(pricing.subtotal_ex_vat)}", st["Body"]))
    if p.rot_applies:
        story.append(Paragraph(f"ROT-avdrag ({int(p.rot_percent*100)}% på arbete): -{fmt_currency(pricing.rot_deduction)}", st["Body"]))
    story.append(Paragraph(f"Moms ({int(p.vat_rate*100)}%): {fmt_currency(pricing.vat_amount)}", st["Body"]))
    story.append(Paragraph(f"<b>Att betala:</b> {fmt_currency(pricing.total_due)}", st["Strong"]))
    doc.build(story)

# =========================================================
# Public URL helpers + file logging
# =========================================================
FILE_LOG: List[Dict[str, Any]] = []  # [{time, type, flow_id, filename, rel_url, abs_url, path}]

def _rel_url_for_path(path: str) -> Optional[str]:
    # Normalize absolute to relative served URL
    abspath = os.path.abspath(path)
    up = os.path.abspath(UPLOAD_DIR)
    gp = os.path.abspath(GENERATED_DIR)
    if abspath.startswith(up + os.sep):
        return f"/assets/{os.path.basename(path)}"
    if abspath.startswith(gp + os.sep):
        return f"/generated/{os.path.basename(path)}"
    # Already a served url?
    if path.startswith("/assets/") or path.startswith("/generated/"):
        return path
    return None

def _abs_url(request: Request, rel_url: Optional[str]) -> Optional[str]:
    if not rel_url: return None
    base = str(request.base_url).rstrip("/")
    return f"{base}{rel_url}"

def _log_file(kind: str, flow_id: Optional[str], path: str, request: Optional[Request] = None):
    rel = _rel_url_for_path(path)
    absu = _abs_url(request, rel) if request else None
    FILE_LOG.append({
        "time": _now_iso(),
        "type": kind,
        "flow_id": flow_id,
        "filename": os.path.basename(path),
        "rel_url": rel,
        "abs_url": absu,
        "path": path,
    })

# =========================================================
# API: Offer/Order/Invoice (existing)
# =========================================================
@app.post("/offer")
def create_offer(payload: OfferPayload, request: Request):
    out = os.path.join(GENERATED_DIR, f"Offert_{payload.customer.name}_{payload.date}.pdf")
    safe_mkdir(out)
    build_offer_template_pdf(offer_payload_to_template_data(payload), out)
    totals = sum(i.line_total() for i in payload.items)
    _log_file("offer_pdf", None, out, request)
    return JSONResponse({"ok": True, "pdf_path": out,
                         "pdf_url": _abs_url(request, _rel_url_for_path(out)),
                         "totals_ex_vat": totals})

@app.post("/order")
def create_order(payload: OrderPayload, request: Request):
    out = os.path.join(GENERATED_DIR, f"Order_{payload.customer.name}_{payload.date}.pdf")
    safe_mkdir(out)
    build_order_pdf(payload, out)
    _log_file("order_pdf", None, out, request)
    return JSONResponse({"ok": True, "pdf_path": out,
                         "pdf_url": _abs_url(request, _rel_url_for_path(out))})

@app.post("/invoice")
def create_invoice(payload: InvoicePayload, request: Request):
    out = os.path.join(GENERATED_DIR, f"Faktura_{payload.customer.name}_{payload.date}.pdf")
    safe_mkdir(out)
    pricing = compute_pricing(payload.items, payload.vat_rate, payload.rot_applies, payload.rot_percent)
    build_invoice_pdf(payload, out)
    _log_file("invoice_pdf", None, out, request)
    return JSONResponse({"ok": True, "pdf_path": out,
                         "pdf_url": _abs_url(request, _rel_url_for_path(out)),
                         "pricing": pricing.dict()})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "docs-service",
        "docs": "/docs",
        "endpoints": [
            "/flow/start", "/flow/create-offer/{id}", "/flow/send-offer/{id}",
            "/flow/generate-order/{id}", "/flow/generate-invoice/{id}",
            "/flow/{id}", "/flows", "/files",
            "/ocr/analyze", "/ocr/offer-draft", "/ocr/receipt-draft", "/flow/start-ocr",
        ]
    }

# =========================================================
# Uploads API
# =========================================================
@app.post("/assets/upload")
async def assets_upload(request: Request, file: UploadFile = File(...), desired_name: Optional[str] = Form(None)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (*IMAGE_EXTS, *PDF_EXTS):
        raise HTTPException(400, f"Unsupported file type: {ext}")
    base = desired_name or file.filename or "file"
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", base)
    fname = f"{uuid4().hex[:8]}_{safe}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as out:
        shutil.copyfileobj(io.BytesIO(await file.read()), out)
    rel = f"/assets/{fname}"
    absu = _abs_url(request, rel)
    _log_file("upload", None, fpath, request)
    return {"ok": True, "path": fpath, "rel_url": rel, "abs_url": absu, "filename": fname}

# =========================================================
# Voice + NLP (existing)
# =========================================================
def _num(s: str) -> float:
    return float(s.replace(" ", "").replace(",", ".").strip())

_HOURS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:h|tim(?:me|mar)?)", re.I)
_RATE_RE  = re.compile(r"(?:à|a|per|/)\s*(\d{2,5}(?:[.,]\d{1,2})?)\s*(?:kr|sek)?|(\d{2,5}(?:[.,]\d{1,2})?)\s*(?:kr|sek)\s*(?:/|per)?\s*(?:h|tim)", re.I)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

def rule_based_offer_from_transcript(txt: str) -> dict:
    hours = 0.0; rate = 0.0
    mh = _HOURS_RE.search(txt); 
    if mh: hours = _num(mh.group(1))
    mr = _RATE_RE.search(txt)
    if mr:
        g = mr.group(1) or mr.group(2)
        if g: rate = _num(g)
    if hours <= 0: hours = 8.0
    if rate  <= 0: rate  = 800.0
    email = (_EMAIL_RE.search(txt) or [None])[0]
    customer_name = "Okänd kund"
    mname = re.search(r"(?:kund|företag|bolag|företagsnamn)[:\s]+(.{3,60})", txt, re.I)
    if mname: customer_name = mname.group(1).strip().strip(".,")
    return {
        "customer": {
            "name": customer_name, "orgnr": "", "address": "",
            "contact": "", "phone": "", "email": email or "",
        },
        "date": str(date.today()),
        "offer_number": f"#O-{date.today().strftime('%y%m%d')}-{uuid4().hex[:4].upper()}",
        "items": [{"title": "Arbete", "hours": hours, "rate": rate, "type": "labour"}],
        "version": "1.0",
        "planned_start": "", "delivery_time": "30 dagar",
        "payment": {"terms": "30 dagar", "late_fee": "8% enligt räntelagen", "method": "Faktura"},
        "gdpr": ("Vi hanterar kunduppgifter enligt Dataskyddsförordningen (GDPR). "
                 "Personuppgifter används endast för att uppfylla avtal, fakturering och kontakt."),
        "validity_days": 30,
        "vat_rate": 0.25, "rot_applies": False, "rot_percent": 0.30,
        "compact": True,
        "images": []
    }

@app.post("/stt")
async def stt(audio: UploadFile = File(None), language: str = Form("sv"), text: Optional[str] = Form(None)):
    if _client and audio is not None:
        data = await audio.read()
        bio = io.BytesIO(data); bio.name = audio.filename or "audio.webm"
        try:
            tr = _client.audio.transcriptions.create(model=OPENAI_MODEL_STT, file=bio, language=language)
            return {"ok": True, "text": tr.text}
        except Exception as e:
            raise HTTPException(500, f"STT failed: {e}")
    if text:
        return {"ok": True, "text": text}
    raise HTTPException(400, "Provide audio (with OPENAI_API_KEY) or 'text' form field.")

@app.post("/nlp/build-offer")
async def nlp_build_offer(payload: dict):
    txt = (payload or {}).get("transcript", "") or ""
    use_llm = bool((payload or {}).get("use_llm", False))
    base_json = rule_based_offer_from_transcript(txt)
    if not use_llm or not _client:
        return {"ok": True, "offer": base_json, "source": "rule-based"}
    try:
        prompt = ("Return ONLY JSON matching the seed OfferPayload keys; keep key names exactly. "
                  "If a field is missing, keep the seed value.")
        resp = _client.responses.create(
            model=OPENAI_MODEL_PARSER,
            input=[
                {"role":"system","content": prompt},
                {"role":"user","content": f"seed: {json.dumps(base_json, ensure_ascii=False)}"},
                {"role":"user","content": f"transcript: {txt}"},
            ],
            response_format={"type":"json_object"}
        )
        refined = json.loads(resp.output_text)
        base_json.update({k: refined.get(k, base_json.get(k)) for k in base_json})
        return {"ok": True, "offer": base_json, "source": "openai+rule-based"}
    except Exception as e:
        return {"ok": True, "offer": base_json, "source": f"rule-based (LLM failed: {e})"}

@app.post("/tts")
def tts(text: str = Form(...), voice: str = Form("alloy"), fmt: str = Form("mp3")):
    if not _client:
        raise HTTPException(500, "OpenAI not configured (set OPENAI_API_KEY).")
    out_path = f"/tmp/tts_{uuid4().hex}.{fmt}"
    try:
        with _client.audio.speech.with_streaming_response.create(
            model=OPENAI_MODEL_TTS, voice=voice, input=text, format=fmt
        ) as resp:
            resp.stream_to_file(out_path)
        media = "audio/mpeg" if fmt == "mp3" else "audio/wav"
        return FileResponse(out_path, media_type=media, filename=f"speech.{fmt}")
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")

# =========================================================
# OCR utilities
# =========================================================
def _pdf_extract_text(pdf_bytes: bytes) -> str:
    if not HAS_PYPDF: return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip(): parts.append(t)
        return "\n\n".join(parts).strip()
    except Exception as e:
        print("[pdf] extract_text failed:", e)
        return ""

def _img_ocr_tesseract(img_bytes: bytes, lang: str = "swe+eng") -> str:
    if not HAS_TESS: return ""
    try:
        im = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(im, lang=lang)
    except Exception as e:
        print("[tesseract] failed:", e)
        return ""

def _pdf_ocr_tesseract(pdf_bytes: bytes, lang: str = "swe+eng") -> str:
    if not (HAS_TESS and HAS_PDF2IMG): return ""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=300)
        out = []
        for p in pages:
            with io.BytesIO() as buf:
                p.save(buf, format="PNG")
                out.append(_img_ocr_tesseract(buf.getvalue(), lang=lang))
        return "\n\n".join(out)
    except Exception as e:
        print("[tesseract] pdf2image failed:", e)
        return ""

def _ocr_openai(img_bytes: bytes, filename: str) -> str:
    if not _client: return ""
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    b64 = base64.b64encode(img_bytes).decode("ascii")
    try:
        resp = _client.responses.create(
            model=OPENAI_MODEL_VISION,
            input=[
                {"role":"user","content":[
                    {"type":"input_text","text":"Extract all text content as plain UTF-8. Keep lines and numbers readable."},
                    {"type":"input_image","image_data":{"data": b64, "mime_type": mime}}
                ]}
            ],
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        print("[openai vision] failed:", e)
        return ""

def _file_ext(fn: str) -> str:
    return (os.path.splitext(fn or "")[1] or "").lower()

# =========================================================
# OCR endpoints
# =========================================================
@app.post("/ocr/analyze")
async def ocr_analyze(file: UploadFile = File(None), method: str = Form("auto"), lang: str = Form("swe+eng")):
    if file is None:
        raise HTTPException(400, "Send a file")
    fn = file.filename or "upload"
    ext = _file_ext(fn)
    data = await file.read()

    text = ""
    used = None

    if method == "tesseract":
        if ext in PDF_EXTS:
            text = _pdf_ocr_tesseract(data, lang=lang); used = "tesseract_pdf"
        else:
            text = _img_ocr_tesseract(data, lang=lang); used = "tesseract_img"

    elif method == "openai":
        if ext in PDF_EXTS:
            t = _pdf_extract_text(data)
            if not t and HAS_TESS and HAS_PDF2IMG:
                text = _pdf_ocr_tesseract(data, lang=lang); used = "openai_fallback_tesseract_pdf"
            elif not t:
                raise HTTPException(400, "OpenAI OCR for PDFs needs page images; install poppler+tesseract or upload as image.")
            else:
                text = t; used = "pdf_text"
        else:
            text = _ocr_openai(data, fn); used = "openai_vision"

    else:  # auto
        if ext in PDF_EXTS:
            t = _pdf_extract_text(data)
            if t and len(t.strip()) > 40:
                text = t; used = "pdf_text"
            else:
                if HAS_TESS and HAS_PDF2IMG:
                    text = _pdf_ocr_tesseract(data, lang=lang); used = "tesseract_pdf"
                else:
                    raise HTTPException(400, "Scanned PDF without tesseract/poppler. Install them or upload as an image.")
        else:
            if HAS_TESS:
                t = _img_ocr_tesseract(data, lang=lang)
                if t.strip():
                    text = t; used = "tesseract_img"
                elif _client:
                    text = _ocr_openai(data, fn); used = "openai_vision_after_tesseract_empty"
            elif _client:
                text = _ocr_openai(data, fn); used = "openai_vision"
            else:
                raise HTTPException(400, "No OCR available. Install tesseract or set OPENAI_API_KEY.")

    return {"ok": True, "text": text, "meta": {"method_used": used, "filename": fn}}

@app.post("/ocr/offer-draft")
async def ocr_offer_draft(file: UploadFile = File(None), text: Optional[str] = Form(None), lang: str = Form("swe+eng")):
    if not text and not file:
        raise HTTPException(400, "Send a file or 'text'.")

    ocr_text = text or ""
    used = None
    if not ocr_text and file:
        res = await ocr_analyze(file=file, method="auto", lang=lang)  # type: ignore
        ocr_text = res["text"]; used = res["meta"]["method_used"]

    draft = rule_based_offer_from_transcript(ocr_text or "")
    return {"ok": True, "offer": draft, "source": used or "text"}

# Focused OCR for receipts → bokföring draft
_DATE_RE = re.compile(r"(20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]20\d{2})")
_CURRENCY_RE = re.compile(r"\b(?:SEK|kr)\b", re.I)
_TOTAL_HINT_RE = re.compile(r"(summa|totalt|att\s*betala|total)", re.I)
_VAT_HINT_RE = re.compile(r"(moms|vat)", re.I)
_NUMBER_RE = re.compile(r"(\d{1,3}(?:[ .]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})")

def _parse_receipt_text(txt: str) -> dict:
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    currency = "SEK" if _CURRENCY_RE.search(txt) else "SEK"
    date_m = _DATE_RE.search(txt)
    date_val = date_m.group(0) if date_m else ""

    total = None
    for ln in lines:
        if _TOTAL_HINT_RE.search(ln):
            nums = [m.group(1) for m in _NUMBER_RE.finditer(ln)]
            vals = []
            for n in nums:
                if "," in n and n.count(",")==1 and "." not in n:
                    vals.append(float(n.replace(" ", "").replace(".", "").replace(",", ".")))
                else:
                    vals.append(float(n.replace(" ", "").replace(",", "")))
            if vals:
                cand = max(vals)
                total = cand if (total is None or cand > total) else total
    if total is None:
        nums = [m.group(1) for m in _NUMBER_RE.finditer(txt)]
        vals = []
        for n in nums:
            if "," in n and n.count(",")==1 and "." not in n:
                vals.append(float(n.replace(" ", "").replace(".", "").replace(",", ".")))
            else:
                vals.append(float(n.replace(" ", "").replace(",", "")))
        if vals:
            total = max(vals)

    vat = None
    for ln in lines:
        if _VAT_HINT_RE.search(ln):
            nums = [m.group(1) for m in _NUMBER_RE.finditer(ln)]
            for n in nums:
                try:
                    if "," in n and n.count(",")==1 and "." not in n:
                        v = float(n.replace(" ", "").replace(".", "").replace(",", "."))
                    else:
                        v = float(n.replace(" ", "").replace(",", ""))
                    vat = v
                except:
                    continue
            if vat is not None:
                break

    merchant = lines[0] if lines else "Okänd"

    return {
        "merchant": merchant,
        "date": date_val,
        "currency": currency,
        "total": total,
        "vat_amount": vat,
        "source_confidence": "heuristic"
    }

@app.post("/ocr/receipt-draft")
async def ocr_receipt_draft(file: UploadFile = File(None), text: Optional[str] = Form(None), lang: str = Form("swe+eng")):
    if not text and not file:
        raise HTTPException(400, "Send a file or 'text'.")
    ocr_text = text or ""
    used = None
    if not ocr_text and file:
        res = await ocr_analyze(file=file, method="auto", lang=lang)  # reuse existing OCR
        ocr_text = res["text"]; used = res["meta"]["method_used"]
    draft = _parse_receipt_text(ocr_text or "")
    return {"ok": True, "receipt": draft, "source": used or "text"}

@app.post("/flow/start-ocr")
async def flow_start_ocr(file: UploadFile = File(...), lang: str = Form("swe+eng")):
    res = await ocr_analyze(file=file, method="auto", lang=lang)  # type: ignore
    text = res["text"]
    base_json = rule_based_offer_from_transcript(text)
    flow_id = uuid4().hex[:12]
    FLOWS[flow_id] = {
        "transcript": text,
        "offer": base_json,
        "offer_pdf": None,
        "order_pdf": None,
        "invoice_pdf": None,
        "pricing": None,
        "status": {
            "offer_sent": False,
            "order_generated": False,
            "invoice_generated": False,
        },
        "timestamps": {"ocr_started_at": _now_iso()},
        "history": [{"time": _now_iso(), "event": "flow_started_ocr"}],
        "ocr_meta": res["meta"]
    }
    return {"ok": True, "flow_id": flow_id, "transcript": text, "offer": base_json, "ocr": res["meta"]}

# =========================================================
# Orchestration flow + history + listing
# =========================================================
FLOWS: Dict[str, Dict[str, Any]] = {}

def _flow_log(flow: Dict[str, Any], event: str, extra: Optional[Dict[str, Any]] = None):
    flow.setdefault("history", []).append({"time": _now_iso(), "event": event, "data": extra or {}})

@app.post("/flow/start")
async def flow_start(audio: UploadFile = File(None),
                     language: str = Form("sv"),
                     text: Optional[str] = Form(None),
                     use_llm: bool = Form(False)):
    if _client and audio is not None:
        data = await audio.read()
        bio = io.BytesIO(data); bio.name = audio.filename or "audio.webm"
        tr = _client.audio.transcriptions.create(model=OPENAI_MODEL_STT, file=bio, language=language)
        transcript = tr.text
    elif text:
        transcript = text
    else:
        raise HTTPException(400, "Provide audio (with OPENAI_API_KEY) or 'text' form field.")
    base_json = rule_based_offer_from_transcript(transcript)
    flow_id = uuid4().hex[:12]
    FLOWS[flow_id] = {
        "transcript": transcript,
        "offer": base_json,
        "offer_pdf": None,
        "order_pdf": None,
        "invoice_pdf": None,
        "pricing": None,
        "status": {
            "offer_sent": False,
            "order_generated": False,
            "invoice_generated": False,
        },
        "timestamps": {"flow_started_at": _now_iso()},
        "history": [{"time": _now_iso(), "event": "flow_started"}]
    }
    return {"ok": True, "flow_id": flow_id, "transcript": transcript, "offer": base_json}

@app.post("/flow/create-offer/{flow_id}")
async def flow_create_offfer(flow_id: str, request: Request, payload: dict):
    f = FLOWS.get(flow_id)
    if not f: raise HTTPException(404, "Flow not found.")
    offer = payload.get("offer") or f["offer"]
    layout = payload.get("layout") or {}
    images = payload.get("images") or offer.get("images") or []
    if "services" not in offer and "items" in offer:
        offer["services"] = [{"title": i.get("title",""), "hours": i.get("hours") or i.get("quantity",0), "rate": i.get("rate",0)} for i in offer["items"]]
    if "services" not in offer:
        offer["services"] = [{"title": "Arbete", "hours": 8, "rate": 800}]
    offer["layout"] = layout
    offer["images"] = images
    out = os.path.join(GENERATED_DIR, f"Offert_{offer['customer']['name']}_{offer['date']}_{flow_id}.pdf")
    safe_mkdir(out)
    build_offer_template_pdf(offer, out)
    f["offer"] = offer; f["offer_pdf"] = out
    totals = sum(int(r["hours"] or 0) * float(r["rate"] or 0) for r in offer["services"])
    _flow_log(f, "offer_generated", {"totals_ex_vat": totals})
    _log_file("offer_pdf", flow_id, out, request)
    return {
        "ok": True, "flow_id": flow_id,
        "offer_pdf": out,
        "offer_pdf_url": _abs_url(request, _rel_url_for_path(out)),
        "totals_ex_vat": totals
    }

@app.post("/flow/confirm/{flow_id}")
async def flow_confirm(flow_id: str, request: Request):
    f = FLOWS.get(flow_id)
    if not f: raise HTTPException(404, "Flow not found.")
    offer = f["offer"]
    items = [Item(title=s["title"], hours=float(s.get("hours",0)), rate=float(s.get("rate",0)), type="labour")
             for s in offer.get("services", [])]
    order_payload = OrderPayload(
        customer=Customer(**offer["customer"]), items=items, date=offer["date"],
        order_number=f"#ORD-{flow_id.upper()}", offer_ref=offer.get("offer_number",""),
        planned_start=offer.get("planned_start",""), delivery_time=offer.get("delivery_time",""),
        payment=Payment(**offer["payment"]), gdpr=offer.get("gdpr",""),
        compact=offer.get("compact", True), images=[ImageSpec(**img) for img in (offer.get("images") or [])]
    )
    order_out = os.path.join(GENERATED_DIR, f"Order_{offer['customer']['name']}_{offer['date']}_{flow_id}.pdf")
    build_order_pdf(order_payload, order_out); f["order_pdf"] = order_out
    f.setdefault("status", {})["order_generated"] = True
    f.setdefault("timestamps", {})["order_generated_at"] = _now_iso()
    _flow_log(f, "order_generated")
    _log_file("order_pdf", flow_id, order_out, request)

    inv_payload = InvoicePayload(
        customer=Customer(**offer["customer"]), items=items, date=offer["date"],
        invoice_number=f"#INV-{flow_id.upper()}", order_ref=order_payload.order_number,
        payment=Payment(**offer["payment"]), gdpr=offer.get("gdpr",""),
        compact=offer.get("compact", True), vat_rate=offer.get("vat_rate",0.25),
        rot_applies=offer.get("rot_applies", False), rot_percent=offer.get("rot_percent",0.30),
        images=[ImageSpec(**img) for img in (offer.get("images") or [])]
    )
    invoice_out = os.path.join(GENERATED_DIR, f"Faktura_{offer['customer']['name']}_{offer['date']}_{flow_id}.pdf")
    pricing = compute_pricing(inv_payload.items, inv_payload.vat_rate, inv_payload.rot_applies, inv_payload.rot_percent)
    build_invoice_pdf(inv_payload, invoice_out)
    f["invoice_pdf"] = invoice_out; f["pricing"] = pricing.dict()
    f["status"]["invoice_generated"] = True
    f["timestamps"]["invoice_generated_at"] = _now_iso()
    _flow_log(f, "invoice_generated", {"pricing": pricing.dict()})
    _log_file("invoice_pdf", flow_id, invoice_out, request)
    return {
        "ok": True, "flow_id": flow_id,
        "order_pdf": order_out, "order_pdf_url": _abs_url(request, _rel_url_for_path(order_out)),
        "invoice_pdf": invoice_out, "invoice_pdf_url": _abs_url(request, _rel_url_for_path(invoice_out)),
        "pricing": pricing.dict()
    }

# =========================================================
# Autofill helpers + new endpoints for 3-column UI
# =========================================================
def _autofill_order_from_offer(offer: dict) -> "OrderPayload":
    items = []
    for s in offer.get("services", []):
        items.append(Item(
            title=s.get("title","Arbete"),
            hours=float(s.get("hours",0)),
            rate=float(s.get("rate",0)),
            type="labour"
        ))
    return OrderPayload(
        customer=Customer(**offer["customer"]),
        items=items,
        date=offer.get("date", str(date.today())),
        order_number=offer.get("order_number") or f"#ORD-{uuid4().hex[:6].upper()}",
        offer_ref=offer.get("offer_number",""),
        planned_start=offer.get("planned_start",""),
        delivery_time=offer.get("delivery_time",""),
        payment=Payment(**offer.get("payment", {})),
        gdpr=offer.get("gdpr",""),
        compact=bool(offer.get("compact", True)),
        logo_path=offer.get("logo_path"),
        logo_dir=offer.get("logo_dir"),
        side_panel=SidePanel(**offer.get("side_panel", {})) if offer.get("side_panel") else SidePanel(),
        layout=offer.get("layout") or {},
        images=[ImageSpec(**img) for img in (offer.get("images") or [])]
    )

def _autofill_invoice_from_order(offer: dict, order: "OrderPayload") -> "InvoicePayload":
    return InvoicePayload(
        customer=order.customer,
        items=order.items,
        date=offer.get("date", str(date.today())),
        invoice_number=offer.get("invoice_number") or f"#INV-{uuid4().hex[:6].upper()}",
        order_ref=order.order_number,
        payment=order.payment,
        gdpr=offer.get("gdpr",""),
        compact=order.compact,
        logo_path=offer.get("logo_path"),
        logo_dir=offer.get("logo_dir"),
        vat_rate=float(offer.get("vat_rate", 0.25)),
        rot_applies=bool(offer.get("rot_applies", False)),
        rot_percent=float(offer.get("rot_percent", 0.30)),
        images=[ImageSpec(**img) for img in (offer.get("images") or [])]
    )

@app.post("/flow/send-offer/{flow_id}")
def flow_send_offer(flow_id: str):
    f = FLOWS.get(flow_id)
    if not f: raise HTTPException(404, "Flow not found.")
    f.setdefault("status", {})["offer_sent"] = True
    f.setdefault("timestamps", {})["offer_sent_at"] = _now_iso()
    _flow_log(f, "offer_sent")
    return {"ok": True, "flow_id": flow_id, "status": f["status"], "timestamps": f["timestamps"]}

@app.post("/flow/generate-order/{flow_id}")
def flow_generate_order(flow_id: str, request: Request):
    f = FLOWS.get(flow_id)
    if not f: raise HTTPException(404, "Flow not found.")
    offer = f["offer"]
    order_payload = _autofill_order_from_offer(offer)
    out = os.path.join(GENERATED_DIR, f"Order_{offer['customer']['name']}_{offer['date']}_{flow_id}.pdf")
    build_order_pdf(order_payload, out)
    f["order_pdf"] = out
    f.setdefault("status", {})["order_generated"] = True
    f.setdefault("timestamps", {})["order_generated_at"] = _now_iso()
    _flow_log(f, "order_generated_step")
    _log_file("order_pdf", flow_id, out, request)
    return {
        "ok": True,
        "flow_id": flow_id,
        "order_pdf": out,
        "order_pdf_url": _abs_url(request, _rel_url_for_path(out)),
        "order": order_payload.model_dump(),
        "status": f["status"]
    }

@app.post("/flow/generate-invoice/{flow_id}")
def flow_generate_invoice(flow_id: str, request: Request):
    f = FLOWS.get(flow_id)
    if not f: raise HTTPException(404, "Flow not found.")
    offer = f["offer"]
    order_payload = _autofill_order_from_offer(offer)
    inv_payload = _autofill_invoice_from_order(offer, order_payload)
    out = os.path.join(GENERATED_DIR, f"Faktura_{offer['customer']['name']}_{offer['date']}_{flow_id}.pdf")
    pricing = compute_pricing(inv_payload.items, inv_payload.vat_rate, inv_payload.rot_applies, inv_payload.rot_percent)
    build_invoice_pdf(inv_payload, out)
    f["invoice_pdf"] = out
    f["pricing"] = pricing.dict()
    f.setdefault("status", {})["invoice_generated"] = True
    f.setdefault("timestamps", {})["invoice_generated_at"] = _now_iso()
    _flow_log(f, "invoice_generated_step", {"pricing": pricing.dict()})
    _log_file("invoice_pdf", flow_id, out, request)
    return {
        "ok": True,
        "flow_id": flow_id,
        "invoice_pdf": out,
        "invoice_pdf_url": _abs_url(request, _rel_url_for_path(out)),
        "invoice": inv_payload.model_dump(),
        "pricing": pricing.dict(),
        "status": f["status"]
    }

# --- Status APIs (polling-friendly) ---
@app.get("/flow/{flow_id}")
def flow_get(flow_id: str, request: Request):
    f = FLOWS.get(flow_id)
    if not f: raise HTTPException(404, "Flow not found.")
    def url_or_none(p): return _abs_url(request, _rel_url_for_path(p)) if p else None
    return {
        "ok": True,
        "flow_id": flow_id,
        "status": f.get("status", {}),
        "timestamps": f.get("timestamps", {}),
        "history": f.get("history", []),
        "offer_pdf": f.get("offer_pdf"),
        "order_pdf": f.get("order_pdf"),
        "invoice_pdf": f.get("invoice_pdf"),
        "offer_pdf_url": url_or_none(f.get("offer_pdf")),
        "order_pdf_url": url_or_none(f.get("order_pdf")),
        "invoice_pdf_url": url_or_none(f.get("invoice_pdf")),
        "pricing": f.get("pricing")
    }

@app.get("/flows")
def flows_list(request: Request):
    out = []
    for fid, f in FLOWS.items():
        def url(p): return _abs_url(request, _rel_url_for_path(p)) if p else None
        out.append({
            "flow_id": fid,
            "customer": f.get("offer", {}).get("customer", {}).get("name"),
            "date": f.get("offer", {}).get("date"),
            "status": f.get("status", {}),
            "last_event": (f.get("history", [])[-1] if f.get("history") else None),
            "offer_pdf_url": url(f.get("offer_pdf")),
            "order_pdf_url": url(f.get("order_pdf")),
            "invoice_pdf_url": url(f.get("invoice_pdf")),
        })
    return {"ok": True, "flows": out}

@app.get("/files")
def files_list():
    # Last 100 files
    return {"ok": True, "files": FILE_LOG[-100:]}

# =========================================================
# Autofill endpoint (text/partial → normalized offer, optional flow update)
# =========================================================
@app.post("/autofill/offer")
async def autofill_offer(request: Request, payload: dict):
    """
    Body can contain:
      - text:    free text (voice transcript / OCR)
      - offer:   partial offer dict to merge with rule-based extraction
      - flow_id: optional; if provided, updates the flow's offer and returns it
    """
    text = (payload or {}).get("text") or ""
    partial = (payload or {}).get("offer") or {}
    flow_id = (payload or {}).get("flow_id")

    seed = rule_based_offer_from_transcript(text) if text else rule_based_offer_from_transcript("")
    # Merge partial fields into seed (shallow merge for known keys)
    for key in ["customer","date","offer_number","items","version","planned_start","delivery_time",
                "payment","gdpr","validity_days","vat_rate","rot_applies","rot_percent","compact",
                "logo_path","logo_dir","layout","images"]:
        if key in partial and partial[key] is not None:
            seed[key] = partial[key]

    # Normalize items/services duality
    if "services" not in seed and "items" in seed:
        seed["services"] = [{"title": i.get("title",""), "hours": i.get("hours") or i.get("quantity",0), "rate": i.get("rate",0)} for i in seed["items"]]
    if "items" not in seed and "services" in seed:
        seed["items"] = [{"title": r.get("title",""), "hours": r.get("hours",0), "rate": r.get("rate",0), "type":"labour"} for r in seed["services"]]

    if flow_id:
        f = FLOWS.get(flow_id)
        if not f: raise HTTPException(404, "Flow not found.")
        f["offer"] = seed
        f.setdefault("timestamps", {})["offer_autofilled_at"] = _now_iso()
        _flow_log(f, "offer_autofilled", {"from_text": bool(text)})
        return {"ok": True, "flow_id": flow_id, "offer": seed}

    return {"ok": True, "offer": seed}

