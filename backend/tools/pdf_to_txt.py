#!/usr/bin/env python
"""
Jednoduchý převod PDF → TXT (UTF-8) pro použití s /ingest_texts.

Použití (Windows PowerShell):

  # 1) Jeden soubor
  #   Vytvoří backend/storage/texts/<nazev>.txt
  python backend/tools/pdf_to_txt.py --input "C:\\cesta\\k\\souboru.pdf"

  # 2) Všechny PDF v adresáři (nerekurzivně)
  python backend/tools/pdf_to_txt.py --indir "C:\\cesta\\k\\adresari"

  # 3) Rekurzivně přes adresář (všechny .pdf)
  python backend/tools/pdf_to_txt.py --indir "C:\\cesta\\k\\adresari" --recursive

Poznámky:
- Výstup se ukládá do backend/storage/texts (lze změnit parametrem --outdir)
- Pokusí se použít PyMuPDF (fitz) pro kvalitnější extrakci; pokud není nainstalován,
  spadne na pypdf (základní extrakce). Pro skenované PDF (obrázky) je potřeba OCR
  (např. 
    1) OCRmyPDF: https://ocrmypdf.readthedocs.io
    2) Tesseract OCR: https://github.com/tesseract-ocr/tesseract
)
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import List

# Cesty relativně ke kořeni backendu
BASE_DIR = Path(__file__).resolve().parents[1]
TEXTS_DIR = BASE_DIR / "storage" / "texts"


def extract_with_pymupdf(pdf_path: Path) -> str:
    try:
        import fitz  # type: ignore
    except Exception:  # PyMuPDF není nainstalováno
        raise

    text_parts: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # "text" = textová extrakce s rozumným rozložením
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)


def extract_with_pypdf(pdf_path: Path) -> str:
    from pypdf import PdfReader
    texts: List[str] = []
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                texts.append(t)
    except Exception as e:
        raise RuntimeError(f"Chyba při čtení PDF pypdf: {e}")
    return "\n".join(texts)


def extract_text(pdf_path: Path) -> str:
    # Nejprve zkus PyMuPDF (pokud je), pak pypdf
    try:
        return extract_with_pymupdf(pdf_path)
    except Exception:
        return extract_with_pypdf(pdf_path)


def convert_one(pdf_path: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / (pdf_path.stem + ".txt")
    txt = extract_text(pdf_path)
    out_path.write_text(txt, encoding="utf-8", errors="ignore")
    return out_path


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="PDF → TXT (UTF-8) converter")
    parser.add_argument("--input", type=str, help="Cesta k PDF souboru")
    parser.add_argument("--indir", type=str, help="Adresář s PDF soubory")
    parser.add_argument("--outdir", type=str, default=str(TEXTS_DIR), help="Výstupní adresář pro .txt")
    parser.add_argument("--recursive", action="store_true", help="Rekurzivní průchod adresářem")

    args = parser.parse_args(argv)

    outdir = Path(args.outdir).resolve()

    if args.input:
        pdf = Path(args.input)
        if not pdf.exists() or not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            print(f"[!] Soubor není PDF nebo neexistuje: {pdf}")
            return 1
        out_path = convert_one(pdf, outdir)
        print(f"[OK] Uloženo: {out_path}")
        return 0

    if args.indir:
        root = Path(args.indir).resolve()
        if not root.exists() or not root.is_dir():
            print(f"[!] Adresář neexistuje: {root}")
            return 1
        pattern = "**/*.pdf" if args.recursive else "*.pdf"
        pdfs = sorted(root.glob(pattern))
        if not pdfs:
            print("[i] Nebyly nalezeny žádné .pdf soubory.")
            return 0
        ok, fail = 0, 0
        for pdf in pdfs:
            try:
                out_path = convert_one(pdf, outdir)
                print(f"[OK] {pdf.name} → {out_path.name}")
                ok += 1
            except Exception as e:
                print(f"[X] {pdf}: {e}")
                fail += 1
        print(f"Hotovo. Úspěšně: {ok}, chyby: {fail}.")
        return 0 if fail == 0 else 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
