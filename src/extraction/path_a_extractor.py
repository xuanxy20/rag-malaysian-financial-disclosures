"""
Path A Extractor — Baseline PDF text extraction.

Uses PyMuPDF (fitz) as the primary engine with pdfplumber as a per-page fallback.
For each document listed in config.yaml the extractor produces a single JSON file at:
    data/processed/path_a/<doc_name>.json

Each JSON file contains a list of page-level records:
    [{"page": 1, "text": "...", "engine": "pymupdf"}, ...]

Scanned pages (no extractable text) are flagged in the record rather than silently
dropped, so downstream chunking can skip or report them explicitly.
"""

import json
import logging
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load and return the central YAML configuration.

    Args:
        config_path: Absolute path to configs/config.yaml.

    Returns:
        Parsed config as a nested dict.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_pdf_path(raw_dir: Path, file_entry: str) -> Path | None:
    """Resolve a document entry (file or folder) to a single PDF path.

    If the entry is a folder, the first .pdf found inside is returned.
    Returns None if nothing usable is found.

    Args:
        raw_dir:    Path to data/raw/.
        file_entry: Value of 'file' from the config documents list.

    Returns:
        A Path pointing to a PDF, or None.
    """
    candidate = raw_dir / file_entry

    if candidate.is_file() and candidate.suffix.lower() == ".pdf":
        return candidate

    if candidate.is_dir():
        pdfs = sorted(candidate.glob("*.pdf"))
        if pdfs:
            logger.info("Folder entry '%s' — using first PDF found: %s", file_entry, pdfs[0].name)
            return pdfs[0]
        logger.warning("Folder '%s' contains no PDF files — skipping.", candidate)
        return None

    logger.warning("Entry '%s' not found in %s — skipping.", file_entry, raw_dir)
    return None


# ---------------------------------------------------------------------------
# Per-page extraction
# ---------------------------------------------------------------------------

def extract_page_pymupdf(page: fitz.Page) -> str:
    """Extract plain text from a single PyMuPDF page object.

    Args:
        page: A fitz.Page instance.

    Returns:
        Extracted text string (may be empty for scanned pages).
    """
    return page.get_text("text")


def extract_page_pdfplumber(pdf_path: Path, page_number: int) -> str:
    """Extract plain text from a single page using pdfplumber.

    Args:
        pdf_path:    Path to the PDF file.
        page_number: 0-based page index.

    Returns:
        Extracted text string (may be empty).
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        return page.extract_text() or ""


# ---------------------------------------------------------------------------
# Document extraction
# ---------------------------------------------------------------------------

def extract_document(
    pdf_path: Path,
    doc_name: str,
    min_chars: int,
    fallback_engine: str,
) -> list[dict]:
    """Extract all pages from a PDF, using pdfplumber as fallback when needed.

    A page is considered "text-poor" (likely scanned) when PyMuPDF returns
    fewer than min_chars characters. The fallback engine is tried once; if it
    also yields too little text the page is flagged as scanned.

    Args:
        pdf_path:       Absolute path to the PDF file.
        doc_name:       Human-readable document name (used in log messages).
        min_chars:      Character count below which fallback is triggered.
        fallback_engine: "pdfplumber" or "none".

    Returns:
        List of page records, each a dict with keys:
            - page (int, 1-based)
            - text (str)
            - engine (str): "pymupdf", "pdfplumber", or "scanned"
            - char_count (int)
    """
    records = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.error("Cannot open '%s' with PyMuPDF: %s", pdf_path.name, exc)
        return records

    num_pages = len(doc)
    logger.info("Extracting '%s' — %d pages", doc_name, num_pages)

    scanned_count = 0

    for i in tqdm(range(num_pages), desc=f"  {doc_name}", unit="page", leave=False):
        page = doc[i]
        text = extract_page_pymupdf(page)
        engine = "pymupdf"

        if len(text.strip()) < min_chars:
            if fallback_engine == "pdfplumber":
                try:
                    fallback_text = extract_page_pdfplumber(pdf_path, i)
                    if len(fallback_text.strip()) >= min_chars:
                        text = fallback_text
                        engine = "pdfplumber"
                    else:
                        engine = "scanned"
                        scanned_count += 1
                except Exception as exc:
                    logger.debug("pdfplumber fallback failed on page %d: %s", i + 1, exc)
                    engine = "scanned"
                    scanned_count += 1
            else:
                engine = "scanned"
                scanned_count += 1

        records.append({
            "page": i + 1,
            "text": text,
            "engine": engine,
            "char_count": len(text.strip()),
        })

    doc.close()

    if scanned_count:
        logger.warning(
            "'%s': %d / %d pages flagged as scanned (no extractable text).",
            doc_name, scanned_count, num_pages,
        )

    return records


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_extraction(records: list[dict], output_path: Path) -> None:
    """Serialise page records to a JSON file.

    Args:
        records:     List of page-level extraction dicts.
        output_path: Destination .json file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    logger.info("Saved extraction → %s  (%d pages)", output_path, len(records))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: Path | None = None) -> None:
    """Run Path A extraction for all documents defined in config.yaml.

    Skips documents whose output JSON already exists (resumable).

    Args:
        config_path: Optional override for the config file location.
                     Defaults to <project_root>/configs/config.yaml.
    """
    project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    cfg = load_config(config_path)

    # Logging setup (respects config level)
    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s — %(message)s"),
        datefmt=log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )

    raw_dir      = project_root / cfg["paths"]["raw_data"]
    output_dir   = project_root / cfg["paths"]["processed_path_a"]
    min_chars    = cfg["path_a"]["extraction"]["min_chars"]
    fallback_eng = cfg["path_a"]["extraction"]["fallback_engine"]
    documents    = cfg["documents"]

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Path A Extraction started (%d documents) ===", len(documents))

    for doc in documents:
        doc_name = doc["name"]
        output_path = output_dir / f"{doc_name}.json"

        if output_path.exists():
            logger.info("'%s' already extracted — skipping (delete file to re-run).", doc_name)
            continue

        pdf_path = resolve_pdf_path(raw_dir, doc["file"])
        if pdf_path is None:
            continue

        records = extract_document(pdf_path, doc_name, min_chars, fallback_eng)
        if records:
            save_extraction(records, output_path)
        else:
            logger.error("No records extracted for '%s' — output not saved.", doc_name)

    logger.info("=== Path A Extraction complete ===")


if __name__ == "__main__":
    run()
