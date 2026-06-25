"""
Path B Extractor — Layout-aware PDF extraction via LlamaParse.

LlamaParse is called ONCE per document and the result is saved as a static JSON file:
    data/processed/path_b/<doc_name>.json

Subsequent pipeline runs load from that file directly — the API is never called again
for an already-parsed document. This preserves API quota and ensures reproducibility.

Output schema per document JSON:
    {
      "doc_name": str,
      "source_file": str,
      "pages": [
        {
          "page": int,           # 1-based
          "text": str,           # markdown text returned by LlamaParse
          "metadata": dict       # raw LlamaParse page metadata (headings, type, etc.)
        },
        ...
      ]
    }

The LLAMA_CLOUD_API_KEY is loaded from the project .env file via python-dotenv.
"""

import json
import logging
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
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
            logger.info("Folder entry '%s' — using first PDF: %s", file_entry, pdfs[0].name)
            return pdfs[0]
        logger.warning("Folder '%s' contains no PDF files — skipping.", candidate)
        return None

    logger.warning("Entry '%s' not found in %s — skipping.", file_entry, raw_dir)
    return None


# ---------------------------------------------------------------------------
# LlamaParse extraction
# ---------------------------------------------------------------------------

def parse_with_llamaparse(
    pdf_path: Path,
    doc_name: str,
    result_type: str,
    language: str,
    api_key: str,
) -> list[dict]:
    """Call LlamaParse API and return a list of page-level records.

    LlamaParse returns documents split by page. Each record captures the
    markdown text and the full metadata dict for use by the structural chunker.

    Args:
        pdf_path:    Absolute path to the source PDF.
        doc_name:    Human-readable name used in log messages.
        result_type: LlamaParse result type, e.g. "markdown".
        language:    Document language hint, e.g. "en".
        api_key:     LlamaParse cloud API key.

    Returns:
        List of page records, each a dict with keys:
            - page (int, 1-based)
            - text (str)
            - metadata (dict)
    """
    # Import here so the module is importable even when llama-parse is not
    # installed (e.g., during unit tests of other modules).
    try:
        from llama_parse import LlamaParse
    except ImportError as exc:
        raise ImportError(
            "llama-parse is not installed. Run: pip install llama-parse"
        ) from exc

    import httpx

    logger.info("Calling LlamaParse for '%s' (result_type=%s) …", doc_name, result_type)

    parser = LlamaParse(
        api_key=api_key,
        result_type=result_type,
        language=language,
        verbose=False,
        custom_client=httpx.AsyncClient(verify=False),
    )

    # LlamaParse returns a list of Document objects (one per page by default).
    try:
        documents = parser.load_data(str(pdf_path))
    except Exception as exc:
        logger.error("LlamaParse failed for '%s': %s", doc_name, exc)
        return []

    records = []
    for idx, doc in enumerate(
        tqdm(documents, desc=f"  {doc_name} (pages)", unit="page", leave=False)
    ):
        # doc.text holds the markdown content; doc.metadata is a dict with
        # structural info (headings, page number, element type, etc.)
        metadata = doc.metadata if hasattr(doc, "metadata") and doc.metadata else {}
        page_num = metadata.get("page_label") or metadata.get("page") or (idx + 1)

        try:
            page_num = int(page_num)
        except (TypeError, ValueError):
            page_num = idx + 1

        records.append({
            "page": page_num,
            "text": doc.text,
            "metadata": metadata,
        })

    logger.info("LlamaParse returned %d page documents for '%s'.", len(records), doc_name)
    return records


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_extraction(
    records: list[dict],
    doc_name: str,
    source_file: str,
    output_path: Path,
) -> None:
    """Save LlamaParse output as a structured JSON file.

    Args:
        records:     List of page-level dicts from parse_with_llamaparse().
        doc_name:    Document name stored in the JSON for traceability.
        source_file: Original filename stored for traceability.
        output_path: Destination .json file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "doc_name": doc_name,
        "source_file": source_file,
        "pages": records,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    logger.info("Saved extraction → %s  (%d pages)", output_path, len(records))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: Path | None = None) -> None:
    """Run Path B extraction for all documents defined in config.yaml.

    Skips documents whose output JSON already exists (resumable).
    Loads LLAMA_CLOUD_API_KEY from the project .env file.

    Args:
        config_path: Optional override for the config file location.
                     Defaults to <project_root>/configs/config.yaml.
    """
    project_root = Path(__file__).resolve().parents[2]

    # Load .env — must happen before reading the API key from os.environ
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)

    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    cfg = load_config(config_path)

    # Logging setup
    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s — %(message)s"),
        datefmt=log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )

    api_key = os.environ.get("LLAMA_CLOUD_API_KEY", "")
    if not api_key:
        logger.error(
            "LLAMA_CLOUD_API_KEY not found. "
            "Ensure it is set in the project .env file."
        )
        return

    raw_dir    = project_root / cfg["paths"]["raw_data"]
    output_dir = project_root / cfg["paths"]["processed_path_b"]
    ext_cfg    = cfg["path_b"]["extraction"]
    documents  = cfg["documents"]

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Path B Extraction started (%d documents) ===", len(documents))

    for doc in documents:
        doc_name = doc["name"]
        output_path = output_dir / f"{doc_name}.json"

        if output_path.exists():
            logger.info(
                "'%s' already parsed — skipping (delete file to re-parse).", doc_name
            )
            continue

        pdf_path = resolve_pdf_path(raw_dir, doc["file"])
        if pdf_path is None:
            continue

        records = parse_with_llamaparse(
            pdf_path=pdf_path,
            doc_name=doc_name,
            result_type=ext_cfg["result_type"],
            language=ext_cfg["language"],
            api_key=api_key,
        )

        if records:
            save_extraction(records, doc_name, doc["file"], output_path)
        else:
            logger.error("No records returned by LlamaParse for '%s' — output not saved.", doc_name)

        # Brief pause between documents to be polite to the API rate limiter.
        time.sleep(2)

    logger.info("=== Path B Extraction complete ===")


if __name__ == "__main__":
    run()
