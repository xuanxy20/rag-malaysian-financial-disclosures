"""
Fixed Token Chunker — Path A baseline chunking strategy.

Reads the per-document JSON files produced by path_a_extractor.py and splits
each page's text into overlapping fixed-size token windows.

Input (per document):
    data/processed/path_a/<doc_name>.json
    — list of {page, text, engine, char_count} records

Output (per document):
    data/processed/path_a/<doc_name>_chunks.json
    — list of chunk records:
      {
        "chunk_id":   str,   # "<doc_name>_<zero-padded index>"
        "doc_name":   str,
        "page":       int,   # source page number (1-based)
        "text":       str,   # chunk text
        "token_count": int,
        "engine":     str    # extraction engine of source page
      }

All chunking parameters (chunk_size, overlap, tokenizer) are read from
configs/config.yaml — nothing is hardcoded here.
"""

import json
import logging
from pathlib import Path

import tiktoken
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
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


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def get_tokenizer(encoding_name: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding by name.

    Args:
        encoding_name: tiktoken encoding identifier, e.g. "cl100k_base".

    Returns:
        A tiktoken.Encoding instance.
    """
    return tiktoken.get_encoding(encoding_name)


def tokenize(text: str, tokenizer: tiktoken.Encoding) -> list[int]:
    """Encode text to a list of token ids.

    Args:
        text:      Input string.
        tokenizer: tiktoken.Encoding instance.

    Returns:
        List of integer token ids.
    """
    return tokenizer.encode(text)


def detokenize(token_ids: list[int], tokenizer: tiktoken.Encoding) -> str:
    """Decode a list of token ids back to a string.

    Args:
        token_ids: List of integer token ids.
        tokenizer: tiktoken.Encoding instance.

    Returns:
        Decoded text string.
    """
    return tokenizer.decode(token_ids)


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def chunk_token_windows(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: tiktoken.Encoding,
) -> list[str]:
    """Split text into overlapping token windows.

    Slides a window of `chunk_size` tokens across the token sequence,
    advancing by (chunk_size - chunk_overlap) tokens each step.

    Args:
        text:          Source text to split.
        chunk_size:    Maximum tokens per chunk.
        chunk_overlap: Number of tokens shared between consecutive chunks.
        tokenizer:     tiktoken.Encoding used for tokenisation.

    Returns:
        List of decoded text chunks. Empty list if text is blank.
    """
    text = text.strip()
    if not text:
        return []

    tokens = tokenize(text, tokenizer)
    if not tokens:
        return []

    stride = chunk_size - chunk_overlap
    if stride <= 0:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = detokenize(tokens[start:end], tokenizer)
        if chunk_text.strip():
            chunks.append(chunk_text)
        start += stride

    return chunks


def chunk_document(
    pages: list[dict],
    doc_name: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: tiktoken.Encoding,
) -> list[dict]:
    """Chunk all pages of a document into fixed-size token windows.

    Pages flagged as 'scanned' (engine == "scanned") are skipped with a
    warning rather than crashing.

    Args:
        pages:         List of page records from the extractor JSON.
        doc_name:      Document name, used for chunk_id generation.
        chunk_size:    Target tokens per chunk.
        chunk_overlap: Overlap tokens between consecutive chunks.
        tokenizer:     tiktoken.Encoding instance.

    Returns:
        List of chunk dicts, each containing:
            chunk_id, doc_name, page, text, token_count, engine.
    """
    all_chunks = []
    chunk_index = 0
    skipped_pages = 0

    for page_record in tqdm(pages, desc=f"  {doc_name}", unit="page", leave=False):
        page_num = page_record.get("page", 0)
        engine   = page_record.get("engine", "unknown")
        text     = page_record.get("text", "")

        if engine == "scanned" or not text.strip():
            skipped_pages += 1
            logger.debug("Page %d of '%s' skipped (engine=%s).", page_num, doc_name, engine)
            continue

        page_chunks = chunk_token_windows(text, chunk_size, chunk_overlap, tokenizer)

        for chunk_text in page_chunks:
            token_count = len(tokenize(chunk_text, tokenizer))
            all_chunks.append({
                "chunk_id":    f"{doc_name}_{chunk_index:05d}",
                "doc_name":    doc_name,
                "page":        page_num,
                "text":        chunk_text,
                "token_count": token_count,
                "engine":      engine,
            })
            chunk_index += 1

    if skipped_pages:
        logger.warning(
            "'%s': %d page(s) skipped (scanned or empty).", doc_name, skipped_pages
        )

    return all_chunks


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_extraction(input_path: Path) -> list[dict]:
    """Load a page-extraction JSON file produced by path_a_extractor.py.

    Args:
        input_path: Path to <doc_name>.json.

    Returns:
        List of page-record dicts.
    """
    with open(input_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Save chunk records to a JSON file.

    Args:
        chunks:      List of chunk dicts.
        output_path: Destination .json file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %d chunks → %s", len(chunks), output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: Path | None = None) -> None:
    """Run fixed-token chunking for all Path A documents.

    For each document, reads <doc_name>.json from data/processed/path_a/,
    produces <doc_name>_chunks.json in the same directory.
    Skips documents whose chunk file already exists (resumable).

    Args:
        config_path: Optional override for the config file location.
                     Defaults to <project_root>/configs/config.yaml.
    """
    project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    cfg = load_config(config_path)

    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s — %(message)s"),
        datefmt=log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )

    processed_dir = project_root / cfg["paths"]["processed_path_a"]
    chunk_cfg     = cfg["path_a"]["chunking"]
    chunk_size    = chunk_cfg["chunk_size"]
    chunk_overlap = chunk_cfg["chunk_overlap"]
    tokenizer_name = chunk_cfg["tokenizer"]
    documents     = cfg["documents"]

    tokenizer = get_tokenizer(tokenizer_name)
    logger.info(
        "=== Path A Chunking started | size=%d overlap=%d tokenizer=%s ===",
        chunk_size, chunk_overlap, tokenizer_name,
    )

    for doc in documents:
        doc_name    = doc["name"]
        input_path  = processed_dir / f"{doc_name}.json"
        output_path = processed_dir / f"{doc_name}_chunks.json"

        if output_path.exists():
            logger.info("'%s' chunks already exist — skipping.", doc_name)
            continue

        if not input_path.exists():
            logger.warning(
                "'%s': extraction file not found at %s — run Path A extraction first.",
                doc_name, input_path,
            )
            continue

        pages  = load_extraction(input_path)
        chunks = chunk_document(pages, doc_name, chunk_size, chunk_overlap, tokenizer)

        if chunks:
            save_chunks(chunks, output_path)
            logger.info("'%s': %d chunks produced.", doc_name, len(chunks))
        else:
            logger.error("'%s': no chunks produced — check extraction output.", doc_name)

    logger.info("=== Path A Chunking complete ===")


if __name__ == "__main__":
    run()
