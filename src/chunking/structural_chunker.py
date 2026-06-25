"""
Structural Chunker — Path B layout-aware chunking strategy.

Reads the per-document JSON files produced by path_b_extractor.py (LlamaParse
markdown output) and splits the content by structural boundaries — headings,
sections, and tables — rather than by a fixed token window.

Chunking logic (in priority order):
  1. Markdown headings (# / ## / ###) mark hard section boundaries.
  2. Table blocks (lines starting with '|') are kept intact as single chunks.
  3. Sections exceeding max_chunk_tokens are recursively split on paragraph
     boundaries (blank lines), then on sentence boundaries as a last resort.
  4. Sections below min_chunk_tokens are merged into the following chunk.

Input (per document):
    data/processed/path_b/<doc_name>.json
    — {doc_name, source_file, pages: [{page, text, metadata}, ...]}

Output (per document):
    data/processed/path_b/<doc_name>_chunks.json
    — list of chunk records:
      {
        "chunk_id":     str,   # "<doc_name>_<zero-padded index>"
        "doc_name":     str,
        "page":         int,   # source page of the first token in the chunk
        "text":         str,   # chunk text (markdown preserved)
        "token_count":  int,
        "chunk_type":   str,   # "heading_section" | "table" | "paragraph" | "merged"
        "heading":      str    # nearest preceding heading, or "" if none
      }

All parameters are read from configs/config.yaml — nothing is hardcoded.
"""

import json
import logging
import re
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
# Tokenizer helpers (mirrors fixed_chunker for consistency)
# ---------------------------------------------------------------------------

def get_tokenizer(encoding_name: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding by name.

    Args:
        encoding_name: tiktoken encoding identifier, e.g. "cl100k_base".

    Returns:
        A tiktoken.Encoding instance.
    """
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, tokenizer: tiktoken.Encoding) -> int:
    """Return the token count for a string.

    Args:
        text:      Input string.
        tokenizer: tiktoken.Encoding instance.

    Returns:
        Integer token count.
    """
    return len(tokenizer.encode(text))


# ---------------------------------------------------------------------------
# Markdown structural parsing
# ---------------------------------------------------------------------------

# Matches Markdown headings: #, ##, ### (ATX style)
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# A line is part of a table if it starts with '|' (after stripping)
def _is_table_line(line: str) -> bool:
    return line.strip().startswith("|")


def split_into_sections(text: str) -> list[dict]:
    """Split a markdown text into structural sections.

    Each section is one of:
      - A heading followed by its body text
      - A freestanding table block
      - A paragraph block between headings

    Args:
        text: Full markdown text for a page or document.

    Returns:
        List of section dicts, each with:
            - "type": "heading_section" | "table" | "paragraph"
            - "heading": str  (the heading text, or "" for non-heading sections)
            - "text": str     (full text of the section including the heading line)
    """
    lines = text.splitlines()
    sections: list[dict] = []
    current_heading = ""
    current_lines: list[str] = []
    in_table = False
    table_lines: list[str] = []

    def flush_current():
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({
                "type": "heading_section" if current_heading else "paragraph",
                "heading": current_heading,
                "text": body,
            })
        current_lines = []

    def flush_table():
        nonlocal table_lines, in_table
        body = "\n".join(table_lines).strip()
        if body:
            sections.append({
                "type": "table",
                "heading": current_heading,
                "text": body,
            })
        table_lines = []
        in_table = False

    for line in lines:
        # --- heading detection ---
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            if in_table:
                flush_table()
            flush_current()
            current_heading = heading_match.group(2).strip()
            current_lines = [line]
            continue

        # --- table detection ---
        if _is_table_line(line):
            if not in_table:
                # entering a table: flush non-table content first
                flush_current()
                in_table = True
            table_lines.append(line)
            continue

        # --- leaving a table ---
        if in_table and not _is_table_line(line):
            flush_table()

        current_lines.append(line)

    # flush remaining content
    if in_table:
        flush_table()
    flush_current()

    return sections


# ---------------------------------------------------------------------------
# Section splitting (for oversized sections)
# ---------------------------------------------------------------------------

def split_on_paragraphs(text: str) -> list[str]:
    """Split text on blank lines (paragraph boundaries).

    Args:
        text: Input text block.

    Returns:
        List of non-empty paragraph strings.
    """
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_on_sentences(text: str, max_tokens: int, tokenizer: tiktoken.Encoding) -> list[str]:
    """Greedily accumulate sentences up to max_tokens, then start a new chunk.

    Used as a last resort when paragraph splitting still leaves chunks that
    exceed max_chunk_tokens.

    Args:
        text:       Input text.
        max_tokens: Hard token cap per chunk.
        tokenizer:  tiktoken.Encoding instance.

    Returns:
        List of text chunks each within max_tokens.
    """
    # Simple sentence splitter on . ! ? followed by whitespace or end of string
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = count_tokens(sentence, tokenizer)
        if current_tokens + s_tokens > max_tokens and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = [sentence]
            current_tokens = s_tokens
        else:
            current_parts.append(sentence)
            current_tokens += s_tokens

    if current_parts:
        chunks.append(" ".join(current_parts))

    return [c for c in chunks if c.strip()]


def split_table_by_rows(
    text: str,
    max_tokens: int,
    tokenizer: tiktoken.Encoding,
) -> list[str]:
    """Split a large markdown table into header-prefixed row-group chunks.

    Each output chunk starts with the original header and separator rows,
    followed by a group of data rows that fit within max_tokens. This
    allows individual rows of large milestone/timeline tables to be
    retrieved independently rather than being buried in one huge chunk.

    Args:
        text:       Full markdown table text.
        max_tokens: Token cap per output chunk.
        tokenizer:  tiktoken.Encoding instance.

    Returns:
        List of table sub-chunks, each repeating the original header row.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return [text]

    # Identify header lines (everything up to and including the separator row)
    header_lines: list[str] = []
    data_start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*\|[\-| :]+\|\s*$", line):  # separator: | --- | --- |
            header_lines = lines[: i + 1]
            data_start = i + 1
            break

    if not header_lines:
        return [text]

    header = "\n".join(header_lines)
    header_tokens = count_tokens(header, tokenizer)
    data_rows = lines[data_start:]

    if not data_rows:
        return [text]

    chunks: list[str] = []
    current_rows: list[str] = []
    current_tokens = header_tokens

    for row in data_rows:
        row_tokens = count_tokens(row, tokenizer)
        if current_tokens + row_tokens > max_tokens and current_rows:
            chunks.append(header + "\n" + "\n".join(current_rows))
            current_rows = [row]
            current_tokens = header_tokens + row_tokens
        else:
            current_rows.append(row)
            current_tokens += row_tokens

    if current_rows:
        chunks.append(header + "\n" + "\n".join(current_rows))

    return chunks if chunks else [text]


def split_oversized_section(
    text: str,
    max_tokens: int,
    tokenizer: tiktoken.Encoding,
) -> list[str]:
    """Recursively split a section that exceeds max_tokens.

    Tries paragraph splits first; falls back to sentence splits for
    paragraphs that are still too large.

    Args:
        text:       Section text to split.
        max_tokens: Hard token cap.
        tokenizer:  tiktoken.Encoding instance.

    Returns:
        List of text sub-chunks each within max_tokens.
    """
    if count_tokens(text, tokenizer) <= max_tokens:
        return [text]

    sub_chunks: list[str] = []
    for para in split_on_paragraphs(text):
        if count_tokens(para, tokenizer) <= max_tokens:
            sub_chunks.append(para)
        else:
            sub_chunks.extend(split_on_sentences(para, max_tokens, tokenizer))

    return sub_chunks if sub_chunks else [text]


# ---------------------------------------------------------------------------
# Chunk assembly with min-size merging
# ---------------------------------------------------------------------------

def assemble_chunks(
    sections: list[dict],
    page_num: int,
    doc_name: str,
    max_tokens: int,
    min_tokens: int,
    preserve_tables: bool,
    tokenizer: tiktoken.Encoding,
    start_index: int,
    max_table_tokens: int | None = None,
) -> list[dict]:
    """Convert structural sections into final chunk records.

    Oversized sections are split; undersized chunks are merged forward.
    Tables are kept intact up to max_table_tokens; larger tables are split
    by row groups so individual rows remain independently retrievable.

    Args:
        sections:         Output of split_into_sections().
        page_num:         Source page number (1-based).
        doc_name:         Document name for chunk_id generation.
        max_tokens:       Hard cap — split sections larger than this.
        min_tokens:       Merge chunks smaller than this with the next.
        preserve_tables:  If True, keep tables intact up to max_table_tokens.
        tokenizer:        tiktoken.Encoding instance.
        start_index:      Global chunk counter offset (for unique chunk_ids).
        max_table_tokens: Hard cap for table chunks — tables above this are
                          split into row groups regardless of preserve_tables.
                          Defaults to max_tokens when not set.

    Returns:
        List of chunk dicts.
    """
    table_cap = max_table_tokens if max_table_tokens is not None else max_tokens

    # Build a flat list of (text, type, heading) tuples first
    flat: list[tuple[str, str, str]] = []

    for sec in sections:
        sec_text    = sec["text"]
        sec_type    = sec["type"]
        sec_heading = sec["heading"]

        if not sec_text.strip():
            continue

        # Tables: preserve within table_cap; split large tables by row groups
        if sec_type == "table":
            t_tokens = count_tokens(sec_text, tokenizer)
            if t_tokens > table_cap:
                # Too large even for preserve_tables — split by rows
                for part in split_table_by_rows(sec_text, table_cap, tokenizer):
                    flat.append((part, "table", sec_heading))
            elif preserve_tables or t_tokens <= max_tokens:
                flat.append((sec_text, "table", sec_heading))
            else:
                for part in split_oversized_section(sec_text, max_tokens, tokenizer):
                    flat.append((part, "table", sec_heading))
            continue

        # Regular sections: split if too large
        if count_tokens(sec_text, tokenizer) > max_tokens:
            parts = split_oversized_section(sec_text, max_tokens, tokenizer)
            for part in parts:
                flat.append((part, sec_type, sec_heading))
        else:
            flat.append((sec_text, sec_type, sec_heading))

    # Merge undersized chunks forward
    merged: list[tuple[str, str, str]] = []
    buffer_text    = ""
    buffer_type    = "paragraph"
    buffer_heading = ""

    for text, ctype, heading in flat:
        tokens = count_tokens(text, tokenizer)
        if not buffer_text:
            buffer_text, buffer_type, buffer_heading = text, ctype, heading
        elif count_tokens(buffer_text, tokenizer) < min_tokens:
            # current buffer is too small — merge this section in
            buffer_text    = buffer_text + "\n\n" + text
            buffer_type    = "merged"
        else:
            merged.append((buffer_text, buffer_type, buffer_heading))
            buffer_text, buffer_type, buffer_heading = text, ctype, heading

    if buffer_text.strip():
        merged.append((buffer_text, buffer_type, buffer_heading))

    # Build final chunk records
    chunks: list[dict] = []
    for i, (text, ctype, heading) in enumerate(merged):
        chunks.append({
            "chunk_id":    f"{doc_name}_{start_index + i:05d}",
            "doc_name":    doc_name,
            "page":        page_num,
            "text":        text.strip(),
            "token_count": count_tokens(text, tokenizer),
            "chunk_type":  ctype,
            "heading":     heading,
        })

    return chunks


# ---------------------------------------------------------------------------
# Cross-page merge pass
# ---------------------------------------------------------------------------

def _cross_page_merge(
    chunks: list[dict],
    min_tokens: int,
    tokenizer: tiktoken.Encoding,
) -> list[dict]:
    """Merge standalone tiny chunks (page title/divider pages) into the next chunk.

    Page-level assembly can produce tiny chunks from near-empty pages such as
    title pages ("SD Guthrie / Integrated Report 2025") or section dividers
    ("Consolidated Financial Statements of the Nestlé Group 2025"). These
    score deceptively high on any company-related query because they contain
    the company name, displacing actual data chunks from the top-k results.

    This pass forwards any chunk below min_tokens into the following chunk,
    prepending its text as context. A chunk at the end of the list with no
    successor is simply dropped.

    Args:
        chunks:     Flat list of chunk dicts from all pages.
        min_tokens: Threshold — chunks below this are merged forward.
        tokenizer:  tiktoken.Encoding instance.

    Returns:
        Filtered and merged chunk list.
    """
    if not chunks:
        return chunks

    result: list[dict] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        if count_tokens(chunk["text"], tokenizer) < min_tokens and i + 1 < len(chunks):
            # Prepend this tiny chunk into the next one and skip it
            nxt = dict(chunks[i + 1])
            nxt["text"] = chunk["text"] + "\n\n" + nxt["text"]
            nxt["token_count"] = count_tokens(nxt["text"], tokenizer)
            nxt["chunk_type"] = "merged"
            chunks[i + 1] = nxt
            i += 1
        else:
            result.append(chunk)
            i += 1

    return result


# ---------------------------------------------------------------------------
# Document-level chunking
# ---------------------------------------------------------------------------

def chunk_document(
    pages: list[dict],
    doc_name: str,
    max_tokens: int,
    min_tokens: int,
    preserve_tables: bool,
    tokenizer: tiktoken.Encoding,
    max_table_tokens: int | None = None,
) -> list[dict]:
    """Structurally chunk all pages of a LlamaParse document.

    Args:
        pages:            List of page records from the Path B extractor JSON.
        doc_name:         Document name.
        max_tokens:       Hard cap — oversized sections are split.
        min_tokens:       Merge threshold — small chunks are merged forward.
        preserve_tables:  Keep table blocks intact when True.
        tokenizer:        tiktoken.Encoding instance.
        max_table_tokens: Hard cap for table chunks — large tables split by rows.

    Returns:
        Flat list of chunk dicts across all pages.
    """
    all_chunks: list[dict] = []
    chunk_index = 0

    for page_record in tqdm(pages, desc=f"  {doc_name}", unit="page", leave=False):
        page_num = page_record.get("page", 0)
        text     = page_record.get("text", "")

        if not text.strip():
            logger.debug("Page %d of '%s' is empty — skipping.", page_num, doc_name)
            continue

        sections = split_into_sections(text)
        page_chunks = assemble_chunks(
            sections=sections,
            page_num=page_num,
            doc_name=doc_name,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            preserve_tables=preserve_tables,
            tokenizer=tokenizer,
            start_index=chunk_index,
            max_table_tokens=max_table_tokens,
        )

        all_chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    # Merge any standalone tiny chunks (page titles, section dividers) into
    # the following chunk so they don't pollute retrieval results.
    all_chunks = _cross_page_merge(all_chunks, min_tokens, tokenizer)

    # Re-index chunk_ids after merging to keep them contiguous
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"{doc_name}_{i:05d}"

    return all_chunks


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_extraction(input_path: Path) -> tuple[str, list[dict]]:
    """Load a Path B extraction JSON file.

    Args:
        input_path: Path to <doc_name>.json produced by path_b_extractor.py.

    Returns:
        Tuple of (doc_name, pages list).
    """
    with open(input_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["doc_name"], data["pages"]


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
    """Run structural chunking for all Path B documents.

    For each document, reads <doc_name>.json from data/processed/path_b/,
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

    processed_dir    = project_root / cfg["paths"]["processed_path_b"]
    chunk_cfg        = cfg["path_b"]["chunking"]
    max_tokens       = chunk_cfg["max_chunk_tokens"]
    min_tokens       = chunk_cfg["min_chunk_tokens"]
    preserve_tables  = chunk_cfg["preserve_tables"]
    max_table_tokens = chunk_cfg.get("max_table_tokens")
    tokenizer_name   = chunk_cfg["tokenizer"]
    documents        = cfg["documents"]

    tokenizer = get_tokenizer(tokenizer_name)
    logger.info(
        "=== Path B Chunking started | max=%d min=%d max_table=%s preserve_tables=%s tokenizer=%s ===",
        max_tokens, min_tokens, max_table_tokens, preserve_tables, tokenizer_name,
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
                "'%s': extraction file not found at %s — run Path B extraction first.",
                doc_name, input_path,
            )
            continue

        _, pages = load_extraction(input_path)
        chunks = chunk_document(
            pages=pages,
            doc_name=doc_name,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            preserve_tables=preserve_tables,
            tokenizer=tokenizer,
            max_table_tokens=max_table_tokens,
        )

        if chunks:
            save_chunks(chunks, output_path)
            logger.info("'%s': %d chunks produced.", doc_name, len(chunks))

            # Log chunk type distribution for research records
            type_counts: dict[str, int] = {}
            for c in chunks:
                type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1
            logger.info("  Chunk types: %s", type_counts)
        else:
            logger.error("'%s': no chunks produced — check extraction output.", doc_name)

    logger.info("=== Path B Chunking complete ===")


if __name__ == "__main__":
    run()
