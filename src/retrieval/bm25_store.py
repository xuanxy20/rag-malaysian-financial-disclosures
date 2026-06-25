"""
BM25Store — keyword-based retrieval using rank_bm25.

One BM25Store covers a single document (parallel to FAISSStore).  It is built
from the same <doc_name>_chunks.json file already on disk and persisted to
<doc_name>_bm25.json so it can be loaded without re-tokenising.

Tokenisation: lowercase whitespace split (fast, no NLTK dependency).
Scores returned in the 'bm25_score' key; callers are responsible for
normalising before combining with dense scores.

Usage:
    store = BM25Store()
    store.build(chunks)                        # from list[dict] with "text" key
    store.save(path / "CIMB-2025_bm25.json")

    store2 = BM25Store()
    store2.load(path / "CIMB-2025_bm25.json")
    results = store2.search("net profit CET1", top_k=20)
"""

import json
import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Store:
    """BM25 keyword index for one document's chunks.

    Attributes:
        chunks:            List of original chunk dicts (with text, chunk_id, …).
        _tokenized_corpus: Parallel list of token lists, aligned with chunks.
        _bm25:             Fitted BM25Okapi model (None until built/loaded).
    """

    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # Tokeniser
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from a list of chunk dicts.

        Args:
            chunks: List of dicts, each must have a 'text' key.
        """
        self.chunks = chunks
        self._tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(
            "BM25Store built: %d chunks, doc='%s'",
            len(chunks),
            chunks[0].get("doc_name", "?") if chunks else "?",
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise the tokenised corpus and chunk metadata to JSON.

        Rebuilding BM25Okapi from the saved corpus is fast, so we don't
        need to pickle the model itself.

        Args:
            path: Destination file path (e.g. <processed_dir>/<doc>_bm25.json).
        """
        if self._bm25 is None:
            raise RuntimeError("Cannot save — call build() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"tokenized_corpus": self._tokenized_corpus, "chunks": self.chunks}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        logger.info("BM25Store saved → %s  (%d chunks)", path, len(self.chunks))

    def load(self, path: Path) -> None:
        """Load a saved BM25 store from disk and rebuild the BM25Okapi model.

        Args:
            path: Path to a previously saved _bm25.json file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 store not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self._tokenized_corpus = data["tokenized_corpus"]
        self.chunks = data["chunks"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(
            "BM25Store loaded ← %s  (%d chunks)", path, len(self.chunks)
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int) -> list[dict]:
        """Return top-k chunks ranked by BM25 score.

        Chunks with a score of 0 are excluded (no query term overlap).

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.

        Returns:
            List of chunk dicts (copies) with an added 'bm25_score' key,
            sorted descending by score.
        """
        if self._bm25 is None:
            raise RuntimeError("Cannot search — call build() or load() first.")

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # argsort descending, take top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            if scores[i] <= 0:
                break  # sorted, so all remaining are also 0
            record = dict(self.chunks[i])
            record["bm25_score"] = float(scores[i])
            results.append(record)

        return results

    # ------------------------------------------------------------------
    # Convenience property
    # ------------------------------------------------------------------

    @property
    def doc_name(self) -> str:
        """Return doc_name of the first chunk, or empty string."""
        if self.chunks:
            return self.chunks[0].get("doc_name", "")
        return ""


# ---------------------------------------------------------------------------
# Pipeline helper: build/load all BM25 stores for one path
# ---------------------------------------------------------------------------

def build_and_save_all_bm25(
    processed_dir: Path,
    documents: list[dict],
) -> list[BM25Store]:
    """Build (or load cached) BM25 indexes for all documents.

    Reads <doc_name>_chunks.json for each document, builds BM25, and saves
    to <doc_name>_bm25.json.  Skips documents whose _bm25.json already exists.

    Args:
        processed_dir: Directory containing <doc_name>_chunks.json files.
        documents:     List of document dicts from config.yaml.

    Returns:
        List of loaded BM25Store instances (one per available document).
    """
    stores: list[BM25Store] = []

    for doc in documents:
        doc_name   = doc["name"]
        bm25_path  = processed_dir / f"{doc_name}_bm25.json"
        chunks_path = processed_dir / f"{doc_name}_chunks.json"

        store = BM25Store()

        if bm25_path.exists():
            store.load(bm25_path)
            stores.append(store)
            continue

        if not chunks_path.exists():
            logger.warning(
                "'%s': chunks file not found at %s — skipping BM25 build.",
                doc_name, chunks_path,
            )
            continue

        with open(chunks_path, "r", encoding="utf-8") as fh:
            chunks = json.load(fh)

        if not chunks:
            logger.error("'%s': chunks file is empty — skipping.", doc_name)
            continue

        store.build(chunks)
        store.save(bm25_path)
        stores.append(store)

    logger.info(
        "BM25: loaded/built %d / %d stores from %s.",
        len(stores), len(documents), processed_dir,
    )
    return stores
