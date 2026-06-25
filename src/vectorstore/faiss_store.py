"""
FAISS Vector Store — shared index build, save, and load (identical for Path A and Path B).

For each document the store writes two files side-by-side:
    <output_dir>/<doc_name>.faiss      — binary FAISS index
    <output_dir>/<doc_name>.meta.json  — parallel list of chunk metadata dicts

The metadata list is index-aligned with the FAISS vectors: row i in the index
corresponds to metadata[i]. This lets the retriever return full chunk records
(text, page, chunk_id, …) alongside similarity scores without a separate DB.

Index type is read from config (default: IndexFlatIP).
IndexFlatIP on L2-normalised vectors gives exact cosine similarity — no
approximation error, appropriate for a corpus of this size (~4 documents).
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import yaml

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
# Index construction
# ---------------------------------------------------------------------------

def build_index(vectors: np.ndarray, index_type: str = "IndexFlatIP") -> faiss.Index:
    """Create and populate a FAISS index from a 2-D float32 array.

    Args:
        vectors:    np.ndarray of shape (n, dim), dtype float32.
                    Vectors should be L2-normalised when using IndexFlatIP.
        index_type: FAISS index class name. Currently only "IndexFlatIP"
                    is supported; extend here if approximate search is needed.

    Returns:
        A populated faiss.Index ready for search.

    Raises:
        ValueError: If vectors array has wrong dtype or dimensions.
        NotImplementedError: If an unsupported index_type is requested.
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    if vectors.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {vectors.shape}.")

    n, dim = vectors.shape

    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dim)
    else:
        raise NotImplementedError(
            f"index_type '{index_type}' is not supported. Use 'IndexFlatIP'."
        )

    index.add(vectors)
    logger.debug("Built %s index: %d vectors, dim=%d.", index_type, n, dim)
    return index


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_index(
    index: faiss.Index,
    metadata: list[dict],
    index_path: Path,
    meta_path: Path,
) -> None:
    """Write a FAISS index and its metadata list to disk.

    Args:
        index:      Populated faiss.Index.
        metadata:   List of chunk dicts, index-aligned with FAISS rows.
        index_path: Destination path for the .faiss binary file.
        meta_path:  Destination path for the .meta.json file.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("Saved FAISS index → %s  (%d vectors)", index_path, index.ntotal)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    logger.info("Saved metadata    → %s  (%d records)", meta_path, len(metadata))


def load_index(index_path: Path, meta_path: Path) -> tuple[faiss.Index, list[dict]]:
    """Load a FAISS index and its metadata list from disk.

    Args:
        index_path: Path to the .faiss binary file.
        meta_path:  Path to the .meta.json file.

    Returns:
        Tuple of (faiss.Index, list[dict]).

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    index = faiss.read_index(str(index_path))
    logger.info("Loaded FAISS index ← %s  (%d vectors)", index_path, index.ntotal)

    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    logger.info("Loaded metadata    ← %s  (%d records)", meta_path, len(metadata))

    return index, metadata


# ---------------------------------------------------------------------------
# High-level store class
# ---------------------------------------------------------------------------

class FAISSStore:
    """Manages a single FAISS index + metadata list for one document.

    Attributes:
        index:    faiss.Index (None until built or loaded).
        metadata: Parallel list of chunk dicts (None until built or loaded).
    """

    def __init__(self) -> None:
        """Initialise an empty FAISSStore."""
        self.index:    faiss.Index | None = None
        self.metadata: list[dict] | None  = None

    def build(self, vectors: np.ndarray, metadata: list[dict], index_type: str = "IndexFlatIP") -> None:
        """Build the index from vectors and attach metadata.

        Args:
            vectors:    np.ndarray shape (n, dim), float32, L2-normalised.
            metadata:   List of n chunk dicts, aligned with vectors rows.
            index_type: FAISS index class name.

        Raises:
            ValueError: If len(metadata) != vectors.shape[0].
        """
        if len(metadata) != vectors.shape[0]:
            raise ValueError(
                f"metadata length ({len(metadata)}) must equal "
                f"number of vectors ({vectors.shape[0]})."
            )
        self.index    = build_index(vectors, index_type)
        self.metadata = metadata

    def save(self, index_path: Path, meta_path: Path) -> None:
        """Persist the index and metadata to disk.

        Args:
            index_path: Destination .faiss file path.
            meta_path:  Destination .meta.json file path.

        Raises:
            RuntimeError: If build() has not been called yet.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Cannot save — store has not been built. Call build() first.")
        save_index(self.index, self.metadata, index_path, meta_path)

    def load(self, index_path: Path, meta_path: Path) -> None:
        """Load index and metadata from disk into this store.

        Args:
            index_path: Path to the .faiss binary file.
            meta_path:  Path to the .meta.json file.
        """
        self.index, self.metadata = load_index(index_path, meta_path)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        """Search the index and return the top-k most similar chunk records.

        Args:
            query_vector: 1-D float32 numpy array, L2-normalised.
            top_k:        Number of results to return.

        Returns:
            List of up to top_k chunk dicts, each augmented with a
            'score' key (float, cosine similarity in [-1, 1]).

        Raises:
            RuntimeError: If the store has not been built or loaded.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Cannot search — store is empty. Call build() or load() first.")

        query_vector = query_vector.astype(np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for unfilled slots
                continue
            record = dict(self.metadata[idx])
            record["score"] = float(score)
            results.append(record)

        return results


# ---------------------------------------------------------------------------
# Pipeline helper: build and save all documents for one path
# ---------------------------------------------------------------------------

def build_and_save_all(
    processed_dir: Path,
    documents: list[dict],
    embedder,                  # src.embeddings.embedder.Embedder
    index_type: str,
) -> None:
    """Embed all chunked documents and write FAISS indexes to processed_dir.

    For each document, reads <doc_name>_chunks.json, embeds the chunk texts,
    and saves <doc_name>.faiss + <doc_name>.meta.json.
    Skips documents whose .faiss file already exists (resumable).

    Args:
        processed_dir: Directory containing <doc_name>_chunks.json files
                       and where .faiss + .meta.json will be written.
        documents:     List of document dicts from config.yaml.
        embedder:      A built src.embeddings.embedder.Embedder instance.
        index_type:    FAISS index class name from config.
    """
    for doc in documents:
        doc_name   = doc["name"]
        chunks_path = processed_dir / f"{doc_name}_chunks.json"
        index_path  = processed_dir / f"{doc_name}.faiss"
        meta_path   = processed_dir / f"{doc_name}.meta.json"

        if index_path.exists() and meta_path.exists():
            logger.info("'%s' index already exists — skipping.", doc_name)
            continue

        if not chunks_path.exists():
            logger.warning(
                "'%s': chunks file not found at %s — run chunking first.",
                doc_name, chunks_path,
            )
            continue

        with open(chunks_path, "r", encoding="utf-8") as fh:
            chunks = json.load(fh)

        if not chunks:
            logger.error("'%s': chunks file is empty — skipping.", doc_name)
            continue

        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks for '%s' …", len(texts), doc_name)
        vectors = embedder.encode(texts, desc=f"  {doc_name}")

        store = FAISSStore()
        store.build(vectors, chunks, index_type)
        store.save(index_path, meta_path)
        logger.info("'%s': index built (%d vectors).", doc_name, len(chunks))
