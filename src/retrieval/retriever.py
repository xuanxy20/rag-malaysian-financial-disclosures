"""
Retriever — shared dense retrieval module (identical for Path A and Path B).

Given a query string, the retriever:
  1. Encodes the query into an L2-normalised vector via the shared Embedder.
  2. Searches all loaded FAISS indexes (one per document) for the top-k chunks
     by cosine similarity.
  3. Merges results across documents, re-ranks by score, and returns the
     global top-k chunks.

This module does not know whether it is operating on Path A or Path B indexes —
it receives pre-loaded FAISSStore instances from the caller. This keeps the
retrieval logic path-agnostic and ensures the only variable between paths is
the content of the indexes.

Usage (from run_pipeline.py or ragas_evaluator.py):
    retriever = Retriever.from_config(stores, embedder)
    results   = retriever.retrieve("What is CIMB's net profit for FY2025?")
    # results → list of up to top_k chunk dicts, each with a 'score' key
"""

import logging
from pathlib import Path

import yaml

from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore

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
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Dense retriever that queries one or more FAISS indexes.

    Each FAISSStore holds the index for a single document. The retriever
    fans out the query across all stores, then merges and re-ranks results
    by cosine similarity to return the global top-k.

    Attributes:
        stores:   List of loaded FAISSStore instances.
        embedder: Shared Embedder instance for query encoding.
        top_k:    Number of results to return per query.
    """

    def __init__(
        self,
        stores: list[FAISSStore],
        embedder: Embedder,
        top_k: int,
    ) -> None:
        """Initialise the Retriever.

        Args:
            stores:   List of populated FAISSStore instances.
            embedder: Embedder instance (model may be loaded lazily).
            top_k:    Number of top chunks to return per query.
        """
        self.stores   = stores
        self.embedder = embedder
        self.top_k    = top_k

    def retrieve(self, query: str) -> list[dict]:
        """Encode a query and return the top-k most relevant chunks.

        Searches all loaded FAISS indexes, merges results, sorts by cosine
        similarity (descending), and returns the top-k records.

        Args:
            query: Natural-language question string.

        Returns:
            List of up to top_k chunk dicts sorted by 'score' descending.
            Each dict contains all fields from the chunk record plus 'score'.
        """
        if not query.strip():
            logger.warning("retrieve() called with an empty query — returning [].")
            return []

        query_vector = self.embedder.encode_query(query)

        all_results: list[dict] = []
        for store in self.stores:
            try:
                results = store.search(query_vector, self.top_k)
                all_results.extend(results)
            except Exception as exc:
                logger.error("Search failed on a store: %s", exc)

        if not all_results:
            logger.warning("No results returned for query: '%s'", query[:80])
            return []

        # Global re-rank across all documents
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[: self.top_k]

        logger.debug(
            "Query: '%s…' → %d results (top score: %.4f)",
            query[:60], len(top_results), top_results[0]["score"],
        )
        return top_results

    def retrieve_batch(self, queries: list[str]) -> list[list[dict]]:
        """Retrieve top-k chunks for a list of queries.

        Args:
            queries: List of query strings.

        Returns:
            List of result lists, one per query (same order as input).
        """
        return [self.retrieve(q) for q in queries]

    @classmethod
    def from_config(
        cls,
        stores: list[FAISSStore],
        embedder: Embedder,
        config_path: Path | None = None,
        top_k: int | None = None,
    ) -> "Retriever":
        """Construct a Retriever from configs/config.yaml.

        Args:
            stores:      List of populated FAISSStore instances.
            embedder:    Shared Embedder instance.
            config_path: Optional config path override.
            top_k:       Optional override for the number of results to return.
                         When provided, takes precedence over config.yaml.

        Returns:
            A configured Retriever instance.
        """
        project_root = Path(__file__).resolve().parents[2]
        if config_path is None:
            config_path = project_root / "configs" / "config.yaml"

        cfg           = load_config(config_path)
        effective_k   = top_k if top_k is not None else cfg["retrieval"]["top_k"]

        logger.info("Retriever initialised | top_k=%d | stores=%d", effective_k, len(stores))
        return cls(stores=stores, embedder=embedder, top_k=effective_k)


# ---------------------------------------------------------------------------
# Store loader helper (used by run_pipeline.py)
# ---------------------------------------------------------------------------

def load_stores(processed_dir: Path, documents: list[dict]) -> list[FAISSStore]:
    """Load all FAISS indexes for a given pipeline path.

    Looks for <doc_name>.faiss + <doc_name>.meta.json in processed_dir.
    Documents missing either file are skipped with a warning.

    Args:
        processed_dir: data/processed/path_a/ or data/processed/path_b/.
        documents:     List of document dicts from config.yaml.

    Returns:
        List of loaded FAISSStore instances (one per available document).
    """
    stores: list[FAISSStore] = []

    for doc in documents:
        doc_name   = doc["name"]
        index_path = processed_dir / f"{doc_name}.faiss"
        meta_path  = processed_dir / f"{doc_name}.meta.json"

        if not index_path.exists() or not meta_path.exists():
            logger.warning(
                "'%s': index files not found in %s — skipping.",
                doc_name, processed_dir,
            )
            continue

        store = FAISSStore()
        store.load(index_path, meta_path)
        stores.append(store)

    logger.info("Loaded %d / %d document indexes from %s.", len(stores), len(documents), processed_dir)
    return stores
