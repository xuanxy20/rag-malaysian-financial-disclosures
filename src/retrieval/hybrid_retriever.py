"""
HybridRetriever — BM25 + dense FAISS retrieval with cross-encoder reranking.

Architecture (Path C):
    Query
      ├── FAISS dense search  (top_k_dense candidates)
      └── BM25 keyword search (top_k_bm25  candidates)
            ↓
      merge + deduplicate by chunk_id
      weighted hybrid score = dense_weight * dense_norm + bm25_weight * bm25_norm
            ↓
      cross-encoder rerank → top_k_final results passed to generator

Score normalisation:
  - Dense (cosine in [-1, 1]) → (score + 1) / 2  → [0, 1]
  - BM25  ([0, ∞))            → score / max_score → [0, 1]

Company-targeted retrieval is preserved: callers swap faiss_stores and
bm25_stores to a single-document list before calling retrieve(), exactly as
run_pipeline.py does for Retriever.stores.

Usage (from run_pipeline.py):
    retriever = HybridRetriever.from_config(faiss_stores, bm25_stores, embedder, config_path)
    retriever.faiss_stores = [target_faiss_store]
    retriever.bm25_stores  = [target_bm25_store]
    chunks = retriever.retrieve(question)
"""

import logging
from pathlib import Path

import yaml

from src.embeddings.embedder import Embedder
from src.retrieval.bm25_store import BM25Store
from src.retrieval.reranker import Reranker
from src.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid BM25 + dense retriever with cross-encoder reranking.

    Attributes:
        faiss_stores:  List of FAISSStore instances to search (mutable for scoping).
        bm25_stores:   List of BM25Store instances to search (mutable for scoping).
        embedder:      Shared Embedder for query encoding.
        reranker:      Reranker instance for final scoring.
        top_k_dense:   FAISS candidates per query.
        top_k_bm25:    BM25 candidates per query.
        top_k_final:   Results returned after reranking.
        dense_weight:  Weight for normalised dense score in hybrid merge.
        bm25_weight:   Weight for normalised BM25 score in hybrid merge.
    """

    def __init__(
        self,
        faiss_stores: list[FAISSStore],
        bm25_stores: list[BM25Store],
        embedder: Embedder,
        reranker: Reranker,
        top_k_dense: int,
        top_k_bm25: int,
        top_k_final: int,
        dense_weight: float,
        bm25_weight: float,
    ) -> None:
        self.faiss_stores = faiss_stores
        self.bm25_stores  = bm25_stores
        self.embedder     = embedder
        self.reranker     = reranker
        self.top_k_dense  = top_k_dense
        self.top_k_bm25   = top_k_bm25
        self.top_k_final  = top_k_final
        self.dense_weight = dense_weight
        self.bm25_weight  = bm25_weight

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict]:
        """Hybrid retrieval pipeline: dense + BM25 → merge → rerank.

        Args:
            query: Natural-language question string.

        Returns:
            List of up to top_k_final chunk dicts sorted by rerank_score.
            Each dict includes 'score' (rerank), '_dense_norm', '_bm25_norm'.
        """
        if not query.strip():
            logger.warning("retrieve() called with empty query — returning [].")
            return []

        # 1. Dense retrieval
        query_vector  = self.embedder.encode_query(query)
        dense_results: list[dict] = []
        for store in self.faiss_stores:
            try:
                dense_results.extend(store.search(query_vector, self.top_k_dense))
            except Exception as exc:
                logger.error("FAISS search failed: %s", exc)

        # 2. BM25 retrieval
        bm25_results: list[dict] = []
        for store in self.bm25_stores:
            try:
                bm25_results.extend(store.search(query, self.top_k_bm25))
            except Exception as exc:
                logger.error("BM25 search failed: %s", exc)

        if not dense_results and not bm25_results:
            logger.warning("No results from either retriever for query: '%s'", query[:80])
            return []

        # 3. Normalise dense scores: cosine ∈ [-1, 1] → [0, 1]
        dense_by_id: dict[str, dict] = {}
        for r in dense_results:
            cid       = r["chunk_id"]
            norm      = (r["score"] + 1.0) / 2.0
            existing  = dense_by_id.get(cid)
            if existing is None or norm > existing["_dense_norm"]:
                rec              = dict(r)
                rec["_dense_norm"] = norm
                dense_by_id[cid] = rec

        # 4. Normalise BM25 scores: [0, ∞) → [0, 1] via max-normalisation
        bm25_by_id: dict[str, dict] = {}
        if bm25_results:
            max_bm25 = max(r["bm25_score"] for r in bm25_results)
            if max_bm25 <= 0:
                max_bm25 = 1.0
            for r in bm25_results:
                cid      = r["chunk_id"]
                norm     = r["bm25_score"] / max_bm25
                existing = bm25_by_id.get(cid)
                if existing is None or norm > existing["_bm25_norm"]:
                    rec             = dict(r)
                    rec["_bm25_norm"] = norm
                    bm25_by_id[cid] = rec

        # 5. Merge: union of chunk_ids, compute weighted hybrid score
        all_ids    = set(dense_by_id) | set(bm25_by_id)
        candidates: list[dict] = []
        for cid in all_ids:
            d_norm = dense_by_id[cid]["_dense_norm"] if cid in dense_by_id else 0.0
            b_norm = bm25_by_id[cid]["_bm25_norm"]   if cid in bm25_by_id  else 0.0
            hybrid = self.dense_weight * d_norm + self.bm25_weight * b_norm

            # Prefer the dense record as the base (richer metadata); fall back to BM25
            base   = dense_by_id.get(cid) or bm25_by_id[cid]
            record = dict(base)
            record["_dense_norm"] = d_norm
            record["_bm25_norm"]  = b_norm
            record["_hybrid_score"] = hybrid
            candidates.append(record)

        # Pre-sort by hybrid score so the reranker receives the strongest candidates first
        candidates.sort(key=lambda x: x["_hybrid_score"], reverse=True)

        logger.debug(
            "Hybrid merge: %d dense + %d BM25 → %d unique candidates",
            len(dense_results), len(bm25_results), len(candidates),
        )

        # 6. Cross-encoder rerank → top_k_final
        reranked = self.reranker.rerank(query, candidates, self.top_k_final)
        return reranked

    def retrieve_batch(self, queries: list[str]) -> list[list[dict]]:
        """Retrieve for a list of queries.

        Args:
            queries: List of question strings.

        Returns:
            List of result lists (same order as input).
        """
        return [self.retrieve(q) for q in queries]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        faiss_stores: list[FAISSStore],
        bm25_stores: list[BM25Store],
        embedder: Embedder,
        reranker: Reranker,
        config_path: Path | None = None,
    ) -> "HybridRetriever":
        """Construct a HybridRetriever from configs/config.yaml path_c section.

        Args:
            faiss_stores: Pre-loaded FAISSStore instances.
            bm25_stores:  Pre-loaded BM25Store instances.
            embedder:     Shared Embedder instance.
            reranker:     Configured Reranker instance.
            config_path:  Optional config path override.

        Returns:
            A configured HybridRetriever.
        """
        project_root = Path(__file__).resolve().parents[2]
        if config_path is None:
            config_path = project_root / "configs" / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        rc = cfg.get("path_c", {}).get("retrieval", {})
        top_k_dense  = rc.get("top_k_dense",  20)
        top_k_bm25   = rc.get("top_k_bm25",   20)
        top_k_final  = rc.get("top_k_final",  10)
        dense_weight = rc.get("dense_weight", 0.7)
        bm25_weight  = rc.get("bm25_weight",  0.3)

        logger.info(
            "HybridRetriever | dense_k=%d bm25_k=%d final_k=%d | "
            "weights: dense=%.1f bm25=%.1f | faiss_stores=%d bm25_stores=%d",
            top_k_dense, top_k_bm25, top_k_final,
            dense_weight, bm25_weight,
            len(faiss_stores), len(bm25_stores),
        )
        return cls(
            faiss_stores=faiss_stores,
            bm25_stores=bm25_stores,
            embedder=embedder,
            reranker=reranker,
            top_k_dense=top_k_dense,
            top_k_bm25=top_k_bm25,
            top_k_final=top_k_final,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
