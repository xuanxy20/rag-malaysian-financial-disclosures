"""
Reranker — cross-encoder wrapper for Path C hybrid retrieval.

Wraps sentence_transformers.CrossEncoder to score (query, chunk) pairs and
return the top-k most relevant chunks.  The model is loaded lazily on first
use so import cost is zero when the reranker is not needed.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking
  - ~22M params, fast on CPU / MPS
  - Output: raw logit (higher = more relevant, unbounded)

Usage:
    reranker = Reranker.from_config()
    top_chunks = reranker.rerank(query, candidate_chunks, top_k=10)
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker for a list of candidate chunks.

    Attributes:
        model_name: HuggingFace model ID for the CrossEncoder.
        device:     Torch device string ('cpu', 'mps', 'cuda').
        _model:     Loaded CrossEncoder instance (None until first use).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device     = device
        self._model     = None

    def _load(self) -> None:
        """Lazily load the CrossEncoder model.

        Sets HF_HUB_DISABLE_SSL_VERIFICATION to avoid SSL errors on machines
        behind a corporate proxy.  The model is expected to be cached locally;
        if not, run with HF_HUB_DISABLE_SSL_VERIFICATION=1 in the shell first.
        """
        if self._model is not None:
            return
        import os
        os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder model '%s' on device '%s' …", self.model_name, self.device)
        self._model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Cross-encoder loaded.")

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """Score each (query, chunk) pair and return the top-k highest-scoring chunks.

        Args:
            query:  Natural-language question string.
            chunks: Candidate chunk dicts (each must have a 'text' key).
            top_k:  Number of results to return.

        Returns:
            List of up to top_k chunk dicts (copies) sorted by 'rerank_score'
            descending.  Each dict has an added 'rerank_score' key.
        """
        if not chunks:
            return []

        self._load()

        pairs  = [(query, c["text"]) for c in chunks]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        results = []
        for score, chunk in ranked[:top_k]:
            record = dict(chunk)
            record["rerank_score"] = float(score)
            # Expose rerank_score as the primary 'score' key so the generator
            # and evaluator can treat all retrievers uniformly.
            record["score"] = float(score)
            results.append(record)

        logger.debug(
            "Reranked %d → %d chunks (top score: %.4f)",
            len(chunks), len(results), results[0]["rerank_score"] if results else 0,
        )
        return results

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: Path | None = None,
    ) -> "Reranker":
        """Construct a Reranker from configs/config.yaml path_c settings.

        Args:
            config_path: Optional override for the config file path.

        Returns:
            A configured Reranker instance (model not yet loaded).
        """
        project_root = Path(__file__).resolve().parents[2]
        if config_path is None:
            config_path = project_root / "configs" / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        retrieval_cfg = cfg.get("path_c", {}).get("retrieval", {})
        model_name    = retrieval_cfg.get("reranker_model", _DEFAULT_MODEL)

        # Cross-encoder runs on CPU; MPS support via sentence-transformers is
        # unreliable for CrossEncoder (only works for bi-encoders).
        device = "cpu"

        logger.info("Reranker configured: model='%s', device='%s'", model_name, device)
        return cls(model_name=model_name, device=device)
