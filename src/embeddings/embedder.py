"""
Embedder — shared embedding module (identical for Path A and Path B).

Loads sentence-transformers/all-MiniLM-L6-v2 locally and encodes a list of
text strings into L2-normalised numpy float32 vectors.

This module is intentionally stateless: callers construct an Embedder instance,
call encode(), and receive a numpy array. No caching or persistence happens here
— that is the responsibility of faiss_store.py.

All parameters (model name, device, batch size, normalisation flag) are read
from configs/config.yaml under the 'embeddings' key.
"""

import logging
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
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
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """Wraps a SentenceTransformer model for batch text encoding.

    Attributes:
        model_name:  HuggingFace model identifier.
        device:      Torch device string ("cpu" or "mps").
        batch_size:  Number of texts encoded per forward pass.
        normalize:   Whether to L2-normalize output vectors.
        _model:      Loaded SentenceTransformer instance (lazy).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        """Initialise the Embedder with configuration parameters.

        The underlying SentenceTransformer model is loaded lazily on the
        first call to encode() to avoid slow imports at module load time.

        Args:
            model_name: HuggingFace model identifier or local path.
            device:     Torch device string, e.g. "cpu" or "mps".
            batch_size: Texts per encoding batch.
            normalize:  If True, L2-normalise all output vectors.
        """
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size
        self.normalize  = normalize
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> None:
        """Load the SentenceTransformer model if not already loaded."""
        if self._model is None:
            logger.info("Loading embedding model '%s' on device '%s' …", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded. Embedding dimension: %d", self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], desc: str = "Embedding") -> np.ndarray:
        """Encode a list of strings into a 2-D float32 numpy array.

        Processes texts in batches of self.batch_size with a tqdm progress bar.
        Empty or whitespace-only strings are replaced with a single space before
        encoding to avoid SentenceTransformer errors on blank inputs.

        Args:
            texts: List of strings to embed.
            desc:  Label shown on the tqdm progress bar.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32.
            Vectors are L2-normalised if self.normalize is True.
        """
        self._load_model()

        if not texts:
            raise ValueError("encode() received an empty list of texts.")

        # Guard against blank strings
        sanitised = [t if t.strip() else " " for t in texts]

        all_vectors: list[np.ndarray] = []

        for start in tqdm(
            range(0, len(sanitised), self.batch_size),
            desc=desc,
            unit="batch",
            leave=False,
        ):
            batch = sanitised[start : start + self.batch_size]
            vectors = self._model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )
            all_vectors.append(vectors.astype(np.float32))

        result = np.vstack(all_vectors)
        logger.debug("Encoded %d texts → shape %s", len(texts), result.shape)
        return result

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a 1-D float32 vector.

        Convenience wrapper around encode() for single-string retrieval queries.

        Args:
            query: Query string.

        Returns:
            np.ndarray of shape (embedding_dim,), dtype float32.
        """
        vectors = self.encode([query], desc="Query")
        return vectors[0]

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model.

        Returns:
            Integer embedding dimension.
        """
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_embedder(config_path: Path | None = None) -> Embedder:
    """Construct an Embedder from configs/config.yaml.

    This is the standard way to create an Embedder in pipeline scripts —
    they call build_embedder() rather than instantiating Embedder directly.

    Args:
        config_path: Optional override for the config file location.
                     Defaults to <project_root>/configs/config.yaml.

    Returns:
        A configured Embedder instance (model not yet loaded).
    """
    project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    cfg     = load_config(config_path)
    emb_cfg = cfg["embeddings"]

    return Embedder(
        model_name=emb_cfg["model_name"],
        device=emb_cfg["device"],
        batch_size=emb_cfg["batch_size"],
        normalize=emb_cfg["normalize"],
    )


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    embedder = build_embedder()
    sample = [
        "What is the total revenue of Bursa Malaysia for FY2026?",
        "Describe the risk management framework of CIMB Group.",
    ]
    vecs = embedder.encode(sample, desc="Smoke test")
    logger.info("Output shape: %s  |  First vector norm: %.6f", vecs.shape, float(np.linalg.norm(vecs[0])))
