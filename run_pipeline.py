"""
run_pipeline.py — Single entry point for the RAG research pipeline.

Usage:
    python run_pipeline.py --path a      # Run Path A (baseline) end to end
    python run_pipeline.py --path b      # Run Path B (layout-aware) end to end
    python run_pipeline.py --path c      # Run Path C (hybrid BM25 + dense + reranking)
    python run_pipeline.py --path all    # Run all paths then generate comparison report

    Optional flags:
    --skip-extraction    Skip extraction stage (use existing processed files)
    --skip-chunking      Skip chunking stage
    --skip-embedding     Skip embedding/index-build stage
    --skip-eval          Skip RAGAS evaluation (pipeline stops after generation)
    --config PATH        Override default configs/config.yaml path

    Note: Path C always skips extraction/chunking/embedding (reuses Path B's processed dir).

Each stage saves its output to disk before the next stage begins. Re-running
the pipeline skips any stage whose output files already exist, making the
pipeline resumable after a crash or interruption.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path so src.* imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedder import build_embedder
from src.evaluation.ragas_evaluator import evaluate_pipeline
from src.evaluation.report_generator import generate_report
from src.generation.generator import build_generator
from src.retrieval.bm25_store import build_and_save_all_bm25
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever, load_stores
from src.vectorstore.faiss_store import build_and_save_all

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & logging helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load and return the central YAML configuration.

    Args:
        config_path: Path to configs/config.yaml.

    Returns:
        Parsed config as a nested dict.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging(cfg: dict) -> None:
    """Configure root logger from config.yaml logging section.

    Args:
        cfg: Full config dict.
    """
    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s — %(message)s"),
        datefmt=log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
        force=True,
    )


# ---------------------------------------------------------------------------
# Eval questions loader
# ---------------------------------------------------------------------------

def load_eval_questions(questions_path: Path) -> tuple[list[str], list[str]]:
    """Load evaluation questions and ground truths from eval_questions.json.

    Args:
        questions_path: Path to questions/eval_questions.json.

    Returns:
        Tuple of (questions list, ground_truths list), both aligned by index.

    Raises:
        FileNotFoundError: If the questions file does not exist.
        ValueError: If no questions are found in the file.
    """
    if not questions_path.exists():
        raise FileNotFoundError(f"Eval questions file not found: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    items = data.get("questions", [])
    if not items:
        raise ValueError(f"No questions found in {questions_path}")

    questions     = [q["question"]     for q in items]
    ground_truths = [q["ground_truth"] for q in items]
    company_ids   = [q.get("company_id", "") for q in items]

    logger.info("Loaded %d evaluation questions from %s", len(questions), questions_path)
    return questions, ground_truths, company_ids


# ---------------------------------------------------------------------------
# Stage: Extraction
# ---------------------------------------------------------------------------

def run_extraction(path_label: str, config_path: Path) -> None:
    """Run the extraction stage for the given path.

    Args:
        path_label:  "a" or "b".
        config_path: Path to config.yaml.
    """
    logger.info("--- [%s] Stage 1: Extraction ---", path_label.upper())
    if path_label == "a":
        from src.extraction.path_a_extractor import run
    else:
        from src.extraction.path_b_extractor import run
    run(config_path=config_path)


# ---------------------------------------------------------------------------
# Stage: Chunking
# ---------------------------------------------------------------------------

def run_chunking(path_label: str, config_path: Path) -> None:
    """Run the chunking stage for the given path.

    Args:
        path_label:  "a" or "b".
        config_path: Path to config.yaml.
    """
    logger.info("--- [%s] Stage 2: Chunking ---", path_label.upper())
    if path_label == "a":
        from src.chunking.fixed_chunker import run
    else:
        from src.chunking.structural_chunker import run
    run(config_path=config_path)


# ---------------------------------------------------------------------------
# Stage: Embedding & Index Build
# ---------------------------------------------------------------------------

def run_embedding(path_label: str, cfg: dict, config_path: Path) -> None:
    """Embed all chunks and build FAISS indexes for the given path.

    Args:
        path_label:  "a" or "b".
        cfg:         Loaded config dict.
        config_path: Path to config.yaml (for embedder factory).
    """
    logger.info("--- [%s] Stage 3: Embedding & Index Build ---", path_label.upper())

    processed_dir_key = "processed_path_a" if path_label == "a" else "processed_path_b"
    processed_dir = PROJECT_ROOT / cfg["paths"][processed_dir_key]
    index_type    = cfg["vectorstore"]["index_type"]
    documents     = cfg["documents"]

    embedder = build_embedder(config_path)
    build_and_save_all(processed_dir, documents, embedder, index_type)


# ---------------------------------------------------------------------------
# Stage: Retrieval + Generation
# ---------------------------------------------------------------------------

def run_generation(
    path_label: str,
    cfg: dict,
    config_path: Path,
    questions: list[str],
    company_ids: list[str] | None = None,
) -> list[dict]:
    """Retrieve context and generate answers for all eval questions.

    Args:
        path_label:  "a" or "b".
        cfg:         Loaded config dict.
        config_path: Path to config.yaml.
        questions:   List of evaluation question strings.
        company_ids: Optional list of company_id strings aligned with questions.
                     When provided, each question searches only its company's index.

    Returns:
        List of RAG result dicts from Generator.generate_batch().
    """
    logger.info("--- [%s] Stage 4: Retrieval & Generation ---", path_label.upper())

    processed_dir_key = "processed_path_a" if path_label == "a" else "processed_path_b"
    processed_dir     = PROJECT_ROOT / cfg["paths"][processed_dir_key]
    results_dir_key   = "results_path_a" if path_label == "a" else "results_path_b"
    results_dir       = PROJECT_ROOT / cfg["paths"][results_dir_key]
    documents         = cfg["documents"]

    # Check for cached generation results (resumable)
    cached_path = results_dir / f"rag_results_{path_label}.json"
    if cached_path.exists():
        logger.info("Cached generation results found — loading from %s", cached_path)
        with open(cached_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    embedder  = build_embedder(config_path)
    stores    = load_stores(processed_dir, documents)

    if not stores:
        logger.error("No FAISS indexes found for path_%s. Run embedding stage first.", path_label)
        return []

    # Build a lookup from doc_name → store for company-targeted retrieval
    store_by_name = {
        s.metadata[0]["doc_name"]: s
        for s in stores
        if s.metadata
    }

    path_top_k = cfg.get("path_b", {}).get("retrieval", {}).get("top_k") if path_label == "b" else None
    retriever = Retriever.from_config(stores, embedder, config_path, top_k=path_top_k)
    generator = build_generator(config_path)

    if not generator.check_connection():
        logger.error(
            "Ollama is not reachable or model '%s' is not available. "
            "Start Ollama and pull the model before running generation.",
            cfg["generation"]["model"],
        )
        return []

    logger.info("Retrieving and generating answers for %d questions …", len(questions))
    rag_results = []
    for i, question in enumerate(tqdm(questions, desc=f"  Path {path_label.upper()} generation", unit="q")):
        company_id = (company_ids[i] if company_ids else "") or ""
        target_store = store_by_name.get(company_id)
        if target_store:
            retriever.stores = [target_store]
        else:
            retriever.stores = stores
            if company_id:
                logger.warning("No index found for company_id='%s' — searching all stores.", company_id)
        chunks = retriever.retrieve(question)
        result = generator.generate(question, chunks)
        rag_results.append(result)

    # Cache results so evaluation can be re-run without re-generating
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(cached_path, "w", encoding="utf-8") as fh:
        json.dump(rag_results, fh, ensure_ascii=False, indent=2)
    logger.info("Cached generation results → %s", cached_path)

    return rag_results


# ---------------------------------------------------------------------------
# Stage: Retrieval + Generation (Path C — hybrid BM25 + dense + reranking)
# ---------------------------------------------------------------------------

def run_generation_c(
    cfg: dict,
    config_path: Path,
    questions: list[str],
    company_ids: list[str] | None = None,
) -> list[dict]:
    """Retrieve context with hybrid retrieval and generate answers for Path C.

    Reuses Path B's processed directory (FAISS indexes + chunk JSON).
    Builds BM25 indexes on first run (cached as <doc>_bm25.json).

    Args:
        cfg:         Loaded config dict.
        config_path: Path to config.yaml.
        questions:   List of evaluation question strings.
        company_ids: Optional list of company_id strings aligned with questions.

    Returns:
        List of RAG result dicts from Generator.generate_batch().
    """
    logger.info("--- [C] Stage 4: Retrieval & Generation (Hybrid) ---")

    # Path C shares Path B's processed dir
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_path_b"]
    results_dir   = PROJECT_ROOT / cfg["paths"]["results_path_c"]
    documents     = cfg["documents"]

    cached_path = results_dir / "rag_results_c.json"
    if cached_path.exists():
        logger.info("Cached generation results found — loading from %s", cached_path)
        with open(cached_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    embedder = build_embedder(config_path)
    faiss_stores = load_stores(processed_dir, documents)
    bm25_stores  = build_and_save_all_bm25(processed_dir, documents)

    if not faiss_stores:
        logger.error("No FAISS indexes found in %s. Run Path B embedding first.", processed_dir)
        return []
    if not bm25_stores:
        logger.error("No BM25 indexes could be built from %s.", processed_dir)
        return []

    # Build per-doc lookups for company-targeted retrieval
    faiss_by_name = {
        s.metadata[0]["doc_name"]: s
        for s in faiss_stores
        if s.metadata
    }
    bm25_by_name = {s.doc_name: s for s in bm25_stores if s.doc_name}

    reranker  = Reranker.from_config(config_path)
    retriever = HybridRetriever.from_config(
        faiss_stores, bm25_stores, embedder, reranker, config_path
    )
    generator = build_generator(config_path)

    if not generator.check_connection():
        logger.error(
            "Ollama is not reachable or model '%s' is not available.",
            cfg["generation"]["model"],
        )
        return []

    logger.info("Retrieving and generating answers for %d questions …", len(questions))
    rag_results = []
    for i, question in enumerate(tqdm(questions, desc="  Path C generation", unit="q")):
        company_id   = (company_ids[i] if company_ids else "") or ""
        target_faiss = faiss_by_name.get(company_id)
        target_bm25  = bm25_by_name.get(company_id)

        if target_faiss and target_bm25:
            retriever.faiss_stores = [target_faiss]
            retriever.bm25_stores  = [target_bm25]
        else:
            retriever.faiss_stores = faiss_stores
            retriever.bm25_stores  = bm25_stores
            if company_id:
                logger.warning(
                    "No index found for company_id='%s' — searching all stores.", company_id
                )

        chunks = retriever.retrieve(question)
        result = generator.generate(question, chunks)
        rag_results.append(result)

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(cached_path, "w", encoding="utf-8") as fh:
        json.dump(rag_results, fh, ensure_ascii=False, indent=2)
    logger.info("Cached generation results → %s", cached_path)

    return rag_results


# ---------------------------------------------------------------------------
# Stage: RAGAS Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    path_label: str,
    rag_results: list[dict],
    ground_truths: list[str],
    config_path: Path,
) -> None:
    """Run RAGAS evaluation for one pipeline path.

    Args:
        path_label:    "a" or "b".
        rag_results:   RAG output dicts from run_generation().
        ground_truths: Aligned ground-truth answer strings.
        config_path:   Path to config.yaml.
    """
    logger.info("--- [%s] Stage 5: RAGAS Evaluation ---", path_label.upper())
    evaluate_pipeline(
        rag_results=rag_results,
        ground_truths=ground_truths,
        path_label=f"path_{path_label}",
        config_path=config_path,
    )


# ---------------------------------------------------------------------------
# Per-path pipeline runner
# ---------------------------------------------------------------------------

def run_path(
    path_label: str,
    cfg: dict,
    config_path: Path,
    args: argparse.Namespace,
) -> list[dict]:
    """Run the complete pipeline for a single path (a or b).

    Args:
        path_label:  "a" or "b".
        cfg:         Loaded config dict.
        config_path: Path to config.yaml.
        args:        Parsed CLI arguments (skip flags).

    Returns:
        List of RAG result dicts (empty list on failure).
    """
    logger.info("========== PATH %s PIPELINE START ==========", path_label.upper())

    if not args.skip_extraction:
        run_extraction(path_label, config_path)
    else:
        logger.info("--- [%s] Stage 1: Extraction — SKIPPED ---", path_label.upper())

    if not args.skip_chunking:
        run_chunking(path_label, config_path)
    else:
        logger.info("--- [%s] Stage 2: Chunking — SKIPPED ---", path_label.upper())

    if not args.skip_embedding:
        run_embedding(path_label, cfg, config_path)
    else:
        logger.info("--- [%s] Stage 3: Embedding — SKIPPED ---", path_label.upper())

    questions, ground_truths, company_ids = load_eval_questions(
        PROJECT_ROOT / cfg["paths"]["eval_questions"]
    )

    rag_results = run_generation(path_label, cfg, config_path, questions, company_ids)

    if not args.skip_eval and rag_results:
        run_eval(path_label, rag_results, ground_truths, config_path)

    logger.info("========== PATH %s PIPELINE COMPLETE ==========", path_label.upper())
    return rag_results


# ---------------------------------------------------------------------------
# Per-path pipeline runner (Path C)
# ---------------------------------------------------------------------------

def run_path_c(
    cfg: dict,
    config_path: Path,
    args: argparse.Namespace,
) -> list[dict]:
    """Run the Path C pipeline (hybrid retrieval — no extraction/chunking/embedding stages).

    Path C always reuses Path B's processed dir.  The extraction, chunking, and
    embedding skip flags are implicit (those stages don't exist for Path C).

    Args:
        cfg:         Loaded config dict.
        config_path: Path to config.yaml.
        args:        Parsed CLI arguments (skip flags).

    Returns:
        List of RAG result dicts (empty list on failure).
    """
    logger.info("========== PATH C PIPELINE START ==========")
    logger.info("--- [C] Stage 1: Extraction — SKIPPED (reuses Path B) ---")
    logger.info("--- [C] Stage 2: Chunking   — SKIPPED (reuses Path B) ---")
    logger.info("--- [C] Stage 3: Embedding  — SKIPPED (reuses Path B) ---")

    questions, ground_truths, company_ids = load_eval_questions(
        PROJECT_ROOT / cfg["paths"]["eval_questions"]
    )

    rag_results = run_generation_c(cfg, config_path, questions, company_ids)

    if not args.skip_eval and rag_results:
        logger.info("--- [C] Stage 5: RAGAS Evaluation ---")
        evaluate_pipeline(
            rag_results=rag_results,
            ground_truths=ground_truths,
            path_label="path_c",
            config_path=config_path,
        )

    logger.info("========== PATH C PIPELINE COMPLETE ==========")
    return rag_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="RAG Research Pipeline — Malaysian Financial Disclosures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --path a
  python run_pipeline.py --path b
  python run_pipeline.py --path c --skip-extraction --skip-chunking --skip-embedding --skip-eval
  python run_pipeline.py --path all
  python run_pipeline.py --path all --skip-extraction --skip-chunking
  python run_pipeline.py --path a --config configs/config.yaml
        """,
    )
    parser.add_argument(
        "--path",
        choices=["a", "b", "c", "all"],
        required=True,
        help="Which pipeline path to run: 'a' (baseline), 'b' (layout-aware), 'c' (hybrid+reranking), or 'all' (all three + comparison report).",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction stage (use existing processed files).",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip chunking stage.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding and FAISS index build stage.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip RAGAS evaluation stage.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
        help="Path to config.yaml (default: configs/config.yaml).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point — parse args and dispatch to the appropriate pipeline."""
    args        = parse_args()
    config_path = args.config.resolve()

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    setup_logging(cfg)

    logger.info("RAG Research Pipeline | path=%s | config=%s", args.path, config_path)

    if args.path in ("a", "all"):
        run_path("a", cfg, config_path, args)

    if args.path in ("b", "all"):
        run_path("b", cfg, config_path, args)

    if args.path in ("c", "all"):
        run_path_c(cfg, config_path, args)

    if args.path == "all" and not args.skip_eval:
        logger.info("========== GENERATING COMPARISON REPORT ==========")
        generate_report(config_path)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
