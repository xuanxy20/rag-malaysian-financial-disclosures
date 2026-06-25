"""
test_pipeline.py — Interactive diagnostic script for the RAG research pipeline.

Runs each pipeline component individually with clear pass/fail output so you
can verify every stage is working before committing to a full run.

Usage:
    python test_pipeline.py              # run all tests
    python test_pipeline.py --stage 4   # run only stage 4 (generation)

Stages:
    1  Config loading
    2  Ollama connectivity
    3  Embedding (encode 2 sentences)
    4  Retrieval (query against existing Path A indexes)
    5  Generation (1 question end-to-end)
    6  RAGAS evaluation (1 question — the slowest stage)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── colour helpers (no extra deps) ──────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {CYAN}→{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def header(title):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")

# ── project root on path ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)   # suppress library noise during tests

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# ============================================================
# Stage 1 — Config
# ============================================================

def test_config() -> dict | None:
    header("Stage 1 — Config loading")
    try:
        import yaml
        with open(CONFIG_PATH) as fh:
            cfg = yaml.safe_load(fh)
        ok(f"config.yaml loaded ({len(cfg)} top-level keys)")
        info(f"  documents : {[d['name'] for d in cfg['documents']]}")
        info(f"  chunk size: {cfg['path_a']['chunking']['chunk_size']} tokens")
        info(f"  top_k     : {cfg['retrieval']['top_k']}")
        info(f"  model     : {cfg['generation']['model']}")
        return cfg
    except Exception as e:
        fail(f"Config load failed: {e}")
        return None


# ============================================================
# Stage 2 — Ollama
# ============================================================

def test_ollama(cfg: dict) -> bool:
    header("Stage 2 — Ollama connectivity")
    try:
        import ollama
        model    = cfg["generation"]["model"]
        base_url = cfg["generation"]["base_url"]
        client   = ollama.Client(host=base_url)

        info(f"Connecting to {base_url} …")
        t0 = time.time()
        models = client.list()
        elapsed = time.time() - t0

        available = [m["name"] for m in models.get("models", [])]
        ok(f"Ollama reachable ({elapsed:.2f}s)")
        info(f"  Available models: {available}")

        if any(model in name for name in available):
            ok(f"Target model '{model}' is available")
            return True
        else:
            fail(f"Model '{model}' NOT found. Run: ollama pull {model}")
            return False
    except Exception as e:
        fail(f"Cannot reach Ollama: {e}")
        info("  Make sure Ollama is running: ollama serve")
        return False


# ============================================================
# Stage 3 — Embedding
# ============================================================

def test_embedding(cfg: dict) -> object | None:
    header("Stage 3 — Embedding model")
    try:
        from src.embeddings.embedder import build_embedder
        import numpy as np

        info("Loading sentence-transformers model …")
        t0 = time.time()
        embedder = build_embedder(CONFIG_PATH)

        sentences = [
            "What is the total revenue of Bursa Malaysia?",
            "Describe the board of directors composition.",
        ]
        vectors = embedder.encode(sentences, desc="test")
        elapsed = time.time() - t0

        ok(f"Model loaded and encoded {len(sentences)} sentences ({elapsed:.2f}s)")
        info(f"  Output shape : {vectors.shape}")
        info(f"  Vector norms : {[round(float(np.linalg.norm(v)), 6) for v in vectors]}")

        norm = float(np.linalg.norm(vectors[0]))
        if abs(norm - 1.0) < 0.001:
            ok("Vectors are L2-normalised ✓")
        else:
            warn(f"Vectors may not be normalised (norm={norm:.4f})")

        return embedder
    except Exception as e:
        fail(f"Embedding test failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================
# Stage 4 — Retrieval
# ============================================================

def test_retrieval(cfg: dict, embedder) -> list | None:
    header("Stage 4 — Retrieval (FAISS search)")

    path_a_dir = PROJECT_ROOT / cfg["paths"]["processed_path_a"]
    index_files = list(path_a_dir.glob("*.faiss"))

    if not index_files:
        warn("No FAISS indexes found in data/processed/path_a/")
        info("Run extraction + chunking + embedding first:")
        info("  python run_pipeline.py --path a --skip-eval")
        return None

    try:
        from src.retrieval.retriever import Retriever, load_stores

        info(f"Found {len(index_files)} index file(s): {[f.stem for f in index_files]}")
        stores = load_stores(path_a_dir, cfg["documents"])

        if not stores:
            fail("No stores loaded — check index files.")
            return None

        retriever = Retriever.from_config(stores, embedder, CONFIG_PATH)

        query = "What is the total revenue?"
        info(f"Query: '{query}'")
        t0 = time.time()
        results = retriever.retrieve(query)
        elapsed = time.time() - t0

        ok(f"Retrieved {len(results)} chunks ({elapsed:.3f}s)")
        for i, r in enumerate(results, 1):
            print(f"\n    [{i}] score={r['score']:.4f} | {r['doc_name']} p.{r['page']}")
            print(f"        {r['text'][:120].replace(chr(10), ' ')} …")

        return results
    except Exception as e:
        fail(f"Retrieval test failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================
# Stage 5 — Generation
# ============================================================

def test_generation(cfg: dict, embedder, chunks: list) -> dict | None:
    header("Stage 5 — Generation (Ollama LLM)")
    try:
        from src.generation.generator import build_generator

        generator = build_generator(CONFIG_PATH)
        question  = "What is the total revenue?"

        info(f"Question : {question}")
        info(f"Context  : {len(chunks)} chunks passed to LLM")
        info("Calling Ollama (may take 10–30s) …")

        t0 = time.time()
        result = generator.generate(question, chunks)
        elapsed = time.time() - t0

        answer = result.get("answer", "")
        if answer:
            ok(f"Answer generated ({elapsed:.1f}s, {len(answer)} chars)")
            print(f"\n    {CYAN}Answer:{RESET}")
            for line in answer.split("\n"):
                print(f"    {line}")
        else:
            fail("Empty answer returned — check Ollama logs")
            return None

        return result
    except Exception as e:
        fail(f"Generation test failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================
# Stage 6 — RAGAS (single question)
# ============================================================

def test_ragas(cfg: dict, result: dict) -> None:
    header("Stage 6 — RAGAS evaluation (1 question)")

    info("This stage calls the LLM judge multiple times — expect 30–120s")
    info("Watch for per-metric scoring messages below …\n")

    # Temporarily set logging to INFO so RAGAS progress is visible
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("src.evaluation").setLevel(logging.INFO)

    try:
        from src.evaluation.ragas_evaluator import build_ragas_dataset, run_evaluation

        ground_truth = "Test ground truth: revenue information from the annual report."
        rag_results  = [result]
        ground_truths = [ground_truth]

        info("Building RAGAS dataset …")
        dataset = build_ragas_dataset(rag_results, ground_truths)
        ok(f"Dataset built: {dataset.num_rows} row(s), columns={dataset.column_names}")

        info("Running RAGAS metrics (faithfulness + answer_relevancy only for speed) …")
        t0 = time.time()
        scores_df = run_evaluation(
            rag_results=rag_results,
            ground_truths=ground_truths,
            metrics=["faithfulness", "answer_relevancy"],
            judge_model=cfg["evaluation"]["judge_model"],
            judge_base_url=cfg["evaluation"]["judge_base_url"],
            emb_model_name=cfg["embeddings"]["model_name"],
            emb_device=cfg["embeddings"]["device"],
        )
        elapsed = time.time() - t0

        ok(f"RAGAS evaluation complete ({elapsed:.1f}s)")
        print(f"\n    {CYAN}Scores:{RESET}")
        print(scores_df.to_string(index=False))

    except Exception as e:
        fail(f"RAGAS test failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        logging.getLogger().setLevel(logging.WARNING)


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RAG pipeline component tests")
    parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="Run only a specific stage (default: all stages)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_all = args.stage is None

    print(f"\n{BOLD}RAG Pipeline Diagnostic{RESET} — {PROJECT_ROOT}")

    cfg = test_config() if (run_all or args.stage == 1) else None
    if cfg is None and run_all:
        fail("Cannot continue without config."); return

    if run_all or args.stage == 2:
        if cfg is None:
            import yaml
            with open(CONFIG_PATH) as fh: cfg = yaml.safe_load(fh)
        test_ollama(cfg)

    embedder = None
    if run_all or args.stage in (3, 4, 5, 6):
        if cfg is None:
            import yaml
            with open(CONFIG_PATH) as fh: cfg = yaml.safe_load(fh)
        embedder = test_embedding(cfg)

    chunks = None
    if run_all or args.stage in (4, 5, 6):
        if embedder:
            chunks = test_retrieval(cfg, embedder)

    result = None
    if run_all or args.stage in (5, 6):
        if embedder and chunks:
            result = test_generation(cfg, embedder, chunks)

    if run_all or args.stage == 6:
        if result:
            test_ragas(cfg, result)
        else:
            warn("Skipping RAGAS test — generation result not available.")

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  Diagnostic complete.{RESET}\n")


if __name__ == "__main__":
    main()
