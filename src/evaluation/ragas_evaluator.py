"""
RAGAS Evaluator — shared evaluation module (identical for Path A and Path B).

Runs RAGAS metrics against the output of a complete RAG pipeline run and saves
results to the path-specific results directory.

Metrics evaluated:
    - context_precision   : are the retrieved chunks relevant to the question?
    - context_recall      : does the retrieved context cover the ground truth?
    - faithfulness        : is the answer grounded in the retrieved context?
    - answer_relevancy    : is the answer relevant to the question?

RAGAS requires an LLM judge. This module reuses the same local Ollama model
configured in config.yaml under evaluation.judge_model, wrapped in a
LangChain-compatible interface that RAGAS accepts.

Input:
    A list of RAG result dicts (from Generator.generate_batch()), each with:
        - 'question'      (str)
        - 'answer'        (str)
        - 'context'       (str)   assembled context string
        - 'chunks'        (list[dict])  raw retrieved chunks
    Plus a list of ground-truth answer strings aligned with the questions.

Output:
    results/<path_a|path_b>/ragas_results.json
    results/<path_a|path_b>/ragas_results.csv
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml


from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Metric name → ragas metric object
_METRIC_MAP = {
    "context_precision": context_precision,
    "context_recall":    context_recall,
    "faithfulness":      faithfulness,
    "answer_relevancy":  answer_relevancy,
}


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
# Ollama LLM wrapper for RAGAS
# ---------------------------------------------------------------------------

def _build_ragas_llm(model: str, base_url: str):
    """Build a LangChain-compatible LLM wrapper for the RAGAS judge.

    RAGAS internally uses LangChain's LLM interface. We wrap the local Ollama
    model so no OpenAI key is required.

    Args:
        model:    Ollama model name, e.g. "llama3.1:8b".
        base_url: Ollama server URL, e.g. "http://127.0.0.1:11434".

    Returns:
        A LangChain ChatOllama instance configured for deterministic output.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Install langchain-ollama or langchain-community to use "
                "local Ollama as the RAGAS judge LLM.\n"
                "  pip install langchain-ollama"
            ) from exc

    return ChatOllama(model=model, base_url=base_url, temperature=0)


def _build_ragas_embeddings(model_name: str, device: str):
    """Build a LangChain-compatible embeddings wrapper for RAGAS.

    RAGAS answer_relevancy metric requires an embeddings model. We reuse
    the same sentence-transformers model as the pipeline to avoid loading
    a second model.

    Args:
        model_name: HuggingFace model identifier.
        device:     Torch device string.

    Returns:
        A LangChain HuggingFaceEmbeddings instance.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Install langchain-huggingface to use local embeddings in RAGAS.\n"
                "  pip install langchain-huggingface"
            ) from exc

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def build_ragas_dataset(
    rag_results: list[dict],
    ground_truths: list[str],
) -> Dataset:
    """Convert RAG pipeline outputs into a RAGAS-compatible HuggingFace Dataset.

    RAGAS expects a Dataset with columns:
        - question        (str)
        - answer          (str)
        - contexts        (list[str])  — one string per retrieved chunk
        - ground_truth    (str)

    The 'contexts' column is built from the raw chunk dicts in each result,
    NOT from the pre-assembled context string, so RAGAS can score individual
    chunks for context_precision and context_recall.

    Args:
        rag_results:   List of dicts from Generator.generate_batch().
        ground_truths: List of ground-truth answer strings, aligned with
                       rag_results.

    Returns:
        A HuggingFace Dataset ready for ragas.evaluate().

    Raises:
        ValueError: If rag_results and ground_truths have different lengths.
    """
    if len(rag_results) != len(ground_truths):
        raise ValueError(
            f"rag_results ({len(rag_results)}) and ground_truths "
            f"({len(ground_truths)}) must have the same length."
        )

    rows = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],
    }

    for result, gt in zip(rag_results, ground_truths):
        contexts = [c["text"] for c in result.get("chunks", []) if c.get("text", "").strip()]
        if not contexts:
            # Fallback: split assembled context string by double newline
            contexts = [
                p.strip()
                for p in result.get("context", "").split("\n\n")
                if p.strip()
            ] or [""]

        rows["question"].append(result["question"])
        rows["answer"].append(result.get("answer", ""))
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(gt)

    return Dataset.from_dict(rows)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    rag_results: list[dict],
    ground_truths: list[str],
    metrics: list[str],
    judge_model: str,
    judge_base_url: str,
    emb_model_name: str,
    emb_device: str,
) -> pd.DataFrame:
    """Run RAGAS evaluation and return a per-question scores DataFrame.

    Args:
        rag_results:    List of RAG result dicts from the generator.
        ground_truths:  Aligned ground-truth answers.
        metrics:        List of metric names from config (e.g. ["faithfulness"]).
        judge_model:    Ollama model name for the LLM judge.
        judge_base_url: Ollama server URL.
        emb_model_name: HuggingFace embeddings model for answer_relevancy.
        emb_device:     Torch device for the embeddings model.

    Returns:
        pd.DataFrame with one row per question and one column per metric,
        plus 'question' and 'answer' columns for traceability.
    """
    logger.info("Building RAGAS dataset (%d samples) …", len(rag_results))
    dataset = build_ragas_dataset(rag_results, ground_truths)

    # Resolve metric objects
    selected_metrics = []
    for name in metrics:
        if name in _METRIC_MAP:
            selected_metrics.append(_METRIC_MAP[name])
        else:
            logger.warning("Unknown metric '%s' — skipping.", name)

    if not selected_metrics:
        raise ValueError("No valid RAGAS metrics selected. Check config.yaml.")

    logger.info("Initialising RAGAS judge LLM (%s) …", judge_model)
    llm        = _build_ragas_llm(judge_model, judge_base_url)
    embeddings = _build_ragas_embeddings(emb_model_name, emb_device)

    logger.info(
        "Running RAGAS evaluation | metrics=%s …",
        [m.name for m in selected_metrics],
    )

    result = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=RunConfig(
            timeout=300,       # 5 min per call — llama3.1:8b on complex prompts can be slow
            max_workers=1,     # sequential: Ollama handles one request at a time
            max_retries=2,
            max_wait=60,
        ),
    )

    scores_df = result.to_pandas()

    # Ensure question and answer columns are present for traceability
    if "question" not in scores_df.columns:
        scores_df.insert(0, "question", [r["question"] for r in rag_results])
    if "answer" not in scores_df.columns:
        scores_df.insert(1, "answer", [r.get("answer", "") for r in rag_results])

    logger.info("RAGAS evaluation complete.")
    logger.info("Mean scores:\n%s", scores_df[[m.name for m in selected_metrics]].mean().to_string())

    return scores_df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(scores_df: pd.DataFrame, output_dir: Path, path_label: str) -> None:
    """Save RAGAS scores to CSV and JSON in the results directory.

    Args:
        scores_df:    DataFrame returned by run_evaluation().
        output_dir:   Destination directory (results/path_a/ or results/path_b/).
        path_label:   "path_a" or "path_b" — included in filenames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = output_dir / f"ragas_results_{path_label}.csv"
    json_path = output_dir / f"ragas_results_{path_label}.json"

    scores_df.to_csv(csv_path, index=False)
    logger.info("Saved CSV  → %s", csv_path)

    scores_df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    logger.info("Saved JSON → %s", json_path)


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------

def _enrich_with_metadata(scores_df: pd.DataFrame, config_path: Path) -> pd.DataFrame:
    """Left-join question metadata (layout_sensitivity, question_type, etc.) into scores.

    Reads eval_questions.json, matches on the 'question' string, and prepends
    question_id, layout_sensitivity, question_type, company_id, sector, doc_type
    columns so every downstream CSV has stratification keys ready.

    Args:
        scores_df:   DataFrame from run_evaluation() with a 'question' column.
        config_path: Path to configs/config.yaml (used to locate eval_questions.json).

    Returns:
        Enriched DataFrame with metadata columns prepended.
    """
    project_root  = config_path.resolve().parents[1]
    cfg           = load_config(config_path)
    questions_path = project_root / cfg["paths"]["eval_questions"]

    if not questions_path.exists():
        logger.warning("eval_questions.json not found at %s — skipping metadata enrichment.", questions_path)
        return scores_df

    with open(questions_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    meta_rows = [
        {
            "question":           q["question"],
            "question_id":        q["id"],
            "layout_sensitivity": q.get("layout_sensitivity", ""),
            "question_type":      q.get("question_type", ""),
            "company_id":         q.get("company_id", ""),
            "sector":             q.get("sector", ""),
            "doc_type":           q.get("doc_type", ""),
        }
        for q in data.get("questions", [])
    ]
    meta_df = pd.DataFrame(meta_rows)

    # Normalise whitespace before joining to avoid silent mismatches
    scores_df = scores_df.copy()
    scores_df["question"] = scores_df["question"].str.strip()
    meta_df["question"]   = meta_df["question"].str.strip()

    enriched = scores_df.merge(meta_df, on="question", how="left")

    front = ["question_id", "question", "answer", "layout_sensitivity",
             "question_type", "company_id", "sector", "doc_type"]
    rest  = [c for c in enriched.columns if c not in front]
    enriched = enriched[[c for c in front if c in enriched.columns] + rest]

    matched = int(enriched["question_id"].notna().sum())
    logger.info("Metadata enrichment: %d / %d questions matched.", matched, len(enriched))
    return enriched


# ---------------------------------------------------------------------------
# High-level runner (called by run_pipeline.py)
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    rag_results: list[dict],
    ground_truths: list[str],
    path_label: str,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """End-to-end RAGAS evaluation for one pipeline path.

    Reads all evaluation parameters from config.yaml, runs RAGAS, saves
    results, and returns the scores DataFrame.

    Args:
        rag_results:    List of RAG result dicts from Generator.generate_batch().
        ground_truths:  Aligned ground-truth answer strings.
        path_label:     "path_a" or "path_b" — controls output directory.
        config_path:    Optional config file path override.

    Returns:
        pd.DataFrame of per-question RAGAS scores.
    """
    project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    cfg     = load_config(config_path)
    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s — %(message)s"),
        datefmt=log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )

    eval_cfg   = cfg["evaluation"]
    emb_cfg    = cfg["embeddings"]
    _output_key_map = {
        "path_a": "results_path_a",
        "path_b": "results_path_b",
        "path_c": "results_path_c",
    }
    output_key = _output_key_map.get(path_label, "results_path_b")
    output_dir = project_root / cfg["paths"][output_key]

    scores_df = run_evaluation(
        rag_results=rag_results,
        ground_truths=ground_truths,
        metrics=eval_cfg["metrics"],
        judge_model=eval_cfg["judge_model"],
        judge_base_url=eval_cfg["judge_base_url"],
        emb_model_name=emb_cfg["model_name"],
        emb_device=emb_cfg["device"],
    )

    scores_df = _enrich_with_metadata(scores_df, config_path)
    save_results(scores_df, output_dir, path_label)
    return scores_df
