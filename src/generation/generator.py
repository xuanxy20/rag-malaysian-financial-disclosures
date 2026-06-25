"""
Generator — shared LLM generation module (identical for Path A and Path B).

Sends a retrieval-augmented prompt to a locally running Ollama server and
returns the model's text response. The prompt template is fixed and loaded
from config.yaml — it is identical for both paths, which is a core research
requirement.

Context is assembled by concatenating the text of the top-k retrieved chunks,
each labelled with its source document and page number for traceability.

All generation parameters (model, temperature, max_tokens, base_url, prompt
template) are read from configs/config.yaml under the 'generation' key.
"""

import logging
from pathlib import Path

import ollama
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
# Context assembly
# ---------------------------------------------------------------------------

def assemble_context(chunks: list[dict]) -> str:
    """Concatenate retrieved chunk texts into a single context string.

    Each chunk is prefixed with a source label (document name + page number)
    so the LLM can attribute statements and so RAGAS faithfulness scoring has
    clean context boundaries.

    Args:
        chunks: List of chunk dicts as returned by Retriever.retrieve().
                Each dict must have keys: 'text', 'doc_name', 'page'.

    Returns:
        A single string with all chunks joined by double newlines.
        Returns empty string if chunks list is empty.
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        doc_name = chunk.get("doc_name", "unknown")
        page     = chunk.get("page", "?")
        text     = chunk.get("text", "").strip()
        parts.append(f"[{i}] (Source: {doc_name}, Page {page})\n{text}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """Wraps the Ollama Python client for RAG-style text generation.

    Attributes:
        model:            Ollama model identifier, e.g. "llama3.1:8b".
        base_url:         Ollama server URL, e.g. "http://127.0.0.1:11434".
        temperature:      Sampling temperature (0.0 for deterministic output).
        max_tokens:       Maximum tokens in the generated response.
        prompt_template:  String template with {context} and {question} slots.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        prompt_template: str,
    ) -> None:
        """Initialise the Generator with Ollama connection parameters.

        Args:
            model:           Ollama model name.
            base_url:        URL of the local Ollama server.
            temperature:     Sampling temperature.
            max_tokens:      Maximum response tokens.
            prompt_template: Prompt string with {context} and {question}.
        """
        self.model           = model
        self.base_url        = base_url
        self.temperature     = temperature
        self.max_tokens      = max_tokens
        self.prompt_template = prompt_template

        self._client = ollama.Client(host=base_url)

    def _build_prompt(self, context: str, question: str) -> str:
        """Fill the fixed prompt template with context and question.

        Args:
            context:  Assembled context string from retrieved chunks.
            question: The evaluation question string.

        Returns:
            Filled prompt string ready for the LLM.
        """
        return self.prompt_template.format(context=context, question=question)

    def generate(self, question: str, chunks: list[dict]) -> dict:
        """Generate an answer for a question given retrieved chunks.

        Assembles context from chunks, fills the prompt template, calls
        the Ollama API, and returns a result dict.

        Args:
            question: Natural-language question string.
            chunks:   List of retrieved chunk dicts (from Retriever.retrieve()).

        Returns:
            Dict with keys:
                - 'answer'   (str): LLM-generated answer text.
                - 'context'  (str): Assembled context passed to the LLM.
                - 'question' (str): Original question (passed through).
                - 'chunks'   (list[dict]): Retrieved chunks with scores.
        """
        context = assemble_context(chunks)
        prompt  = self._build_prompt(context, question)

        try:
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            answer = response["response"].strip()
        except Exception as exc:
            logger.error("Ollama generation failed for question '%s…': %s", question[:60], exc)
            answer = ""

        logger.debug("Generated answer (%d chars) for: '%s…'", len(answer), question[:60])

        return {
            "answer":   answer,
            "context":  context,
            "question": question,
            "chunks":   chunks,
        }

    def generate_batch(self, questions: list[str], chunks_list: list[list[dict]]) -> list[dict]:
        """Generate answers for a batch of questions.

        Args:
            questions:   List of question strings.
            chunks_list: List of retrieved chunk lists, aligned with questions.

        Returns:
            List of result dicts (same order as input), one per question.

        Raises:
            ValueError: If questions and chunks_list lengths differ.
        """
        if len(questions) != len(chunks_list):
            raise ValueError(
                f"questions ({len(questions)}) and chunks_list ({len(chunks_list)}) "
                "must have the same length."
            )
        results = []
        for question, chunks in zip(questions, chunks_list):
            results.append(self.generate(question, chunks))
        return results

    def check_connection(self) -> bool:
        """Verify that the Ollama server is reachable and the model is available.

        Returns:
            True if the server responds and the configured model is listed,
            False otherwise.
        """
        try:
            models = self._client.list()
            available = [m["name"] for m in models.get("models", [])]
            if not any(self.model in name for name in available):
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s",
                    self.model, available,
                )
                return False
            logger.info("Ollama connection OK — model '%s' is available.", self.model)
            return True
        except Exception as exc:
            logger.error("Cannot reach Ollama at %s: %s", self.base_url, exc)
            return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_generator(config_path: Path | None = None) -> Generator:
    """Construct a Generator from configs/config.yaml.

    Args:
        config_path: Optional override for the config file location.
                     Defaults to <project_root>/configs/config.yaml.

    Returns:
        A configured Generator instance.
    """
    project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"

    cfg     = load_config(config_path)
    gen_cfg = cfg["generation"]

    return Generator(
        model=gen_cfg["model"],
        base_url=gen_cfg["base_url"],
        temperature=gen_cfg["temperature"],
        max_tokens=gen_cfg["max_tokens"],
        prompt_template=gen_cfg["prompt_template"],
    )


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    gen = build_generator()
    if gen.check_connection():
        dummy_chunks = [
            {"doc_name": "Test", "page": 1, "text": "Bursa Malaysia reported revenue of RM 500 million in FY2026."}
        ]
        result = gen.generate("What was Bursa Malaysia's revenue in FY2026?", dummy_chunks)
        logger.info("Answer: %s", result["answer"])
