"""
Catalog loader: reads shl_catalog.json, builds a FAISS semantic search index,
and provides search + lookup utilities for the agent.
"""
import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── test_type code mapping ────────────────────────────────────────────────────
KEYS_TO_CODE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Competencies": "C",
    "Simulations": "S",
    "Development & 360": "D",
    "Biodata & Situational Judgment": "B",
    "Assessment Exercises": "E",
}


def _derive_test_type(keys: list[str]) -> str:
    seen: list[str] = []
    for key in keys:
        code = KEYS_TO_CODE.get(key)
        if code and code not in seen:
            seen.append(code)
    return ",".join(seen) if seen else "K"


def _text_for_embedding(item: dict) -> str:
    """Rich text blob used to build the FAISS embedding."""
    parts: list[str] = [item.get("name", "")]
    desc = item.get("description", "")
    if desc:
        parts.append(desc)
    keys = item.get("keys", [])
    if keys:
        parts.append("Categories: " + ", ".join(keys))
    jl = item.get("job_levels", [])
    if jl:
        parts.append("Job levels: " + ", ".join(jl))
    return " ".join(parts)


def _format_for_context(item: dict) -> str:
    """Human-readable block injected into the LLM prompt."""
    name = item.get("name", "")
    url = item.get("link", "")
    test_type = item.get("_test_type", "K")
    keys = ", ".join(item.get("keys", []))
    duration = item.get("duration") or "—"
    jl = ", ".join(item.get("job_levels", [])) or "All levels"
    langs = item.get("languages", [])
    lang_str = ", ".join(langs[:5])
    if len(langs) > 5:
        lang_str += f" (+{len(langs) - 5} more)"
    description = item.get("description", "")
    return (
        f"Name: {name}\n"
        f"URL: {url}\n"
        f"Test Type Code: {test_type}  |  Keys: {keys}\n"
        f"Duration: {duration}  |  Job Levels: {jl}\n"
        f"Languages: {lang_str or 'English (USA)'}\n"
        f"Description: {description}"
    )


class Catalog:
    """Singleton that owns the catalog data and FAISS index."""

    def __init__(self) -> None:
        data_path = Path(__file__).parent.parent / "data" / "shl_catalog.json"
        logger.info("Loading catalog from %s", data_path)
        with open(data_path, "r", encoding="utf-8") as fh:
            raw: list[dict] = json.loads(fh.read(), strict=False)

        # Keep only items that scraped successfully
        self.items: list[dict] = [i for i in raw if i.get("status") == "ok"]
        logger.info("Catalog: %d usable items", len(self.items))

        # Pre-compute test_type codes and stash them on the item dict
        for item in self.items:
            item["_test_type"] = _derive_test_type(item.get("keys", []))

        # Lookup tables
        self.by_url: dict[str, dict] = {
            item["link"]: item for item in self.items if item.get("link")
        }
        self.valid_urls: set[str] = set(self.by_url.keys())
        # lower-cased name → item  (for comparison queries)
        self.by_name: dict[str, dict] = {
            item["name"].lower(): item for item in self.items if item.get("name")
        }

        # ── Build FAISS index ──────────────────────────────────────────────
        logger.info("Loading sentence-transformer model …")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

        texts = [_text_for_embedding(i) for i in self.items]
        logger.info("Encoding %d items …", len(texts))
        embeddings: np.ndarray = self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=64,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        # Inner-product on L2-normalised vectors == cosine similarity
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        logger.info("FAISS index ready: %d vectors, dim=%d", self._index.ntotal, dim)

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """Return up to top_k catalog items most similar to query."""
        q_vec = self._model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        _, indices = self._index.search(q_vec, min(top_k, len(self.items)))
        return [self.items[i] for i in indices[0] if 0 <= i < len(self.items)]

    def lookup_by_name(self, name: str) -> dict | None:
        """Fuzzy name lookup: exact → prefix → substring."""
        key = name.lower().strip()
        if key in self.by_name:
            return self.by_name[key]
        for stored_key, item in self.by_name.items():
            if stored_key.startswith(key) or key.startswith(stored_key):
                return item
        for stored_key, item in self.by_name.items():
            if key in stored_key or stored_key in key:
                return item
        return None

    def is_valid_url(self, url: str) -> bool:
        return url in self.valid_urls

    def format_context(self, items: list[dict]) -> str:
        return "\n\n---\n\n".join(_format_for_context(i) for i in items)


# Module-level singleton — loaded once at process start
catalog = Catalog()
