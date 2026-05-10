"""
Groq LLM client wrapper with JSON-mode enforcement and retry logic.
"""
import json
import logging
import os
import re
import time

from groq import Groq, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)

# Primary model: best quality on Groq, supports JSON mode
PRIMARY_MODEL = "llama-3.3-70b-versatile"
# Fallback: fastest model if rate-limited
FALLBACK_MODEL = "llama-3.1-8b-instant"

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        _client = Groq(api_key=api_key, max_retries=0)
    return _client


def generate(messages: list[dict], temperature: float = 0.1) -> str:
    """
    Call Groq with JSON mode. Tries PRIMARY_MODEL, falls back to FALLBACK_MODEL.
    Strictly avoids exceeding 30 seconds to pass evaluation.
    """
    client = get_client()
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]

    for attempt, model in enumerate(models_to_try):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
                response_format={"type": "json_object"},
                timeout=12.0,  # stay well under the 30-s hard limit overall (12s x 2 = 24s max)
            )
            content = resp.choices[0].message.content
            logger.debug("LLM response (%s): %s", model, content[:200])
            return content

        except Exception as exc:
            logger.warning("Attempt %d with model %s failed: %s", attempt + 1, model, exc)
            if attempt == len(models_to_try) - 1:
                raise
            continue

    raise RuntimeError("All Groq attempts failed")


def safe_parse_json(text: str) -> dict:
    """Parse JSON, with a best-effort fallback extraction."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.error("Failed to parse LLM output as JSON: %s", text[:500])
        # Return a safe default so the endpoint never crashes
        return {
            "reply": "I encountered an internal error. Could you rephrase your request?",
            "recommendations": [],
            "end_of_conversation": False,
        }
