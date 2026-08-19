"""
llm.py — Gemini interaction using the official google-genai SDK.

Responsibility:
    * Talk to the Gemini API to generate answers.
    * Map API errors (missing key, rate limits, quota, network) into
      friendly, human-readable messages.
"""

from __future__ import annotations

import logging

from . import config

logger = logging.getLogger(__name__)

MISSING_KEY_MESSAGE = (
    "The Gemini API key is missing. Create a .env file from .env.example "
    "and set GEMINI_API_KEY (see README.md)."
)


class LLMError(Exception):
    """Raised when the LLM cannot generate an answer (friendly message for users)."""


class GeminiClient:
    """Thin wrapper around Google's Gemini API."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = (api_key if api_key is not None else config.GEMINI_API_KEY).strip()
        self.model = model or config.GEMINI_MODEL
        self._client = None

    def _get_client(self):
        """Create the SDK client lazily (only if a key is present)."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                logger.exception("Failed to initialise the Gemini SDK client")
                raise LLMError("The Gemini client could not be initialised.") from exc
        return self._client

    def is_configured(self) -> bool:
        """True when an API key is available."""
        return bool(self.api_key)

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send the prompt to Gemini and return the generated answer text.

        Raises LLMError with a friendly message on any failure.
        """
        if not self.is_configured():
            logger.warning("Gemini call attempted without an API key")
            raise LLMError(MISSING_KEY_MESSAGE)

        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config={
                    "system_instruction": system_prompt,
                    "temperature": config.TEMPERATURE,
                    "max_output_tokens": config.MAX_OUTPUT_TOKENS,
                },
            )
            text = (response.text or "").strip()
            if not text:
                raise LLMError("Gemini returned an empty response. Please try again.")
            return text
        except LLMError:
            raise
        except Exception as exc:
            raise self._friendly_error(exc) from exc

    def _friendly_error(self, exc: Exception) -> LLMError:
        """Map an SDK exception to a friendly, human-readable LLMError."""
        message = str(exc).lower()
        status = getattr(exc, "code", None)
        if status is None:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)

        # 5xx — the service is temporarily overloaded or down (not our fault).
        if status is not None and 500 <= int(status) < 600:
            logger.warning("Gemini server error (5xx): %s", exc)
            return LLMError(
                "Gemini is temporarily overloaded (high demand). "
                "Please wait a moment and try again."
            )
        # 429 / quota / rate-limit errors
        if status == 429 or "429" in message or "quota" in message or "rate limit" in message or "resource_exhausted" in message:
            logger.warning("Gemini rate limit / quota error: %s", exc)
            return LLMError(
                "Gemini rate limit or quota reached. Wait a moment and try again."
            )
        # Authentication problems (401/403 or key-related messages)
        if status in (401, 403) or "api key" in message or "permission" in message or "unauthorized" in message or "invalid key" in message:
            logger.warning("Gemini authentication error: %s", exc)
            return LLMError(
                "The Gemini API key seems invalid or unauthorised. Check GEMINI_API_KEY in .env."
            )
        # Model not found (404 or explicit message)
        if status == 404 or ("not found" in message and "model" in message):
            logger.warning("Gemini model error: %s", exc)
            return LLMError(
                f"The Gemini model '{self.model}' is not available. "
                "Update GEMINI_MODEL in .env to a current model name."
            )
        logger.exception("Gemini API call failed")
        return LLMError(
            "Could not reach the Gemini API. Check your internet connection and try again."
        )
