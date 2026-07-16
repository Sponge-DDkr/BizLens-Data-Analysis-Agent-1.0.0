"""DeepSeek API Client for BizLens.

Supports per-call model override — code Agents use deepseek-reasoner (R1)
while Insight Agent uses deepseek-chat (V3) for better Chinese writing.
"""

import json
import logging
from typing import Any, Optional

import httpx

from backend.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger(__name__)


class LLMClient:
    """Encapsulates DeepSeek API chat completion calls."""

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = DEEPSEEK_MODEL,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Max tokens in the response.
            response_format: Optional {"type": "json_object"} for structured output.
            model: Optional model override (e.g. "deepseek-reasoner" for code gen).
                   Defaults to self.model (from DEEPSEEK_MODEL env).

        Returns:
            The model's text response.
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            # Capture response body for debugging
            resp_text = ""
            try:
                resp_text = e.response.text[:500]
            except Exception:
                resp_text = "(response body unavailable)"
            logger.error(
                f"DeepSeek API HTTP {e.response.status_code}: {resp_text}"
            )
            raise
        except httpx.HTTPError as e:
            logger.error(f"DeepSeek API request failed: {type(e).__name__}: {e}")
            raise

    async def chat_structured(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Chat with JSON mode for structured output parsing.

        Returns:
            Parsed JSON dict from the response.
        """
        text = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            model=model,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {text[:200]}...")
            # Retry without JSON mode as fallback
            text = await self.chat(
                messages=messages + [
                    {"role": "user", "content": "Please output valid JSON only. No markdown, no explanation."}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            )
            return json.loads(text)

    async def close(self):
        await self._client.aclose()


# Module-level singleton factory
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
