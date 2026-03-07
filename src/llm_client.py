"""
Multi-provider LLM client.

Priority (auto-detected from env vars):
  1. Gemini  (GOOGLE_API_KEY)     — ideal in local env
  2. Groq    (GROQ_API_KEY)       — free, fast, works everywhere
  3. OpenRouter (OPENROUTER_API_KEY) — routes to many models including Gemini
  4. Anthropic  (ANTHROPIC_API_KEY)  — fallback

Set LLM_PROVIDER=gemini|groq|openrouter|anthropic to force a provider.
"""

import os
from typing import Generator
from loguru import logger


# ── Provider detection ────────────────────────────────────────────────────────

def _detect_provider() -> str:
    forced = os.getenv("LLM_PROVIDER", "").lower()
    if forced in ("gemini", "groq", "openrouter", "anthropic"):
        return forced

    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "gemini"


PROVIDER = _detect_provider()
logger.info(f"LLM provider: {PROVIDER}")


# ── Gemini client ─────────────────────────────────────────────────────────────

class GeminiClient:
    def __init__(self):
        from google import genai
        from google.genai import types as gtypes
        self._gtypes = gtypes
        api_key = os.getenv("GOOGLE_API_KEY", "")
        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.default_history: list = []

    def call(self, prompt: str, system: str, max_tokens: int = 8192) -> str:
        from google.genai import types as t
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=t.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=0.7,
            ),
        )
        return response.text or ""

    def call_with_history(self, messages: list[dict], max_tokens: int = 2048) -> str:
        from google.genai import types as t
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        contents = [
            t.Content(role=m["role"], parts=[t.Part(text=m["content"])])
            for m in messages if m["role"] != "system"
        ]
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=t.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=0.7,
            ),
        )
        return response.text or ""

    def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        from google.genai import types as t

        contents = [
            t.Content(role=m["role"], parts=[t.Part(text=m["content"])])
            for m in messages
        ]
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=t.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=0.7,
            ),
        ):
            yield chunk.text or ""

    def make_message(self, role: str, content: str) -> dict:
        return {"role": role, "content": content}

    def assistant_role(self) -> str:
        return "model"


# ── OpenAI-compatible client (Groq / OpenRouter) ──────────────────────────────

class OpenAICompatClient:
    """Works for both Groq and OpenRouter (both are OpenAI-compatible)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        try:
            from openai import OpenAI
            # max_retries=0: disables silent auto-retries that cause 2-3 min hangs
            # on Groq rate limit (429). We handle retries explicitly in the caller.
            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=45.0, max_retries=0)
        except ImportError:
            import httpx, json as _json

            class _FallbackClient:
                def __init__(self, base_url, api_key):
                    self._base_url = base_url.rstrip("/")
                    self._api_key = api_key
                    self._headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }

                def chat_complete(self, body):
                    with httpx.Client(timeout=120) as hc:
                        r = hc.post(
                            f"{self._base_url}/chat/completions",
                            headers=self._headers,
                            json=body,
                        )
                        r.raise_for_status()
                        return r.json()["choices"][0]["message"]["content"]

            self.client = _FallbackClient(base_url, api_key)

        self.model = model

    def call(self, prompt: str, system: str, max_tokens: int = 8192) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.call_with_history(messages, max_tokens)

    def call_with_history(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """Send a full messages list (with history) and return response."""
        if hasattr(self.client, "chat"):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content or ""
        else:
            return self.client.chat_complete(
                {"model": self.model, "messages": messages, "max_tokens": max_tokens}
            )

    # Keep call_with_history as an alias for the fallback path above (handled inside call_with_history)

    def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        if hasattr(self.client, "chat"):
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=0.7,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
            # Fallback non-streaming
            result = self.client.chat_complete(
                {"model": self.model, "messages": full_messages, "max_tokens": max_tokens}
            )
            yield result

    def make_message(self, role: str, content: str) -> dict:
        return {"role": role, "content": content}

    def assistant_role(self) -> str:
        return "assistant"


# ── Anthropic client ──────────────────────────────────────────────────────────

class AnthropicClient:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    def call(self, prompt: str, system: str, max_tokens: int = 8192) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text or ""

    def call_with_history(self, messages: list[dict], max_tokens: int = 2048) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        history = [m for m in messages if m["role"] != "system"]
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=history,
        )
        return msg.content[0].text or ""

    def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as s:
            yield from s.text_stream

    def make_message(self, role: str, content: str) -> dict:
        return {"role": role, "content": content}

    def assistant_role(self) -> str:
        return "assistant"


# ── Factory ───────────────────────────────────────────────────────────────────

def _try_gemini():
    """Try to initialise Gemini; return client or raise."""
    c = GeminiClient()
    # Quick connectivity probe
    c.call("ping", "reply pong", max_tokens=10)
    return c


def build_llm_client():
    """
    Return the best available LLM client.
    Tries the configured provider; on failure automatically tries the others.
    """
    providers_order = [PROVIDER] + [
        p for p in ("gemini", "groq", "openrouter", "anthropic") if p != PROVIDER
    ]

    for provider in providers_order:
        try:
            client = _build(provider)
            if client:
                logger.info(f"Using LLM provider: {provider}")
                return client
        except Exception as e:
            logger.warning(f"Provider '{provider}' unavailable: {e}")

    raise RuntimeError(
        "No LLM provider available. Set GOOGLE_API_KEY, GROQ_API_KEY, "
        "OPENROUTER_API_KEY or ANTHROPIC_API_KEY in your .env file."
    )


def _build(provider: str):
    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            return None
        return _try_gemini()

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return None
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return OpenAICompatClient(
            base_url="https://api.groq.com/openai/v1",
            api_key=key,
            model=model,
        )

    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            return None
        model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
        return OpenAICompatClient(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            model=model,
        )

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            return None
        return AnthropicClient()

    return None
