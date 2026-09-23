from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Any


DEFAULT_BASE_URLS = ["https://api.openai.com/v1"]


@dataclass
class APIClientConfig:
    api_key: str
    base_urls: list[str] = field(default_factory=lambda: list(DEFAULT_BASE_URLS))
    model: str = "glm-5.1"
    temperature: float = 0.2
    timeout: float = 120.0
    max_tokens: int = 2048
    retries_per_url: int = 1
    chat_completions_path: str = "/v1/chat/completions"
    thinking_disabled: bool = False
    user_agent: str = "CARVE/1.0"

    @classmethod
    def from_env(cls) -> "APIClientConfig":
        key = os.environ.get("CARVE_API_KEY")
        if not key:
            raise RuntimeError("CARVE_API_KEY is required for API model calls")
        base_urls = os.environ.get("CARVE_BASE_URLS")
        urls = [u.strip().rstrip("/") for u in base_urls.split(",") if u.strip()] if base_urls else list(DEFAULT_BASE_URLS)
        return cls(
            api_key=key,
            base_urls=urls,
            model=os.environ.get("CARVE_MODEL", "glm-5.1"),
            temperature=float(os.environ.get("CARVE_TEMPERATURE", "0.2")),
            timeout=float(os.environ.get("CARVE_API_TIMEOUT", "120")),
            max_tokens=int(os.environ.get("CARVE_MAX_TOKENS", "2048")),
            retries_per_url=int(os.environ.get("CARVE_RETRIES_PER_URL", "1")),
            chat_completions_path=os.environ.get("CARVE_CHAT_COMPLETIONS_PATH", "/v1/chat/completions"),
            thinking_disabled=os.environ.get("CARVE_THINKING_DISABLED", "0") == "1",
            user_agent=os.environ.get("CARVE_USER_AGENT", "CARVE/1.0"),
        )


class OpenAICompatibleClient:
    """OpenAI-compatible Chat Completions adapter with base URL failover."""

    def __init__(
        self,
        config: APIClientConfig,
        opener: Callable[[urllib.request.Request, float], Any] | None = None,
    ):
        self.config = config
        self.opener = opener or urllib.request.urlopen
        self._cursor = 0
        self._last_completion_telemetry: dict[str, Any] = {}

    def complete(self, role: str, prompt: str, seed: int) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": f"You are the {role} agent in a CARVE multi-agent orchestration trace."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": seed,
        }
        if self.config.thinking_disabled:
            payload["thinking"] = {"type": "disabled"}
        data = json.dumps(payload).encode("utf-8")
        errors: list[str] = []
        urls = self._ordered_urls()
        request_attempts = 0
        started_at = time.perf_counter()
        path = "/" + self.config.chat_completions_path.lstrip("/")
        for base_url in urls:
            endpoint = f"{base_url.rstrip(chr(47))}{path}"
            for attempt in range(self.config.retries_per_url):
                request_attempts += 1
                request = urllib.request.Request(
                    endpoint,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.config.api_key}",
                        "User-Agent": self.config.user_agent,
                    },
                    method="POST",
                )
                try:
                    with self.opener(request, timeout=self.config.timeout) as response:
                        raw = response.read().decode("utf-8")
                    parsed = json.loads(raw)
                    usage = parsed.get("usage") or {}
                    message = parsed["choices"][0]["message"]
                    content = message.get("content") or message.get("reasoning_content")
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("empty message content")
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    provider_usage = isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
                    self._last_completion_telemetry = {
                        "api_calls": 1,
                        "api_request_attempts": request_attempts,
                        "input_tokens": int(prompt_tokens) if provider_usage else None,
                        "output_tokens": int(completion_tokens) if provider_usage else None,
                        "token_source": "provider_usage" if provider_usage else "unavailable",
                        "wall_clock_latency_ms": (time.perf_counter() - started_at) * 1000.0,
                        "endpoint": endpoint,
                    }
                    return content
                except (
                    TimeoutError,
                    socket.timeout,
                    urllib.error.URLError,
                    urllib.error.HTTPError,
                    KeyError,
                    IndexError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    errors.append(f"{endpoint} attempt {attempt + 1}: {exc}")
                    time.sleep(min(2.0, 0.25 * (attempt + 1)))
        raise RuntimeError("all CARVE API endpoints failed: " + " | ".join(errors[-4:]))

    def last_completion_telemetry(self) -> dict[str, Any]:
        return dict(self._last_completion_telemetry)

    def _ordered_urls(self) -> list[str]:
        urls = self.config.base_urls
        if not urls:
            raise RuntimeError("at least one base URL is required")
        start = self._cursor % len(urls)
        self._cursor += 1
        return urls[start:] + urls[:start]
