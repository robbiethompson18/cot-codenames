"""Thin OpenRouter chat client. Returns the raw assistant message (incl. `reasoning` / `reasoning_details`) plus usage."""

import os
import time

import httpx

# short name -> (OpenRouter slug, pinned provider), no fallbacks. All open weights (hugging_face_id on OpenRouter).
# First-party where possible. DeepSeek's own endpoint is blocked by our account's no-training privacy setting, so
# Together (DeepInfra caps its output at 16k tokens, which truncated reasoning). Qwen3.8-27B on Alibaba drops
# prior-turn reasoning, so DeepInfra. See docs/stage-0.md.
MODELS = {
    "kimi-k3": ("moonshotai/kimi-k3", "moonshotai"),
    "glm-5.3": ("z-ai/glm-5.3", "z-ai"),
    "qwen3.8-2.4t": ("qwen/qwen3.8-2.4t-a95b", "alibaba"),
    "deepseek-v4-pro": ("deepseek/deepseek-v4-pro-0813", "together"),
    "qwen3.8-27b": ("qwen/qwen3.8-27b", "deepinfra/bf16"),  # cheap dev-loop model
    "deepseek-v4.1-flash": ("deepseek/deepseek-v4.1-flash", "deepinfra/fp8"),
}
# OpenRouter rejects reasoning={"enabled": False} for these on every provider we tried ("Reasoning is mandatory").
THINKING_MANDATORY = {"glm-5.3", "qwen3.8-2.4t"}
# What we run and show from stage 1 on: the models with a no-CoT arm (so not THINKING_MANDATORY), minus the dev model.
DEFAULT_MODELS = ["kimi-k3", "deepseek-v4-pro", "deepseek-v4.1-flash"]

URL = "https://openrouter.ai/api/v1/chat/completions"
# httpx defaults to 100 pooled connections, which would silently queue (and time out) calls above that.
_http = httpx.Client(timeout=900, limits=httpx.Limits(max_connections=500, max_keepalive_connections=500))
RETRYABLE = {408, 429, 500, 502, 503, 504}


def tool_choice(thinking: bool) -> str:
    # "required" + thinking is rejected by several pinned endpoints, so CoT uses "auto". Without CoT, "auto" let
    # DeepSeek V4.1 Flash reason in its visible reply (stage 1), so force the tool call there.
    return "auto" if thinking else "required"


def chat(model: str, messages: list[dict], tools: list[dict], thinking: bool) -> dict:
    """One completion. Returns {"message": assistant msg dict, "usage": ..., "provider": ..., "latency_s": ...,
    "failed_attempts": [error strings of retried attempts]}."""
    slug, provider = MODELS[model]
    body = {
        "model": slug,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice(thinking),
        "reasoning": {"enabled": thinking},
        "max_tokens": 65536,  # some endpoints default lower (DeepInfra DeepSeek: 16384, which truncated reasoning)
        "provider": {"order": [provider], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    err, failed = "", []
    for attempt in range(10):
        t0 = time.time()
        try:
            resp = _http.post(URL, json=body, headers=headers)
        except httpx.TransportError as e:
            err = repr(e)
        else:
            try:
                data = resp.json()
            except ValueError:  # non-JSON or truncated body: treat like any other bad response
                data = {}
            if resp.status_code == 200 and data.get("choices"):
                return {
                    "id": data.get("id"),  # OpenRouter generation id: /api/v1/generation?id= gives native tokens + provider
                    "model": data.get("model"),  # exact served model version
                    "message": data["choices"][0]["message"],
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "usage": data.get("usage"),
                    "provider": data.get("provider"),
                    "latency_s": round(time.time() - t0, 1),
                    "failed_attempts": failed,
                }
            err = f"{resp.status_code} {resp.text[:500]}"
            if resp.status_code not in RETRYABLE and resp.status_code != 200:  # 400 "reasoning is mandatory", 402, 404 routing
                break
        failed.append(err[:300])
        print(f"[client] {model} attempt {attempt}: {err}", flush=True)
        time.sleep(min(2**attempt, 60))  # ~6 min total: upstream 429s from shared provider pools can last minutes
    raise RuntimeError(f"{model}: gave up after retries: {err}")
