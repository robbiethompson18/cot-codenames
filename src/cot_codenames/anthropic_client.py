"""Direct Anthropic Messages API backend for `client.chat` (Claude models never go through OpenRouter: Robbie has Anthropic
credits). The game speaks OpenAI-format messages, so this converts them on the way in and the reply on the way out.

Each assistant message also keeps its native content blocks under "anthropic_content", and those are what we send back:
thinking blocks must be replayed unchanged (their signatures are checked), so they can't be rebuilt from the OpenAI fields.

Anthropic never returns the raw chain of thought. With display "summarized", message["reasoning"] is a summary of it.
"""

import json
import time

import anthropic
from anthropic.types import MessageParam, OutputConfigParam, ThinkingConfigParam, ToolParam

# $/MTok (input, output), Anthropic first-party rates. No prompt caching, so no cache pricing needed.
PRICES = {"claude-sonnet-5-5": (2, 10), "claude-opus-5-5": (4, 20), "claude-fable-5-1": (10, 50)}

# The SDK retries 408/409/429/5xx and connection errors with backoff.
_client = anthropic.Anthropic(max_retries=8, timeout=900)


def to_anthropic(messages: list[dict]) -> tuple[str, list[MessageParam]]:
    """OpenAI-format messages -> (system, Anthropic messages). Tool results and user text become blocks of one user turn."""
    system, out = "", []  # out holds plain dicts shaped like MessageParam
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "assistant":
            out.append({"role": "assistant", "content": m["anthropic_content"]})
        else:
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            else:
                block = {"type": "text", "text": m["content"]}
            if out and out[-1]["role"] == "user":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return system, out


def to_openai(blocks: list[dict]) -> dict:
    """Anthropic content blocks -> OpenAI-format assistant message (plus the blocks themselves, for replay)."""
    text = "".join(b["text"] for b in blocks if b["type"] == "text")
    reasoning = "\n\n".join(b["thinking"] for b in blocks if b["type"] == "thinking" and b["thinking"])
    tool_calls = [
        {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
        for b in blocks
        if b["type"] == "tool_use"
    ]
    return {
        "role": "assistant",
        "content": text or None,
        "reasoning": reasoning or None,
        "tool_calls": tool_calls or None,
        "anthropic_content": blocks,
    }


def chat(model: str, messages: list[dict], tools: list[dict], condition: dict) -> dict:
    """Same contract as client.chat: returns {"id", "model", "message", "finish_reason", "usage", "provider", "latency_s",
    "failed_attempts"} and raises RuntimeError when the API gives up."""
    # Opus 5.5 and Fable 5.1 can't turn thinking off. Sonnet 5.5 can ({"type": "between_tools"}), but the `visible`
    # prompt then gets refused by the reasoning_extraction classifier, so all three are THINKING_MANDATORY.
    if not condition["thinking"]:
        raise ValueError(f"{model} can't run with thinking off")
    thinking: ThinkingConfigParam = {"type": "adaptive", "display": "summarized"}
    system, msgs = to_anthropic(messages)
    functions = [t["function"] for t in tools]
    anthropic_tools: list[ToolParam] = [
        {"name": f["name"], "description": f.get("description", ""), "input_schema": f["parameters"]} for f in functions
    ]
    output_config: OutputConfigParam | anthropic.Omit = {"effort": condition["effort"]} if condition.get("effort") else anthropic.omit
    t0 = time.time()
    try:
        with _client.messages.stream(  # the SDK requires streaming at these max_tokens
            model=model,
            system=system,
            messages=msgs,
            tools=anthropic_tools,
            tool_choice={"type": "auto"},  # forced tool use ("any"/"tool") is a 400 on these models
            thinking=thinking,
            output_config=output_config,
            max_tokens=128000 if condition.get("effort") == "high" else 65536,  # same caps as the OpenRouter path
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.APIError as e:
        raise RuntimeError(f"{model}: {e}") from e
    # A safety-classifier refusal has no content. Fail the call rather than nudge the model, and no fallback model: it
    # would silently put another model's answer in this model's game.
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"{model}: refusal ({msg.stop_details.category if msg.stop_details else None})")
    u = msg.usage
    p_in, p_out = PRICES[model]
    usage = {
        "prompt_tokens": u.input_tokens,
        "completion_tokens": u.output_tokens,
        "completion_tokens_details": {"reasoning_tokens": u.output_tokens_details.thinking_tokens if u.output_tokens_details else None},
        "cost": (u.input_tokens * p_in + u.output_tokens * p_out) / 1e6,
        "anthropic": u.to_dict(),
    }
    return {
        "id": msg.id,
        "model": msg.model,
        "message": to_openai([b.to_dict() for b in msg.content]),
        "finish_reason": msg.stop_reason,
        "usage": usage,
        "provider": "anthropic-api",
        "latency_s": round(time.time() - t0, 1),
        "failed_attempts": [],
    }
