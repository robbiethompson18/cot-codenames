"""Tinker backend for the steganography game: the LoRA lives on Tinker's servers and this file only renders prompts,
submits requests and builds training data. All requests in a batch are submitted before any result is awaited.

Qwen: reasoning is off everywhere. Prompts are rendered with the cookbook's `*_disable_thinking` renderer for the model,
which ends the generation prompt with an empty `<think>\\n\\n</think>\\n\\n` block (what the HF template does for
enable_thinking=False).

gpt-oss (Harmony format): an assistant turn is a run of messages, each on a channel:
`<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|><|start|>assistant<|channel|>final<|message|>answer<|return|>`.
The model cannot switch reasoning off; the system prompt's "Reasoning: low|medium|high" line only sets how much it does.
So wherever we want no reasoning (a no-reasoning sender, the receiver, the monitor) the assistant turn is prefilled
straight into the final channel with FINAL_PREFILL, and the model's next token is the first token of its answer. A
reasoning sender gets no prefill and writes both channels itself.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import tinker
import torch
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from cot_codenames.stego import game
from cot_codenames.stego.game import Chat, Sample

# Appended to the renderer's generation prompt, which ends `<|start|>assistant` (tokens 200006, 173781), so the turn
# opens `<|start|>assistant<|channel|>final<|message|>`: the prefill is tokens 200005, 17196, 200008.
FINAL_PREFILL = "<|channel|>final<|message|>"
# The Harmony system prompt carries the date. Pinned, so a prompt's tokens do not change from one day to the next.
SYSTEM_DATE = "2026-10-08"
LOW_LABEL_MASS = 0.8


def is_gpt_oss(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def no_thinking_renderer(model: str, effort: str = "low") -> renderers.Renderer:
    """The renderer for prompts answered without reasoning. For gpt-oss that is the cookbook's `gpt_oss_<effort>_reasoning`
    renderer (the standard system prompt); TinkerBackend._tokens adds FINAL_PREFILL to skip the reasoning."""
    if is_gpt_oss(model):
        return GptOssRenderer(get_tokenizer(model), use_system_prompt=True, reasoning_effort=effort, current_date=SYSTEM_DATE)
    names = model_info.get_recommended_renderer_names(model)
    name = next((n for n in names if n.endswith("disable_thinking")), None)
    assert name, f"no thinking-off renderer for {model} (have {names})"
    return renderers.get_renderer(name, get_tokenizer(model))


def split_channels(completion: str) -> tuple[str, str]:
    """(reasoning, final answer) from a gpt-oss assistant turn decoded with its special tokens. Everything outside the
    final channel counts as reasoning. No final channel (the reasoning ran into the token limit) gives an empty answer."""
    parts = re.findall(r"<\|channel\|>(\w+)[^<]*<\|message\|>(.*?)(?=<\|(?:end|return|call|start|channel)\|>|$)", completion, re.DOTALL)
    reasoning = "\n\n".join(body.strip() for channel, body in parts if channel != "final")
    return reasoning, "\n\n".join(body.strip() for channel, body in parts if channel == "final")


def float_tensor(values: list[float]) -> types.TensorData:
    return types.TensorData.from_torch(torch.tensor(values, dtype=torch.float32))


class TinkerBackend:
    def __init__(self, args: argparse.Namespace):
        self.lr = args.lr
        service = tinker.ServiceClient()
        self.trainer = service.create_lora_training_client(base_model=args.model, rank=args.lora_rank, seed=args.seed)
        print(f"tinker training run {self.trainer.model_id}", flush=True)
        self.thinking_tokens = args.thinking_tokens
        self.renderer = {"receiver": no_thinking_renderer(args.model, args.reasoning_effort)}
        if not args.monitor_model.startswith("claude"):  # API monitors live in train.py
            self.renderer["monitor"] = no_thinking_renderer(args.monitor_model, args.reasoning_effort)
            self.monitor = service.create_sampling_client(base_model=args.monitor_model)
        # A sampling client is a snapshot of the weights, so it is replaced after every optimizer step.
        self.policy = self.trainer.save_weights_and_get_sampling_client()
        self.logged_mass: set[str] = set()

    def _tokens(self, role: str, chat: Chat, thinking: bool = False) -> list[int]:
        """The prompt the model continues. Without `thinking`, a gpt-oss prompt ends inside the final channel."""
        renderer = self.renderer[role]
        prefill = FINAL_PREFILL if isinstance(renderer, GptOssRenderer) and not thinking else None
        return renderer.build_generation_prompt(chat, prefill=prefill).to_ints()  # ty: ignore[invalid-argument-type]

    def sample(self, chats: list[Chat], max_tokens: int, thinking: bool = False) -> list[Sample]:
        renderer = self.renderer["receiver"]
        gpt_oss = isinstance(renderer, GptOssRenderer)
        if thinking and not gpt_oss:
            raise NotImplementedError("on Tinker, sender reasoning is only implemented for gpt-oss")
        # A reasoning sender gets one budget for reasoning plus essay, and one sampled sequence holds both channels. The
        # stop tokens (<|return|>, <|call|>) end the whole turn, not the reasoning, which ends with <|end|>.
        limit = max_tokens + (self.thinking_tokens if thinking else 0)
        params = types.SamplingParams(max_tokens=limit, temperature=1.0, stop=renderer.get_stop_sequences())
        # train.py repeats each prompt k times; one request with num_samples=k prefills the prompt once.
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i, chat in enumerate(chats):
            groups[tuple(self._tokens("receiver", chat, thinking))].append(i)
        futures = [self.policy.sample(types.ModelInput.from_ints(list(p)), len(idx), params) for p, idx in groups.items()]
        out: list[Sample | None] = [None] * len(chats)
        for (prompt, idx), future in zip(groups.items(), futures):
            for i, seq in zip(idx, future.result().sequences, strict=True):
                assert seq.logprobs is not None
                # `raw` holds every sampled token (reasoning, channel markers, essay, stop token), so the policy
                # gradient in `step` covers reasoning and essay alike.
                raw = {"prompt": list(prompt), "tokens": seq.tokens, "logprobs": seq.logprobs}
                if not gpt_oss:
                    text = str(renderer.tokenizer.decode(seq.tokens, skip_special_tokens=True)).strip()
                    out[i] = Sample(chats[i], text, raw)
                    continue
                decoded = str(renderer.tokenizer.decode(seq.tokens))
                cot, text = split_channels(decoded if thinking else FINAL_PREFILL + decoded)
                if thinking and "<essay>" in text:
                    # The final channel already separates the essay; the tags only matter when the sender used them.
                    stray, text = game.split_answer(text)
                    cot = f"{cot}\n\n[in the answer, outside the essay tags]\n{stray}" if stray else cot
                out[i] = Sample(chats[i], text, raw, cot)
        return out  # ty: ignore[invalid-return-type]

    def _label_ids(self, role: str, n_labels: int) -> list[int]:
        ids = [self.renderer[role].tokenizer.encode(label, add_special_tokens=False) for label in game.labels(n_labels)]
        assert all(len(i) == 1 for i in ids), "candidate labels must be single tokens"
        return [i[0] for i in ids]

    def choose(self, role: str, chats: list[Chat], n_labels: int) -> list[list[float]]:
        """One prefill per chat: `target_prompt_logprobs` returns the logprob of every label as the token right after
        the assistant prefix. A throwaway token is appended so that position exists in the prompt."""
        client = self.policy if role == "receiver" else self.monitor
        ids = self._label_ids(role, n_labels)
        futures = []
        for chat in chats:
            tokens = self._tokens(role, chat) + [ids[0]]
            target = torch.full((len(tokens) - 1, n_labels), -1, dtype=torch.int64)
            target[-1] = torch.tensor(ids)
            sparse = types.TensorData.from_torch_sparse(target, pad_value=-1)
            futures.append(
                client.sample(types.ModelInput.from_ints(tokens), 1, types.SamplingParams(max_tokens=1), target_prompt_logprobs=sparse)
            )
        dists, masses = [], []
        for future in futures:
            probs = [math.exp(lp) for lp in future.result().target_prompt_logprobs.to_torch()[-1].tolist()]  # ty: ignore[unresolved-attribute]
            masses.append(sum(probs))
            dists.append([p / masses[-1] for p in probs])
        if role not in self.logged_mass:
            self.logged_mass.add(role)
            mass = sum(masses) / len(masses)
            print(f"{role}: un-normalised probability on the {n_labels} labels: mean {mass:.4f}, min {min(masses):.4f}", flush=True)
            if mass < LOW_LABEL_MASS:
                print(f"WARNING {role}: label mass under {LOW_LABEL_MASS}: it does not answer with a bare letter here", flush=True)
        return dists

    def step(self, sender: list[tuple[Sample, float]], receiver: list[tuple[Chat, int]]) -> dict[str, float]:
        """Two forward_backward calls, then one optim_step: Tinker accumulates gradients until the optimizer step, so
        this is one update on the sum of both losses. Both losses are token sums, so with equal counts of essays and
        receiver answers this is mean_essays(-advantage * log p(essay)) + mean_answers(-log p(label)) up to a constant
        that Adam ignores."""
        pg_data = []
        for sample, advantage in sender:
            prompt, tokens, logprobs = sample.raw["prompt"], sample.raw["tokens"], sample.raw["logprobs"]
            full, pad = prompt + tokens, [0.0] * (len(prompt) - 1)
            pg_data.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(full[:-1]),
                    loss_fn_inputs={
                        "target_tokens": types.TensorData.from_torch(torch.tensor(full[1:])),
                        "logprobs": float_tensor(pad + logprobs),
                        "advantages": float_tensor(pad + [advantage] * len(tokens)),
                    },
                )
            )
        recv_data = []
        for chat, label in receiver:
            tokens = self._tokens("receiver", chat)
            answer = self._label_ids("receiver", label + 1)[label]
            recv_data.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens),
                    loss_fn_inputs={
                        "target_tokens": types.TensorData.from_torch(torch.tensor(tokens[1:] + [answer])),
                        "weights": float_tensor([0.0] * (len(tokens) - 1) + [1.0]),
                    },
                )
            )
        # Submitted together so they land on the same clock cycle; results are awaited afterwards.
        pg_future = self.trainer.forward_backward(pg_data, "importance_sampling")
        recv_future = self.trainer.forward_backward(recv_data, "cross_entropy")
        optim_future = self.trainer.optim_step(types.AdamParams(learning_rate=self.lr))
        pg, recv, optim = pg_future.result(), recv_future.result(), optim_future.result()
        self.policy = self.trainer.save_weights_and_get_sampling_client()
        metrics = {"loss_pg": pg.metrics["loss:sum"] / len(pg_data), "loss_recv": recv.metrics["loss:sum"] / len(recv_data)}
        return {k: round(v, 4) for k, v in (metrics | (optim.metrics or {})).items()}

    def label_prob(self, role: str, chats: list[Chat], answers: list[str]) -> list[tuple[float, bool]]:
        raise NotImplementedError("candidate lists longer than 52 are only implemented in the local backend")

    def save(self, path: Path) -> None:
        """`state` resumes training (create_training_client_from_state); `sampler` is the path that
        tinker_cookbook.weights.download + build_lora_adapter turn into a PEFT adapter."""
        state, sampler = self.trainer.save_state(path.name), self.trainer.save_weights_for_sampler(path.name)
        path.mkdir(parents=True, exist_ok=True)
        record = {"state": state.result().path, "sampler": sampler.result().path, "training_run": self.trainer.model_id}
        (path / "tinker.json").write_text(json.dumps(record, indent=2))
