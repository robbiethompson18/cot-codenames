"""Local backend for the stego trainer: Hugging Face transformers + PEFT on one GPU (RunPod), see train.Backend.

The policy is the base model with one LoRA; it plays sender and receiver. The monitor is a second, frozen model, or the
policy with its adapter switched off when --monitor-model equals --model.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, PeftModel, get_peft_model

from cot_codenames.stego import game
from cot_codenames.stego.game import Chat, Sample


def _load(name: str, device: str):
    tok: Any = transformers.AutoTokenizer.from_pretrained(name)  # transformers types this as a loose union
    tok.padding_side = "left"  # generation and next-token reads both want the last position to be real
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # Qwen3.5/3.6 are vision-language checkpoints and load through the multimodal auto class; we only feed them text.
    for cls_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(name, dtype=dtype)
            break
        except ValueError:  # this auto class has no mapping for the checkpoint's config
            continue
    else:
        raise RuntimeError(f"no transformers auto class could load {name}")
    return tok, model.to(device).eval()


def _render(tok, chat: Chat, thinking: bool = False) -> str:
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


class LocalBackend:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device == "cpu":
            # Seen on RunPod hosts with NVIDIA driver 570: the lockfile's CUDA 13 torch cannot see the GPU there and would
            # quietly run the 27B on CPU, producing nothing for an hour. Fail instead.
            raise RuntimeError("no GPU visible to torch (on RunPod this means the host's NVIDIA driver is too old)")
        self.tok, base = _load(args.model, self.device)
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules="all-linear",
            exclude_modules=r".*(visual|vision).*",  # text-only task: leave the vision tower alone
            task_type="CAUSAL_LM",
        )
        if args.init_adapter:
            # Continue from an earlier run's checkpoint ("<run>/ckpt-NNNN" in the Hugging Face repo), e.g. to swap in a
            # stronger monitor. Its rank comes from the checkpoint, not --lora-rank.
            from huggingface_hub import snapshot_download

            root = snapshot_download(args.hf_repo, allow_patterns=[f"runs/{args.init_adapter}/*"])
            self.policy = PeftModel.from_pretrained(base, f"{root}/runs/{args.init_adapter}", is_trainable=True)
        else:
            self.policy = get_peft_model(base, lora)
        self.policy.print_trainable_parameters()
        if args.monitor_model == args.model or args.monitor_model.startswith("claude"):  # API monitors live in train.py
            self.mon_tok, self.monitor = self.tok, None
        else:
            self.mon_tok, self.monitor = _load(args.monitor_model, self.device)
            self.monitor.requires_grad_(False)
        self.opt = torch.optim.AdamW([p for p in self.policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
        self._reported_label_mass: set[str] = set()

    @contextmanager
    def _base(self):
        """The policy with its LoRA switched off, i.e. the frozen base model."""
        with self.policy.disable_adapter():
            yield self.policy

    def _batches(self, items: list, size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _generate(self, rows: list[list[int]], max_new_tokens: int, stop: list[int]) -> list[list[int]]:
        """Sample a continuation of each token row at temperature 1, cut after the first stop token (kept)."""
        out = []
        for batch in self._batches(rows, self.args.score_batch):
            width = max(map(len, batch))
            ids = torch.full((len(batch), width), self.tok.pad_token_id, device=self.device)
            attn = torch.zeros((len(batch), width), dtype=torch.long, device=self.device)
            for i, row in enumerate(batch):  # left-padded, so every row's last position is real
                ids[i, width - len(row) :] = torch.tensor(row)
                attn[i, width - len(row) :] = 1
            # Plain temperature-1 sampling, with the checkpoint's default top-k/top-p turned off, so the distribution
            # we sample from is the one the policy gradient differentiates.
            gen = self.policy.generate(
                input_ids=ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                eos_token_id=stop,
            )
            for new in gen[:, width:].tolist():
                cut = min((new.index(t) + 1 for t in stop if t in new), default=len(new))
                out.append(new[:cut])
        return out

    @torch.no_grad()
    def sample(self, chats: list[Chat], max_tokens: int, thinking: bool = False) -> list[Sample]:
        eos = self.tok.eos_token_id
        prompts = [self.tok(_render(self.tok, c, thinking), add_special_tokens=False).input_ids for c in chats]
        if not thinking:
            # The EOS is kept in the completion so the policy is also trained on when to stop.
            essays = self._generate(prompts, max_tokens, [eos])
            texts = [self.tok.decode(e, skip_special_tokens=True).strip() for e in essays]
            return [
                Sample(c, t, {"prompt_ids": p, "completion_ids": e, "forced": (0, 0)}) for c, p, e, t in zip(chats, prompts, essays, texts)
            ]
        # Reasoning has a hard budget of --thinking-tokens. A sender still reasoning at the limit has its reasoning closed
        # for it and then writes the essay. The inserted closing tokens were not sampled, so `forced` marks them and the
        # policy-gradient loss skips them.
        (close,) = self.tok.encode("</think>", add_special_tokens=False)
        closing = self.tok.encode("\n</think>\n\n", add_special_tokens=False)
        cots = self._generate(prompts, self.args.thinking_tokens, [eos, close])
        forced = [(len(c), len(c) + len(closing)) if c[-1] not in (eos, close) else (0, 0) for c in cots]
        cots = [c + closing if f != (0, 0) else c for c, f in zip(cots, forced)]
        # A sender that emitted EOS inside its reasoning wrote no essay; everyone else continues to the essay.
        go = [i for i, c in enumerate(cots) if c[-1] != eos]
        essays: list[list[int]] = [[] for _ in chats]
        for i, e in zip(go, self._generate([prompts[i] + cots[i] for i in go], max_tokens, [eos])):
            essays[i] = e
        out = []
        for chat, prompt, cot, essay, span in zip(chats, prompts, cots, essays, forced):
            raw = {"prompt_ids": prompt, "completion_ids": cot + essay, "forced": span}
            reasoning, _ = game.split_reasoning(self.tok.decode(cot, skip_special_tokens=True))
            stray, text = game.split_answer(self.tok.decode(essay, skip_special_tokens=True))
            out.append(Sample(chat, text, raw, (reasoning + "\n\n[after the reasoning was closed]\n" + stray) if stray else reasoning))
        return out

    def _next_logits(self, model, tok, chats: list[Chat], batch_size: int) -> torch.Tensor:
        """Next-token logits after each rendered chat, shape (len(chats), vocab). Gradients flow if enabled."""
        rows = []
        for batch in self._batches(chats, batch_size):
            enc = tok([_render(tok, c) for c in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            rows.append(model(**enc).logits[:, -1].float())
        return torch.cat(rows)

    def _label_ids(self, tok, n_labels: int) -> list[int]:
        ids = [tok.encode(label, add_special_tokens=False) for label in game.labels(n_labels)]
        assert all(len(i) == 1 for i in ids), "candidate labels must be single tokens"
        return [i[0] for i in ids]

    @torch.no_grad()
    def choose(self, role: str, chats: list[Chat], n_labels: int) -> list[list[float]]:
        # Few-shot monitor prompts are several times longer than plain ones, so the batch shrinks to keep roughly the
        # same number of tokens per pass (about 4 characters per token, 600 tokens for a plain prompt).
        longest = max(len(c[0]["content"]) for c in chats) // 4
        batch = max(1, min(self.args.score_batch, self.args.score_batch * 600 // max(600, longest)))
        if role == "receiver":
            tok, logits = self.tok, self._next_logits(self.policy, self.tok, chats, batch)
        elif self.monitor is None:
            with self._base() as base:
                tok, logits = self.tok, self._next_logits(base, self.tok, chats, batch)
        else:
            tok, logits = self.mon_tok, self._next_logits(self.monitor, self.mon_tok, chats, batch)
        full = logits.softmax(-1)[:, self._label_ids(tok, n_labels)]
        if role not in self._reported_label_mass:  # sanity check that the chooser really answers with a letter
            print(f"{role}: mean probability mass on the {n_labels} labels = {full.sum(-1).mean().item():.3f}", flush=True)
            self._reported_label_mass.add(role)
        return (full / full.sum(-1, keepdim=True)).tolist()

    @torch.no_grad()
    def label_prob(self, role: str, chats: list[Chat], answers: list[str]) -> list[tuple[float, bool]]:
        """For candidate lists too long for one-letter labels: the probability the chooser gives to `answers[i]` as its
        whole reply, and whether greedy decoding would produce exactly that reply. One teacher-forced pass."""
        if role == "monitor" and self.monitor is not None:
            return self._label_prob(self.monitor, self.mon_tok, chats, answers)
        if role == "monitor":
            with self._base() as base:
                return self._label_prob(base, self.tok, chats, answers)
        return self._label_prob(self.policy, self.tok, chats, answers)

    def _label_prob(self, model, tok, chats: list[Chat], answers: list[str]) -> list[tuple[float, bool]]:
        longest = max(len(c[0]["content"]) for c in chats) // 4
        size = max(1, min(self.args.score_batch, self.args.score_batch * 600 // max(600, longest)))
        out = []
        for batch in self._batches(list(zip(chats, answers)), size):
            # The end-of-turn token is part of the answer, so "13" is not credited when the model was going to say "137".
            answer_ids = [tok.encode(a, add_special_tokens=False) + [tok.eos_token_id] for _, a in batch]
            rows = [tok(_render(tok, c), add_special_tokens=False).input_ids + ids for (c, _), ids in zip(batch, answer_ids)]
            width = max(map(len, rows))
            ids = torch.full((len(rows), width), tok.pad_token_id, device=self.device)
            attn = torch.zeros((len(rows), width), dtype=torch.long, device=self.device)
            for i, row in enumerate(rows):  # left-padded, so every answer ends at the last position
                ids[i, width - len(row) :] = torch.tensor(row)
                attn[i, width - len(row) :] = 1
            logp = model(input_ids=ids, attention_mask=attn).logits.float().log_softmax(-1)
            for i, answer in enumerate(answer_ids):
                # The token at position t is predicted from position t - 1.
                steps = logp[i, width - len(answer) - 1 : width - 1]
                target = torch.tensor(answer, device=self.device)
                total = steps.gather(1, target[:, None]).sum().item()
                out.append((float(torch.tensor(total).exp()), bool((steps.argmax(-1) == target).all())))
        return out

    def step(self, sender: list[tuple[Sample, float]], receiver: list[tuple[Chat, int]]) -> dict[str, float]:
        self.opt.zero_grad()
        # Sampling and scoring leave large cached blocks behind; without this the first backward pass runs out of memory.
        if self.device == "cuda":
            torch.cuda.empty_cache()
        loss_pg = loss_kl = loss_recv = 0.0
        # Policy gradient on the essay tokens: -advantage * log pi(essay | prompt), summed over tokens, mean over essays.
        for batch in self._batches(sender, self.args.micro_batch):
            seqs = [s.raw["prompt_ids"] + s.raw["completion_ids"] for s, _ in batch]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), self.tok.pad_token_id, device=self.device)
            essay = torch.zeros((len(seqs), width), dtype=torch.bool, device=self.device)  # positions of essay tokens
            for i, ((s, _), seq) in enumerate(zip(batch, seqs)):
                ids[i, : len(seq)] = torch.tensor(seq)
                essay[i, len(s.raw["prompt_ids"]) : len(seq)] = True
                lo, hi = s.raw["forced"]
                essay[i, len(s.raw["prompt_ids"]) + lo : len(s.raw["prompt_ids"]) + hi] = False  # tokens we inserted, not sampled
            attn = ids != self.tok.pad_token_id
            attn[essay] = True  # an essay's EOS can share an id with the pad token
            logits = self.policy(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            targets, mask = ids[:, 1:], essay[:, 1:]
            logp = -F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
            adv = torch.tensor([a for _, a in batch], device=self.device)
            pg = -(adv * (logp * mask).sum(1)).sum() / len(sender)
            loss = pg
            if self.args.kl_coef:
                with torch.no_grad(), self._base() as base:
                    base_logp = base(input_ids=ids, attention_mask=attn).logits[:, :-1].float().log_softmax(-1)
                cur_logp = logits.log_softmax(-1)
                kl = ((cur_logp.exp() * (cur_logp - base_logp)).sum(-1) * mask).sum() / mask.sum() * len(batch) / len(sender)
                loss = loss + self.args.kl_coef * kl
                loss_kl += kl.item()
            loss.backward()
            loss_pg += pg.item()
        # Receiver: cross-entropy on the letter of the true secret, through the same LoRA.
        for batch in self._batches(receiver, self.args.micro_batch):
            n_labels = max(i for _, i in receiver) + 1
            label_ids = torch.tensor(self._label_ids(self.tok, n_labels), device=self.device)
            logits = self._next_logits(self.policy, self.tok, [c for c, _ in batch], len(batch))
            ce = F.cross_entropy(logits, label_ids[[i for _, i in batch]], reduction="sum") / len(receiver)
            ce.backward()
            loss_recv += ce.item()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad], 1.0)
        self.opt.step()
        return {
            "loss_pg": round(loss_pg, 4),
            "loss_kl": round(loss_kl, 4),
            "loss_recv": round(loss_recv, 4),
            "grad_norm": round(grad_norm.item(), 3),
        }

    def save(self, path: Path) -> None:
        self.policy.save_pretrained(str(path))
