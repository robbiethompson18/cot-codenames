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
from peft import LoraConfig, get_peft_model

from cot_codenames.stego import game
from cot_codenames.stego.game import Chat, Sample

NO_GRAD_BATCH = 64  # sequences per pass when sampling or scoring; --micro-batch only bounds passes that keep gradients


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


def _render(tok, chat: Chat) -> str:
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)


class LocalBackend:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok, base = _load(args.model, self.device)
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules="all-linear",
            exclude_modules=r".*(visual|vision).*",  # text-only task: leave the vision tower alone
            task_type="CAUSAL_LM",
        )
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

    @torch.no_grad()
    def sample(self, chats: list[Chat], max_tokens: int) -> list[Sample]:
        out = []
        eos = self.tok.eos_token_id
        for batch in self._batches(chats, NO_GRAD_BATCH):
            enc = self.tok([_render(self.tok, c) for c in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(
                self.device
            )
            # Plain temperature-1 sampling, with the checkpoint's default top-k/top-p turned off, so the distribution
            # we sample from is the one the policy gradient differentiates.
            gen = self.policy.generate(**enc, max_new_tokens=max_tokens, do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
            for chat, prompt_ids, mask, row in zip(batch, enc.input_ids, enc.attention_mask, gen[:, enc.input_ids.shape[1] :].tolist()):
                # Keep tokens up to and including the first EOS, so the policy is also trained on when to stop.
                completion = row[: row.index(eos) + 1] if eos in row else row
                text = self.tok.decode(completion, skip_special_tokens=True).strip()
                raw = {"prompt_ids": prompt_ids[mask.bool()].tolist(), "completion_ids": completion}
                out.append(Sample(chat, text, raw))
        return out

    def _next_logits(self, model, tok, chats: list[Chat], batch_size: int) -> torch.Tensor:
        """Next-token logits after each rendered chat, shape (len(chats), vocab). Gradients flow if enabled."""
        rows = []
        for batch in self._batches(chats, batch_size):
            enc = tok([_render(tok, c) for c in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            rows.append(model(**enc).logits[:, -1].float())
        return torch.cat(rows)

    def _label_ids(self, tok, n_labels: int) -> list[int]:
        ids = [tok.encode(label, add_special_tokens=False) for label in game.LABELS[:n_labels]]
        assert all(len(i) == 1 for i in ids), "candidate labels must be single tokens"
        return [i[0] for i in ids]

    @torch.no_grad()
    def choose(self, role: str, chats: list[Chat], n_labels: int) -> list[list[float]]:
        # Few-shot monitor prompts are several times longer than plain ones, so the batch shrinks to keep roughly the
        # same number of tokens per pass (about 4 characters per token, 600 tokens for a plain prompt).
        longest = max(len(c[0]["content"]) for c in chats) // 4
        batch = max(1, min(NO_GRAD_BATCH, NO_GRAD_BATCH * 600 // max(600, longest)))
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
