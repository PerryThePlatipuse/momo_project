"""Class-conditional language model generator used by DiLM and Vanilla LM.

Adapted from `DiLM-main/src/generator.py`. The generator is a causal LM
(GPT-2-family) extended with:

- a `<pad>` token for batching;
- a `<sep>` token for separating sentence pairs (QQP, MNLI);
- one `<bos_y>` per label, each initialized from the model's original BOS so
  it starts in a sensible region of embedding space.

After training (LM phase + optional DC phase) the generator is sampled from
each `<bos_y>` to produce a class-conditional distilled dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SEP_TOKEN = "<sep>"


@dataclass
class GeneratorConfig:
    model_name: str = "gpt2"
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float = 1.0
    generate_batch_size: int = 32
    generate_max_length: int = 64
    generate_bf16: bool = False
    generate_fp16: bool = False


class GeneratorModel(nn.Module):
    """Class-conditional causal LM. See module docstring for the token layout."""

    def __init__(
        self,
        config: GeneratorConfig,
        num_labels: int,
        sentence_keys: tuple[str, ...],
    ):
        super().__init__()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.config = config
        self.num_labels = num_labels
        self.sentence_keys = sentence_keys

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        if SEP_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(SEP_TOKEN)
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)

        self.bos_tokens = {label: f"<bos_{label}>" for label in range(num_labels)}
        for bos in self.bos_tokens.values():
            if bos not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens(bos)
        self.bos_ids = {
            label: self.tokenizer.convert_tokens_to_ids(token)
            for label, token in self.bos_tokens.items()
        }

        self.model.resize_token_embeddings(len(self.tokenizer))

        # Init each <bos_y> from the model's original BOS so sampling starts in
        # a usable region of embedding space instead of random.
        ref_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        with torch.no_grad():
            ref = self.model.get_input_embeddings().weight[ref_id].clone()
            for bos_id in self.bos_ids.values():
                self.model.get_input_embeddings().weight[bos_id].copy_(ref)

    # ---------- Training-side helpers ----------

    def build_training_text(self, example: dict, label: int) -> str:
        """Format one example as ``<bos_y> sent_a <sep> sent_b`` for LM training."""
        body = f" {SEP_TOKEN} ".join(str(example[k]) for k in self.sentence_keys)
        return f"{self.bos_tokens[label]} {body}"

    def encode_training_batch(self, texts: Iterable[str], max_length: int) -> dict[str, torch.Tensor]:
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = encodings["input_ids"].clone()
        labels[encodings["attention_mask"] == 0] = -100
        encodings["labels"] = labels
        return encodings

    def compute_loss(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        **_unused,
    ) -> torch.Tensor:
        """Per-sample (not averaged) causal-LM loss. Required for DiLM DC weighting."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if labels is None:
            labels = input_ids
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        )
        return loss.reshape(shift_labels.shape).mean(-1)  # (batch_size,)

    # ---------- Sampling-side helpers ----------

    @torch.inference_mode()
    def sample(
        self,
        label: int,
        n: int,
        max_length: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> list[str]:
        """Sample `n` continuations from the `<bos_label>` prompt and return decoded texts."""
        max_length = max_length or self.config.generate_max_length
        top_p = top_p if top_p is not None else self.config.top_p

        bad_words = [[bos_id] for bos_id in self.bos_ids.values()]
        device = next(self.model.parameters()).device

        prompt = torch.full((n, 1), self.bos_ids[label], dtype=torch.long, device=device)
        out = self.model.generate(
            prompt,
            do_sample=True,
            top_p=top_p,
            top_k=self.config.top_k,
            temperature=temperature,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            bad_words_ids=bad_words,
            repetition_penalty=self.config.repetition_penalty,
        )
        return self.tokenizer.batch_decode(out[:, 1:], skip_special_tokens=True)

    def decode_to_example(self, text: str) -> dict[str, str]:
        """Split a generated string back into sentence fields by `<sep>`."""
        parts = [p.strip() for p in text.split(SEP_TOKEN, maxsplit=len(self.sentence_keys) - 1)]
        while len(parts) < len(self.sentence_keys):
            parts.append("")
        return {k: (p or " ") for k, p in zip(self.sentence_keys, parts)}

    def device(self) -> torch.device:
        return next(self.model.parameters()).device
