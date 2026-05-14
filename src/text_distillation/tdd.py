"""TDD — Text Dataset Distillation in continuous embedding space.

Maekawa et al. «Dataset Distillation with Attention Labels for Fine-tuning BERT»
(ACL 2023). Two related methods from the DiLM paper's comparison table:

- **TDD embed**: distill into `(K × seq_len × hidden_dim)` tensors trained by
  matching attention maps + gradients of a downstream BERT classifier. The
  distilled artifact is *not text* — it lives in BERT's embedding space and
  cannot be transferred to other architectures.
- **TDD text**: decode the distilled embeddings back to tokens via nearest
  vocabulary embedding. On the DiLM table this collapses to ~50% on SST-2
  (vs ~89% for TDD embed) — the trick is that the information is in the
  continuous vectors, not the discrete projection.

Status in this project: **not implemented**.

Reasons:
- The reference implementation is not in `DiLM-main/` (that repo covers Vanilla
  LM and DiLM). TDD has its own repo:
  https://github.com/arumaekawa/dataset-distillation-with-attention-labels
- A from-scratch implementation requires a custom training loop on `inputs_embeds`
  and an attention-label objective; integrating it cleanly with our existing
  pipeline (`train_text_classifier` expects token ids, not embedding tensors) is
  a small project on its own.

Recommended path when implementing:
1. Vendor the reference repo into this project (e.g. as `TDD-implementation/`).
2. Add `distill_tdd_embed(dataset, ...) -> tuple[Tensor, Tensor]` returning
   `(synthetic_embeds, synthetic_labels)`.
3. Add a sibling `train_text_classifier_on_embeds(...)` that calls
   `model(inputs_embeds=synthetic_embeds, labels=synthetic_labels)`.
4. Add `distill_tdd_text(...)` that snaps each embedding to its nearest token
   id in `model.get_input_embeddings().weight` and returns a regular `Dataset`.
"""

from __future__ import annotations


_TDD_REFERENCE = (
    "https://github.com/arumaekawa/dataset-distillation-with-attention-labels"
)


def distill_tdd_embed(*_args, **_kwargs):
    raise NotImplementedError(
        "TDD embed is not implemented in this project. The reference repo is "
        f"{_TDD_REFERENCE} (NOT shipped in DiLM-main/). See module docstring "
        "for the recommended integration path."
    )


def distill_tdd_text(*_args, **_kwargs):
    raise NotImplementedError(
        "TDD text is not implemented in this project. It requires distill_tdd_embed "
        "first, then a nearest-vocab-embedding decode. See module docstring."
    )
