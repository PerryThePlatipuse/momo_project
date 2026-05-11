from __future__ import annotations


def load_tokenizer(model_name: str, **kwargs):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def load_sequence_classifier(
    model_name: str,
    num_labels: int,
    label_names: list[str] | None = None,
    ignore_mismatched_sizes: bool = True,
    **kwargs,
):
    from transformers import AutoModelForSequenceClassification

    id2label = None
    label2id = None
    if label_names:
        id2label = {index: label for index, label in enumerate(label_names)}
        label2id = {label: index for index, label in id2label.items()}

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        **kwargs,
    )
