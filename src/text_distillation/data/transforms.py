from __future__ import annotations

from typing import Any


def tokenize_text_dataset(
    dataset: Any,
    tokenizer: Any,
    text_column: str = "text",
    label_column: str = "label",
    max_length: int = 128,
    padding: bool | str = "max_length",
    remove_text: bool = True,
):
    """Tokenize a text classification dataset for Hugging Face Trainer."""

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch[text_column],
            padding=padding,
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = batch[label_column]
        return tokenized

    remove_columns = []
    if remove_text and text_column in dataset.column_names:
        remove_columns.append(text_column)
    if label_column in dataset.column_names:
        remove_columns.append(label_column)

    return dataset.map(tokenize_batch, batched=True, remove_columns=remove_columns)
