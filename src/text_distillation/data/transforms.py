from __future__ import annotations

from collections.abc import Sequence
from typing import Any


TextColumns = str | Sequence[str]


def tokenize_text_dataset(
    dataset: Any,
    tokenizer: Any,
    text_column: str = "text",
    text_columns: TextColumns | None = None,
    label_column: str = "label",
    max_length: int = 128,
    padding: bool | str = "max_length",
    remove_text: bool = True,
):
    """Tokenize a single-text or sentence-pair classification dataset."""
    columns = normalize_text_columns(text_column=text_column, text_columns=text_columns)

    def tokenize_batch(batch):
        if len(columns) == 1:
            tokenized = tokenizer(
                batch[columns[0]],
                padding=padding,
                truncation=True,
                max_length=max_length,
            )
        elif len(columns) == 2:
            tokenized = tokenizer(
                batch[columns[0]],
                batch[columns[1]],
                padding=padding,
                truncation=True,
                max_length=max_length,
            )
        else:
            texts = [_join_text_parts(parts) for parts in zip(*(batch[column] for column in columns))]
            tokenized = tokenizer(
                texts,
                padding=padding,
                truncation=True,
                max_length=max_length,
            )
        tokenized["labels"] = batch[label_column]
        return tokenized

    remove_columns = []
    if remove_text:
        remove_columns.extend(column for column in columns if column in dataset.column_names)
    if label_column in dataset.column_names:
        remove_columns.append(label_column)

    return dataset.map(tokenize_batch, batched=True, remove_columns=remove_columns)


def normalize_text_columns(
    text_column: str = "text",
    text_columns: TextColumns | None = None,
) -> tuple[str, ...]:
    if text_columns is None:
        return (text_column,)
    if isinstance(text_columns, str):
        return (text_columns,)
    columns = tuple(text_columns)
    if not columns:
        raise ValueError("text_columns must contain at least one column.")
    return columns


def join_text_columns(dataset: Any, text_columns: TextColumns, separator: str = " [SEP] ") -> list[str]:
    columns = normalize_text_columns(text_columns=text_columns)
    if len(columns) == 1:
        return [str(value) for value in dataset[columns[0]]]
    return [
        separator.join(str(part) for part in parts)
        for parts in zip(*(dataset[column] for column in columns))
    ]


def _join_text_parts(parts: tuple[Any, ...], separator: str = " [SEP] ") -> str:
    return separator.join(str(part) for part in parts)
