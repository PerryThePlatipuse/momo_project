from types import SimpleNamespace

from datasets import Dataset

from text_distillation.data.dataloaders import create_text_dataloader
from text_distillation.model.training import _model_input_columns


def test_training_batches_exclude_metadata_columns():
    tokenized = Dataset.from_dict(
        {
            "input_ids": [[101, 102], [101, 103]],
            "attention_mask": [[1, 1], [1, 1]],
            "labels": [0, 1],
            "idx": [10, 11],
        }
    )
    tokenizer = SimpleNamespace(model_input_names=["input_ids", "attention_mask", "token_type_ids"])

    tokenized.set_format(type="torch", columns=_model_input_columns(tokenized, tokenizer))
    batch = next(iter(create_text_dataloader(tokenized, batch_size=2)))

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
