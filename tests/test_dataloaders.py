import torch
from datasets import Dataset

from text_distillation.data.dataloaders import create_text_dataloader


def test_create_text_dataloader_stacks_tokenized_features():
    dataset = Dataset.from_dict(
        {
            "input_ids": [[101, 10, 102, 0], [101, 20, 30, 102]],
            "attention_mask": [[1, 1, 1, 0], [1, 1, 1, 1]],
            "labels": [0, 1],
        }
    )
    dataset.set_format(type="torch")

    dataloader = create_text_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].dtype == torch.int64
    assert batch["labels"].tolist() == [0, 1]

