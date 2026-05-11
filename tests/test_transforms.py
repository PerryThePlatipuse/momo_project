from datasets import Dataset

from text_distillation.data.transforms import join_text_columns, tokenize_text_dataset


class DummyTokenizer:
    def __call__(self, *texts, padding, truncation, max_length):
        batch_size = len(texts[0])
        if len(texts) == 2:
            lengths = [min(max_length, len(left.split()) + len(right.split()) + 3) for left, right in zip(*texts)]
        else:
            lengths = [min(max_length, len(text.split()) + 2) for text in texts[0]]
        return {
            "input_ids": [[1] * length + [0] * (max_length - length) for length in lengths],
            "attention_mask": [[1] * length + [0] * (max_length - length) for length in lengths],
        }


def test_tokenize_text_dataset_supports_sentence_pairs():
    dataset = Dataset.from_dict(
        {
            "premise": ["A dog runs.", "A cat sleeps."],
            "hypothesis": ["An animal moves.", "A person sings."],
            "label": [0, 2],
        }
    )

    tokenized = tokenize_text_dataset(
        dataset,
        tokenizer=DummyTokenizer(),
        text_columns=("premise", "hypothesis"),
        max_length=8,
    )

    assert tokenized.column_names == ["input_ids", "attention_mask", "labels"]
    assert tokenized["labels"] == [0, 2]
    assert len(tokenized["input_ids"][0]) == 8


def test_join_text_columns_for_sentence_pairs():
    dataset = Dataset.from_dict(
        {
            "question1": ["hello there"],
            "question2": ["general kenobi"],
        }
    )

    assert join_text_columns(dataset, ("question1", "question2")) == ["hello there [SEP] general kenobi"]
