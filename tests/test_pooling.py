"""Pooling-strategy tests that don't require any model download.

The pooling helper consumes only `last_hidden_state` and `attention_mask`,
so we can exercise all three strategies (first_token / last_token / mean)
with tiny synthetic tensors and assert the right slicing behavior. This
catches regressions in the XLNet path where the last *non-padding* token
must be used, not literally position -1.
"""

import torch

from text_distillation.distillation import _pool_hidden_states


def test_first_token_returns_position_0():
    hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
    out = _pool_hidden_states(hidden, attention_mask=mask, pooling="first_token")
    assert out.shape == (2, 4)
    assert torch.equal(out, hidden[:, 0, :])


def test_last_token_uses_last_non_padding_position():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    # row 0 has 3 real tokens (positions 0,1,2), row 1 has 4 (positions 0..3).
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    out = _pool_hidden_states(hidden, attention_mask=mask, pooling="last_token")
    assert out.shape == (2, 3)
    assert torch.equal(out[0], hidden[0, 2, :])
    assert torch.equal(out[1], hidden[1, 3, :])


def test_last_token_falls_back_to_position_minus_one_without_mask():
    hidden = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    out = _pool_hidden_states(hidden, attention_mask=None, pooling="last_token")
    assert torch.equal(out, hidden[:, -1, :])


def test_mean_pooling_respects_mask():
    hidden = torch.ones(1, 4, 2)
    hidden[0, 3, :] = 100.0  # padding position with garbage values
    mask = torch.tensor([[1, 1, 1, 0]])
    out = _pool_hidden_states(hidden, attention_mask=mask, pooling="mean")
    # Padding token must be excluded from the mean.
    assert torch.allclose(out, torch.tensor([[1.0, 1.0]]))


def test_unknown_pooling_raises():
    hidden = torch.zeros(1, 1, 1)
    try:
        _pool_hidden_states(hidden, attention_mask=None, pooling="bogus")  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown pooling strategy")
