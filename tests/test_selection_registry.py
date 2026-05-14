import pytest

from text_distillation.distillation import (
    SELECTION_REGISTRY,
    get_selection_fn,
    list_selection_methods,
    register_selection,
    select_random,
    select_stratified_random,
)


def test_all_baselines_registered():
    # Importing `text_distillation` should pull in vanilla_lm / dilm side modules
    # via the package __init__ so they register themselves.
    import text_distillation  # noqa: F401

    methods = set(list_selection_methods())
    expected = {
        "random", "stratified_random",
        "kcenter_tfidf", "kcenter_cls",
        "herding",
        "vanilla_lm", "dilm", "dilm_official",
    }
    assert expected.issubset(methods), f"missing: {expected - methods}"


def test_get_selection_fn_returns_callable():
    assert get_selection_fn("random") is select_random
    assert get_selection_fn("stratified_random") is select_stratified_random


def test_get_selection_fn_unknown_raises():
    with pytest.raises(KeyError):
        get_selection_fn("definitely_not_a_method")


def test_register_selection_rejects_duplicates():
    with pytest.raises(ValueError):
        register_selection("random")(lambda *a, **kw: None)
    # Sanity: registry was not mutated by the failed registration.
    assert get_selection_fn("random") is select_random


def test_register_selection_round_trip():
    name = "_unit_test_selection_method"
    try:
        @register_selection(name)
        def _dummy(dataset, *, seed=0, **_kwargs):
            return dataset

        assert get_selection_fn(name) is _dummy
        assert name in list_selection_methods()
    finally:
        SELECTION_REGISTRY.pop(name, None)
