from text_distillation.model.registry import (
    MODEL_REGISTRY,
    ModelProfile,
    get_model_profile,
    list_models,
    register_model,
)


REQUIRED_MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "albert-base-v2",
    "microsoft/deberta-v3-base",
    "xlnet-base-cased",
]


def test_required_models_are_registered():
    registered = list_models()
    for name in REQUIRED_MODELS:
        assert name in registered, f"{name} missing from MODEL_REGISTRY"


def test_xlnet_uses_last_token_pooling():
    assert get_model_profile("xlnet-base-cased").embedding_pooling == "last_token"


def test_bert_family_uses_first_token_pooling():
    for name in ["bert-base-uncased", "bert-large-uncased", "roberta-base", "albert-base-v2",
                 "microsoft/deberta-v3-base"]:
        assert get_model_profile(name).embedding_pooling == "first_token", name


def test_bert_large_has_smaller_recommended_batches():
    base = get_model_profile("bert-base-uncased")
    large = get_model_profile("bert-large-uncased")
    assert large.recommended_train_batch_size < base.recommended_train_batch_size
    assert large.recommended_embedding_batch_size < base.recommended_embedding_batch_size


def test_deberta_v3_marked_unsafe_for_fp16():
    # DeBERTa v3 leaves some tensors in fp16 after from_pretrained, which
    # breaks torch.amp.GradScaler. The profile must flag this so callers
    # can opt out of mixed precision.
    assert get_model_profile("microsoft/deberta-v3-base").supports_fp16 is False
    assert get_model_profile("bert-base-uncased").supports_fp16 is True


def test_unknown_model_returns_bert_like_default():
    profile = get_model_profile("some-unregistered/model")
    assert profile.embedding_pooling == "first_token"
    assert profile.family == "bert"


def test_register_model_prevents_silent_overwrite():
    profile = ModelProfile(
        model_name="bert-base-uncased",
        family="bert",
        embedding_pooling="first_token",
    )
    try:
        register_model(profile)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when re-registering an existing model")
    # Overwrite=True must succeed without mutating registry-wide state.
    register_model(profile, overwrite=True)
    assert MODEL_REGISTRY["bert-base-uncased"].embedding_pooling == "first_token"
