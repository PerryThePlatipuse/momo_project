from .distilled_data import DistilledDataConfig
from .embedding_data import (
    EmbeddingDistillationDataConfig,
    EmbeddingLearnerTrainConfig,
    SyntheticEmbeddingDataset,
)
from .embedding_trainer import (
    EmbeddingDistillationDataModule,
    EmbeddingDistillationDataModuleConfig,
    EmbeddingDistillationEvaluator,
    EmbeddingDistillationTrainConfig,
    EmbeddingDistillationTrainer,
    run_embedding_distillation,
)

__all__ = [
    "DistilledDataConfig",
    "EmbeddingDistillationDataConfig",
    "EmbeddingLearnerTrainConfig",
    "SyntheticEmbeddingDataset",
    "EmbeddingDistillationDataModule",
    "EmbeddingDistillationDataModuleConfig",
    "EmbeddingDistillationEvaluator",
    "EmbeddingDistillationTrainConfig",
    "EmbeddingDistillationTrainer",
    "run_embedding_distillation",
    "TrainConfig",
    "TrainerBase",
    "TrainerDC",
    "TrainerLM",
    "get_trainer",
]


def __getattr__(name: str):
    if name in {"TrainConfig", "TrainerBase"}:
        from .trainer_base import TrainConfig, TrainerBase

        return {"TrainConfig": TrainConfig, "TrainerBase": TrainerBase}[name]

    if name == "TrainerDC":
        from .trainer_dc import TrainerDC

        return TrainerDC

    if name == "TrainerLM":
        from .trainer_lm import TrainerLM

        return TrainerLM

    if name == "get_trainer":
        return get_trainer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_trainer(config, distilled_data_config):
    from .trainer_dc import TrainerDC
    from .trainer_lm import TrainerLM

    trainer_classes = {
        "lm": TrainerLM,
        "dc": TrainerDC,
    }
    assert config.train_type in trainer_classes
    return trainer_classes[config.train_type](config, distilled_data_config)
