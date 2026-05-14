# PLAN.md

## Описание проекта

Проект посвящен сравнению методов дистилляции текстовых датасетов для задачи классификации. В первой версии фокусируемся на AG News и проверяем, насколько хорошо модели сохраняют качество при обучении не на полном датасете, а на его сжатой версии.

Исследуем:

- качество моделей при обучении на полном и сжатом датасете;
- зависимость Accuracy и F1-Macro от размера дистиллированного датасета;
- переносимость сжатых датасетов между архитектурами;
- сравнение методов дистилляции с простыми baseline-подходами;
- стоимость методов: время построения датасета и время обучения модели.

Главный принцип репозитория: notebook задает эксперимент, `src/` содержит переиспользуемую логику, а методы не дублируются внутри notebooks.

## Что уже сделано

### Структура репозитория

- Создан `AGENTS.md` с описанием проекта, правил и ожиданий для coding agents.
- Созданы основные папки:
  - `notebooks/`
  - `src/text_distillation/`
  - `data/`
  - `artifacts/`
  - `tests/`
- Добавлены README для важных папок.
- Добавлены `requirements.txt`, `pyproject.toml`, `.gitignore`.

### Реализован reusable-код в `src/text_distillation/`

- `data/datasets.py`
  - загрузка AG News через Hugging Face `datasets`;
  - tiny subsets для быстрых smoke-checks;
  - получение label names.
- `data/transforms.py`
  - токенизация текстового датасета.
- `data/dataloaders.py`
  - простой PyTorch DataLoader для токенизированных датасетов.
- `model/loading.py`
  - загрузка tokenizer;
  - загрузка sequence classification model.
- `model/training.py`
  - простой training loop на PyTorch без `Trainer`, `Hydra`, `MLflow`, `accelerate`.
- `distillation.py`
  - `select_random(...)`;
  - `select_stratified_random(...)`;
  - `select_kcenter_tfidf(...)`;
  - `select_kcenter_embeddings(...)`;
  - вычисление `[CLS]` embeddings.
- `evaluation.py`
  - Accuracy;
  - F1-Macro;
  - базовая evaluation-функция.
- `saving.py`
  - сохранение `config.json`;
  - сохранение `metrics.json`;
  - сохранение distilled dataset.
- `utils.py`
  - seed;
  - JSON;
  - директории;
  - git hash;
  - выбор устройства `cuda -> cpu`.

### Реализованные notebooks

- `notebooks/00_check_setup.ipynb`
  - проверка окружения, импортов, загрузки AG News.
- `notebooks/01_full_data_baseline.ipynb`
  - full-data baseline на AG News.
- `notebooks/02_stratified_random_baseline.ipynb`
  - Stratified Random Coreset.
- `notebooks/03_kcenter_embedding_baseline.ipynb`
  - K-Center over BERT `[CLS]` embeddings.

### Тесты и проверки

- Добавлены тесты:
  - `tests/test_data.py`;
  - `tests/test_distillation.py`;
  - `tests/test_metrics.py`.
- Проверено:
  - `pytest -q`: проходит;
  - `python -m compileall -q src tests`: проходит;
  - AG News загружается;
  - tiny training smoke-check проходит;
  - tiny K-Center over CLS smoke-check проходит.

### Важные технические решения

- Для стабильности на macOS убрана зависимость от Hugging Face `Trainer`.
- `transformers` закреплен как `>=4.40,<4.45`, потому что более свежие версии в текущем conda env ловили segfault.
- По умолчанию устройство выбирается как `cuda -> cpu`; MPS лучше включать явно после отдельной проверки.

### Расширение baseline-инфраструктуры (2026-05-13)

- Добавлен `src/text_distillation/model/registry.py` с профилями для `bert-base-uncased`, `bert-large-uncased`, `roberta-base`, `albert-base-v2`, `microsoft/deberta-v3-base`, `xlnet-base-cased`. Каждый профиль фиксирует `embedding_pooling` (`first_token` для BERT-family, `last_token` для XLNet) и рекомендованные `batch_size`/`max_length`.
- Добавлен `src/text_distillation/timing.py` с `TimingTracker`. Тайминги встраиваются в `train_text_classifier` (`tokenization_sec`, `training_sec`, `evaluation_sec`) и `compute_text_embeddings` (`embedding_load_sec`, `embedding_tokenize_sec`, `embedding_forward_sec`), сохраняются в `metrics.json` под ключом `timings`.
- `compute_cls_embeddings` переименован в `compute_text_embeddings` (старое имя оставлено как alias). Добавлен параметр `pooling: "first_token" | "last_token" | "mean"` с дефолтом из реестра моделей.
- Добавлен registry методов селекции через декоратор `@register_selection(name)`. Все 4 baseline-метода зарегистрированы: `random`, `stratified_random`, `kcenter_tfidf`, `kcenter_cls`.
- Добавлен `src/text_distillation/experiments.py` с `ExperimentConfig` и `run_baseline_experiment(config, runs_dir)`. End-to-end: select → train → eval → save с автоматическим сбором таймингов. Notebooks `01–05` рефакторены в тонкие обёртки (4 ячейки), копипасты `train_and_save` больше нет.
- Все 5 baseline-notebooks теперь свипуют по 6 моделям из реестра и 4 датасетам.
- Добавлен `src/text_distillation/analysis.py` с `collect_runs(runs_dir) -> pd.DataFrame`. Поля `timings` разворачиваются в колонки `timings_<name>`.
- Добавлен `notebooks/templates/baseline_template.ipynb` + раздел «How to add a new baseline» в [AGENTS.md](AGENTS.md).
- Тесты: `test_model_registry.py`, `test_selection_registry.py`, `test_timing.py`, `test_pooling.py`, `test_collect_runs.py`. `test_training_pipeline_smoke.py` параметризован по `{tiny-random-bert, tiny-random-roberta}`.
- Добавлен пятый baseline-метод: `@register_selection("herding")` — Welling 2009 поверх encoder-эмбеддингов. Закрывает gap до DiLM Table 1 (Random / K-centers / Herding). Notebook `06_herding_baseline.ipynb`, тесты `_greedy_herding` (детерминированность, избегание outlier'а).
- Упрощение API (2026-05-13): training-гиперпараметры вынесены в `TrainingConfig` (вложенный в `ExperimentConfig`), `experiment_name` сделан опциональным с автогенерацией из `(prefix, dataset, model, k)`. Notebooks ужаты с ~75 строк до ~30 — CAPS_LOCK блок теперь содержит только то, что задаёт конкретный эксперимент. Удалены: alias `compute_cls_embeddings`, `_is_unserializable`, ручные валидации в `train_text_classifier` для `mixed_precision`/`gradient_accumulation_steps`, самописный `_null_context` (заменён на `contextlib.nullcontext`), sys.path-инжект в notebooks (проект ставится через `pip install -e .`).
- Переход на явный pipeline в notebooks (2026-05-13): `run_baseline_experiment`/`ExperimentConfig`/`TrainingConfig` удалены. Их место заняли два минимальных helper'а в `src/text_distillation/experiments.py`: `load_baseline_data(...)` собирает `BaselineData` (dataset_info / train_pool / eval_dataset / label_names / num_labels), `save_baseline_run(...)` атомарно пишет `config.json` + `metrics.json` + `distilled_dataset/`. Между ними notebook явно показывает шаги: селекция методом `select_*`, обучение через `train_text_classifier`. T4-дефолты подняты прямо в `train_text_classifier`. Селекция теперь выполняется один раз на (dataset), а не на (dataset × model) — ускорение запусков. `TimingTracker.merge(other)` позволяет копировать тайминги селекции в per-model tracker.
- Явная инициализация модели (2026-05-13): `train_text_classifier` теперь принимает pre-loaded `model` и `tokenizer` вместо `model_name`/`num_labels`/`label_names`. Notebooks отдельным шагом вызывают `load_tokenizer(model_name)` + `load_sequence_classifier(model_name, num_labels=..., label_names=...)` — pipeline получает явный «3. Model» step, между селекцией и обучением. Видно какой токенизатор/архитектура/classifier head используются; можно вставить debugging (заморозка слоёв, инспекция config).
- Объединение в один notebook (2026-05-13): `01_*..06_*` склеены в один `notebooks/baselines.ipynb`. Общие параметры (DATASET_NAMES, MODEL_NAMES, SEED, K_PER_CLASS, EMBEDDING_MODEL_NAME) задаются один раз. Сверху — видимый helper `run_all_models(...)` с explicit `tokenizer/model/train/save` шагами. Каждая baseline-секция — markdown-заголовок + единственный блок с селекцией и циклом по датасетам. В конце — `collect_runs(RUNS_DIR)` для агрегации результатов.
- DeBERTa v3 + fp16 fix (2026-05-14): при `mixed_precision="auto"` `train_text_classifier` автоматически отключает AMP для моделей с `config.model_type=="deberta-v2"` (так report-ится DeBERTa v3). Причина — после `from_pretrained` часть тензоров остаётся в fp16, и `torch.amp.GradScaler.unscale_` срывается на их градиентах с `ValueError: Attempting to unscale FP16 gradients.` В реестре `microsoft/deberta-v3-base` теперь имеет `supports_fp16=False` для документирования; runtime-логика в `training.py` работает по `model.config.model_type`.
- Small-K training schedule (2026-05-14): `K_PER_CLASS=20` в `baselines.ipynb` для прямого сравнения с DiLM Table 1. Добавлены `NUM_TRAIN_EPOCHS=20` и `TRAIN_BATCH_SIZE=16` — без них на DPC=20 модель получает 3 шага SGD за всё обучение и не сходится. Helper `run_all_models` принимает эти параметры аргументами: full-data использует дефолты (3 эпохи / batch 64), distillation-методы — small-K schedule. В `train_text_classifier` добавлен 10% warmup (`num_warmup_steps = total_steps // 10`) — без него RoBERTa-family застревает в degenerate solution (всегда предсказывает мажоритарный класс) на маленьких subsets. Это была главная причина расхождения с DiLM-таблицей (плюс multi-seed averaging, ещё не реализован).

## Что нужно сделать дальше

## Ближайшая цель: реализовать все baseline-эксперименты

Нужно довести baseline matrix до полного минимального набора:

1. Full-data baseline.
2. Random Coreset.
3. Stratified Random Coreset.
4. K-Center over TF-IDF.
5. K-Center over BERT `[CLS]` embeddings.
6. Transfer evaluation между архитектурами.
7. Results analysis.

Текущий статус:

| Baseline | Код метода | Notebook | Smoke-check | Полный запуск |
|---|---:|---:|---:|---:|
| Full-data baseline | done | done | partial | todo |
| Random Coreset | done | done | done via tests | todo |
| Stratified Random Coreset | done | done | done via tests | todo |
| K-Center TF-IDF | done | done | done via tests | todo |
| K-Center BERT CLS | done | done | done | todo |
| Transfer evaluation | partial primitives | todo | todo | todo |
| Results analysis | saving primitives | todo | todo | todo |

Baseline notebooks `01-05` now run `ag_news`, `sst2`, `qqp`, and `mnli-m` sequentially by default.

## Baseline notebooks to add

### `02_random_coreset_baseline.ipynb`

Goal: выбрать `K_TOTAL` случайных примеров из train split без стратификации.

Статус: добавлен.

Осталось:

- сравнивать с stratified random при том же размере `K`.

### `03_stratified_random_baseline.ipynb`

Текущий notebook уже есть, но если добавляем pure random как `02`, лучше переименовать текущий stratified notebook в `03`.

Статус: добавлен ранее, нумерация согласована.

Осталось:

- убедиться, что `K_PER_CLASS` и итоговый `K` явно сохраняются в config;
- добавить короткий markdown-вывод в конце notebook.

### `04_kcenter_tfidf_baseline.ipynb`

Goal: K-Center Greedy поверх TF-IDF признаков.

Статус: добавлен.

Осталось:

- прогнать tiny smoke-check;
- прогнать полный эксперимент.

### `05_kcenter_embedding_baseline.ipynb`

Текущий notebook перенесен на финальное имя `05_kcenter_embedding_baseline.ipynb`.

Нужно:

- проверить работу на `bert-base-uncased`;
- добавить параметры:
  - `EMBEDDING_MODEL_NAME`;
  - `EMBEDDING_BATCH_SIZE`;
  - `MAX_TRAIN_POOL_SAMPLES` для smoke-check;
- сохранить время построения embeddings и отбора.

### `06_transfer_evaluation.ipynb`

Goal: проверить переносимость distilled datasets между архитектурами.

Минимальная сетка:

- source dataset:
  - stratified random;
  - k-center TF-IDF;
  - k-center BERT CLS.
- target models:
  - `bert-base-uncased`;
  - `roberta-base`;
  - возможно `albert-base-v2`;
  - DeBERTa добавить позже, если окружение стабильно.

Нужно:

- загрузить сохраненный distilled dataset из `artifacts/runs/...`;
- обучить target model;
- сохранить metrics;
- собрать таблицу transfer results.

### `07_results_analysis.ipynb`

Goal: собрать результаты в таблицы и графики.

Нужно:

- прочитать `artifacts/runs/*/config.json` и `metrics.json`;
- построить таблицу:
  - method;
  - model;
  - K;
  - compression ratio;
  - accuracy;
  - f1_macro;
  - training time;
  - distillation time.
- построить графики:
  - Accuracy vs K;
  - F1-Macro vs K;
  - Performance Gap vs K;
  - метод vs время построения датасета.

## Экспериментальная сетка

### Dataset

Поддержано в коде:

- AG News.
- SST-2.
- QQP.
- MNLI-m.

Позже можно добавить:

- DBpedia;
- Yahoo Answers;
- IMDB, если нужна binary classification задача.

### Model architectures

Первая версия:

- `bert-base-uncased`.

Для transfer:

- `roberta-base`;
- `albert-base-v2`;
- `microsoft/deberta-v3-base`, если зависимости и tokenizer стабильно работают.

### Dataset sizes

Для AG News удобно задавать размер через `K_PER_CLASS`.

Рекомендуемая первая сетка:

- `K_PER_CLASS = 1`;
- `K_PER_CLASS = 5`;
- `K_PER_CLASS = 10`;
- `K_PER_CLASS = 50`;
- `K_PER_CLASS = 100`.

Итоговый размер для AG News:

```text
K_TOTAL = K_PER_CLASS * 4
```

## Метрики

Основные:

- Accuracy;
- F1-Macro.

Относительные:

- Distillation Ratio: `N / K`;
- Performance Gap: `S_full - S_distilled`;
- Relative Efficiency: `S_method / S_random_same_K`;
- Transferability Loss: `S_source_arch - S_target_arch`.

Вычислительные:

- training time;
- distillation / selection time;
- embedding computation time;
- device;
- model name;
- seed.

## Методы, которые стоит добавить после baseline

### Embedding-level dataset distillation

Идея из работы Maekawa et al. про distillation with attention labels: дистиллировать не тексты, а непрерывные embeddings.

Нужно отдельно решить:

- как хранить synthetic embeddings;
- как обучать модель на `inputs_embeds`;
- как оценивать переносимость, если embeddings завязаны на конкретную архитектуру;
- какие ограничения честно описывать в отчете.

### Gradient Matching

Цель: оптимизировать synthetic embeddings так, чтобы градиенты на synthetic set приближали градиенты на real mini-batch.

Нужно:

- внутренний цикл обновления модели;
- внешний цикл обновления synthetic embeddings;
- критерий gradient matching;
- регуляризация embeddings;
- сохранение synthetic tensors и labels.

### Trajectory Matching

Цель: согласовывать не один gradient step, а короткие участки траектории обучения.

Это более сложный этап, его стоит делать только после стабильных baseline и Gradient Matching.

### DiLM-style text-level distillation

В корне уже есть `DiLM-implementation` как reference.

Нужно:

- изучить, какие части можно переиспользовать;
- не переносить Hydra/сложные runners в основной research-first код без необходимости;
- сначала сделать wrapper notebook для запуска/оценки, потом решать вопрос интеграции.

## План работ

### Этап 1. Завершить baseline layer

- Добавить `02_random_coreset_baseline.ipynb`.
- Переименовать/согласовать номера notebooks.
- Добавить `04_kcenter_tfidf_baseline.ipynb`.
- Довести `05_kcenter_embedding_baseline.ipynb`.
- Добавить сохранение времени экспериментов.
- Запустить smoke-check для всех baseline на tiny subsets.

### Этап 2. Полные baseline-запуски на AG News

- Запустить full-data baseline.
- Запустить random baseline для сетки `K_PER_CLASS`.
- Запустить stratified random baseline для сетки `K_PER_CLASS`.
- Запустить TF-IDF k-center baseline для сетки `K_PER_CLASS`.
- Запустить BERT CLS k-center baseline для сетки `K_PER_CLASS`.
- Сохранить все результаты в `artifacts/runs/`.

### Этап 3. Анализ baseline-результатов

- Собрать таблицу результатов.
- Посчитать compression ratio.
- Посчитать performance gap относительно full-data baseline.
- Посчитать relative efficiency относительно random baseline.
- Построить графики.

### Этап 4. Transfer evaluation

- Выбрать сохраненные distilled datasets.
- Обучить разные target architectures на одних и тех же distilled datasets.
- Посчитать transferability loss.
- Сформировать матрицу переносимости.

### Этап 5. Реализация embedding-level distillation

- Спроектировать формат synthetic embeddings.
- Реализовать обучение classifier head / model на `inputs_embeds`.
- Реализовать gradient matching baseline.
- Сравнить с coreset methods.

### Этап 6. Итоговый отчет

- Обновить раздел постановки задачи.
- Добавить таблицы baseline.
- Добавить графики scaling laws.
- Добавить выводы о переносимости.
- Описать ограничения:
  - discrete text space;
  - architecture-specific embeddings;
  - cost of embedding selection;
  - sensitivity to seeds and training budget.

## Definition of Done для первой рабочей версии

Первая рабочая версия считается готовой, когда:

- все baseline notebooks существуют и запускаются на tiny mode;
- все baseline methods используют функции из `src/`, а не локальные реализации в notebooks;
- для AG News есть результаты хотя бы для `K_PER_CLASS in {1, 10, 50, 100}`;
- есть full-data baseline для `bert-base-uncased`;
- есть таблица сравнения Accuracy/F1-Macro;
- есть график качества от `K`;
- результаты сохраняются в `artifacts/runs/`;
- `pytest -q` проходит.
