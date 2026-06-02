# HANDOFF — что доделать (для Павла / его Claude Code)

Этот файл — самодостаточная вводная. Можно показать Claude Code: в нём контекст,
две задачи, точные команды, конфиги и подводные камни (которые уже всплыли и решены).

## Контекст

Проект: дистилляция текстовых датасетов (курс OptML, НИУ ВШЭ).
Сравниваем coreset-методы (`random`, `k_centers`, `herding`) и DiLM на AG News / SST-2 (и MNLI/QQP).

Ветка **`timoxa`**, открыт **PR #2 → `master`**.

Что уже сделано (Тимофей, в этом PR):
- `random` + `k_centers` scaling laws: **AG News + SST-2, bert-base + roberta-base, DPC 1–1000, 3 повтора**.
  Отчёт и графики: `scaling_laws_coreset_report/`, числа: `results/scaling_laws/scaling_laws_all_methods.csv`.
- `herding` — берётся из соседнего `scaling_laws_herding_report/` (твой прошлый прогон), мёржится в графики.
- DiLM на AG News: **LM-претрейн (40k шагов) завершён**, стадия gradient matching (DC) упала по OOM.

Вывод по scaling: `k_centers ≈ herding ≫ random` на малых бюджетах, к DPC≥500 все сходятся (~0.91).

---

## Задача 1 — полный прогон scaling laws (7 моделей × 4 задачи)

Сейчас посчитаны только 3 модели × 2 задачи. Нужно добить до полной сетки, как в herding-отчёте:

- Методы: `random`, `k_centers` (herding уже есть — **не дублировать**).
- Модели: `bert-base-uncased, bert-large-uncased, roberta-base, roberta-large, microsoft/deberta-v3-base, albert-base-v2, xlnet-base-cased`.
- Задачи: `ag_news, sst2, mnli, qqp`.
- DPC: `1, 5, 10, 20, 50, 100, 500, 1000`, по 3 повтора.

Конфиг уже прописан в `notebooks/scaling_laws.py` (секция `# Config`). Скрипт сам:
- берёт только модели/задачи, которые реально доступны (есть в `hf_cache/` или качаются онлайн);
- пропускает уже посчитанные точки (`SKIP_EXISTING` — по наличию `summary.json`), так что **прогон можно прерывать и продолжать**;
- в конце строит графики и мёржит herding из `scaling_laws_herding_report/raw_results.csv`.

## Задача 2 — добить DiLM на AG News

LM-стадия готова (чекпоинты в `results/ag_news/dilm/dpc10_lm40000_dc10000_seed42/lm/`).
Стадия DC (gradient matching) упала с **CUDA OOM** — на сервере была общая карта, память заняли чужие процессы.

Что сделать:
- запустить `notebooks/dilm_ag_news.py` на **свободной** A100 (нужно ~40–60 ГБ свободной памяти под bi-level grad);
- если снова OOM — снизить в `dilm_ag_news.py`: `GM_REAL_DPC` (100→50), `GM_SYN_DPC` (64→32), `train_batch_size` (128→64), и выставить `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- скрипт сам сделает generate + evaluate + сохранит `summary.json` и дистиллированный датасет.

---

## Среда

Зависимости — `requirements.txt`. Важно про версии:
- Код использует `attn_implementation="eager"` (в `src/learner.py`, `src/generator.py`) — **требует transformers ≥ 4.36**. Если у тебя ровно 4.30 из requirements и падает на этом kwarg — обнови transformers (на сервере у нас был ~4.4x).
- Для токенизаторов **deberta-v3-base и xlnet** нужен `sentencepiece` (поставь `pip install sentencepiece`); для deberta на новых transformers может понадобиться `tiktoken`.

## Вариант А — онлайн (проще всего, раз у тебя интернет)

Не создавай `hf_cache/` — тогда скрипты работают в обычном режиме и transformers сам качает модели:

```bash
git clone https://github.com/PerryThePlatipuse/momo_project.git
cd momo_project && git checkout timoxa
pip install -r requirements.txt sentencepiece          # +tiktoken при необходимости

mkdir -p results
CUDA_VISIBLE_DEVICES=0 python -u notebooks/scaling_laws.py 2>&1 | tee results/scaling.log &
CUDA_VISIBLE_DEVICES=1 python -u notebooks/dilm_ag_news.py 2>&1 | tee results/dilm.log &
wait
```

> Замечание: `scaling_laws.py` включает офлайн-режим, ТОЛЬКО если рядом есть папка `hf_cache/`.
> Без неё — онлайн, модели тянутся из интернета. Это для тебя оптимально.

## Вариант Б — офлайн из GitHub-релиза (если интернета к HF нет)

Все ассеты (7 моделей + 4 датасета, ~5.2 ГБ) лежат в релизе `hf-cache-v1` (216 частей):
```bash
bash fetch_hf_cache.sh        # качает все части, склеивает, распаковывает hf_cache/, чистит битые симлинки
```
Дальше так же запускаешь `scaling_laws.py` / `dilm_ag_news.py` (они увидят `hf_cache/` и пойдут офлайн).

---

## Подводные камни (уже решены в коде ветки `timoxa`, но чтобы понимать)

1. **`attn_implementation`**: исходно был `"sdpa"` — на нашей версии transformers gpt2/bert/roberta/albert это не поддерживают → меняли на `"eager"`. (Требует transformers ≥4.36.)
2. **Метрики офлайн**: `evaluate.load("accuracy"/"glue")` лезет в HF Hub. Переписано на локальный `sklearn` (`src/evaluator.py`, класс `Metric`) — accuracy + f1/combined_score.
3. **OpenBLAS на многоядерном сервере**: падал при >128 потоков. В `run_all.sh` и в начале скриптов выставлен лимит `OPENBLAS_NUM_THREADS=32` и т.п. (до импорта numpy).
4. **gradient checkpointing**: albert и xlnet его не поддерживают → в `src/learner.py` обёрнуто в try/except (раньше падало с ValueError).
5. **eval-сплит**: `DataModule` называет валидационный сплит `'validation'` (даже для ag_news, где исходно `test`). В коде это учтено.
6. **DiLM DC OOM**: см. Задачу 2 — нужен свободный GPU.

## Проверка результатов и заливка в PR

```bash
# scaling: должна расти таблица
cat results/scaling_laws/scaling_laws_all_methods.csv

# DiLM: должен появиться summary.json
cat results/ag_news/dilm/*/summary.json

# залить в PR #2 (results/ в .gitignore → через -f, и НЕ коммить тяжёлые чекпоинты lm/dc)
git add -f results/scaling_laws/scaling_laws_all_methods.csv results/scaling_laws/*.png
find results -name summary.json -exec git add -f {} +
# отчёт обнови, если хочешь (scaling_laws_coreset_report/)
git commit -m "Full scaling laws (7 models x 4 tasks) + DiLM AG News result"
git push origin timoxa
```

После заливки PR #2 обновится — останется смёржить в `master`.

> ⚠️ Не запускать на банковской инфраструктуре (комплаенс). Личное железо / облако / твоя A100.
