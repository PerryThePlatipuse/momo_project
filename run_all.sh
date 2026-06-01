#!/bin/bash
# Скачивает базовый hf_cache и запускает оба обучения параллельно на 2 GPU.
# Для запуска в tmux, чтобы можно было отключиться и уйти:
#
#   tmux new -d -s momo 'bash run_all.sh'
#   tmux attach -t momo     # посмотреть прогресс
#   (Ctrl+b, затем d — отключиться, обучение продолжится)
set -u

cd "$(dirname "$0")"
mkdir -p results

echo "=== [1/3] обновляю репозиторий ==="
git pull || echo "git pull не прошёл — продолжаю с тем что есть"

echo "=== [2/3] качаю базовый hf_cache (если ещё нет) ==="
if [ ! -d hf_cache/hub ]; then
  BASE_ONLY=1 bash fetch_hf_cache.sh || { echo "не удалось скачать hf_cache"; exit 1; }
else
  echo "hf_cache уже на месте — пропускаю"
fi

echo "=== [3/3] запускаю обучение на 2 GPU ==="
CUDA_VISIBLE_DEVICES=0 python -u notebooks/scaling_laws.py  > results/scaling.log 2>&1 &
PID_S=$!
CUDA_VISIBLE_DEVICES=1 python -u notebooks/dilm_ag_news.py  > results/dilm.log    2>&1 &
PID_D=$!

echo "scaling_laws PID $PID_S  (лог: results/scaling.log)"
echo "dilm_ag_news PID $PID_D  (лог: results/dilm.log)"
echo "жду завершения обоих ..."
wait $PID_S; echo "scaling_laws завершён (код $?)"
wait $PID_D; echo "dilm_ag_news завершён (код $?)"
echo "=== ВСЁ ГОТОВО. Результаты в results/ ==="
