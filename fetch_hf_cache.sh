#!/bin/bash
# Скачивает hf_cache (разбитый на части) из GitHub-релиза hf-cache-v1,
# склеивает и распаковывает в корень проекта.
#
#   bash fetch_hf_cache.sh
#
# Требует: curl, tar, python3. Токен НЕ нужен (релиз публичный).
set -eu

REPO="PerryThePlatipuse/momo_project"
TAG="hf-cache-v1"
WORK="hf_parts_dl"

cd "$(dirname "$0")"
mkdir -p "$WORK"

echo "Получаю список частей релиза $TAG ..."
# hfc_ = базовый набор (bert-base, roberta-base, albert, gpt2 + ag_news, sst2)
# hfx_ = доп. модели/датасеты (bert-large, roberta-large, deberta, xlnet + mnli, qqp)
# BASE_ONLY=1 — качать только базовый набор (для слабого интернета / к дедлайну).
PREFIXES="('hfc_','hfx_')"
[ "${BASE_ONLY:-0}" = "1" ] && PREFIXES="('hfc_',)" && echo "режим BASE_ONLY: только базовый набор"
curl -fsSL "https://api.github.com/repos/$REPO/releases/tags/$TAG" -o /tmp/rel.json
parts=$(python3 -c "import json;d=json.load(open('/tmp/rel.json'));print('\n'.join(sorted(a['name'] for a in d['assets'] if a['name'].startswith($PREFIXES))))")

n=$(echo "$parts" | grep -c . || true)
echo "Частей: $n"

i=0
for name in $parts; do
  i=$((i+1))
  out="$WORK/$name"
  if [ -f "$out" ]; then echo "[$i/$n] $name уже скачан"; continue; fi
  echo "[$i/$n] качаю $name ..."
  # до 20 попыток на часть (на случай нестабильной сети)
  k=0
  until curl -fsSL --speed-limit 2000 --speed-time 30 \
      "https://github.com/$REPO/releases/download/$TAG/$name" -o "$out"; do
    k=$((k+1)); echo "   retry $k"; [ $k -ge 20 ] && { echo "СДАЛСЯ на $name"; exit 1; }; sleep 5
  done
done

echo "Склеиваю и распаковываю в ./ ..."
cat "$WORK"/hfc_* | tar -xzf -                      # базовый набор
ls "$WORK"/hfx_* >/dev/null 2>&1 && cat "$WORK"/hfx_* | tar -xzf -   # доп. модели/датасеты

echo "Готово. Проверка:"
ls hf_cache/hub/ 2>/dev/null && echo "OK: hf_cache на месте" || echo "ВНИМАНИЕ: hf_cache не найден"
echo "Можно удалить временную папку: rm -rf $WORK"
