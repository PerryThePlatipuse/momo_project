#!/bin/bash
# Скачивает hf_cache (разбитый на части) из GitHub-релиза hf-cache-v1,
# склеивает и распаковывает в корень проекта.
#
#   bash fetch_hf_cache.sh              # базовый + доп. набор
#   BASE_ONLY=1 bash fetch_hf_cache.sh  # только базовый (1.3 ГБ)
#
# Требует: curl, tar, python3. Токен НЕ нужен (релиз публичный).
# Качает части напрямую с github.com (как git clone) — не зависит от api.github.com.
set -u

REPO="PerryThePlatipuse/momo_project"
TAG="hf-cache-v1"
WORK="hf_parts_dl"
DL="https://github.com/$REPO/releases/download/$TAG"

cd "$(dirname "$0")"
mkdir -p "$WORK"

# Скачать один URL в файл. Возвращает HTTP-код. Ретраи на сетевые/5xx; 404 = стоп.
fetch_one() {  # $1=url  $2=out
  local k=0 code
  while :; do
    code=$(curl -sL --speed-limit 2000 --speed-time 30 -o "$2" -w "%{http_code}" "$1" || echo 000)
    case "$code" in
      200) return 0 ;;
      404) return 44 ;;                       # нет такого ассета — конец последовательности
      *)   k=$((k+1)); [ $k -ge 25 ] && return 1
           echo "   код $code, retry $k"; sleep 5 ;;
    esac
  done
}

# Перебор суффиксов split: aa, ab, ..., az, ba, ... — качаем до первого 404.
download_prefix() {  # $1=prefix (hfc_ | hfx_)
  local pref="$1" got=0
  for a in {a..z}; do for b in {a..z}; do
    local name="${pref}${a}${b}" out="$WORK/${pref}${a}${b}"
    if [ -s "$out" ]; then got=$((got+1)); continue; fi
    fetch_one "$DL/$name" "$out"; rc=$?
    if [ $rc -eq 0 ]; then got=$((got+1)); echo "  $name ok ($got)";
    elif [ $rc -eq 44 ]; then rm -f "$out"; echo "  $pref: всего $got частей"; return 0;
    else echo "  СДАЛСЯ на $name"; return 1; fi
  done; done
}

echo "Качаю базовый набор (hfc_) ..."
download_prefix "hfc_" || exit 1

if [ "${BASE_ONLY:-0}" != "1" ]; then
  echo "Качаю доп. набор (hfx_) ..."
  download_prefix "hfx_" || exit 1
fi

echo "Склеиваю и распаковываю в ./ ..."
cat "$WORK"/hfc_* | tar -xzf -                                   # базовый набор
ls "$WORK"/hfx_* >/dev/null 2>&1 && cat "$WORK"/hfx_* | tar -xzf -   # доп. набор

# Удаляем битые симлинки (мёртвые tflite/onnx/lfs-указатели) — ломают копирование папки
echo "Чищу битые симлинки ..."
find hf_cache -type l ! -exec test -e {} \; -delete 2>/dev/null || true

echo "Готово. Проверка:"
ls hf_cache/hub/ 2>/dev/null && echo "OK: hf_cache на месте" || echo "ВНИМАНИЕ: hf_cache не найден"
echo "Можно удалить временную папку: rm -rf $WORK"
