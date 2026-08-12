#!/usr/bin/env bash
# Пилот §4.4: две крайние ячейки на реальной модели.
#
# Задача пилота — не результат, а оценка фактической дисперсии φ^A и
# пересчёт требуемого n. Смотреть раздел «Пилот и мощность» в RUN_REPORT.md.
# Это точка невозврата (неделя 6): если требуемое n непосильно, менять
# сеттинг — упрощать сценарии, снижать T, — а не гнать основной прогон.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Через PYTHONPATH, а не через установленный скрипт: так работает и на свежем
# клоне без `pip install -e .`.
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PY=("${ROOT}/.venv/bin/python" -m harness_asymmetry.cli)
RUN_DIR="${1:-${ROOT}/outputs/pilot}"
REPORT_DIR="${2:-${ROOT}/reports/pilot}"

if [[ ! -f "${ROOT}/.env" ]]; then
  echo "Нет .env — скопируйте .env.example и пропишите OPENROUTER_API_KEY." >&2
  exit 1
fi

"${PY[@]}" doctor --config "${ROOT}/configs/pilot.yaml" --ping

# Каталог не чистим: повторный запуск догоняет недостающие сессии по
# чекпоинтам и не платит за уже сделанное.
"${PY[@]}" run \
  --config "${ROOT}/configs/pilot.yaml" \
  --experiment pilot \
  --provider openrouter \
  --output "${RUN_DIR}" \
  --report "${REPORT_DIR}"

echo
echo "Требуемое n — в ${REPORT_DIR}/RUN_REPORT.md, раздел «Пилот и мощность»."
