#!/usr/bin/env bash
# Смоук инфраструктуры на mock-провайдере: ни одного вызова API.
# Проверяет, что протокол, гейты, чекпоинты, метрики и все четыре отчёта
# работают end-to-end. Научной ценности не имеет — отчёты помечаются баннером.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Через PYTHONPATH, а не через установленный скрипт: так работает и на свежем
# клоне без `pip install -e .`.
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PY=("${ROOT}/.venv/bin/python" -m harness_asymmetry.cli)
RUN_DIR="${1:-${ROOT}/outputs/smoke}"
REPORT_DIR="${2:-${ROOT}/reports/smoke}"

rm -rf "${RUN_DIR}" "${REPORT_DIR}"
"${PY[@]}" run \
  --config "${ROOT}/configs/smoke.yaml" \
  --experiment all \
  --provider mock \
  --output "${RUN_DIR}" \
  --report "${REPORT_DIR}"

echo
echo "Открыть отчёт: ${REPORT_DIR}/report.html"
