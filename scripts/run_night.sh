#!/usr/bin/env bash
# Ночной прогон: пилот → скрининг компонентов (Э2) → симметричная база (Э1).
#
# Порядок не случаен и соответствует §4.4–4.3 дизайн-дока:
#   1. ПИЛОТ — оценить фактическую дисперсию φ^A и пересчитать требуемое n.
#      Это точка невозврата: если утром окажется, что n непосилен, надо менять
#      сеттинг, а не гнать основной эксперимент в надежде.
#   2. Э2 — скрининг Плакетта–Бёрмана: какие компоненты вообще работают.
#      Даёт список выживших для Э3, который запускается уже осознанно.
#   3. Э1 — симметричная лестница на трёх весовых классах: насыщение (H5)
#      и разложение дисперсии «модель против харнесса» (§6.2, главный слайд).
#
# Каждый этап пишет чекпоинты и собирает отчёты сразу после себя. Прогон
# прерываемый: повторный запуск той же команды догоняет недостающее и не
# платит за уже сделанное.
#
# Запуск через ``python -m``, а не через entry_point .venv/bin/harness-asymmetry:
# на macOS .pth-файлам pip-editable проставляется флаг hidden, и site.py их
# пропускает. PYTHONPATH обходит это гарантированно (та же засада описана в
# соседнем проекте mas_managerial_hypothesis_verifying).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y%m%d)}"
OUT="${OUT:-outputs/night_${STAMP}}"
REP="${REP:-reports/night_${STAMP}}"
WORKERS="${WORKERS:-24}"
REPEATS="${REPEATS:-40}"
STAGES="${STAGES:-pilot E2 E1}"

if [[ ! -f .env ]]; then
  echo "Нет .env — скопируйте .env.example и пропишите OPENROUTER_API_KEY." >&2
  exit 1
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PY=("${ROOT}/.venv/bin/python" -m harness_asymmetry.cli)

# Подстраховка от macOS-флага hidden на .pth (если кто-то захочет entry_point).
find .venv/lib/python3.11/site-packages -maxdepth 1 -name "*.pth" \
  -exec chflags nohidden {} + 2>/dev/null || true

# --- Сон машины ------------------------------------------------------------
# caffeinate: -d дисплей, -i idle, -m диск, -s система (только от сети).
# Главное: caffeinate НЕ блокирует clamshell sleep — аппаратный сон при
# закрытой крышке. Чтобы пережить закрытую крышку, нужно либо держать мак на
# зарядке во внешнем мониторе, либо SUDO_DISABLE_SLEEP=1, либо не закрывать.
if command -v caffeinate >/dev/null 2>&1; then
  WRAPPER=(caffeinate -dims)
else
  WRAPPER=()
fi

if command -v pmset >/dev/null 2>&1 && pmset -g ps | head -1 | grep -qi "battery"; then
  cat <<'WARN'
⚠️  Мак на БАТАРЕЕ. При закрытой крышке система уснёт — caffeinate тут бессилен.
   Поставьте на зарядку либо не закрывайте крышку, иначе прогон растянется.
WARN
fi

if [[ "${SUDO_DISABLE_SLEEP:-}" == "1" ]]; then
  echo "[night] отключаю sleep через pmset (нужен sudo)…"
  sudo pmset -a disablesleep 1
  trap 'sudo pmset -a disablesleep 0; echo "[night] вернул pmset disablesleep=0"' EXIT
fi

echo "=================================================================="
echo "  НОЧНОЙ ПРОГОН"
echo "  этапы:   ${STAGES}"
echo "  повторов на ячейку: ${REPEATS}, параллельно ячеек: ${WORKERS}"
echo "  output:  ${OUT}"
echo "  старт:   $(date '+%F %T')"
echo "=================================================================="

"${PY[@]}" doctor --ping

for STAGE in ${STAGES}; do
  # У пилота свой конфиг и свои повторы: n_scenarios=32 там подобрано под
  # repeats=32. Навязать ему 40 повторов значило бы пустить сценарии по
  # второму кругу и сломать общие случайные числа внутри ячейки.
  case "$STAGE" in
    pilot) CFG=configs/pilot.yaml;   REPEATS_ARG=() ;;
    *)     CFG=configs/default.yaml; REPEATS_ARG=(--repeats "$REPEATS") ;;
  esac
  echo
  echo "------------------------------------------------------------------"
  echo "[night] этап ${STAGE} — старт $(date '+%F %T')"
  echo "------------------------------------------------------------------"
  # Каждый этап в своём каталоге: чекпоинты и манифесты не перемешиваются,
  # а упавший этап можно догнать, не трогая соседние.
  "${WRAPPER[@]}" "${PY[@]}" run \
    --log-level warning \
    --config "$CFG" \
    --experiment "$STAGE" \
    --provider openrouter \
    "${REPEATS_ARG[@]}" \
    --workers "$WORKERS" \
    --output "${OUT}/${STAGE}" \
    --report "${REP}/${STAGE}" \
    --no-progress || echo "[night] этап ${STAGE} завершился с ошибкой — иду дальше"
  echo "[night] этап ${STAGE} — финиш $(date '+%F %T')"
done

echo
echo "=================================================================="
echo "  ЗДОРОВЬЕ ПРОГОНА (годен ли результат)"
"${ROOT}/.venv/bin/python" - "$OUT" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
for meta_path in sorted(root.glob("*/run_meta.json")):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stage = meta_path.parent.name
    total = meta.get("sessions_total", 0)
    failed = meta.get("sessions_failed", 0)
    rate = 100 * failed / total if total else 0.0
    print(f"  [{stage}] сессий {total}, сбоев {failed} ({rate:.1f}%), "
          f"{meta.get('elapsed_s', 0):.0f} с")
    for model, h in (meta.get("llm_health") or {}).items():
        print(f"      {model}: ok={h.get('calls_ok')} fail={h.get('calls_failed')} "
              f"пустых={h.get('empty_responses', 0)} "
              f"failure_rate={h.get('failure_rate_pct')}% "
              f"кеш={h.get('cache_hit_rate_pct')}% "
              f"≈${h.get('estimated_cost_usd', 0)}")
    if rate > 10:
        print(f"      ❌ сбоев >10% — провайдер сыпался, часть выборки потеряна. "
              f"Догоните этап повторным запуском той же команды.")
    else:
        print("      ✅ выборка чистая")
PY
echo "  финиш: $(date '+%F %T')"
echo "=================================================================="
echo "  Утром смотреть:"
echo "    ${REP}/pilot/RUN_REPORT.md  — раздел «Пилот и мощность»: требуемое n"
echo "    ${REP}/E2/RESULTS.md        — выжившие компоненты для Э3"
echo "    ${REP}/E1/RESULTS.md        — разложение дисперсии, насыщение"
echo "    open ${REP}/E1/report.html"
echo "=================================================================="
