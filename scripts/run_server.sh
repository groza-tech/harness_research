#!/usr/bin/env bash
# Полный протокол на счётном сервере. Запускается ВНУТРИ tmux и переживает
# обрыв ssh — именно на этом уже погиб один пробный прогон: процесс висел на
# управляющем терминале и умер вместе с ним.
#
# Порядок этапов — не вкусовщина, а зависимость по данным (§4.3–4.4):
#   Э2 скрининг  → даёт список выживших компонентов
#   Э3 градиент  → идёт ПО ЭТИМ выжившим, из свежего Э2, а не из прошлого прогона
#   Э1 лестница  → разложение дисперсии «модель против харнесса»
#   Э5 рынок     → H4a против H4b
#   Э4 курс обмена — отдельно: это две трети счёта и две трети времени.
#
# Каждый этап пишет чекпоинты после каждой сессии. Повторный запуск той же
# команды догоняет недостающее и не платит за уже сделанное — оборвать прогон
# безопасно в любой момент.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGES="${STAGES:-E2 E3 E1 E5}"
WORKERS="${WORKERS:-24}"
CFG="${CFG:-configs/main.yaml}"
OUT="${OUT:-outputs/server_run}"
REP="${REP:-reports/server_run}"
AGENTS="${AGENTS:-24}"
PERIODS="${PERIODS:-8}"
MODEL_CLASSES="${MODEL_CLASSES:-}"
LOG_DIR="${LOG_DIR:-logs}"
STATUS="${OUT}/STATUS.txt"

mkdir -p "$LOG_DIR" "$OUT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PY=("${ROOT}/.venv/bin/python" -m harness_asymmetry.cli)

say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/run_server.log"; }

{
  echo "запуск:  $(date '+%F %T')"
  echo "этапы:   ${STAGES}"
  echo "воркеры: ${WORKERS}"
  echo "конфиг:  ${CFG}"
} > "$STATUS"

say "=== СТАРТ === этапы: ${STAGES}, воркеров: ${WORKERS}"

for STAGE in ${STAGES}; do
  STAGE_ARGS=()
  [[ -n "$MODEL_CLASSES" ]] && STAGE_ARGS+=(--model-classes $MODEL_CLASSES)
  # Выжившие берутся из Э2 ЭТОГО прогона: n=256 может отобрать не тот набор,
  # что пилотные 40 повторов прошлой ночи, и подставлять старый список значило
  # бы считать градиент по компонентам, которые в свежем скрининге не прошли.
  if [[ "$STAGE" == "E3" ]]; then
    if [[ -d "${OUT}/E2/sessions" ]]; then
      STAGE_ARGS+=(--survivors-from "${OUT}/E2")
    else
      say "!! Э3 пропущен: нет ${OUT}/E2 — сначала должен пройти скрининг"
      continue
    fi
  fi
  # Э5 — популяционный этап, у него свои параметры вместо весовых классов.
  if [[ "$STAGE" == "E5" ]]; then
    STAGE_ARGS=(--agents "$AGENTS" --periods "$PERIODS")
    [[ -n "$MODEL_CLASSES" ]] && STAGE_ARGS+=(--model-classes $MODEL_CLASSES)
  fi

  say "--- этап ${STAGE}: старт"
  echo "этап ${STAGE}: идёт с $(date '+%F %T')" >> "$STATUS"
  "${PY[@]}" run \
    --log-level warning \
    --config "$CFG" \
    --experiment "$STAGE" \
    --provider openrouter \
    ${STAGE_ARGS[@]+"${STAGE_ARGS[@]}"} \
    --workers "$WORKERS" \
    --output "${OUT}/${STAGE}" \
    --report "${REP}/${STAGE}" \
    --no-progress >> "${LOG_DIR}/${STAGE}.log" 2>&1
  RC=$?
  if [[ $RC -ne 0 ]]; then
    say "!! этап ${STAGE} упал с кодом ${RC} — иду дальше, догонится повторным запуском"
    echo "этап ${STAGE}: УПАЛ (код ${RC}) в $(date '+%F %T')" >> "$STATUS"
  else
    say "--- этап ${STAGE}: готов"
    echo "этап ${STAGE}: готов $(date '+%F %T')" >> "$STATUS"
  fi
done

say "=== ФИНИШ ==="
{
  echo
  echo "финиш: $(date '+%F %T')"
} >> "$STATUS"

# Итоговая сводка здоровья и денег — то, с чем утром сверяются в первую очередь.
"${ROOT}/.venv/bin/python" - "$OUT" >> "$STATUS" 2>&1 <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
total_cost = 0.0
print("\n=== ЗДОРОВЬЕ ПРОГОНА ===")
for meta_path in sorted(root.glob("*/run_meta.json")):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stage = meta_path.parent.name
    total = meta.get("sessions_total", 0)
    failed = meta.get("sessions_failed", 0)
    rate = 100 * failed / total if total else 0.0
    print(f"[{stage}] сессий {total}, сбоев {failed} ({rate:.1f}%), {meta.get('elapsed_s', 0):.0f} с")
    for model, h in (meta.get("llm_health") or {}).items():
        cost = h.get("estimated_cost_usd", 0) or 0
        total_cost += cost
        print(f"    {model}: ok={h.get('calls_ok')} fail={h.get('calls_failed')} "
              f"пустых={h.get('empty_responses', 0)} ≈${cost}")
    print("    " + ("❌ сбоев >10%: часть выборки потеряна, догоните этап"
                    if rate > 10 else "✅ выборка чистая"))
print(f"\nИТОГО потрачено ≈ ${total_cost:.2f}")
PY

cat "$STATUS"
