#!/usr/bin/env bash
# Управление прогоном на арендованном счётном сервере.
#
#   bash scripts/remote.sh deploy          — залить код и ключ, собрать venv
#   bash scripts/remote.sh start [этапы]   — запустить прогон в tmux (переживает обрыв ssh)
#   bash scripts/remote.sh status          — сколько сделано, сколько потрачено
#   bash scripts/remote.sh logs [этап]     — хвост лога
#   bash scripts/remote.sh fetch           — забрать результаты к себе
#   bash scripts/remote.sh stop            — остановить (данные сохранятся, догонится resume)
#   bash scripts/remote.sh ssh             — просто зайти на сервер
#
# Доступ берётся из .deploy.env (он в .gitignore). Пароль нужен ровно один раз,
# при первом deploy: дальше работает ключ ~/.ssh/harness_deploy.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[[ -f .deploy.env ]] || { echo "Нет .deploy.env — там адрес сервера и доступ." >&2; exit 1; }
set -a; . ./.deploy.env; set +a
: "${REMOTE_HOST:?}" "${REMOTE_USER:?}" "${REMOTE_DIR:=/opt/harness_research}"

# Каталог прогона: разные варианты счёта не должны смешиваться в одном.
# Resume опознаёт сессии по session_id, а модель в него не входит — залить
# лёгкий прогон поверх тяжёлого значило бы получить ячейки из двух моделей.
OUT="${OUT:-outputs/server_run}"
REP="${REP:-reports/$(basename "${OUT:-server_run}")}"

KEY="${HOME}/.ssh/harness_deploy"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
          -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ConnectTimeout=25)
TMUX_SESSION="harness"

rsh() { ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "$@"; }

# Ключ ставится один раз через expect: пароль не оседает ни в истории, ни в логах.
install_key() {
  [[ -f "$KEY" ]] || ssh-keygen -t ed25519 -N '' -C 'harness-asymmetry-deploy' -f "$KEY" -q
  if rsh -o BatchMode=yes true 2>/dev/null; then echo "ключ уже стоит"; return; fi
  : "${REMOTE_PASSWORD:?Нужен REMOTE_PASSWORD в .deploy.env для первичной установки ключа}"
  PUB="$(cat "${KEY}.pub")" expect -c '
    set timeout 60
    spawn ssh -o StrictHostKeyChecking=no -o PubkeyAuthentication=no \
      $env(REMOTE_USER)@$env(REMOTE_HOST) \
      "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && grep -qF \"$env(PUB)\" ~/.ssh/authorized_keys || echo \"$env(PUB)\" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; echo KEY_OK"
    expect { -re "(?i)password:" { send "$env(REMOTE_PASSWORD)\r"; exp_continue }
             "KEY_OK" {} timeout { exit 1 } eof {} }
    expect eof' >/dev/null
  echo "ключ установлен"
}

cmd_deploy() {
  install_key
  echo "== заливаю код =="
  rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
    --exclude '.venv/' --exclude 'outputs/' --exclude 'reports/' --exclude 'logs/' \
    --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.git/' \
    --exclude '.DS_Store' --exclude '.deploy.env' \
    ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
  echo "== заливаю ключ провайдера =="
  scp "${SSH_OPTS[@]}" -q .env "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/.env"
  rsh "chmod 600 ${REMOTE_DIR}/.env && chown -R ${REMOTE_USER}:${REMOTE_USER} ${REMOTE_DIR}"
  echo "== окружение =="
  # Зависимости — только готовыми колёсами: на одном ядре сборка scipy из
  # исходников заняла бы больше, чем весь эксперимент. Сам пакет ставится
  # отдельно и без зависимостей, иначе --only-binary отвергает и его.
  rsh "cd ${REMOTE_DIR} && { [ -x .venv/bin/python ] || python3 -m venv .venv; } && \
       .venv/bin/pip install -q --upgrade pip && \
       .venv/bin/pip install -q --only-binary=:all: openai pandas numpy scipy statsmodels matplotlib python-dotenv rich tabulate pyyaml pytest && \
       .venv/bin/pip install -q --no-deps -e . && \
       PYTHONPATH=src .venv/bin/python -m pytest -q 2>&1 | tail -2"
}

cmd_start() {
  local stages="${*:-E2 E3 E1 E5}"
  if rsh "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
    echo "Прогон уже идёт. Сначала stop, либо смотрите status." >&2; exit 1
  fi
  rsh "cd ${REMOTE_DIR} && tmux new-session -d -s ${TMUX_SESSION} \
       'STAGES=\"${stages}\" WORKERS=${WORKERS:-24} OUT=\"${OUT}\" REP=\"${REP}\" \
        MODEL_CLASSES=\"${MODEL_CLASSES:-}\" HA_MODEL=\"${HA_MODEL:-}\" \
        bash scripts/run_server.sh'"
  sleep 3
  echo "Запущено в tmux-сессии «${TMUX_SESSION}». Этапы: ${stages}"
  rsh "tmux has-session -t ${TMUX_SESSION} 2>/dev/null && echo 'сессия жива' || echo 'СЕССИЯ НЕ ПОДНЯЛАСЬ — смотрите logs'"
}

cmd_status() {
  rsh "tmux has-session -t ${TMUX_SESSION} 2>/dev/null && echo '● прогон ИДЁТ' || echo '○ прогон не запущен'"
  rsh "cd ${REMOTE_DIR} && uptime && OUT='${OUT}' .venv/bin/python - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ['OUT'])
if not root.exists():
    print('данных пока нет'); raise SystemExit
grand_n = 0; grand_cost = 0.0
for stage in sorted(p for p in root.iterdir() if p.is_dir()):
    n = tin = tout = fails = 0
    for f in (stage / 'sessions').glob('*.jsonl') if (stage / 'sessions').exists() else []:
        for line in f.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            try: s = json.loads(line)
            except json.JSONDecodeError: continue
            n += 1; tin += s.get('prompt_tokens', 0) or 0; tout += s.get('completion_tokens', 0) or 0
            fails += bool(s.get('technical_failure'))
    prices = {}
    man = stage / 'run_manifest.json'
    if man.exists():
        try: prices = (json.loads(man.read_text(encoding='utf-8')).get('prices') or {}).get('models', {})
        except json.JSONDecodeError: pass
    pin = sum(v['in'] for v in prices.values()) / len(prices) if prices else 0.14
    pout = sum(v['out'] for v in prices.values()) / len(prices) if prices else 0.28
    cost = tin / 1e6 * pin + tout / 1e6 * pout
    grand_n += n; grand_cost += cost
    print(f'{stage.name:>6}: сессий {n:>6}, сбоев {fails:>3}, ≈\${cost:.2f}')
print(f'{\"ИТОГО\":>6}: сессий {grand_n:>6}, потрачено ≈\${grand_cost:.2f}')
PY"
  rsh "tail -3 ${REMOTE_DIR}/${OUT}/STATUS.txt 2>/dev/null || true"
}

cmd_logs() { rsh "tail -n ${LINES:-40} ${REMOTE_DIR}/logs/${1:-run_server}.log"; }

cmd_fetch() {
  mkdir -p outputs reports
  rsync -az -e "ssh ${SSH_OPTS[*]}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${OUT}" outputs/
  rsync -az -e "ssh ${SSH_OPTS[*]}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${REP}" reports/
  rsync -az -e "ssh ${SSH_OPTS[*]}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/logs" "outputs/$(basename "$OUT")/" 2>/dev/null || true
  echo "забрано в outputs/$(basename "$OUT") и reports/$(basename "$REP")"
}

cmd_stop() {
  rsh "tmux kill-session -t ${TMUX_SESSION} 2>/dev/null; pkill -f harness_asymmetry.cli || true"
  echo "остановлено; сделанные сессии сохранены, повторный start догонит остальное"
}

case "${1:-}" in
  deploy) shift; cmd_deploy "$@" ;;
  start)  shift; cmd_start "$@" ;;
  status) shift; cmd_status "$@" ;;
  logs)   shift; cmd_logs "$@" ;;
  fetch)  shift; cmd_fetch "$@" ;;
  stop)   shift; cmd_stop "$@" ;;
  ssh)    shift; ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "$@" ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac
