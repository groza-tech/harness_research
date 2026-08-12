"""Наблюдаемость харнесса: JSONL-события, трассировка, предохранители.

Это тот слой, ради которого дизайн-док §7.1 велит не брать тяжёлый фреймворк:
нужна полная наблюдаемость и возможность точечно отключать компоненты.
Инженерные решения взяты из практики harness engineering 2026:

* **Одно событие — одна строка JSON, один файл на прогон.** Отладка сводится
  к ``grep session_id | jq``, а не к внешнему вендору обсервабилити.
* **Схема события** содержит идентификаторы (``trace_id``/``span_id``/
  ``parent_span_id`` в духе OTel GenAI), учёт токенов, ``latency_ms``,
  ``stop_reason``.
* **Сырьё не логируем.** Промпты и ответы уходят в лог хешем и длиной;
  полный транскрипт живёт отдельно, в ``sessions/*.jsonl``, и содержит
  только синтетические данные эксперимента.
* **Предохранители живут в коде харнесса, а не в промпте.** Модель не может
  «уговорить» себя выйти за лимит раундов, токенов или времени.
* **Детектор залипания.** Отпечаток хода (сторона + действие + цена) не
  должен повторяться подряд: агент, третий раз подряд предлагающий ту же
  цену, не торгуется, а зациклился.

``stop_reason`` — самое недооценённое поле: именно по нему потом строится
отчёт о здоровье прогона и отбираются сессии-регрессии для эвалов.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Типы событий
# ---------------------------------------------------------------------------

EVENT_RUN_START = "run.start"
EVENT_RUN_END = "run.end"
EVENT_SESSION_START = "session.start"
EVENT_SESSION_END = "session.end"
EVENT_TURN_START = "turn.start"
EVENT_TURN_END = "turn.end"
EVENT_LLM_CALL = "llm.call"
EVENT_LLM_RETRY = "llm.retry"
EVENT_LLM_ERROR = "llm.error"
EVENT_PARSE_FAIL = "action.parse_fail"
EVENT_COMPONENT = "harness.component"
EVENT_GATE_VIOLATION = "harness.gate_violation"
EVENT_CIRCUIT_BREAKER = "harness.circuit_breaker"
EVENT_STUCK = "harness.stuck"


def sha256_short(payload: Any, *, length: int = 16) -> str:
    """Хеш содержимого — то, что уходит в лог вместо самого содержимого."""

    if not isinstance(payload, (str, bytes)):
        payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Логгер событий
# ---------------------------------------------------------------------------


class EventLog:
    """Потокобезопасный JSONL-логгер событий харнесса.

    Пишет в ``events.jsonl`` построчно с немедленным ``flush`` — прогон на
    сотни тысяч вызовов упадёт посередине, и всё, что было до падения,
    обязано остаться на диске (дизайн-док §7.2 про чекпоинты).
    """

    def __init__(self, path: Path | None, *, run_id: str, echo: bool = False) -> None:
        self.run_id = run_id
        self.path = path
        self.echo = echo
        self._lock = threading.Lock()
        self._fh = None
        self._counts: Counter[str] = Counter()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # append: resume дописывает в тот же файл, история прогона цельная.
            self._fh = path.open("a", encoding="utf-8")

    # -- запись -------------------------------------------------------------

    def emit(self, event_type: str, **fields: Any) -> None:
        record = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "run_id": self.run_id,
            "event_type": event_type,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
        }
        record.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._counts[event_type] += 1
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()
        if self.echo:  # pragma: no cover - только для ручной отладки
            print(line)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class NullEventLog(EventLog):
    """Заглушка для тестов: события считает, на диск ничего не пишет."""

    def __init__(self, run_id: str = "test") -> None:
        super().__init__(None, run_id=run_id)


# ---------------------------------------------------------------------------
# Предохранители
# ---------------------------------------------------------------------------


class CircuitBreakerTripped(RuntimeError):
    """Сессия остановлена жёстким лимитом харнесса."""

    def __init__(self, reason: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


@dataclass(slots=True)
class CircuitBreaker:
    """Жёсткие лимиты сессии. Живут в коде, не в промпте.

    Проверяются перед каждым ходом. Сработавший предохранитель не «чинит»
    сессию, а честно её обрывает: такая сессия помечается техническим сбоем
    и исключается из анализа, а её доля публикуется в отчёте о прогоне.
    """

    max_turns: int = 24
    max_tokens: int = 120_000
    max_wall_s: float = 900.0
    max_repeated_fingerprints: int = 3
    #: Обрывать ли сессию при залипании. По умолчанию НЕТ — см. :meth:`after_turn`.
    abort_on_stuck: bool = False

    started_at: float = field(default_factory=time.monotonic)
    turns: int = 0
    tokens: int = 0
    _fingerprints: dict[str, deque[str]] = field(default_factory=dict)

    def reset(self) -> None:
        self.started_at = time.monotonic()
        self.turns = 0
        self.tokens = 0
        self._fingerprints.clear()

    def before_turn(self) -> None:
        if self.turns >= self.max_turns:
            raise CircuitBreakerTripped(
                "max_turns", {"turns": self.turns, "limit": self.max_turns}
            )
        if self.tokens >= self.max_tokens:
            raise CircuitBreakerTripped(
                "max_tokens", {"tokens": self.tokens, "limit": self.max_tokens}
            )
        elapsed = time.monotonic() - self.started_at
        if elapsed >= self.max_wall_s:
            raise CircuitBreakerTripped(
                "max_wall_s", {"elapsed_s": round(elapsed, 1), "limit": self.max_wall_s}
            )

    def after_turn(self, *, tokens: int, side: str, fingerprint: str) -> bool:
        """Учитывает ход и сигнализирует о залипании. Возвращает ``True``, если залип.

        Отпечаток формирует протокол; в него входит и **предложение
        контрагента**, стоящее на столе. Иначе детектор путает две разные
        ситуации: агент, который держится за свою цену, пока контрагент
        уступает (это стратегия), и агент, у которого вообще ничего не
        меняется (это зависание). История ведётся отдельно по сторонам:
        ходы чередуются, и в общей очереди три подряд одинаковых отпечатка
        не встретятся никогда.

        **Сессию по залипанию не обрываем** (``abort_on_stuck=False``).
        Протокол и так ограничен $2T$ ходами, так что бесконечного цикла
        быть не может, а досрочный обрыв стоил бы дороже, чем сэкономил:
        давление дедлайна растёт к последнему раунду и регулярно расшивает
        тупик, в котором стороны простояли середину торга. Обрывать —
        значит подменять честное «сделки не было» техническим сбоем и
        смещать выборку против компонентов, чья суть в том, чтобы стоять
        на своём (H3 «обязательство», H4 «регламент»).

        Сигнал при этом не теряется: протокол пишет событие ``harness.stuck``
        и помечает сессию, а отчёт о прогоне показывает долю зависших.
        Кому важнее экономия — включает ``abort_on_stuck``.
        """

        self.turns += 1
        self.tokens += int(tokens)
        history = self._fingerprints.setdefault(side, deque(maxlen=8))
        history.append(fingerprint)
        window = list(history)[-self.max_repeated_fingerprints :]
        stuck = len(window) == self.max_repeated_fingerprints and len(set(window)) == 1
        if stuck and self.abort_on_stuck:
            raise CircuitBreakerTripped(
                "stuck_repeated_action",
                {
                    "side": side,
                    "fingerprint": fingerprint,
                    "repeats": self.max_repeated_fingerprints,
                },
            )
        return stuck


# ---------------------------------------------------------------------------
# Трассировка
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TraceContext:
    """Минимальная трассировка в духе OTel: сессия — trace, ход — span."""

    trace_id: str
    span_id: str | None = None
    parent_span_id: str | None = None

    def child(self, prefix: str = "span") -> "TraceContext":
        return TraceContext(
            trace_id=self.trace_id,
            span_id=new_id(prefix),
            parent_span_id=self.span_id,
        )

    def as_fields(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }


def iter_events(path: Path) -> Iterator[dict[str, Any]]:
    """Читает ``events.jsonl`` устойчиво к обрыву последней строки."""

    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Файл оборвался на середине строки — прогон убили сигналом.
                continue
