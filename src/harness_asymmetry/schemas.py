"""Ядро типов: вектор харнесса, сценарий, действие, записи хода и сессии.

Здесь живёт вся «валюта» исследования. Два правила, которые нельзя нарушать:

1. **Харнесс — это вектор бит, а не скалярный индекс.** Дизайн-док §8 прямо
   предупреждает: композитный индекс произволен, рецензент спросит «почему вы
   так взвесили компоненты». Поэтому основной анализ идёт по ``HarnessVector``
   покомпонентно; ``level`` (число включённых) существует только для
   визуализации и подписей осей.

2. **Язык и действие разделены.** ``Action`` извлекается строгим парсером из
   структурированного JSON-поля, а не регуляркой из свободного текста
   (дизайн-док §4.2, «именно на этом ломаются наивные реализации»).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Компоненты харнесса
# ---------------------------------------------------------------------------

#: Порядок компонентов зафиксирован навсегда: он определяет битовую строку,
#: имена колонок в regression-матрице и порядок столбцов Плакетта–Бёрмана.
COMPONENT_KEYS: tuple[str, ...] = (
    "memory",
    "market",
    "commitment",
    "verifier",
    "planner",
    "full_log",
)

#: Код компонента из дизайн-дока §3.3 → техническое имя поля.
COMPONENT_CODES: dict[str, str] = {
    "memory": "H1",
    "market": "H2",
    "commitment": "H3",
    "verifier": "H4",
    "planner": "H5",
    "full_log": "H6",
}

#: Экономическое имя компонента. Требование дизайн-дока §3.3: каждый компонент
#: обязан иметь экономическую категорию, иначе получится инженерная статья с
#: экономическими декорациями.
COMPONENT_ECONOMICS_RU: dict[str, str] = {
    "memory": "Репутационный механизм (односторонний переход к повторяющейся игре)",
    "market": "Снижение неопределённости о резервной цене контрагента",
    "commitment": "Commitment device; стратегическое связывание рук (Шеллинг)",
    "verifier": "Бюджетное ограничение в духе ZI-C (Годе–Сандер); мониторинг агента принципалом",
    "planner": "Способность к последовательной рациональности",
    "full_log": "Издержки обработки информации; ограниченная рациональность как параметр",
}

COMPONENT_LABELS_RU: dict[str, str] = {
    "memory": "H1 Память о контрагенте",
    "market": "H2 Рыночный ретрив",
    "commitment": "H3 Механизм обязательства",
    "verifier": "H4 Верификационный гейт",
    "planner": "H5 Планировщик уступок",
    "full_log": "H6 Полный лог (vs компакция)",
}


@dataclass(frozen=True, slots=True)
class HarnessVector:
    """Битовый вектор включённых компонентов обвязки, $h \\in \\{0,1\\}^6$.

    ``full_log=False`` означает **компакцию** контекста, а не отсутствие
    контекста: агент видит детерминированную сводку вместо полного
    транскрипта. Это осознанный выбор кодировки — «голая» конфигурация
    (все нули) должна быть дешёвой во всех смыслах, а компакция как раз
    удешевляет прогон. Дизайн-док §3.3 отмечает H6 как кандидата на
    компонент с *отрицательной* отдачей: экономия токенов покупается
    уступкой излишка.
    """

    memory: bool = False
    market: bool = False
    commitment: bool = False
    verifier: bool = False
    planner: bool = False
    full_log: bool = False

    # -- конструкторы -------------------------------------------------------

    @classmethod
    def bare(cls) -> "HarnessVector":
        """Голая модель: ни одного компонента (базовая точка всех сравнений)."""

        return cls()

    @classmethod
    def full(cls) -> "HarnessVector":
        """Полная обвязка: все шесть компонентов."""

        return cls(**{k: True for k in COMPONENT_KEYS})

    @classmethod
    def from_bits(cls, bits: Sequence[int | bool]) -> "HarnessVector":
        if len(bits) != len(COMPONENT_KEYS):
            raise ValueError(
                f"Ожидалось {len(COMPONENT_KEYS)} бит в порядке {COMPONENT_KEYS}, "
                f"получено {len(bits)}."
            )
        return cls(**{k: bool(b) for k, b in zip(COMPONENT_KEYS, bits)})

    @classmethod
    def from_code(cls, code: str) -> "HarnessVector":
        """``"101010"`` → вектор. Обратно — :meth:`code`."""

        cleaned = code.strip()
        if not set(cleaned) <= {"0", "1"}:
            raise ValueError(f"Код харнесса должен состоять из 0/1, получено {code!r}.")
        return cls.from_bits([int(ch) for ch in cleaned])

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "HarnessVector":
        """``["memory", "verifier"]`` → вектор. Удобно для конфигов YAML."""

        wanted = {n.strip() for n in names if n and n.strip()}
        unknown = wanted - set(COMPONENT_KEYS)
        if unknown:
            raise ValueError(f"Неизвестные компоненты харнесса: {sorted(unknown)}.")
        return cls(**{k: (k in wanted) for k in COMPONENT_KEYS})

    # -- представления ------------------------------------------------------

    def to_bits(self) -> tuple[int, ...]:
        return tuple(int(getattr(self, k)) for k in COMPONENT_KEYS)

    def code(self) -> str:
        """Компактный человекочитаемый ключ ячейки плана: ``"101010"``."""

        return "".join(str(b) for b in self.to_bits())

    def active(self) -> tuple[str, ...]:
        return tuple(k for k in COMPONENT_KEYS if getattr(self, k))

    def as_dict(self) -> dict[str, bool]:
        return {k: bool(getattr(self, k)) for k in COMPONENT_KEYS}

    @property
    def level(self) -> int:
        """Число включённых компонентов. **Только для визуализации** (§8)."""

        return sum(self.to_bits())

    def with_(self, **overrides: bool) -> "HarnessVector":
        data = self.as_dict()
        data.update({k: bool(v) for k, v in overrides.items()})
        return HarnessVector(**data)

    def __str__(self) -> str:  # pragma: no cover - косметика
        return self.code()


def asymmetry(h_a: HarnessVector, h_b: HarnessVector) -> tuple[int, ...]:
    """$\\Delta = h^A - h^B \\in \\{-1,0,1\\}^6$."""

    return tuple(a - b for a, b in zip(h_a.to_bits(), h_b.to_bits()))


def asymmetry_l1(h_a: HarnessVector, h_b: HarnessVector) -> int:
    """$|\\Delta|_1$ — число компонентов, по которым стороны различаются."""

    return sum(abs(d) for d in asymmetry(h_a, h_b))


# ---------------------------------------------------------------------------
# Сценарий и роли
# ---------------------------------------------------------------------------


class InfoRegime(str, Enum):
    """Контрольная ось, разводящая харнессную асимметрию с информационной.

    ``I0`` — полная информация: обе стороны знают $v$ и $c$; работает эталон
    Рубинштейна. Эффект харнесса, выживающий в I0, невозможно объяснить
    переодетой информационной асимметрией — это главный аргумент §8.

    ``I1`` — приватная информация: каждый знает своё, распределение чужого
    известно; работает граница Майерсона–Сатертуэйта.
    """

    FULL = "I0"
    PRIVATE = "I1"


class Role(str, Enum):
    BUYER = "buyer"
    SELLER = "seller"

    def opposite(self) -> "Role":
        return Role.SELLER if self is Role.BUYER else Role.BUYER


@dataclass(frozen=True, slots=True)
class Scenario:
    """Розыгрыш $(v, c)$ и параметры торга.

    Издержки и ценность разыгрываются нами, поэтому максимальный излишек
    известен аналитически — никакой разметки и экспертов не требуется
    (дизайн-док §4.1, главное операционное преимущество дизайна).
    """

    scenario_id: str
    v: float  # резервная ценность покупателя
    c: float  # издержки продавца
    discount: float  # δ ∈ (0,1)
    max_rounds: int
    info_regime: InfoRegime
    v_low: float  # границы распределения, известные обеим сторонам в I1
    v_high: float
    c_low: float
    c_high: float
    unit: str = "руб./т"
    good: str = "партия сырья"

    @property
    def surplus(self) -> float:
        """$S = v - c$. По построению генератора всегда > 0."""

        return self.v - self.c

    def share_seller(self, price: float) -> float:
        return (price - self.c) / self.surplus

    def share_buyer(self, price: float) -> float:
        return (self.v - price) / self.surplus

    def share_for(self, role: Role, price: float) -> float:
        return self.share_seller(price) if role is Role.SELLER else self.share_buyer(price)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "v": self.v,
            "c": self.c,
            "surplus": self.surplus,
            "discount": self.discount,
            "max_rounds": self.max_rounds,
            "info_regime": self.info_regime.value,
        }


def rubinstein_price(scenario: Scenario, first_mover: Role) -> float:
    """Теоретический эталон: равновесие Рубинштейна (1982) при общем δ.

    В бесконечной игре с чередующимися предложениями и общим дисконтом
    первый ходящий получает долю $1/(1+\\delta)$ излишка. Это нулевая линия
    §3.1: отклонение симметричных голых харнессов от неё — само по себе
    результат (поведенческое отклонение LLM-агентов от равновесия).
    """

    first_share = 1.0 / (1.0 + scenario.discount)
    if first_mover is Role.SELLER:
        return scenario.c + first_share * scenario.surplus
    return scenario.v - first_share * scenario.surplus


# ---------------------------------------------------------------------------
# Действия
# ---------------------------------------------------------------------------


def anchor_reference(
    scenario: Scenario, role: Role, opponent_reservation: float | None
) -> float:
    """Дальняя граница переговорного диапазона стороны.

    В I0 это точная величина контрагента. В I1 — граница **известного
    распределения**, а не истинное значение: иначе через якорную точку в
    машиночитаемый блок промпта утекла бы приватная информация, и режим I1
    перестал бы отличаться от I0. Проверяется
    ``tests/test_protocol.py::test_private_regime_does_not_leak``.
    """

    if scenario.info_regime is InfoRegime.FULL:
        if opponent_reservation is None:
            raise ValueError("Режим I0 требует резервной величины контрагента.")
        return opponent_reservation
    return scenario.v_high if role is Role.SELLER else scenario.c_low


class ActionType(str, Enum):
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(slots=True)
class Action:
    """Структурированное действие агента — единственный канал влияния на исход.

    Свободный текст (``message``) идёт контрагенту и в лог, но НИКОГДА не
    парсится на предмет цены.
    """

    type: ActionType
    price: float | None = None
    message: str = ""

    def is_valid(self) -> bool:
        if self.type is ActionType.OFFER:
            return self.price is not None and math.isfinite(self.price)
        return True

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.type.value, "price": self.price, "message": self.message}


class MalformedActionError(RuntimeError):
    """Модель не выдала валидное структурированное действие за все попытки.

    Несёт с собой частично заполненную запись хода: сорвавшийся ход всё
    равно потратил токены и время, и они обязаны попасть в учёт сессии.
    Иначе доля невалидных выводов — метрика, которую §7.2 велит публиковать, —
    систематически занижалась бы ровно на тех конфигурациях, где формат
    ломается чаще всего.
    """

    def __init__(self, message: str, turn: "TurnRecord | None" = None) -> None:
        super().__init__(message)
        self.turn = turn


# ---------------------------------------------------------------------------
# Записи для логов и аналитики
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TurnRecord:
    """Один ход одной стороны: что запросили, что вернулось, что сделал гейт."""

    turn_index: int
    round_index: int
    side: str  # "A" | "B"
    role: str
    action_type: str
    price: float | None
    price_before_gate: float | None = None
    gate_violations: list[str] = field(default_factory=list)
    gate_repaired: bool = False
    gate_clamped: bool = False
    message: str = ""
    raw_text: str = ""
    invalid_outputs: int = 0  # сколько раз JSON не распарсился на этом ходу
    components_active: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    stop_reason: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "round_index": self.round_index,
            "side": self.side,
            "role": self.role,
            "action_type": self.action_type,
            "price": self.price,
            "price_before_gate": self.price_before_gate,
            "gate_violations": list(self.gate_violations),
            "gate_repaired": self.gate_repaired,
            "gate_clamped": self.gate_clamped,
            "message": self.message,
            "raw_text": self.raw_text,
            "invalid_outputs": self.invalid_outputs,
            "components_active": list(self.components_active),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "stop_reason": self.stop_reason,
        }


@dataclass(slots=True)
class SessionRecord:
    """Полный результат одной сессии переговоров — строка датасета.

    ``technical_failure=True`` означает, что сессия исключается из анализа
    (дизайн-док §4.2 п.5): агент трижды не выдал валидное действие, упал
    провайдер и т.п. Доля таких сессий логируется и публикуется, а не
    заметается под ковёр.
    """

    session_id: str
    run_id: str
    experiment: str
    cell_id: str
    repeat_index: int
    pair_id: str

    # сценарий
    scenario_id: str
    v: float
    c: float
    surplus: float
    discount: float
    max_rounds: int
    info_regime: str

    # конфигурация сторон
    harness_a: str
    harness_b: str
    model_a: str
    model_b: str
    role_a: str
    first_mover_side: str
    delta_bits: tuple[int, ...] = ()
    abs_delta: int = 0

    # исход
    deal: bool = False
    price: float | None = None
    agreement_round: int | None = None
    phi_a: float | None = None
    phi_b: float | None = None
    phi_a_discounted: float | None = None
    efficiency: float = 0.0
    rubinstein_price: float | None = None
    rubinstein_gap: float | None = None
    anchor_a: float | None = None
    anchor_b: float | None = None
    anchor_aggressiveness_a: float | None = None
    concession_rate_a: float | None = None
    concession_rate_b: float | None = None

    # издержки и здоровье
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    llm_calls: int = 0
    invalid_outputs: int = 0
    gate_violations_a: int = 0
    gate_violations_b: int = 0
    commitment_a: float | None = None
    commitment_b: float | None = None
    #: Сделка заключена за пределами $[c, v]$ — одна из сторон согласилась на
    #: условия хуже собственной резервной величины, и её доля вышла за
    #: $[0,1]$. Это не сбой стенда, а наблюдаемое поведение: агент без
    #: верификационного гейта (H4) может нарушить бюджетное ограничение —
    #: ровно контраст ZI-U против ZI-C у Годе–Сандера. Метрика публикуется.
    budget_violation: bool = False
    technical_failure: bool = False
    failure_reason: str = ""
    stop_reason: str = "completed"
    #: Сколько ходов детектор счёл залипанием (действие и предложение
    #: контрагента не менялись). Сессию это не обрывает — только сигнал.
    stuck_turns: int = 0

    #: Метаданные ячейки плана (класс модели, уровень асимметрии, номер рынка…).
    #: Попадают в плоскую таблицу как обычные колонки — на них опирается
    #: аналитика Э2/Э4/Э5, которой нужен не только битовый вектор.
    meta: dict[str, Any] = field(default_factory=dict)

    turns: list[TurnRecord] = field(default_factory=list)

    def flat(self) -> dict[str, Any]:
        """Плоская строка для ``sessions.csv`` (без транскрипта)."""

        row: dict[str, Any] = {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "cell_id": self.cell_id,
            "repeat_index": self.repeat_index,
            "pair_id": self.pair_id,
            "scenario_id": self.scenario_id,
            "v": self.v,
            "c": self.c,
            "surplus": self.surplus,
            "discount": self.discount,
            "max_rounds": self.max_rounds,
            "info_regime": self.info_regime,
            "harness_a": self.harness_a,
            "harness_b": self.harness_b,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "role_a": self.role_a,
            "first_mover_side": self.first_mover_side,
            "abs_delta": self.abs_delta,
            "deal": self.deal,
            "price": self.price,
            "agreement_round": self.agreement_round,
            "phi_a": self.phi_a,
            "phi_b": self.phi_b,
            "phi_a_discounted": self.phi_a_discounted,
            "efficiency": self.efficiency,
            "rubinstein_price": self.rubinstein_price,
            "rubinstein_gap": self.rubinstein_gap,
            "anchor_a": self.anchor_a,
            "anchor_b": self.anchor_b,
            "anchor_aggressiveness_a": self.anchor_aggressiveness_a,
            "concession_rate_a": self.concession_rate_a,
            "concession_rate_b": self.concession_rate_b,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "llm_calls": self.llm_calls,
            "invalid_outputs": self.invalid_outputs,
            "gate_violations_a": self.gate_violations_a,
            "gate_violations_b": self.gate_violations_b,
            "commitment_a": self.commitment_a,
            "commitment_b": self.commitment_b,
            "budget_violation": self.budget_violation,
            "technical_failure": self.technical_failure,
            "failure_reason": self.failure_reason,
            "stop_reason": self.stop_reason,
            "stuck_turns": self.stuck_turns,
            "harness_level_a": sum(int(ch) for ch in self.harness_a),
            "harness_level_b": sum(int(ch) for ch in self.harness_b),
        }
        # Покомпонентные дамми — это и есть регрессоры основной спецификации §6.1.
        for idx, key in enumerate(COMPONENT_KEYS):
            row[f"a_{key}"] = int(self.harness_a[idx])
            row[f"b_{key}"] = int(self.harness_b[idx])
            row[f"d_{key}"] = int(self.harness_a[idx]) - int(self.harness_b[idx])
        # Метаданные ячейки не должны затирать вычисленные поля: при коллизии
        # имён побеждает то, что посчитал протокол.
        for key, value in self.meta.items():
            row.setdefault(key, value)
        return row

    def as_dict(self) -> dict[str, Any]:
        """Полная запись с транскриптом — для ``sessions/*.jsonl``."""

        data = self.flat()
        data["delta_bits"] = list(self.delta_bits)
        data["meta"] = dict(self.meta)
        data["turns"] = [t.as_dict() for t in self.turns]
        return data
