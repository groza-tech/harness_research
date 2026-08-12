"""Общие типы компонентов харнесса.

Компонент — это объект с тремя необязательными способностями:

* ``context_block(state)`` — добавить блок в промпт своей стороны;
* ``gate(action, state)`` — детерминированно проверить/поправить действие;
* ``on_session_end(record)`` — записать что-то в персистентное состояние.

Больше компоненту ничего не позволено: он не видит ни контрагента, ни его
приватных величин. Это не стилистика, а защита от самой частой ошибки
реализации — утечки состояния между сторонами (дизайн-док §7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness_asymmetry.schemas import Action, Role, Scenario


@dataclass(frozen=True, slots=True)
class SideIdentity:
    """Кто эта сторона. ``party_id`` персистентен между сессиями пары.

    ``party_id`` — то, по чему компонент памяти (H1) узнаёт контрагента:
    именно он превращает разовую игру в повторяющуюся для одной стороны.
    """

    side: str  # "A" | "B"
    party_id: str
    counterparty_id: str


@dataclass(slots=True)
class NegotiationState:
    """Всё, что компонент вправе знать о торге на текущем ходу.

    Приватные величины контрагента сюда не попадают никогда: в режиме I0
    известное значение приходит через ``opponent_reservation``, и туда его
    кладёт протокол, а не компонент.
    """

    session_id: str
    scenario: Scenario
    identity: SideIdentity
    role: Role
    reservation: float  # своя резервная величина: c у продавца, v у покупателя
    anchor_reference: float  # дальняя граница переговорного диапазона
    opponent_reservation: float | None  # заполнено только в режиме I0
    round_index: int
    turn_index: int
    max_rounds: int
    standing_offer: float | None = None  # последнее предложение контрагента
    own_last_offer: float | None = None
    own_offers: list[float] = field(default_factory=list)
    opponent_offers: list[float] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    own_commitment: float | None = None
    opponent_commitment: float | None = None
    plan: list[float] = field(default_factory=list)

    @property
    def span(self) -> float:
        """Размах переговорного диапазона от своей резервной до якорной точки."""

        return abs(self.anchor_reference - self.reservation)

    def share_of(self, price: float) -> float:
        """Доля излишка, которую даёт цена этой стороне (для гейта и метрик)."""

        surplus = self.scenario.surplus
        if surplus <= 0:
            return 0.0
        if self.role is Role.SELLER:
            return (price - self.scenario.c) / surplus
        return (self.scenario.v - price) / surplus

    def is_worse_than_reservation(self, price: float) -> bool:
        """Нарушает ли цена бюджетное ограничение стороны (ZI-C у Годе–Сандера)."""

        if self.role is Role.SELLER:
            return price < self.reservation
        return price > self.reservation

    def machine_state(self) -> dict[str, Any]:
        """Машиночитаемый блок для промпта — числа, а не пересказ прозой."""

        return {
            "session_id": self.session_id,
            "round": self.round_index,
            "max_rounds": self.max_rounds,
            "my_role": self.role.value,
            "my_reservation": round(self.reservation, 2),
            "anchor_reference": round(self.anchor_reference, 2),
            "standing_offer": round(self.standing_offer, 2)
            if self.standing_offer is not None
            else None,
            "my_last_offer": round(self.own_last_offer, 2)
            if self.own_last_offer is not None
            else None,
            "my_commitment": self.own_commitment,
            "counterparty_commitment": self.opponent_commitment,
        }


@dataclass(slots=True)
class GateResult:
    """Вердикт детерминированной проверки действия.

    ``repaired_price`` заполняется только при ``clamped=True`` — когда гейт
    сам чинит цену после исчерпания попыток переспросить модель. Различать
    «модель поправилась сама» и «за неё поправил гейт» обязательно: это
    разные экономические механизмы и разные строки в отчёте.
    """

    ok: bool
    violations: list[str] = field(default_factory=list)
    repaired_price: float | None = None
    clamped: bool = False

    @classmethod
    def passed(cls) -> "GateResult":
        return cls(ok=True)


class Component:
    """База компонента. Все хуки — no-op, потомок переопределяет нужные."""

    key: str = ""
    code: str = ""

    def context_block(self, state: NegotiationState) -> str | None:
        return None

    def gate(self, action: Action, state: NegotiationState) -> GateResult:
        return GateResult.passed()

    def on_session_end(self, state: NegotiationState, outcome: dict[str, Any]) -> None:
        return None
