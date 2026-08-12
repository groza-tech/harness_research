"""H1. Память о контрагенте — репутационный механизм.

Экономическая категория: односторонний переход от разовой игры к
повторяющейся. Сторона с памятью видит историю прошлых сделок именно с этим
партнёром; сторона без памяти каждый раз начинает с чистого листа. Это
классический источник асимметрии в теории репутации, и, по гипотезе H2
дизайн-дока, — наибольший вклад в переговорную ренту.

**Изоляция — критична.** Самая частая ошибка реализации: общая память
утекает обеим сторонам, и вся асимметрия схлопывается (§7.2). Поэтому:

* у каждой стороны свой экземпляр :class:`MemoryStore`;
* запись помечается ``owner_id``, а чтение чужого владельца бросает
  ``MemoryIsolationError`` — тихо вернуть пустой список было бы хуже: баг
  прошёл бы незамеченным и обнулил бы половину плана эксперимента.

Проверяется тестом ``tests/test_memory_isolation.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness_asymmetry.harness.components.base import Component, NegotiationState
from harness_asymmetry.schemas import Role


class MemoryIsolationError(RuntimeError):
    """Попытка прочитать память чужой стороны."""


@dataclass(frozen=True, slots=True)
class DealMemory:
    """Одна запись о прошлой сделке с этим контрагентом."""

    session_id: str
    counterparty_id: str
    role: str
    deal: bool
    price: float | None
    my_share: float | None
    rounds: int
    counterparty_first_offer: float | None
    counterparty_final_offer: float | None

    def render(self) -> str:
        if not self.deal:
            return (
                f"- сессия {self.session_id}: сделки не было за {self.rounds} раунд(ов); "
                f"первое предложение контрагента {_fmt(self.counterparty_first_offer)}, "
                f"последнее {_fmt(self.counterparty_final_offer)}"
            )
        return (
            f"- сессия {self.session_id}: сделка по {_fmt(self.price)} в раунде {self.rounds}; "
            f"ваша доля выигрыша {self.my_share:.0%}; "
            f"контрагент стартовал с {_fmt(self.counterparty_first_offer)}, "
            f"закончил на {_fmt(self.counterparty_final_offer)}"
        )


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}".replace(",", " ")


@dataclass(slots=True)
class MemoryStore:
    """Персистентный лог сделок ОДНОЙ стороны, разложенный по контрагентам."""

    owner_id: str
    records: dict[str, list[DealMemory]] = field(default_factory=dict)

    def append(self, owner_id: str, memory: DealMemory) -> None:
        if owner_id != self.owner_id:
            raise MemoryIsolationError(
                f"Запись в чужую память: store принадлежит {self.owner_id!r}, "
                f"пишет {owner_id!r}."
            )
        self.records.setdefault(memory.counterparty_id, []).append(memory)

    def recall(self, owner_id: str, counterparty_id: str, *, window: int) -> list[DealMemory]:
        if owner_id != self.owner_id:
            raise MemoryIsolationError(
                f"Чтение чужой памяти: store принадлежит {self.owner_id!r}, "
                f"читает {owner_id!r}."
            )
        history = self.records.get(counterparty_id, [])
        return history[-window:] if window > 0 else list(history)

    def size(self) -> int:
        return sum(len(v) for v in self.records.values())


class MemoryComponent(Component):
    """Компонент H1: подмешивает историю сделок с этим контрагентом."""

    key = "memory"
    code = "H1"

    def __init__(self, store: MemoryStore, *, window: int) -> None:
        self.store = store
        self.window = window

    def context_block(self, state: NegotiationState) -> str | None:
        history = self.store.recall(
            state.identity.party_id,
            state.identity.counterparty_id,
            window=self.window,
        )
        if not history:
            # Первая встреча с этим контрагентом: блок не показываем вовсе,
            # чтобы не сообщать «истории нет» — это тоже информация.
            return None
        lines = [
            f"[ПАМЯТЬ О КОНТРАГЕНТЕ] Ваши прошлые переговоры с партнёром "
            f"{state.identity.counterparty_id} ({len(history)} записей):"
        ]
        lines.extend(record.render() for record in history)
        deals = [r for r in history if r.deal and r.my_share is not None]
        if deals:
            mean_share = sum(r.my_share for r in deals) / len(deals)  # type: ignore[misc]
            lines.append(
                f"Средняя ваша доля выигрыша в прошлых сделках с ним: {mean_share:.0%}."
            )
        return "\n".join(lines)

    def on_session_end(self, state: NegotiationState, outcome: dict[str, Any]) -> None:
        """Фиксирует исход. Пишем всегда — и сделку, и её отсутствие."""

        self.store.append(
            state.identity.party_id,
            DealMemory(
                session_id=outcome.get("session_id", state.session_id),
                counterparty_id=state.identity.counterparty_id,
                role=state.role.value,
                deal=bool(outcome.get("deal")),
                price=outcome.get("price"),
                my_share=outcome.get("my_share"),
                rounds=int(outcome.get("rounds") or 0),
                counterparty_first_offer=outcome.get("counterparty_first_offer"),
                counterparty_final_offer=outcome.get("counterparty_final_offer"),
            ),
        )


def make_stores(party_ids: list[str]) -> dict[str, MemoryStore]:
    """Реестр персональных сторов. Один объект на сторону, не на пару."""

    return {pid: MemoryStore(owner_id=pid) for pid in party_ids}


def role_share(role: Role, price: float, v: float, c: float) -> float | None:
    """Доля излишка стороны — используется при записи исхода в память."""

    surplus = v - c
    if surplus <= 0:
        return None
    return (price - c) / surplus if role is Role.SELLER else (v - price) / surplus
