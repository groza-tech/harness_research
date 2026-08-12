"""H3. Механизм обязательства — commitment device (Шеллинг, 1960).

Экономическая категория: стратегическое связывание рук. До начала торга
сторона публикует резервную цену; обвязка принудительно её соблюдает, а
контрагенту объявление сообщается.

Важное для §2.2: этот компонент **не сводится к информационной асимметрии**.
Он не про знание, а про способность связать себе руки — и именно поэтому он
отдельно обсуждается в разграничении с направлением Б.

«Принудительно соблюдаемая» здесь означает буквально: гейт не пропустит ни
одного предложения хуже объявленного, даже если модель передумает. Проверка
— ``tests/test_commitment.py``.
"""

from __future__ import annotations

from harness_asymmetry.harness.components.base import (
    Component,
    GateResult,
    NegotiationState,
)
from harness_asymmetry.schemas import Action, ActionType, Role


class CommitmentComponent(Component):
    key = "commitment"
    code = "H3"

    def context_block(self, state: NegotiationState) -> str | None:
        if state.own_commitment is None:
            return None
        return (
            "[ВАШЕ ПУБЛИЧНОЕ ОБЯЗАТЕЛЬСТВО] Вы публично объявили, что не согласитесь "
            f"на условия хуже {state.own_commitment:,.0f}. Объявление доведено до "
            "контрагента и обязательно для вас: система отклонит любое ваше "
            "предложение хуже этой цены.".replace(",", " ")
        )

    def gate(self, action: Action, state: NegotiationState) -> GateResult:
        """Обязательство связывает и предложения, и приём чужой цены."""

        if state.own_commitment is None or action.price is None:
            return GateResult.passed()
        if not _worse_than(state.role, action.price, state.own_commitment):
            return GateResult.passed()
        if action.type is ActionType.ACCEPT:
            # Принять цену хуже объявленной нельзя — это и есть связывание рук.
            return GateResult(
                ok=False,
                violations=[
                    f"приём цены {action.price:,.0f} хуже вашего публичного "
                    f"обязательства {state.own_commitment:,.0f}".replace(",", " ")
                ],
                repaired_price=state.own_commitment,
            )
        return GateResult(
            ok=False,
            violations=[
                f"предложение {action.price:,.0f} хуже вашего публичного "
                f"обязательства {state.own_commitment:,.0f}".replace(",", " ")
            ],
            repaired_price=state.own_commitment,
        )


def opponent_commitment_block(commitment: float | None, counterparty_id: str) -> str | None:
    """Блок для промпта КОНТРАГЕНТА: он видит чужое объявление.

    Живёт отдельно от компонента: объявление контрагента видно всем, даже
    стороне без собственного H3. Иначе обязательство не работало бы как
    механизм — связывать руки имеет смысл только публично.
    """

    if commitment is None:
        return None
    return (
        f"[ОБЯЗАТЕЛЬСТВО КОНТРАГЕНТА] Партнёр {counterparty_id} публично и "
        f"связывающе объявил, что не согласится на условия хуже "
        f"{commitment:,.0f}. Его обвязка не пропустит предложений хуже этой "
        "цены.".replace(",", " ")
    )


def _worse_than(role: Role, price: float, reference: float) -> bool:
    """Хуже ли ``price`` для стороны, чем ``reference``."""

    return price < reference if role is Role.SELLER else price > reference


def clamp_commitment(role: Role, proposed: float, reservation: float, anchor: float) -> float:
    """Не даёт объявить заведомо неисполнимое обязательство.

    Объявление хуже собственной резервной цены бессмысленно (оно ничего не
    связывает), а объявление за пределами переговорного диапазона гарантирует
    провал сделки и превращает компонент в генератор отказов. Зажимаем в
    коридор [резервная; якорная] — это по-прежнему сильное обязательство.
    """

    low, high = (reservation, anchor) if reservation <= anchor else (anchor, reservation)
    return min(max(proposed, low), high)
