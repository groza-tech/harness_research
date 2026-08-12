"""H4. Верификационный гейт — бюджетное ограничение в духе ZI-C.

Это самый содержательный компонент дизайна. У Годе и Сандера (1993) ZI-C
отличался от ZI-U ровно наличием бюджетного ограничения — и именно оно
давало почти стопроцентную аллокативную эффективность при нулевом интеллекте
агентов. Наш гейт — современная реализация той же идеи: детерминированная
проверка, что предложение не хуже собственной резервной цены, а уступка не
превышает допустимый шаг.

Экономически это ещё и технология мониторинга агента принципалом: фирма не
доверяет переговорщику полностью, а ограничивает его мандат.

Если H4 окажется сильнейшим компонентом — классический результат
воспроизведён на новом объекте. Если нет — содержательное расхождение с
классикой, которое надо объяснять. Оба исхода публикабельны (§3.3).
"""

from __future__ import annotations

from harness_asymmetry.harness.components.base import (
    Component,
    GateResult,
    NegotiationState,
)
from harness_asymmetry.schemas import Action, ActionType, Role


class VerifierComponent(Component):
    key = "verifier"
    code = "H4"

    def __init__(self, *, max_concession_frac: float) -> None:
        self.max_concession_frac = max_concession_frac

    def context_block(self, state: NegotiationState) -> str | None:
        limit = self.max_concession_frac * state.span
        return (
            "[РЕГЛАМЕНТ СДЕЛКИ] Ваши полномочия ограничены и проверяются автоматически:\n"
            f"- нельзя предлагать и принимать условия хуже {state.reservation:,.0f};\n"
            f"- нельзя уступать более {limit:,.0f} за один раунд.\n"
            "Нарушающее ответ будет отклонено до отправки контрагенту.".replace(",", " ")
        )

    def gate(self, action: Action, state: NegotiationState) -> GateResult:
        if action.price is None:
            return GateResult.passed()

        violations: list[str] = []
        repaired: float | None = None

        # (1) Бюджетное ограничение — то самое ZI-C.
        if state.is_worse_than_reservation(action.price):
            violations.append(
                f"цена {action.price:,.0f} хуже вашей резервной "
                f"{state.reservation:,.0f}".replace(",", " ")
            )
            repaired = state.reservation

        # (2) Ограничение шага уступки. Приём чужой цены уступкой не считается:
        # accept — это согласие с уже стоящим на столе предложением, и запрещать
        # его размером шага значило бы запрещать сделку как таковую.
        if (
            action.type is ActionType.OFFER
            and state.own_last_offer is not None
            and state.span > 0
        ):
            concession = _concession(state.role, state.own_last_offer, action.price)
            limit = self.max_concession_frac * state.span
            if concession > limit:
                violations.append(
                    f"уступка {concession:,.0f} за раунд превышает допустимые "
                    f"{limit:,.0f}".replace(",", " ")
                )
                capped = _apply_concession(state.role, state.own_last_offer, limit)
                repaired = capped if repaired is None else _worse_of(state.role, repaired, capped)

        if not violations:
            return GateResult.passed()
        return GateResult(ok=False, violations=violations, repaired_price=repaired)


def _concession(role: Role, previous: float, current: float) -> float:
    """Насколько сторона сдвинулась в сторону контрагента (>0 — уступка)."""

    return previous - current if role is Role.SELLER else current - previous


def _apply_concession(role: Role, previous: float, amount: float) -> float:
    return previous - amount if role is Role.SELLER else previous + amount


def _worse_of(role: Role, a: float, b: float) -> float:
    """Из двух починенных цен берём более консервативную для стороны."""

    return max(a, b) if role is Role.SELLER else min(a, b)
