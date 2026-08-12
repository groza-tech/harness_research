"""H5. Планировщик — способность к последовательной рациональности.

Экономическая категория: агент без планировщика решает каждый раунд заново
и потому уязвим к тактике «дожать в последнем раунде». Агент с планировщиком
один раз формирует траекторию уступок и держится её, отклоняясь только по
факту.

Технически: план строится одним LLM-вызовом до начала торга и затем
персистентно подмешивается в контекст с указанием цели на текущий раунд.
План — рекомендация, а не гейт: принудительно его никто не соблюдает, иначе
компонент дублировал бы H4.
"""

from __future__ import annotations

from harness_asymmetry.harness.components.base import Component, NegotiationState


class PlannerComponent(Component):
    key = "planner"
    code = "H5"

    def __init__(self, *, horizon: int) -> None:
        self.horizon = horizon

    def context_block(self, state: NegotiationState) -> str | None:
        if not state.plan:
            return None
        target = state.plan[min(state.round_index - 1, len(state.plan) - 1)]
        steps = ", ".join(f"р{i + 1}: {p:,.0f}" for i, p in enumerate(state.plan))
        return (
            "[ПЛАН УСТУПОК] Ваш план на переговоры, составленный до их начала: "
            f"{steps}.\nЦель текущего раунда: {target:,.0f}. Отклоняйтесь от плана "
            "только если этого требует поведение контрагента.".replace(",", " ")
        )


def sanitize_plan(
    plan: list[float],
    *,
    reservation: float,
    anchor: float,
    horizon: int,
    is_seller: bool,
) -> list[float]:
    """Приводит план от модели в исполнимый вид.

    Модель регулярно возвращает план, уходящий за собственную резервную цену
    или немонотонный. Чинить это внутри планировщика правильно: планировщик
    отвечает за последовательность, а не за границы. Границы — забота H4,
    который может быть выключен, поэтому минимальную санитацию делаем здесь
    же, но без гейта: план остаётся рекомендацией.
    """

    if not plan:
        return []
    low, high = (reservation, anchor) if reservation <= anchor else (anchor, reservation)
    cleaned: list[float] = []
    for value in plan[:horizon]:
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        cleaned.append(min(max(price, low), high))
    if not cleaned:
        return []
    # Монотонность: продавец не повышает цену по ходу торга, покупатель не снижает.
    monotone = [cleaned[0]]
    for price in cleaned[1:]:
        prev = monotone[-1]
        monotone.append(min(price, prev) if is_seller else max(price, prev))
    return monotone
