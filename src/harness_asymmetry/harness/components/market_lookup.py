"""H2. Рыночный ретрив — снижение неопределённости о резервной цене контрагента.

Экономическая категория: доступ к сопоставимым сделкам сужает субъективную
неопределённость относительно того, где проходит граница контрагента.
Технически — выборка цен вокруг конкурентной точки $(v+c)/2$ с шумом,
детерминированная по ``scenario_id``.

Сознательное ограничение: ретрив показывает **статистику рынка**, а не
подсказку «предложи столько-то». Иначе компонент превратился бы в оракула, и
измеряли бы мы не обвязку, а качество подсказки.
"""

from __future__ import annotations

from statistics import median

from harness_asymmetry.harness.components.base import Component, NegotiationState
from harness_asymmetry.scenarios import market_comparables


class MarketLookupComponent(Component):
    key = "market"
    code = "H2"

    def __init__(self, *, n: int, noise_frac: float) -> None:
        self.n = n
        self.noise_frac = noise_frac

    def context_block(self, state: NegotiationState) -> str | None:
        comps = market_comparables(
            state.scenario, n=self.n, noise_frac=self.noise_frac
        )
        if not comps:
            return None
        prices = [c.price for c in comps]
        lines = [
            "[РЫНОЧНЫЕ СОПОСТАВИМЫЕ] Закрытые сделки по сходным поставкам за последний период:"
        ]
        lines.extend(
            f"- {c.deal_id}: {c.price:,.0f} ({c.volume_note})".replace(",", " ") for c in comps
        )
        lines.append(
            f"Медиана по выборке: {median(prices):,.0f}; диапазон "
            f"{min(prices):,.0f}–{max(prices):,.0f}.".replace(",", " ")
        )
        return "\n".join(lines)
