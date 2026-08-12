"""Сборка харнесса по битовому вектору.

Единственное место, где бит вектора превращается в объект. Ничто другое в
коде не имеет права спрашивать «а включена ли память» — иначе компоненты
перестают быть независимо отключаемыми, и план эксперимента становится
неисполнимым (дизайн-док §7.1).

Инвариант, который проверяет ``tests/test_assembler.py``: для любого из 64
векторов сборка проходит, а множество активных ключей в точности совпадает с
множеством единиц вектора (H6 присутствует всегда как рендерер контекста, но
его режим определяется битом).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness_asymmetry.config import HarnessConfig
from harness_asymmetry.harness.components.base import (
    Component,
    GateResult,
    NegotiationState,
)
from harness_asymmetry.harness.components.commitment import (
    CommitmentComponent,
    opponent_commitment_block,
)
from harness_asymmetry.harness.components.context import ContextComponent
from harness_asymmetry.harness.components.market_lookup import MarketLookupComponent
from harness_asymmetry.harness.components.memory import MemoryComponent, MemoryStore
from harness_asymmetry.harness.components.planner import PlannerComponent
from harness_asymmetry.harness.components.verifier import VerifierComponent
from harness_asymmetry.observability import EVENT_COMPONENT, EventLog
from harness_asymmetry.schemas import Action, HarnessVector


@dataclass(slots=True)
class Harness:
    """Собранная обвязка одной стороны.

    Обратите внимание: ``vector`` хранится целиком, и все отчёты ссылаются
    именно на него. Скалярный «уровень обвязки» нигде не участвует в
    вычислениях — только в подписях к графикам (§8, риск произвольного
    композитного индекса).
    """

    vector: HarnessVector
    components: list[Component] = field(default_factory=list)
    context_renderer: ContextComponent | None = None
    event_log: EventLog | None = None
    session_id: str = ""

    # -- свойства плана -----------------------------------------------------

    @property
    def uses_commitment(self) -> bool:
        return self.vector.commitment

    @property
    def uses_planner(self) -> bool:
        return self.vector.planner

    @property
    def active_keys(self) -> list[str]:
        return list(self.vector.active())

    # -- работа на ходу -----------------------------------------------------

    def context_blocks(
        self,
        state: NegotiationState,
        *,
        extra_blocks: list[str] | None = None,
    ) -> list[str]:
        """Блоки промпта от всех включённых компонентов, в фиксированном порядке.

        Порядок детерминирован порядком ``COMPONENT_KEYS``: два прогона с
        одинаковой конфигурацией обязаны дать побайтово одинаковый промпт,
        иначе ломается кеш и воспроизводимость.
        """

        blocks: list[str] = []
        for component in self.components:
            block = component.context_block(state)
            if block:
                blocks.append(block)
                self._emit(
                    "context_block",
                    component=component.key,
                    code=component.code,
                    chars=len(block),
                )
        # Блок контекста (H6) рендерится последним: история торга должна идти
        # непосредственно перед вопросом «ваш ход».
        if self.context_renderer is not None:
            rendered = self.context_renderer.context_block(state)
            if rendered:
                blocks.append(rendered)
        if extra_blocks:
            blocks.extend(b for b in extra_blocks if b)
        return blocks

    def gate(self, action: Action, state: NegotiationState) -> GateResult:
        """Прогоняет действие через все гейты и склеивает вердикт.

        Если хотя бы один компонент отклонил действие, результат — отказ, а
        починенная цена берётся у последнего отклонившего гейта: порядок
        компонентов фиксирован, так что это детерминированно.
        """

        violations: list[str] = []
        repaired: float | None = None
        for component in self.components:
            result = component.gate(action, state)
            if result.ok:
                continue
            violations.extend(result.violations)
            if result.repaired_price is not None:
                repaired = result.repaired_price
            self._emit(
                "gate_violation",
                component=component.key,
                code=component.code,
                violations=result.violations,
            )
        if not violations:
            return GateResult.passed()
        return GateResult(ok=False, violations=violations, repaired_price=repaired)

    def on_session_end(self, state: NegotiationState, outcome: dict[str, Any]) -> None:
        for component in self.components:
            component.on_session_end(state, outcome)

    # -- внутреннее ---------------------------------------------------------

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.event_log is not None:
            self.event_log.emit(
                EVENT_COMPONENT,
                session_id=self.session_id,
                harness=self.vector.code(),
                kind=kind,
                **fields,
            )


def assemble_harness(
    vector: HarnessVector,
    *,
    config: HarnessConfig,
    memory_store: MemoryStore | None = None,
    event_log: EventLog | None = None,
    session_id: str = "",
) -> Harness:
    """Собирает обвязку строго по вектору. Единственная точка ветвления."""

    components: list[Component] = []

    if vector.memory:
        if memory_store is None:
            raise ValueError(
                "Компонент памяти включён, но не передан MemoryStore. Память "
                "обязана быть персональной: общий стор схлопывает асимметрию."
            )
        components.append(MemoryComponent(memory_store, window=config.memory_window))
    if vector.market:
        components.append(
            MarketLookupComponent(
                n=config.market_comparables, noise_frac=config.market_noise_frac
            )
        )
    if vector.commitment:
        components.append(CommitmentComponent())
    if vector.verifier:
        components.append(
            VerifierComponent(max_concession_frac=config.verifier_max_concession_frac)
        )
    if vector.planner:
        components.append(PlannerComponent(horizon=config.planner_horizon))

    context_renderer = ContextComponent(
        full_log=vector.full_log, tail_turns=config.compaction_tail_turns
    )

    return Harness(
        vector=vector,
        components=components,
        context_renderer=context_renderer,
        event_log=event_log,
        session_id=session_id,
    )


__all__ = ["Harness", "assemble_harness", "opponent_commitment_block"]
