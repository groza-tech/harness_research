"""H6. Полный лог vs компакция — издержки обработки информации.

Экономическая категория: ограниченная рациональность как управляемый
параметр. Сторона с ``full_log=1`` видит весь транскрипт; сторона с
``full_log=0`` — детерминированную сводку плюс хвост из последних ходов.

Дизайн-док §3.3 отмечает H6 как кандидата на компонент с **отрицательной
отдачей**: компакция удешевляет прогон, но может ухудшать переговорную
позицию. Если подтвердится — красивый побочный результат: экономия на
токенах покупается уступкой излишка, чего в токеномической литературе никто
не считал. Именно поэтому расход токенов логируется по каждому ходу.

Компакция намеренно **детерминированная**, без LLM-суммаризатора: иначе в
компонент проникла бы вариативность модели, и H6 перестал бы быть чистым
контрастом «весь контекст / часть контекста».
"""

from __future__ import annotations

from typing import Any

from harness_asymmetry.harness.components.base import Component, NegotiationState


class ContextComponent(Component):
    """Рендерер истории торга. Присутствует всегда, режим задаётся битом H6."""

    key = "full_log"
    code = "H6"

    def __init__(self, *, full_log: bool, tail_turns: int) -> None:
        self.full_log = full_log
        self.tail_turns = tail_turns

    def context_block(self, state: NegotiationState) -> str | None:
        if not state.transcript:
            return "[ХОД ТОРГА] Раунд первый, предложений ещё не было."
        if self.full_log:
            return self._render_full(state.transcript)
        return self._render_compacted(state)

    # -- режимы -------------------------------------------------------------

    def _render_full(self, transcript: list[dict[str, Any]]) -> str:
        lines = ["[ХОД ТОРГА] Полная стенограмма:"]
        for turn in transcript:
            lines.append(_render_turn(turn, with_message=True))
        return "\n".join(lines)

    def _render_compacted(self, state: NegotiationState) -> str:
        transcript = state.transcript
        tail = transcript[-self.tail_turns :] if self.tail_turns > 0 else []
        head_count = len(transcript) - len(tail)

        lines = ["[ХОД ТОРГА] Сжатая сводка:"]
        lines.append(
            f"- проведено ходов: {len(transcript)}, текущий раунд {state.round_index} "
            f"из {state.max_rounds}"
        )
        if state.own_offers:
            lines.append(
                f"- ваши предложения: первое {_fmt(state.own_offers[0])}, "
                f"последнее {_fmt(state.own_offers[-1])}"
            )
        if state.opponent_offers:
            lines.append(
                f"- предложения контрагента: первое {_fmt(state.opponent_offers[0])}, "
                f"последнее {_fmt(state.opponent_offers[-1])}"
            )
            if len(state.opponent_offers) >= 2:
                step = abs(state.opponent_offers[-1] - state.opponent_offers[-2])
                lines.append(f"- шаг уступки контрагента на прошлом ходу: {_fmt(step)}")
        if head_count > 0:
            lines.append(f"- ранние {head_count} ход(ов) свёрнуты")
        if tail:
            lines.append("Последние ходы:")
            lines.extend(_render_turn(turn, with_message=False) for turn in tail)
        return "\n".join(lines)


def _render_turn(turn: dict[str, Any], *, with_message: bool) -> str:
    who = "вы" if turn.get("mine") else "контрагент"
    action = turn.get("action_type", "?")
    price = turn.get("price")
    base = f"  р{turn.get('round_index', '?')} {who}: {action}"
    if price is not None:
        base += f" {_fmt(price)}"
    message = (turn.get("message") or "").strip()
    if with_message and message:
        base += f" — «{message}»"
    return base


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}".replace(",", " ")
