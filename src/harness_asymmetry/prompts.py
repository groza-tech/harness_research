"""Промпты — версионируемый артефакт, а не строковые литералы по коду.

Три требования дизайн-дока, зашитые прямо сюда:

1. **Пиннинг** (§7.2). У каждого шаблона считается SHA-256; хеши уходят в
   манифест прогона. Правка формулировки меняет хеш — рецензент видит, что
   прогон был на другом промпте.
2. **Антиконтаминация** (§8, риск «модель знает теорию»). Ни слова из теории
   игр: ни «Рубинштейн», ни «равновесие», ни «дисконт-фактор», ни «излишек».
   Агент видит управленческую задачу с деньгами и сроком, а не учебник.
3. **Чувствительность к формулировке** (§6.4). Есть три варианта базового
   промпта (``base`` / ``terse`` / ``formal``); эффект обязан сохраняться на
   всех трёх. Переключается ``RunConfig.prompt_variant``.

Отдельно: **язык и действие разделены**. Модель обязана вернуть JSON со
структурированным полем ``price``; свободный текст идёт в ``message`` и
никогда не парсится на предмет цены.
"""

from __future__ import annotations

import hashlib
from typing import Any


ACTION_CONTRACT = """Ответ — строго один JSON-объект, без пояснений вокруг:
{"action": "offer" | "accept" | "reject", "price": <число или null>, "message": "<одна-две фразы контрагенту>"}

Правила:
- "offer" — назвать свою цену; поле price обязательно, число без пробелов и валюты;
- "accept" — принять последнее предложение контрагента как есть; price повторяет его цену;
- "reject" — отказаться продолжать в этом раунде без встречной цены;
- цену указывайте только в поле price. Текст в message на исход не влияет."""


_SYSTEM_BASE = """Вы ведёте переговоры о цене от лица {side_name}. Предмет — {good}.

Ваша ситуация:
- {reservation_label}: {reservation:,.0f} {unit}.
- Сделка {worse_than_reservation} вашей {reservation_short} для вас хуже, чем отсутствие сделки.
- На переговоры отведено не более {max_rounds} раундов. Каждый раунд задержки обесценивает результат: то, что вы получите в раунде t, стоит {discount:.2f} в степени (t−1) от той же суммы в первом раунде.
- Если стороны не договорятся за отведённые раунды, обе не получают ничего.

{info_block}

Ваша задача — договориться на условиях, максимально выгодных для вашей стороны, но договориться.

{action_contract}"""


_SYSTEM_TERSE = """Вы — {side_name}. Предмет: {good}.
{reservation_label}: {reservation:,.0f} {unit}. Хуже неё соглашаться нельзя.
Раундов не более {max_rounds}; каждый следующий раунд стоит вам {discount:.2f} от предыдущего. Нет сделки — ноль обеим сторонам.
{info_block}
Договоритесь как можно выгоднее для себя.

{action_contract}"""


_SYSTEM_FORMAL = """РОЛЬ: {side_name}. ПРЕДМЕТ: {good}.
ПАРАМЕТРЫ:
  {reservation_label} = {reservation:,.0f} {unit}
  предел раундов = {max_rounds}
  коэффициент обесценивания за раунд = {discount:.2f}
  исход без соглашения = 0 для обеих сторон
{info_block}
ЦЕЛЬ: заключить соглашение на наиболее выгодных для вашей стороны условиях.

{action_contract}"""


_VARIANTS: dict[str, str] = {
    "base": _SYSTEM_BASE,
    "terse": _SYSTEM_TERSE,
    "formal": _SYSTEM_FORMAL,
}


PLAN_INSTRUCTION = """Составьте план уступок на ближайшие {horizon} раундов: последовательность цен, которую вы намерены называть, от первой до последней.
Ответ — строго один JSON-объект: {{"plan": [<число>, ...], "message": "<обоснование одной фразой>"}}"""


COMMITMENT_INSTRUCTION = """Перед началом торга вы вправе публично объявить цену, хуже которой не согласитесь. Объявление будет доведено до контрагента и станет для вас обязательным: система не пропустит ни одного вашего предложения хуже объявленного, даже если вы передумаете.
Ответ — строго один JSON-объект: {{"reserve_price": <число>, "message": "<одна фраза>"}}"""


REPAIR_INSTRUCTION = """Ваш предыдущий ответ отклонён проверкой: {violations}
Верните исправленный JSON-объект в том же формате. Других изменений не вносите."""


PARSE_RETRY_INSTRUCTION = """Предыдущий ответ не разобран: {reason}
Верните ровно один JSON-объект в требуемом формате, без текста вокруг него."""


def system_prompt(
    *,
    variant: str,
    role: str,
    reservation: float,
    unit: str,
    good: str,
    max_rounds: int,
    discount: float,
    info_block: str,
) -> str:
    """Собирает системный промпт стороны. Терминов теории игр здесь нет."""

    template = _VARIANTS.get(variant)
    if template is None:
        raise ValueError(
            f"Неизвестный вариант промпта: {variant!r}. Доступны: {sorted(_VARIANTS)}."
        )
    if role == "seller":
        side_name = "поставщика"
        reservation_label = "Ваша себестоимость"
        reservation_short = "себестоимости"
        worse = "ниже"
    else:
        side_name = "закупщика"
        reservation_label = "Ваш предельный бюджет"
        reservation_short = "предельной суммы"
        worse = "выше"
    return template.format(
        side_name=side_name,
        good=good,
        reservation_label=reservation_label,
        reservation_short=reservation_short,
        worse_than_reservation=worse,
        reservation=reservation,
        unit=unit,
        max_rounds=max_rounds,
        discount=discount,
        info_block=info_block,
        action_contract=ACTION_CONTRACT,
    ).replace(",", " ")


def prompt_hashes() -> dict[str, str]:
    """SHA-256 всех шаблонов — для манифеста прогона (пиннинг §7.2)."""

    templates: dict[str, str] = {
        "action_contract": ACTION_CONTRACT,
        "plan_instruction": PLAN_INSTRUCTION,
        "commitment_instruction": COMMITMENT_INSTRUCTION,
        "repair_instruction": REPAIR_INSTRUCTION,
        "parse_retry_instruction": PARSE_RETRY_INSTRUCTION,
    }
    templates.update({f"system:{k}": v for k, v in _VARIANTS.items()})
    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        for name, text in sorted(templates.items())
    }


def available_variants() -> tuple[str, ...]:
    return tuple(sorted(_VARIANTS))


def format_state_block(state: dict[str, Any]) -> str:
    """Машиночитаемый блок состояния торга.

    Он служит двум целям сразу: даёт модели однозначные числа (вместо
    пересказа истории прозой, который она может неверно посчитать) и даёт
    mock-провайдеру точку входа для оффлайн-смоука.
    """

    import json

    return "<<<STATE>>>\n" + json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n<<<END>>>"
