"""Прайс-лист моделей: тянем с провайдера и пиннимся к нему.

Дизайн-док §7.2 требует фиксировать и версионировать прайс-лист наравне с
версиями моделей. Одна общая цена на все модели — не экономия, а ошибка:
в ночном прогоне 12.08.2026 она занизила стоимость вдвое, потому что цена
дешёвой модели применялась к тяжёлой, которая дороже в пять раз.

Поэтому цены берутся с ``/models`` провайдера по факту прогона и целиком
уходят в манифест. Сеть недоступна или модель не найдена — падаем на
значения из окружения и честно помечаем источник: отчёт должен показывать,
откуда взялась цифра, а не создавать видимость точности.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Iterable

logger = logging.getLogger(__name__)


#: Цена в долларах за 1M токенов: ``{слаг: (вход, выход)}``.
PriceTable = dict[str, tuple[float, float]]


def fetch_prices(
    models: Iterable[str],
    *,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: float = 20.0,
) -> tuple[PriceTable, str]:
    """Возвращает ``(таблица цен, источник)``. Не бросает — прогон важнее."""

    wanted = {m for m in models if m}
    if not wanted:
        return {}, "empty"
    url = f"{base_url.rstrip('/')}/models"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "harness-asymmetry"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 — прайс не повод не считать эксперимент
        logger.warning("Не удалось получить прайс-лист с %s: %s", url, exc)
        return {}, f"unavailable:{type(exc).__name__}"

    table: PriceTable = {}
    for item in payload.get("data", []):
        slug = item.get("id")
        if slug not in wanted:
            continue
        pricing = item.get("pricing") or {}
        try:
            price_in = float(pricing.get("prompt", 0) or 0) * 1e6
            price_out = float(pricing.get("completion", 0) or 0) * 1e6
        except (TypeError, ValueError):
            continue
        table[slug] = (price_in, price_out)

    missing = wanted - set(table)
    if missing:
        logger.warning("Прайс не найден для моделей: %s", sorted(missing))
    return table, "openrouter"


def as_manifest(table: PriceTable, source: str) -> dict[str, object]:
    """Представление для ``run_manifest.json``."""

    return {
        "source": source,
        "unit": "USD per 1M tokens",
        "models": {slug: {"in": pin, "out": pout} for slug, (pin, pout) in sorted(table.items())},
    }
