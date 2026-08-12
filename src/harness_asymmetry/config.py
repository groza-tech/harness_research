"""Загрузка окружения и параметров прогона.

Всё, что влияет на результат, обязано жить в конфиге и попадать в манифест
прогона: версии моделей, промпты, сиды, температура, цены токенов
(дизайн-док §7.2 «пиннинг всего»). Магических чисел в коде быть не должно —
рецензент проверяет именно это.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


class ConfigError(RuntimeError):
    """Конфиг несовместим сам с собой — падаем до первого вызова LLM."""


def _optional_int(raw: str | None) -> int | None:
    """Пустая строка и ``0`` означают «без ограничения», а не ноль токенов."""

    if raw is None or not raw.strip():
        return None
    value = int(raw)
    return value if value > 0 else None


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Транспортные настройки провайдера.

    ``temperature`` фиксированная и низкая, но не ноль: на нуле теряется
    вариативность, нужная для оценки распределений (дизайн-док §4.4).
    """

    model: str
    api_key: str
    base_url: str
    temperature: float = 0.4
    top_p: float = 1.0
    #: Потолок ответа. ``None`` — не отправлять параметр вовсе, то есть НЕ
    #: ограничивать модель. Это дефолт по существу дела: в реальном контуре
    #: переговорного агента никто искусственно не режет, а у рассуждающей
    #: модели ``max_tokens`` покрывает и reasoning-токены, и видимый ответ —
    #: тесный потолок приводит к ``finish_reason=length`` с пустым content,
    #: то есть выглядит как сбой провайдера, будучи ошибкой конфигурации.
    #: Реальный предохранитель от разгона — бюджет токенов на сессию в
    #: ``RunnerConfig``, он живёт в харнессе и считает по факту.
    max_tokens: int | None = None
    #: Режим рассуждений OpenRouter: ``None``/"default" — не трогать, модель
    #: работает как обычно; "off" | "low" | "medium" | "high" — задать явно.
    #: Значение пиннится в манифест: это часть определения «модели» в смысле
    #: дизайн-дока и обязано быть одинаковым во всех ячейках плана.
    reasoning: str | None = None
    timeout_s: float = 60.0
    # Внешний ретрай поверх SDK: сколько раз пытаться получить валидное
    # структурированное действие, прежде чем признать ход техническим сбоем.
    max_attempts: int = 4
    backoff_base_s: float = 2.0
    backoff_max_s: float = 30.0
    # Цена за 1M токенов — только для отчёта об издержках. Пиннится вместе
    # с моделью: прайс-лист меняется, отчёт должен остаться воспроизводимым.
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0

    def redacted(self) -> dict[str, Any]:
        """Версия для манифеста: ключ не пишем даже в приватный лог."""

        data = asdict(self)
        data["api_key"] = f"<redacted:{len(self.api_key)}chars>" if self.api_key else "<absent>"
        return data


def load_llm_settings(
    *,
    model: str | None = None,
    require_key: bool = True,
    overrides: dict[str, Any] | None = None,
) -> LLMSettings:
    """Подхватывает OpenRouter-настройки из ``.env``; падает при пустом ключе."""

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if require_key and not api_key:
        raise ConfigError(
            "OPENROUTER_API_KEY не задан. Скопируйте .env.example в .env и пропишите ключ, "
            "либо запускайте с --provider mock (смоук инфраструктуры без сети)."
        )
    settings = LLMSettings(
        model=model or os.getenv("HA_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
        temperature=float(os.getenv("HA_TEMPERATURE", "0.4")),
        max_tokens=_optional_int(os.getenv("HA_MAX_TOKENS")),
        reasoning=(os.getenv("HA_REASONING") or "").strip().lower() or None,
        price_in_per_mtok=float(os.getenv("HA_PRICE_IN_PER_MTOK", "0") or 0),
        price_out_per_mtok=float(os.getenv("HA_PRICE_OUT_PER_MTOK", "0") or 0),
        timeout_s=float(os.getenv("HA_TIMEOUT_S", "60")),
        max_attempts=int(os.getenv("HA_MAX_ATTEMPTS", "4")),
        backoff_base_s=float(os.getenv("HA_BACKOFF_BASE_S", "2")),
        backoff_max_s=float(os.getenv("HA_BACKOFF_MAX_S", "30")),
    )
    if overrides:
        settings = LLMSettings(**{**asdict(settings), **overrides})
    return settings


def model_registry() -> dict[str, str]:
    """Три весовых класса для Э1/Э4 («курс обмена модель ↔ харнесс»)."""

    load_dotenv()
    return {
        "light": os.getenv("HA_MODEL_LIGHT", DEFAULT_MODEL),
        "mid": os.getenv("HA_MODEL_MID", "qwen/qwen3-72b-instruct"),
        "heavy": os.getenv("HA_MODEL_HEAVY", "anthropic/claude-sonnet-4.5"),
    }


# ---------------------------------------------------------------------------
# Параметры сценариев и протокола
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Распределения $(v,c)$ и параметры торга.

    Держим $v$ и $c$ в разных, но перекрывающихся диапазонах и отбрасываем
    розыгрыши с $v \\le c$: сессия без взаимовыгодной сделки не даёт
    информации о распределении излишка, зато засоряет метрику ``D``.
    """

    v_low: float = 1_000_000.0
    v_high: float = 1_600_000.0
    c_low: float = 700_000.0
    c_high: float = 1_300_000.0
    min_surplus: float = 50_000.0
    discount: float = 0.9
    max_rounds: int = 10
    n_scenarios: int = 40  # пул общих случайных чисел (CRN)
    seed: int = 20260812
    unit: str = "руб. за партию"
    good: str = "квартальная поставка компонентов"


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """Настройки компонентов обвязки.

    Все пороги здесь, а не внутри компонентов: план эксперимента должен
    оставаться исполнимым при их изменении без правки кода.
    """

    # H2: сколько сопоставимых сделок показывает ретрив и с каким шумом.
    market_comparables: int = 5
    market_noise_frac: float = 0.06
    # H4: максимальная уступка за ход, доля от размаха переговоров.
    verifier_max_concession_frac: float = 0.35
    verifier_max_repairs: int = 2
    # H5: сколько раундов вперёд планирует планировщик.
    planner_horizon: int = 4
    # H6: сколько последних ходов остаётся в компактном контексте.
    compaction_tail_turns: int = 2
    # H1: сколько прошлых сделок с контрагентом показывать.
    memory_window: int = 6


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Оркестрация прогона: повторы, параллелизм, предохранители."""

    repeats: int = 40  # n ≥ 40 на ячейку (дизайн-док §4.4)
    max_workers: int = 4
    counterbalance_roles: bool = True
    # Предохранители харнесса (не промпта!) — см. observability.CircuitBreaker.
    max_turns_per_session: int = 24
    # Бюджет токенов на сессию — единственный реальный предохранитель от
    # разгона, раз ответ модели не режется потолком. Считаем по факту:
    # рассуждающая модель тратит ~5 тыс. токенов на ход, 24 хода ⇒ ~130 тыс.,
    # поэтому запас четырёхкратный. Слишком тесный бюджет обрывал бы сессии
    # как технические сбои ровно на самых «думающих» конфигурациях.
    max_tokens_per_session: int = 500_000
    # Потолок времени сессии. Рассуждающая модель тратит ~100 с на ход, и
    # при 24 ходах сессия идёт ~40 минут — тесный лимит обрывал бы её как
    # технический сбой ровно там, где модель думает дольше всего.
    max_wall_s_per_session: float = 7200.0
    # Сколько повторов ячейки образуют одну «пару» с общей памятью. Внутри
    # пары порядок строгий (H1 копит историю), но разные пары независимы и
    # идут параллельно. Значение ≥ memory_window: глубже окна история всё
    # равно не читается, поэтому гнать все 40 повторов одной парой — значит
    # ничего не выиграть в репутационном механизме и потерять весь
    # параллелизм на самых дорогих ячейках.
    pair_chunk: int = 8
    max_consecutive_failures: int = 12
    fail_fast: bool = False
    # Обрывать ли сессию, когда сторона перестала что-либо менять. По
    # умолчанию нет: протокол и так ограничен 2T ходами, а досрочный обрыв
    # подменял бы честное «сделки не было» техническим сбоем и смещал бы
    # выборку против H3/H4, чья суть — стоять на своём. См. observability.
    abort_on_stuck: bool = False


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Полный конфиг прогона — попадает в манифест целиком."""

    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    info_regimes: tuple[str, ...] = ("I0", "I1")
    prompt_variant: str = "base"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenarios": asdict(self.scenarios),
            "harness": asdict(self.harness),
            "runner": asdict(self.runner),
            "info_regimes": list(self.info_regimes),
            "prompt_variant": self.prompt_variant,
        }


def _merge(section: dict[str, Any] | None, cls):
    if not section:
        return cls()
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(section) - known
    if unknown:
        raise ConfigError(f"{cls.__name__}: неизвестные ключи {sorted(unknown)}.")
    return cls(**section)


def load_run_config(path: str | Path | None = None) -> RunConfig:
    """Читает YAML-конфиг; отсутствующие секции берутся из дефолтов."""

    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path).expanduser()
    if not path.exists():
        raise ConfigError(f"Конфиг не найден: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Конфиг {path} должен быть YAML-словарём.")
    cfg = RunConfig(
        scenarios=_merge(raw.get("scenarios"), ScenarioConfig),
        harness=_merge(raw.get("harness"), HarnessConfig),
        runner=_merge(raw.get("runner"), RunnerConfig),
        info_regimes=tuple(raw.get("info_regimes", ("I0", "I1"))),
        prompt_variant=str(raw.get("prompt_variant", "base")),
    )
    if cfg.scenarios.max_rounds < 2:
        raise ConfigError("max_rounds < 2: торг невозможен.")
    if not 0 < cfg.scenarios.discount < 1:
        raise ConfigError("discount должен лежать в (0,1).")
    bad = set(cfg.info_regimes) - {"I0", "I1"}
    if bad:
        raise ConfigError(f"Неизвестные режимы информации: {sorted(bad)}.")
    return cfg
