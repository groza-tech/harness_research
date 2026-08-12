"""Клиенты LLM: OpenRouter (боевой) и детерминированный mock (смоук).

Тонкая обёртка поверх ``openai``-SDK с ``base_url`` OpenRouter — так же, как
в соседних проектах ``phd/mas_*``, только без LangChain: харнесс-исследованию
нужен прямой доступ к ``usage``, ``latency`` и ``finish_reason``, а не ещё
один слой абстракции над ними.

Что здесь есть сверх «вызвать модель»:

* экспоненциальный backoff с потолком и честный счётчик неудачных попыток;
* позитивный и негативный кеш по SHA-256 промпта — при общих случайных
  числах один и тот же промпт встречается в разных ячейках плана, и повторно
  платить за него незачем (кешированный ответ отдаётся с нулевыми токенами:
  реального вызова не было, и отчёт об издержках не должен врать);
* сводка здоровья провайдера за прогон — она уходит в отчёт целиком.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from harness_asymmetry.config import LLMSettings
from harness_asymmetry.observability import (
    EVENT_LLM_CALL,
    EVENT_LLM_ERROR,
    EVENT_LLM_RETRY,
    EventLog,
    sha256_short,
)


logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Провайдер не дал ответа за все попытки.

    Бросается вместо тихой заглушки: «нет ответа LLM» должно приводить к
    техническому сбою сессии и честному счётчику, а не к синтетическому
    предложению, которое загрязнило бы датасет.
    """


@dataclass(slots=True)
class TokenUsage:
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt += other.prompt
        self.completion += other.completion
        self.total += other.total


@dataclass(slots=True)
class LLMResponse:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    cached: bool = False
    model: str = ""
    stop_reason: str = "end_turn"


class LLMClient(Protocol):
    """Контракт, который видит агент. Всё остальное — детали транспорта."""

    model: str

    def chat(self, *, system: str, user: str, tag: str = "") -> LLMResponse: ...

    def health(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Боевой клиент
# ---------------------------------------------------------------------------


class OpenRouterClient:
    """Синхронный клиент OpenRouter. Потокобезопасен: кеш под замком."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        event_log: EventLog | None = None,
        enable_cache: bool = True,
    ) -> None:
        from openai import OpenAI  # импорт внутри: mock-прогон не требует SDK

        self.settings = settings
        self.model = settings.model
        self.event_log = event_log
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_s,
            max_retries=0,  # ретраим сами — нам нужен лог каждой попытки
        )
        self._enable_cache = enable_cache
        self._cache: dict[str, LLMResponse] = {}
        self._negative_cache: set[str] = set()
        self._lock = threading.Lock()

        self.total_usage = TokenUsage()
        self.cache_hits = 0
        self.cache_misses = 0
        self.attempts_failed = 0
        self.calls_ok = 0
        self.calls_failed = 0
        self.negative_cache_skips = 0
        self.empty_responses = 0  # HTTP 200 с пустым content — троттлинг провайдера
        self.latency_ms_total = 0.0

    # -- публичный API ------------------------------------------------------

    def chat(self, *, system: str, user: str, tag: str = "") -> LLMResponse:
        key = sha256_short(
            f"{self.model}\x00{self.settings.temperature}\x00{system}\x00{user}", length=32
        )

        if self._enable_cache:
            with self._lock:
                hit = self._cache.get(key)
                if hit is not None:
                    self.cache_hits += 1
                    return LLMResponse(
                        text=hit.text,
                        usage=TokenUsage(),  # реального вызова не было
                        latency_ms=0.0,
                        cached=True,
                        model=self.model,
                        stop_reason=hit.stop_reason,
                    )
                if key in self._negative_cache:
                    self.negative_cache_skips += 1
                    raise LLMUnavailableError(
                        "Идентичный промпт уже не удался ранее (negative-cache)."
                    )
                self.cache_misses += 1

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_reason = ""
        # Потолок ответа растёт при обрыве по длине: повторять запрос с тем
        # же тесным лимитом бессмысленно — модель снова упрётся в него и
        # снова вернёт пустоту, только за деньги. При ``None`` (дефолт)
        # потолок вообще не отправляется, и эта ветка не работает.
        max_tokens = self.settings.max_tokens
        for attempt in range(1, self.settings.max_attempts + 1):
            started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.settings.temperature,
                    "top_p": self.settings.top_p,
                    "extra_body": self._extra_body(),
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                completion = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 — транспорт бывает любым
                latency = (time.perf_counter() - started) * 1000
                with self._lock:
                    self.attempts_failed += 1
                last_reason = f"{type(exc).__name__}: {exc}"
                self._emit(
                    EVENT_LLM_RETRY,
                    tag=tag,
                    attempt=attempt,
                    latency_ms=round(latency, 1),
                    error=type(exc).__name__,
                    prompt_hash=key,
                )
                logger.warning(
                    "LLM попытка %d/%d упала: %s", attempt, self.settings.max_attempts, exc
                )
                self._sleep_backoff(attempt)
                continue

            latency = (time.perf_counter() - started) * 1000
            choice = completion.choices[0] if completion.choices else None
            text = (choice.message.content if choice and choice.message else "") or ""
            stop_reason = (choice.finish_reason if choice else "unknown") or "unknown"
            usage = _extract_usage(completion)

            if not text.strip():
                # Пустой ответ — не ответ. DeepSeek-flash под троттлингом
                # регулярно отдаёт HTTP 200 с пустым content; та же беда
                # описана в соседнем проекте (scripts/server_run.sh). Считаем
                # это неудачной ПОПЫТКОЙ и уходим в backoff: провайдер обычно
                # отвечает со второго-третьего раза. Трактовать пустоту как
                # «модель не смогла в формат» — значит списать деградацию
                # провайдера на модель и потерять сессию на ровном месте.
                with self._lock:
                    self.attempts_failed += 1
                    self.empty_responses += 1
                    self.total_usage.add(usage)
                last_reason = f"пустой ответ (finish_reason={stop_reason})"
                self._emit(
                    EVENT_LLM_RETRY,
                    tag=tag,
                    attempt=attempt,
                    latency_ms=round(latency, 1),
                    error="empty_response",
                    stop_reason=stop_reason,
                    max_tokens=max_tokens,
                    prompt_hash=key,
                )
                if stop_reason == "length" and max_tokens:
                    # Упёрлись в наш же потолок — на следующей попытке даём
                    # вдвое больше и не спим: провайдер здоров, проблема наша.
                    max_tokens = min(max_tokens * 2, 32_000)
                    continue
                self._sleep_backoff(attempt)
                continue

            response = LLMResponse(
                text=text,
                usage=usage,
                latency_ms=latency,
                cached=False,
                model=self.model,
                stop_reason=stop_reason,
            )
            with self._lock:
                self.total_usage.add(usage)
                self.latency_ms_total += latency
                self.calls_ok += 1
                # В кеш кладём только непустой ответ. Закешированная пустота
                # отравляла бы все последующие идентичные промпты: ретраи
                # возвращались бы из кеша мгновенно и с тем же пустым текстом,
                # то есть механизм повторов молча выключался бы.
                if self._enable_cache:
                    self._cache[key] = response
            self._emit(
                EVENT_LLM_CALL,
                tag=tag,
                model=self.model,
                attempt=attempt,
                latency_ms=round(latency, 1),
                prompt_hash=key,
                response_hash=sha256_short(text),
                response_chars=len(text),
                input_tokens=usage.prompt,
                output_tokens=usage.completion,
                stop_reason=stop_reason,
            )
            return response

        with self._lock:
            self.calls_failed += 1
            if self._enable_cache:
                self._negative_cache.add(key)
        self._emit(EVENT_LLM_ERROR, tag=tag, prompt_hash=key, reason=last_reason)
        raise LLMUnavailableError(
            f"LLM не ответил за {self.settings.max_attempts} попыток "
            f"(последняя причина: {last_reason})."
        )

    def health(self) -> dict[str, Any]:
        """Сводка здоровья провайдера — уходит в отчёт о прогоне целиком."""

        with self._lock:
            resolved = self.calls_ok + self.calls_failed
            total_cache = self.cache_hits + self.cache_misses
            in_cost = self.total_usage.prompt / 1e6 * self.settings.price_in_per_mtok
            out_cost = self.total_usage.completion / 1e6 * self.settings.price_out_per_mtok
            return {
                "provider": "openrouter",
                "model": self.model,
                "calls_ok": self.calls_ok,
                "calls_failed": self.calls_failed,
                "attempts_failed": self.attempts_failed,
                "failure_rate_pct": round(100 * self.calls_failed / resolved, 2) if resolved else 0.0,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate_pct": round(100 * self.cache_hits / total_cache, 2)
                if total_cache
                else 0.0,
                "negative_cache_skips": self.negative_cache_skips,
                "empty_responses": self.empty_responses,
                "prompt_tokens": self.total_usage.prompt,
                "completion_tokens": self.total_usage.completion,
                "total_tokens": self.total_usage.total,
                "latency_ms_total": round(self.latency_ms_total, 1),
                "latency_ms_mean": round(self.latency_ms_total / self.calls_ok, 1)
                if self.calls_ok
                else 0.0,
                "estimated_cost_usd": round(in_cost + out_cost, 4),
            }

    # -- внутреннее ---------------------------------------------------------

    def _extra_body(self) -> dict[str, Any]:
        """Параметр ``reasoning`` OpenRouter — не часть OpenAI-схемы.

        Выключенные рассуждения экономят основную массу выходных токенов и,
        главное, избавляют от пустого ``content`` при ``finish_reason=length``:
        рассуждающая модель успевает израсходовать весь потолок на мысли и
        не оставить места ответу.
        """

        mode = (self.settings.reasoning or "").lower()
        if mode in {"", "default", "auto"}:
            return {}  # не трогаем: модель работает в своём штатном режиме
        if mode in {"off", "none", "false", "disabled"}:
            return {"reasoning": {"enabled": False}}
        if mode in {"low", "medium", "high"}:
            return {"reasoning": {"effort": mode}}
        return {}

    def _emit(self, event: str, **fields: Any) -> None:
        if self.event_log is not None:
            self.event_log.emit(event, **fields)

    def _sleep_backoff(self, attempt: int) -> None:
        if attempt >= self.settings.max_attempts:
            return
        delay = min(
            self.settings.backoff_base_s * (2 ** (attempt - 1)),
            self.settings.backoff_max_s,
        )
        time.sleep(delay)


def _extract_usage(completion: Any) -> TokenUsage:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return TokenUsage()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", prompt + completion_tokens) or 0)
    return TokenUsage(prompt=prompt, completion=completion_tokens, total=total)


# ---------------------------------------------------------------------------
# Детерминированный mock
# ---------------------------------------------------------------------------

_STATE_RE = re.compile(r"<<<STATE>>>\s*(\{.*?\})\s*<<<END>>>", flags=re.DOTALL)


class MockClient:
    """Оффлайн-переговорщик для смоука инфраструктуры. **Не наука.**

    Читает машиночитаемый блок состояния, который агент и так вкладывает в
    промпт, и отвечает по простому правилу: якорь → линейная уступка →
    приём, когда встречное предложение перекрывает порог приемлемости.
    Порог и агрессивность якоря сдвигаются от того, какие блоки харнесса
    присутствуют в промпте.

    Это значит, что на mock-провайдере эффект харнесса возникает **по
    построению**. Такой прогон годится ровно для одного: проверить, что
    протокол, гейты, чекпоинты, метрики и отчёты работают end-to-end, не
    тратя ни рубля на API. Любой отчёт, собранный на mock, помечается
    баннером и не может служить результатом исследования — за этим следит
    ``reports.py``.
    """

    def __init__(self, *, seed: int = 0, event_log: EventLog | None = None) -> None:
        self.model = "mock/scripted-negotiator"
        self.seed = seed
        self.event_log = event_log
        self._lock = threading.Lock()
        self.calls_ok = 0
        self.total_usage = TokenUsage()

    def chat(self, *, system: str, user: str, tag: str = "") -> LLMResponse:
        started = time.perf_counter()
        state = _parse_state(user)
        if "план уступок" in user.lower() and state.get("request") == "plan":
            payload = self._plan(state, user)
        elif state.get("request") == "commitment":
            payload = self._commitment(state, user)
        else:
            payload = self._act(state, user)

        text = json.dumps(payload, ensure_ascii=False)
        latency = (time.perf_counter() - started) * 1000
        usage = TokenUsage(
            prompt=max(1, len(system) + len(user)) // 4,
            completion=max(1, len(text)) // 4,
        )
        usage.total = usage.prompt + usage.completion
        with self._lock:
            self.calls_ok += 1
            self.total_usage.add(usage)
        if self.event_log is not None:
            self.event_log.emit(
                EVENT_LLM_CALL,
                tag=tag,
                model=self.model,
                latency_ms=round(latency, 3),
                prompt_hash=sha256_short(user),
                response_hash=sha256_short(text),
                input_tokens=usage.prompt,
                output_tokens=usage.completion,
                stop_reason="end_turn",
            )
        return LLMResponse(
            text=text,
            usage=usage,
            latency_ms=latency,
            model=self.model,
            stop_reason="end_turn",
        )

    def health(self) -> dict[str, Any]:
        return {
            "provider": "mock",
            "model": self.model,
            "calls_ok": self.calls_ok,
            "calls_failed": 0,
            "failure_rate_pct": 0.0,
            "prompt_tokens": self.total_usage.prompt,
            "completion_tokens": self.total_usage.completion,
            "total_tokens": self.total_usage.total,
            "estimated_cost_usd": 0.0,
            "warning": "MOCK-провайдер: данные синтетические, выводы делать нельзя.",
        }

    # -- поведение ----------------------------------------------------------

    def _rng(self, state: dict[str, Any], salt: str) -> random.Random:
        key = f"{self.seed}|{state.get('session_id', '')}|{state.get('round', 0)}|{salt}"
        return random.Random(int(sha256_short(key), 16))

    def _harness_bonus(self, user: str) -> float:
        """Насколько «увереннее» держится сторона, видящая блоки обвязки."""

        bonus = 0.0
        if "[ПАМЯТЬ О КОНТРАГЕНТЕ]" in user:
            bonus += 0.10
        if "[РЫНОЧНЫЕ СОПОСТАВИМЫЕ]" in user:
            bonus += 0.07
        if "[ВАШЕ ПУБЛИЧНОЕ ОБЯЗАТЕЛЬСТВО]" in user:
            bonus += 0.09
        if "[ПЛАН УСТУПОК]" in user:
            bonus += 0.05
        if "[ОБЯЗАТЕЛЬСТВО КОНТРАГЕНТА]" in user:
            bonus -= 0.08
        return bonus

    def _act(self, state: dict[str, Any], user: str) -> dict[str, Any]:
        role = state.get("my_role", "seller")
        reservation = float(state.get("my_reservation", 0.0))
        anchor_ref = float(state.get("anchor_reference", reservation))
        rnd = int(state.get("round", 1))
        max_rounds = max(2, int(state.get("max_rounds", 10)))
        standing = state.get("standing_offer")
        bonus = self._harness_bonus(user)
        jitter = self._rng(state, "act").uniform(-0.02, 0.02)

        span = abs(anchor_ref - reservation)
        # Доля излишка, на которой сторона стоит в этом раунде: линейный
        # спуск от агрессивного якоря к резервной цене к последнему раунду.
        progress = (rnd - 1) / (max_rounds - 1)
        target_share = max(0.05, (0.92 + bonus) * (1 - progress) + (0.30 + bonus) * progress)
        target_share = min(0.98, target_share + jitter)

        if role == "seller":
            target = reservation + target_share * span
        else:
            target = reservation - target_share * span

        if standing is not None:
            standing = float(standing)
            # Порог приёмлемости смягчается к концу горизонта.
            accept_share = max(0.20, (0.62 + bonus) - 0.35 * progress)
            offered_share = (
                (standing - reservation) / span if role == "seller" else (reservation - standing) / span
            ) if span > 0 else 0.0
            if offered_share >= accept_share or rnd >= max_rounds:
                if offered_share > 0:
                    return {
                        "action": "accept",
                        "price": standing,
                        "message": "Принимаю предложение — дальнейшая задержка съедает выгоду.",
                    }
        return {
            "action": "offer",
            "price": round(target, 2),
            "message": f"Предлагаю {target:,.0f}. Раунд {rnd} из {max_rounds}.".replace(",", " "),
        }

    def _plan(self, state: dict[str, Any], user: str) -> dict[str, Any]:
        reservation = float(state.get("my_reservation", 0.0))
        anchor_ref = float(state.get("anchor_reference", reservation))
        horizon = int(state.get("horizon", 4))
        role = state.get("my_role", "seller")
        span = abs(anchor_ref - reservation)
        steps = []
        for i in range(horizon):
            share = 0.90 - 0.15 * i
            price = reservation + share * span if role == "seller" else reservation - share * span
            steps.append(round(price, 2))
        return {"plan": steps, "message": "Пошаговый план уступок на горизонт."}

    def _commitment(self, state: dict[str, Any], user: str) -> dict[str, Any]:
        reservation = float(state.get("my_reservation", 0.0))
        anchor_ref = float(state.get("anchor_reference", reservation))
        span = abs(anchor_ref - reservation)
        role = state.get("my_role", "seller")
        price = reservation + 0.55 * span if role == "seller" else reservation - 0.55 * span
        return {
            "reserve_price": round(price, 2),
            "message": "Публикую связывающую резервную цену.",
        }


def _parse_state(user: str) -> dict[str, Any]:
    match = _STATE_RE.search(user)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:  # pragma: no cover - блок собираем сами
        return {}


def build_client(
    *,
    provider: str,
    settings: LLMSettings | None,
    event_log: EventLog | None = None,
    seed: int = 0,
) -> LLMClient:
    """Фабрика клиента по имени провайдера (``openrouter`` | ``mock``)."""

    if provider == "mock":
        return MockClient(seed=seed, event_log=event_log)
    if provider == "openrouter":
        if settings is None:
            raise ValueError("Для провайдера openrouter нужны LLMSettings.")
        return OpenRouterClient(settings, event_log=event_log)
    raise ValueError(f"Неизвестный провайдер: {provider!r}. Ожидались openrouter|mock.")


def parse_json_object(text: str) -> dict[str, Any]:
    """Терпимый к обрамлению извлекатель JSON-объекта из ответа модели.

    Модели любят fenced-блоки и префиксы вроде «Вот JSON:». Возвращает
    пустой словарь, если объекта нет — решение о том, считать ли это сбоем,
    принимает вызывающий код.
    """

    if not text:
        return {}
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return {}
    try:
        parsed = json.loads(cleaned[first : last + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def coerce_price(value: Any) -> float | None:
    """Приводит цену к float, отбрасывая мусор вроде ``"1 200 000 руб."``.

    Строгость намеренная: если модель не смогла положить число в поле
    ``price``, это невалидный вывод, а не повод угадывать цену регуляркой
    из текста (дизайн-док §4.2).
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        stripped = value.replace(" ", "").replace(" ", "").replace(",", ".")
        stripped = re.sub(r"[^0-9.\-]", "", stripped)
        if not stripped or stripped in {"-", ".", "-."}:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None
