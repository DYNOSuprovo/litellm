import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter
from redis.exceptions import RedisError

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.llms.anthropic.prompt_cache_prediction import NativePredictionTarget
from litellm.proxy.spend_tracking.autorouter_baseline_cache import (
    BaselineCacheEstimate,
    BaselineCacheEstimator,
    BaselineReservation,
    _HistoryStore,  # pyright: ignore[reportPrivateUsage]  # inject faults at the authoritative store boundary
    _Snapshot,  # pyright: ignore[reportPrivateUsage]  # inspect the persisted reservation lifecycle
    _State,  # pyright: ignore[reportPrivateUsage]  # preserve the typed atomic state transition in the fault store
    _StoreFailure,  # pyright: ignore[reportPrivateUsage]  # simulate expected storage failures without class monkeypatching
)

_MODEL: Final = "claude-sonnet-5"
_TARGET: Final = NativePredictionTarget(_MODEL, "test-provider-key")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")


@dataclass
class _Clock:
    now: float = 10000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Counter:
    def __init__(self, unavailable: bool = False) -> None:
        self.unavailable = unavailable
        self.calls = 0

    async def __call__(self, model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        self.calls += 1
        return None if self.unavailable else json.dumps(_JSON_OBJECT.validate_python(body)).count("token ")


def _wire(ttl: str = "1h", *, growth: int = 0, changed: bool = False) -> httpx.Request:
    grown: Final = f'{{"type":"text","text":"{"token " * growth}"}},' if growth else ""
    return httpx.Request(
        "POST",
        "https://configured-native-provider.test/v1/messages",
        headers=MappingProxyType({"anthropic-version": "2023-06-01"}),
        content=f'''{{
            "model":"{_MODEL}","max_tokens":10,"messages":[{{"role":"user","content":[
                {{"type":"text","text":"{("changed " if changed else "") + "token " * 6000}"}},
                {grown}
                {{"type":"text","text":"end","cache_control":{{"type":"ephemeral","ttl":"{ttl}"}}}}
            ]}}]
        }}''',
    )


def _prepare(
    estimator: BaselineCacheEstimator,
    request: str,
    *,
    caller: str = "caller",
    session: str = "session",
    target: NativePredictionTarget = _TARGET,
) -> BaselineReservation:
    reservation: Final = estimator.prepare(
        caller_key_hash=caller,
        session_id=session,
        router_id="router",
        baseline_deployment_id="baseline",
        target=target,
        request_id=request,
    )
    assert isinstance(reservation, BaselineReservation)
    return reservation


async def _reserve(
    estimator: BaselineCacheEstimator,
    request: str,
    *,
    caller: str = "caller",
    session: str = "session",
    target: NativePredictionTarget = _TARGET,
) -> BaselineReservation:
    reservation: Final = _prepare(estimator, request, caller=caller, session=session, target=target)
    assert await estimator.reserve(reservation) is None
    return reservation


async def _run(
    estimator: BaselineCacheEstimator,
    clock: _Clock,
    request: str,
    wire: httpx.Request,
) -> BaselineCacheEstimate:
    reservation: Final = await _reserve(estimator, request)
    started: Final = clock.now
    clock.advance(0.1)
    return await estimator.finalize(reservation, wire=wire, request_started_at=started, available_at=clock.now)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ttl,duration,bucket",
    (("5m", 300, "cache_creation_5m_input_tokens"), ("1h", 3600, "cache_creation_1h_input_tokens")),
)
async def test_established_expiry_never_receives_a_hypothetical_read(ttl: str, duration: int, bucket: str) -> None:
    clock: Final = _Clock()
    counter: Final = _Counter()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, counter)
    first: Final = await _run(estimator, clock, "first", _wire(ttl))
    assert (first.status, first.reason) == ("unknown", "history_unavailable")
    clock.now = 10001.0
    warm: Final = await _run(estimator, clock, "warm", _wire(ttl))
    assert warm.cache_read_input_tokens == 6000
    assert warm.cache_creation_1h_input_tokens == warm.cache_creation_5m_input_tokens == 0
    clock.now = 10001.0 + duration
    expired: Final = await _run(estimator, clock, "expired", _wire(ttl))
    assert expired.status == "estimated"
    assert expired.reason == "cache_prefix_expired"
    assert expired.cache_read_input_tokens == 0
    assert expired.metadata()[bucket] == 6000
    usage: Final = expired.usage(10)
    assert usage is not None and usage.prompt_tokens == 6000 and usage.total_tokens == 6010
    assert counter.calls <= 4


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl,duration", (("5m", 300), ("1h", 3600)))
@pytest.mark.parametrize("cancelled", (False, True))
async def test_invalidation_keeps_unknown_cache_effects_through_latest_retry_completion(
    ttl: str, duration: int, cancelled: bool
) -> None:
    clock: Final = _Clock()
    counter: Final = _Counter()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, counter)
    await _run(estimator, clock, "seed", _wire(ttl))
    clock.now = 10001.0
    warm: Final = await _run(estimator, clock, "warm", _wire(ttl))
    assert warm.cache_read_input_tokens == 6000
    clock.now = 10005.0
    failed: Final = await _reserve(estimator, "retrying")
    if cancelled:
        await estimator.cancel(failed)
    calls: Final = counter.calls
    clock.now = 10010.0
    abandoned: Final = await estimator.invalidate(failed, "upstream_request_failed")
    assert (abandoned.status, abandoned.reason) == ("unknown", "upstream_request_failed")
    clock.now = 10010.0 + duration
    retried: Final = await estimator.invalidate(failed, "retried_upstream_request")
    assert (retried.status, retried.reason) == ("unknown", "retried_upstream_request")
    assert counter.calls == calls
    clock.now = 10020.0 + duration
    following: Final = await _run(estimator, clock, "following", _wire(ttl))
    assert (following.status, following.reason) == ("unknown", "history_unavailable")
    assert following.cache_read_input_tokens is None
    clock.now = 10020.0 + 2 * duration
    expired: Final = await _run(estimator, clock, "expired", _wire(ttl))
    assert (expired.status, expired.reason) == ("estimated", "cache_prefix_expired")
    assert expired.cache_read_input_tokens == 0


@pytest.mark.asyncio
async def test_prefix_change_is_cold_after_history_horizon_and_unknown_before_it() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "first", _wire())
    early: Final = await _run(estimator, clock, "changed-early", _wire(changed=True))
    assert early.status == "unknown"
    clock.now = 13601.0
    changed: Final = await _run(estimator, clock, "changed", _wire(growth=2000))
    assert changed.status == "estimated"
    assert changed.cache_read_input_tokens == 0
    assert changed.cache_creation_1h_input_tokens == 8000


@pytest.mark.asyncio
async def test_lookback_reads_prior_marker_and_writes_only_growth() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "first", _wire())
    clock.now = 13600.0
    await _run(estimator, clock, "cold", _wire())
    growing_wire: Final = httpx.Request(
        "POST",
        "https://provider.test/v1/messages",
        content=f'''{{
            "model":"{_MODEL}","max_tokens":10,"messages":[{{"role":"user","content":[
                {{"type":"text","text":"{"token " * 6000}"}},
                {{"type":"text","text":"end"}},
                {{"type":"text","text":"{"token " * 2000}","cache_control":{{"type":"ephemeral","ttl":"1h"}}}}
            ]}}]
        }}''',
    )
    grown: Final = await _run(estimator, clock, "grown", growing_wire)
    assert grown.status == "estimated"
    assert grown.cache_read_input_tokens == 6000
    assert grown.cache_creation_1h_input_tokens == 2000
    clock.now = 17200.05
    old_prefix: Final = await _run(estimator, clock, "old-prefix", _wire())
    assert old_prefix.cache_read_input_tokens == 6000


@pytest.mark.asyncio
async def test_mixed_ttl_keeps_new_hour_and_five_minute_writes_separate() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "seed", _wire())
    clock.now = 13600.0
    wire: Final = httpx.Request(
        "POST",
        "https://provider.test/v1/messages",
        content=f'''{{
            "model":"{_MODEL}","max_tokens":10,"messages":[{{"role":"user","content":[
                {{"type":"text","text":"{"token " * 5000}","cache_control":{{"type":"ephemeral","ttl":"1h"}}}},
                {{"type":"text","text":"{"token " * 2000}","cache_control":{{"type":"ephemeral","ttl":"5m"}}}},
                {{"type":"text","text":"{"token " * 100}"}}
            ]}}]
        }}''',
    )
    result: Final = await _run(estimator, clock, "mixed", wire)
    assert result.status == "estimated"
    assert result.input_tokens == 100
    assert result.cache_read_input_tokens == 0
    assert result.cache_creation_1h_input_tokens == 5000
    assert result.cache_creation_5m_input_tokens == 2000


@pytest.mark.asyncio
async def test_parallel_pending_requests_and_late_completions_do_not_self_hit_or_regress_refresh() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    first: Final = await _reserve(estimator, "first")
    clock.now = 10001.0
    second: Final = await _reserve(estimator, "second")
    clock.now = 10002.0
    second_result: Final = await estimator.finalize(
        second, wire=_wire(), request_started_at=10001.0, available_at=10002.0
    )
    assert second_result.reason == "pending_request"
    first_result: Final = await estimator.finalize(
        first, wire=_wire(), request_started_at=10000.0, available_at=10002.0
    )
    assert first_result.cache_read_input_tokens is None
    duplicate: Final = await estimator.finalize(
        second, wire=_wire(changed=True), request_started_at=10001.0, available_at=10002.0
    )
    assert duplicate == second_result
    clock.now = 13600.5
    still_warm: Final = await _run(estimator, clock, "still-warm", _wire())
    assert still_warm.cache_read_input_tokens == 6000


@pytest.mark.asyncio
@pytest.mark.parametrize("complete,cache_hit", ((False, False), (True, True)))
async def test_failed_incomplete_or_gateway_cache_responses_do_not_warm_state(complete: bool, cache_hit: bool) -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    reservation: Final = await _reserve(estimator, "ignored")
    result: Final = await estimator.finalize(
        reservation,
        wire=_wire(),
        request_started_at=clock.now,
        available_at=clock.now,
        completed=complete,
        cache_hit=cache_hit,
    )
    assert result.status == "unknown"
    following: Final = await _run(estimator, clock, "following", _wire())
    assert following.reason == "history_unavailable"


@pytest.mark.asyncio
async def test_provider_counts_are_memoized_and_failures_remain_unknown() -> None:
    clock: Final = _Clock()
    counter: Final = _Counter(unavailable=True)
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, counter)
    unavailable: Final = await _run(estimator, clock, "unavailable", _wire())
    assert unavailable.reason == "token_count_unavailable"
    assert unavailable.usage(5) is None
    counter.unavailable = False
    first: Final = await _run(estimator, clock, "first", _wire())
    assert first.reason == "history_unavailable"
    calls: Final = counter.calls
    second: Final = await _run(estimator, clock, "second", _wire())
    assert second.cache_read_input_tokens == 6000
    assert counter.calls == calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller,session,target",
    (
        ("other", "session", _TARGET),
        ("caller", "other", _TARGET),
        ("caller", "session", NativePredictionTarget(_MODEL, "other-key")),
        ("caller", "session", NativePredictionTarget(_MODEL, "test-provider-key", "https://other.test")),
    ),
)
async def test_scope_isolates_callers_sessions_and_baseline_credentials(
    caller: str, session: str, target: NativePredictionTarget
) -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "first", _wire())
    isolated: Final = await _reserve(estimator, "second", caller=caller, session=session, target=target)
    result: Final = await estimator.finalize(
        isolated, wire=_wire(), request_started_at=clock.now, available_at=clock.now
    )
    assert result.reason == "history_unavailable"


@pytest.mark.asyncio
async def test_uncacheable_prompt_does_not_need_history() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    wire: Final = httpx.Request(
        "POST",
        "https://provider.test/v1/messages",
        content=f'''{{
            "model":"{_MODEL}","max_tokens":10,"messages":[{{"role":"user","content":[
                {{"type":"text","text":"token ","cache_control":{{"type":"ephemeral","ttl":"1h"}}}}
            ]}}]
        }}''',
    )
    result: Final = await _run(estimator, clock, "short", wire)
    assert result.reason == "below_cache_minimum"
    assert result.input_tokens == 1
    assert result.cache_read_input_tokens == result.cache_creation_1h_input_tokens == 0


@pytest.mark.asyncio
async def test_ttl_changes_remain_unknown_until_every_possible_refreshed_entry_expires() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "first", _wire("1h"))
    warm: Final = await _run(estimator, clock, "warm", _wire("1h"))
    assert warm.cache_read_input_tokens == 6000
    clock.now = 10002.0
    changed: Final = await _run(estimator, clock, "changed", _wire("5m"))
    assert (changed.status, changed.reason) == ("unknown", "cache_ttl_changed")
    clock.now = 10303.0
    ambiguous: Final = await _run(estimator, clock, "ambiguous", _wire("5m"))
    assert ambiguous.reason == "cache_ttl_changed"
    clock.now = 13602.5
    refreshed: Final = await _run(estimator, clock, "refreshed", _wire("5m"))
    assert refreshed.reason == "cache_ttl_changed"
    clock.now = 17204.0
    expired: Final = await _run(estimator, clock, "expired", _wire("5m"))
    assert expired.status == "estimated"
    assert expired.cache_read_input_tokens == 0
    assert expired.cache_creation_5m_input_tokens == 6000


@pytest.mark.asyncio
async def test_unmarked_provider_cache_activity_is_unknown_but_plain_usage_is_estimable() -> None:
    clock: Final = _Clock()
    counter: Final = _Counter()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, counter)
    wire: Final = httpx.Request(
        "POST",
        "https://provider.test/v1/messages",
        content=f'''{{
            "model":"{_MODEL}","max_tokens":10,"messages":[{{"role":"user","content":"{"token " * 6000}"}}]
        }}''',
    )
    reservation: Final = await _reserve(estimator, "implicit")
    implicit: Final = await estimator.finalize(
        reservation,
        wire=wire,
        request_started_at=clock.now,
        available_at=clock.now,
        observed_cache_tokens=6000,
    )
    assert implicit.reason == "implicit_cache_without_breakpoints"
    assert counter.calls == 0
    ordinary: Final = await _run(estimator, clock, "ordinary", wire)
    assert ordinary.status == "estimated" and ordinary.input_tokens == 6000
    assert ordinary.cache_read_input_tokens == 0


@pytest.mark.asyncio
async def test_pruned_pending_reservation_cannot_advance_cache_state() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    first: Final = await _reserve(estimator, "first")
    for index in range(256):
        await _reserve(estimator, f"pending-{index}")
    pruned: Final = await estimator.finalize(
        first,
        wire=_wire(),
        request_started_at=clock.now,
        available_at=clock.now,
    )
    assert pruned.reason == "reservation_unavailable"


@pytest.mark.asyncio
async def test_long_running_request_keeps_the_cache_history_needed_at_its_start() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    await _run(estimator, clock, "seed", _wire("5m"))
    clock.now = 10001.0
    delayed: Final = await _reserve(estimator, "delayed")
    clock.now = 15000.0
    result: Final = await estimator.finalize(
        delayed,
        wire=_wire("5m"),
        request_started_at=10001.0,
        available_at=15000.0,
    )
    assert result.status == "estimated"
    assert result.cache_read_input_tokens == 6000


@pytest.mark.asyncio
async def test_redis_is_authoritative_and_faults_never_fall_back_to_local_warmth() -> None:
    clock: Final = _Clock()
    backend: Final = RedisCache(host="127.0.0.1", port=6398, namespace="baseline-state-test")
    cache: Final = DualCache(redis_cache=backend)
    first_process: Final = BaselineCacheEstimator(cache, clock, _Counter())
    second_process: Final = BaselineCacheEstimator(cache, clock, _Counter())
    try:
        await backend.async_delete_cache("unused")
    except (RedisError, OSError):
        pytest.skip("isolated integration Redis is unavailable")
    session: Final = str(id(first_process))
    first: Final = await _reserve(first_process, "first", session=session)
    clock.now += 0.1
    await first_process.finalize(first, wire=_wire(), request_started_at=first.reserved_at, available_at=clock.now)
    second: Final = await _reserve(second_process, "second", session=session)
    warm: Final = await second_process.finalize(
        second, wire=_wire(), request_started_at=clock.now, available_at=clock.now
    )
    assert warm.cache_read_input_tokens == 6000
    await backend.async_delete_cache(first.scope)
    lost: Final = await _reserve(first_process, "lost", session=session)
    unknown: Final = await first_process.finalize(
        lost, wire=_wire(), request_started_at=clock.now, available_at=clock.now
    )
    assert unknown.reason == "history_unavailable"
    broken_cache: Final = DualCache(redis_cache=RedisCache(host="127.0.0.1", port=1))
    broken: Final = BaselineCacheEstimator(broken_cache, clock, _Counter())
    prepared: Final = broken.prepare(
        caller_key_hash="caller",
        session_id=session,
        router_id="router",
        baseline_deployment_id="baseline",
        target=_TARGET,
        request_id="broken",
    )
    assert isinstance(prepared, BaselineReservation)
    refused: Final = await broken.reserve(prepared)
    assert isinstance(refused, BaselineCacheEstimate) and refused.reason == "state_unavailable"


class _FaultStore(_HistoryStore):
    def __init__(self) -> None:
        super().__init__(DualCache())
        self.fault: Literal["read", "before", "after", "conflict"] | None = None
        self.remaining = 0
        self.after_write: Callable[[], None] | None = None

    def arm(self, fault: Literal["read", "before", "after", "conflict"]) -> None:
        self.fault = fault
        self.remaining = 4 if fault == "conflict" else 1

    async def read(self, scope: str, now: float) -> _Snapshot | _StoreFailure:
        if self.fault == "read" and self.remaining:
            self.remaining -= 1
            return _StoreFailure()
        return await super().read(scope, now)

    async def exchange(self, scope: str, before: _Snapshot, after: _State, now: float) -> bool | _StoreFailure:
        if self.remaining and self.fault in ("before", "conflict"):
            self.remaining -= 1
            return _StoreFailure() if self.fault == "before" else False
        applied: Final = await super().exchange(scope, before, after, now)
        callback: Final = self.after_write
        if callback is not None:
            self.after_write = None
            callback()
        if self.remaining and self.fault == "after":
            self.remaining -= 1
            return _StoreFailure()
        return applied


def _fault_estimator(clock: _Clock) -> tuple[BaselineCacheEstimator, _FaultStore]:
    store: Final = _FaultStore()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    estimator.store = store
    return estimator, store


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("reserve", "finalize", "invalidate", "cancel"))
@pytest.mark.parametrize("fault", ("read", "before", "after", "conflict"))
async def test_storage_failure_retires_exact_request_on_recovery(
    operation: str, fault: Literal["read", "before", "after", "conflict"]
) -> None:
    clock: Final = _Clock()
    estimator, store = _fault_estimator(clock)
    await _run(estimator, clock, "seed", _wire())
    failed: Final = _prepare(estimator, "failed")
    if operation != "reserve":
        assert await estimator.reserve(failed) is None
    store.arm(fault)
    if operation == "reserve":
        result: Final = await estimator.reserve(failed)
        assert result is not None and result.status == "unknown"
        clock.advance(10)
        await estimator.invalidate(failed, "registration_failed", completed=True)
    elif operation == "finalize":
        finalized: Final = await estimator.finalize(
            failed, wire=_wire(), request_started_at=clock.now, available_at=clock.now
        )
        assert finalized.status == "unknown"
    elif operation == "invalidate":
        invalidated: Final = await estimator.invalidate(failed, "final_response_unknown")
        assert invalidated.status == "unknown"
    else:
        await estimator.cancel(failed)
    recovered: Final = await _run(estimator, clock, "recovered", _wire())
    assert recovered.reason != "pending_request"
    if operation == "cancel":
        assert recovered.cache_read_input_tokens == 6000
    else:
        assert recovered.reason == "history_unavailable"
    snapshot: Final = await store.read(failed.scope, clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    assert not any(item.request_id == failed.request_id for item in snapshot.state.pending)
    assert any(item.request_id == failed.request_id for item in snapshot.state.completed)


@pytest.mark.asyncio
async def test_active_repair_is_durable_and_delayed_finalizer_cannot_warm_or_close_it() -> None:
    clock: Final = _Clock()
    estimator, store = _fault_estimator(clock)
    await _run(estimator, clock, "seed", _wire())
    active: Final = await _reserve(estimator, "retrying")
    store.arm("after")
    await estimator.invalidate(active, "retrying_request", completed=False)
    second: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    second.store = store
    delayed: Final = await second.finalize(
        active, wire=_wire(), request_started_at=clock.now, available_at=clock.now
    )
    assert delayed.reason == "retrying_request"
    following: Final = await _run(second, clock, "while-active", _wire())
    assert following.reason == "pending_request"
    clock.advance(10)
    await estimator.invalidate(active, "retry_complete", completed=True)
    after: Final = await _run(second, clock, "after-active", _wire())
    assert after.reason == "history_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering", ("active_persisted", "finish_applied", "finish_deferred"))
async def test_failed_old_finalizer_preserves_newer_retry_in_every_atomic_order(ordering: str) -> None:
    clock: Final = _Clock()
    estimator, store = _fault_estimator(clock)
    await _run(estimator, clock, "seed", _wire())
    retrying: Final = await _reserve(estimator, "retry-race")
    if ordering == "active_persisted":
        await estimator.invalidate(retrying, "retry_active", completed=False)
        store.arm("read")
    elif ordering == "finish_applied":
        def activate_during_ack() -> None:
            estimator.defer(retrying, "retry_active", completed=False)

        store.after_write = activate_during_ack
        store.arm("after")
    else:
        store.arm("before")
    failed: Final = await estimator.finalize(
        retrying, wire=_wire(), request_started_at=clock.now, available_at=clock.now
    )
    assert failed.status == "unknown"
    if ordering == "finish_deferred":
        estimator.defer(retrying, "retry_active", completed=False)
    following: Final = await _run(estimator, clock, "during-retry", _wire())
    assert following.reason == "pending_request"
    snapshot: Final = await store.read(retrying.scope, clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    own: Final = next(item for item in snapshot.state.pending if item.request_id == retrying.request_id)
    assert own.invalidated_reason == "retry_active"
    clock.advance(10)
    await estimator.invalidate(retrying, "retry_terminal", completed=True)
    terminal: Final = await _run(estimator, clock, "after-retry-terminal", _wire())
    assert terminal.reason == "history_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", (False, True))
async def test_active_invalidation_supersedes_acknowledged_provisional_finish_but_not_logical_terminal(
    unsupported: bool,
) -> None:
    clock: Final = _Clock()
    estimator, store = _fault_estimator(clock)
    completed: Final = await _reserve(estimator, "provisional")
    wire: Final = httpx.Request("POST", "https://provider.test", content="invalid") if unsupported else _wire()
    await estimator.finalize(
        completed, wire=wire, request_started_at=clock.now, available_at=clock.now
    )
    await estimator.invalidate(completed, "new_retry", completed=False)
    during: Final = await _run(estimator, clock, "during-new-retry", _wire())
    assert during.reason == "pending_request"
    await estimator.invalidate(completed, "logical_terminal", completed=True)
    await estimator.invalidate(completed, "late_failure_callback", completed=False)
    snapshot: Final = await store.read(completed.scope, clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    assert not any(item.request_id == completed.request_id for item in snapshot.state.pending)
    final: Final = next(item for item in snapshot.state.completed if item.request_id == completed.request_id)
    assert final.terminal and final.estimate.status == "unknown"


@pytest.mark.asyncio
async def test_acknowledged_older_repair_cannot_erase_new_terminal_debt() -> None:
    clock: Final = _Clock()
    estimator, store = _fault_estimator(clock)
    active: Final = await _reserve(estimator, "overlap")
    estimator.defer(active, "still_active", completed=False)

    def complete_while_acknowledging() -> None:
        clock.advance(10)
        estimator.defer(active, "now_terminal", completed=True)

    store.after_write = complete_while_acknowledging
    intermediate: Final = await estimator.reserve(active)
    assert intermediate is not None and intermediate.reason == "still_active"
    recovered: Final = await _run(estimator, clock, "after-overlap", _wire())
    assert recovered.reason == "history_unavailable"
    snapshot: Final = await store.read(active.scope, clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    assert not any(item.request_id == active.request_id for item in snapshot.state.pending)
    completed: Final = next(item for item in snapshot.state.completed if item.request_id == active.request_id)
    assert completed.estimate.reason == "now_terminal"
    assert snapshot.state.uncertain_before >= 10010.0


@pytest.mark.asyncio
@pytest.mark.parametrize("early_result", ("pending", "completed"))
async def test_idempotent_results_publish_other_cleanup_without_removing_live_requests(early_result: str) -> None:
    clock: Final = _Clock()
    estimator, store = _fault_estimator(clock)
    existing: Final = await _reserve(estimator, "existing")
    if early_result == "completed":
        await estimator.finalize(existing, wire=_wire(), request_started_at=clock.now, available_at=clock.now)
    abandoned: Final = await _reserve(estimator, "abandoned")
    unrelated: Final = await _reserve(estimator, "unrelated-live")
    estimator.defer(abandoned, "abandoned", completed=True)
    if early_result == "pending":
        assert await estimator.reserve(existing) is None
    else:
        duplicate: Final = await estimator.finalize(
            existing, wire=_wire(), request_started_at=clock.now, available_at=clock.now
        )
        assert duplicate.status == "unknown"
    snapshot: Final = await store.read(existing.scope, clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    pending_ids: Final = frozenset(item.request_id for item in snapshot.state.pending)
    assert abandoned.request_id not in pending_ids
    assert unrelated.request_id in pending_ids
    assert (existing.request_id in pending_ids) == (early_result == "pending")


@pytest.mark.asyncio
async def test_terminal_repairs_survive_one_cache_ttl_and_late_active_callbacks_cannot_resurrect_pending() -> None:
    clock: Final = _Clock()
    estimator, store = _fault_estimator(clock)
    ended: Final = await _reserve(estimator, "ended")
    estimator.defer(ended, "terminal", completed=True)
    clock.advance(3601)
    other_scope: Final = _prepare(estimator, "other-scope", session="other")
    estimator.defer_cancel(other_scope)
    await _run(estimator, clock, "recover-after-hour", _wire())
    await estimator.invalidate(ended, "late_active_callback", completed=False)
    snapshot: Final = await store.read(ended.scope, clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    assert not any(item.request_id == ended.request_id for item in snapshot.state.pending)
    after: Final = await _run(estimator, clock, "after-late-callback", _wire())
    assert after.reason == "history_unavailable"


@pytest.mark.asyncio
async def test_restart_without_unpublished_repairs_preserves_pending_uncertainty() -> None:
    clock: Final = _Clock()
    estimator, store = _fault_estimator(clock)
    pending: Final = await _reserve(estimator, "lost-terminal")
    store.arm("before")
    await estimator.invalidate(pending, "terminal")
    restarted: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    restarted.store = store
    unknown: Final = await _run(restarted, clock, "after-restart", _wire())
    assert unknown.reason == "pending_request"
    clock.advance(7201)
    pruned: Final = await _run(restarted, clock, "after-retention", _wire())
    assert pruned.reason == "history_unavailable"


@pytest.mark.asyncio
async def test_repair_overflow_preserves_unrelated_pending_and_conservative_history() -> None:
    clock: Final = _Clock()
    estimator, _ = _fault_estimator(clock)
    pending: Final = await _reserve(estimator, "overflowed")
    estimator.defer(pending, "lost_repair", completed=True)
    for index in range(1024):
        estimator.defer(
            _prepare(estimator, f"overflow-{index}", session=f"scope-{index}"), "uncertain", completed=True
        )
    assert len(estimator.repairs) == 1024
    recovered: Final = await _run(estimator, clock, "overflow-recovery", _wire())
    assert recovered.reason == "pending_request"
    clock.advance(7201)
    bounded: Final = await _run(estimator, clock, "after-overflow-retention", _wire())
    assert bounded.reason == "history_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("read", "before", "after", "conflict"))
async def test_confirmed_no_upstream_cancel_removes_reserve_fault_without_cache_effect(
    fault: Literal["read", "before", "after", "conflict"]
) -> None:
    clock: Final = _Clock()
    estimator, store = _fault_estimator(clock)
    await _run(estimator, clock, "seed", _wire())
    cancelled: Final = _prepare(estimator, "never-dispatched")
    store.arm(fault)
    assert await estimator.reserve(cancelled) is not None
    await estimator.cancel(cancelled)
    following: Final = await _run(estimator, clock, "after-cancel", _wire())
    assert following.cache_read_input_tokens == 6000


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("read", "after"))
async def test_failed_reserve_recovery_tracks_live_request_until_terminal_callback(
    fault: Literal["read", "after"],
) -> None:
    clock: Final = _Clock()
    estimator, store = _fault_estimator(clock)
    await _run(estimator, clock, "seed", _wire())
    active: Final = _prepare(estimator, "unacknowledged")
    store.arm(fault)
    failed: Final = await estimator.reserve(active)
    assert failed is not None and failed.status == "unknown"
    during: Final = await _run(estimator, clock, "during-unacknowledged", _wire())
    assert during.reason == "pending_request"
    clock.advance(10)
    await estimator.invalidate(active, "last_observed_completion", completed=True)
    after: Final = await _run(estimator, clock, "after-unacknowledged", _wire())
    assert after.reason == "history_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_last", (False, True))
async def test_cancel_and_invalidation_debt_order_never_resurrects_pending(cancel_last: bool) -> None:
    clock: Final = _Clock()
    estimator, _ = _fault_estimator(clock)
    await _run(estimator, clock, "seed", _wire())
    reservation: Final = await _reserve(estimator, "ordered")
    if cancel_last:
        estimator.defer(reservation, "registration_unknown", completed=False)
        estimator.defer_cancel(reservation)
    else:
        estimator.defer_cancel(reservation)
        estimator.defer(reservation, "late_upstream_evidence", completed=False)
    following: Final = await _run(estimator, clock, "after-ordered", _wire())
    assert following.reason != "pending_request"
    if cancel_last:
        assert following.cache_read_input_tokens == 6000
    else:
        assert following.reason == "history_unavailable"


class _FailingClock:
    def __call__(self) -> float:
        raise RuntimeError("clock unavailable")


def test_preparation_and_emergency_deferral_survive_clock_failure() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    reservation: Final = _prepare(estimator, "clock-fault")
    estimator.clock = _FailingClock()
    failed: Final = estimator.prepare(
        caller_key_hash="caller", session_id="session", router_id="router",
        baseline_deployment_id="baseline", target=_TARGET, request_id="failed-prepare",
    )
    assert isinstance(failed, BaselineCacheEstimate) and failed.status == "unknown"
    estimator.defer(reservation, "clock_fault", completed=True)
    repair: Final = next(iter(estimator.repairs.values()))
    assert repair.reason == "clock_fault" and repair.kind == "terminal"
    assert repair.observed_at >= reservation.reserved_at
    estimator.defer_cancel(reservation)
    cancelled: Final = next(iter(estimator.repairs.values()))
    assert cancelled.reason is None and cancelled.kind == "cancel"


@pytest.mark.asyncio
@pytest.mark.parametrize("publish_fault_first", (False, True))
@pytest.mark.parametrize("available_at", (10001.0, 10002.0, 10003.0))
async def test_independent_success_cannot_cross_fault_cutoff_by_finalizer_order(
    publish_fault_first: bool, available_at: float
) -> None:
    clock: Final = _Clock()
    writer: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    invalidator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    invalidator.store = writer.store
    seed: Final = await _reserve(writer, "delayed-independent-success")
    clock.now = 10001.0
    failed: Final = await _reserve(invalidator, "terminal-failure")
    clock.now = 10002.0
    invalidator.defer(failed, "failed_request", completed=True)
    if publish_fault_first:
        assert await invalidator.reserve(failed) is not None
    clock.now = 10004.0
    await writer.finalize(seed, wire=_wire(), request_started_at=10000.0, available_at=available_at)
    if not publish_fault_first:
        assert await invalidator.reserve(failed) is not None
    snapshot: Final = await writer.store.read(seed.scope, clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    assert snapshot.state.invalidated_before == 10002.0
    assert not snapshot.state.pending
    assert next(item for item in snapshot.state.completed if item.request_id == failed.request_id).terminal
    if available_at > 10002.0:
        assert len(snapshot.state.versions) == 1
        assert snapshot.state.versions[0].expires_at == 13600.0
    else:
        assert not snapshot.state.versions
    clock.now = 10005.0
    following: Final = await _run(writer, clock, "after-terminal-and-independent-success", _wire())
    if available_at > 10002.0:
        assert following.cache_read_input_tokens == 6000
    else:
        assert following.reason == "history_unavailable"
    matching: Final = await _run(writer, clock, "matching-after-restored-evidence", _wire())
    assert matching.cache_read_input_tokens == 6000


@pytest.mark.asyncio
async def test_initial_history_timestamp_and_no_upstream_cancel_are_not_fault_cutoffs() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    initial: Final = await _reserve(estimator, "instantaneous-first-success")
    await estimator.finalize(initial, wire=_wire(), request_started_at=clock.now, available_at=clock.now)
    cancelled: Final = await _reserve(estimator, "never-dispatched")
    await estimator.cancel(cancelled)
    snapshot: Final = await estimator.store.read(initial.scope, clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    assert snapshot.state.invalidated_before is None
    matching: Final = await _run(estimator, clock, "matches-initial-history-timestamp", _wire())
    assert matching.cache_read_input_tokens == 6000


@pytest.mark.asyncio
async def test_persisted_pre_cutoff_version_is_removed_before_matching() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    initial: Final = await _reserve(estimator, "old-persisted-success")
    await estimator.finalize(initial, wire=_wire(), request_started_at=clock.now, available_at=clock.now)
    snapshot: Final = await estimator.store.read(initial.scope, clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    cutoff_state: Final = snapshot.state.model_copy(update=MappingProxyType({"invalidated_before": 10001.0}))
    assert await estimator.store.exchange(initial.scope, snapshot, cutoff_state, clock.now) is True
    clock.now = 10002.0
    following: Final = await _run(estimator, clock, "must-not-read-persisted-old-evidence", _wire())
    assert following.reason == "history_unavailable"
    matching: Final = await _run(estimator, clock, "fresh-evidence-after-persisted-cutoff", _wire())
    assert matching.cache_read_input_tokens == 6000


@pytest.mark.asyncio
async def test_overflow_fault_cutoff_survives_later_missing_history_initialization_and_cancel() -> None:
    clock: Final = _Clock(now=10004.0)
    estimator: Final = BaselineCacheEstimator(DualCache(), clock, _Counter())
    estimator.uncertainty_floor = 10002.0
    reservation: Final = await _reserve(estimator, "missing-scope-after-overflow")
    await estimator.cancel(reservation)
    snapshot: Final = await estimator.store.read(reservation.scope, clock.now)
    assert isinstance(snapshot, _Snapshot) and snapshot.state is not None
    assert snapshot.state.uncertain_before == 10004.0
    assert snapshot.state.invalidated_before == 10002.0


def test_legacy_state_does_not_infer_an_explicit_fault_from_missing_history() -> None:
    snapshot: Final = _State.model_validate_json('{"uncertain_before":10000.0}')
    assert snapshot.invalidated_before is None
