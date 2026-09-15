import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal, Protocol, runtime_checkable

import httpx
import pytest
from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.prompt_cache_prediction import NativePredictionTarget
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # inject the native provider's existing HTTP client owner
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.autorouter_baseline_cache import (
    AutoRouterBaselineCache,
    BaselineCacheContext,
    cancel_baseline_cache,
    finalize_baseline_cache,
    invalidate_baseline_cache,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.proxy.spend_tracking.autorouter_baseline_cache import (
    BaselineCacheEstimator,
    BaselineReservation,
    _HistoryStore,  # pyright: ignore[reportPrivateUsage]  # inject faults into the existing storage dependency
    _Snapshot,  # pyright: ignore[reportPrivateUsage]  # retain the storage operation's typed contract
    _State,  # pyright: ignore[reportPrivateUsage]  # retain the storage operation's typed contract
    _StoreFailure,  # pyright: ignore[reportPrivateUsage]  # distinguish returned storage faults from raised exceptions
    unknown_estimate,
)
from litellm.proxy.utils import InternalUsageCache, ProxyLogging
from litellm.router import Router
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.router import RetryPolicy
from litellm.types.utils import CallTypes, ModelResponse, StandardLoggingRoutingDecision, Usage

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OBJECTS: Final = TypeAdapter(dict[str, object])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_EVENTS_ADAPTER: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])

_EVENTS: Final = _EVENTS_ADAPTER.validate_json("""
[
  {
    "type": "message_start",
    "message": {
      "id": "msg_baseline_test",
      "type": "message",
      "role": "assistant",
      "model": "claude-sonnet-5",
      "content": [],
      "stop_reason": null,
      "stop_sequence": null,
      "usage": {
        "input_tokens": 1000,
        "output_tokens": 0,
        "cache_creation_input_tokens": 5000,
        "cache_read_input_tokens": 0,
        "cache_creation": {
          "ephemeral_5m_input_tokens": 0,
          "ephemeral_1h_input_tokens": 5000
        }
      }
    }
  },
  {
    "type": "content_block_start",
    "index": 0,
    "content_block": {
      "type": "text",
      "text": ""
    }
  },
  {
    "type": "content_block_delta",
    "index": 0,
    "delta": {
      "type": "text_delta",
      "text": "OK"
    }
  },
  {
    "type": "content_block_stop",
    "index": 0
  },
  {
    "type": "message_delta",
    "delta": {
      "stop_reason": "end_turn",
      "stop_sequence": null
    },
    "usage": {
      "output_tokens": 10
    }
  },
  {
    "type": "message_stop"
  }
]
""")

_COMPLETED: Final = _JSON_OBJECT.validate_json("""
{
  "id": "msg_baseline_test",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-5",
  "content": [
    {
      "type": "text",
      "text": "OK"
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 1000,
    "output_tokens": 10,
    "cache_creation_input_tokens": 5000,
    "cache_read_input_tokens": 0,
    "cache_creation": {
      "ephemeral_5m_input_tokens": 0,
      "ephemeral_1h_input_tokens": 5000
    }
  }
}
""")

_MODELS_JSON: Final = """
[
  {
    "model_name": "opus",
    "litellm_params": {
      "model": "anthropic/claude-opus-5",
      "api_key": "test-baseline"
    },
    "model_info": {
      "id": "baseline"
    }
  }
]
"""

_MESSAGES_JSON: Final = """
[
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "stable",
        "cache_control": {
          "type": "ephemeral",
          "ttl": "1h"
        }
      },
      {
        "type": "text",
        "text": "question"
      }
    ]
  }
]
"""


@runtime_checkable
class _NativeStream(Protocol):
    def __aiter__(self) -> AsyncIterator[bytes]: ...


async def _consume_stream(stream: _NativeStream) -> tuple[bytes, ...]:
    return tuple([chunk async for chunk in stream])


class _Capture(CustomLogger):
    def __init__(self, call_ids: frozenset[str]) -> None:
        self.payloads: asyncio.Queue[Mapping[str, object]] = asyncio.Queue()
        self.failures: asyncio.Queue[tuple[Mapping[str, object], Exception]] = asyncio.Queue()
        self.call_ids = call_ids

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if kwargs.get("litellm_call_id") not in self.call_ids:
            return
        payload: Final = _OBJECTS.validate_python(kwargs.get("standard_logging_object"))
        self.payloads.put_nowait(payload)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if kwargs.get("litellm_call_id") not in self.call_ids:
            return
        exception: Final = kwargs.get("exception")
        assert isinstance(exception, Exception)
        self.failures.put_nowait((_OBJECTS.validate_python(kwargs.get("standard_logging_object")), exception))


async def _count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
    assert model == "claude-opus-5"
    return 6000 if "question" in json.dumps(_JSON_OBJECT.validate_python(body)) else 5000


def _upstream(request: httpx.Request) -> httpx.Response:
    if _JSON_OBJECT.validate_json(request.content).get("stream") is True:
        return httpx.Response(
            200,
            content="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in _EVENTS),
            headers=MappingProxyType({"content-type": "text/event-stream"}),
            request=request,
        )
    return httpx.Response(200, json=_COMPLETED, request=request)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("trusted_stamp", (True, False))
async def test_native_dispatch_reserves_before_upstream_and_stamps_before_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    trusted_stamp: bool,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    router: Final = Router(model_list=_MESSAGES.validate_json(_MODELS_JSON))

    def get_router() -> Router:
        return router

    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    request_prefix: Final = f"native-baseline-{stream}-{trusted_stamp}"
    capture: Final = _Capture(frozenset((f"{request_prefix}-0", f"{request_prefix}-1")))
    monkeypatch.setattr(litellm, "callbacks", [hook, capture])  # mutable-ok: LiteLLM mutates its callback registries
    monkeypatch.setattr(litellm, "success_callback", [])  # mutable-ok: LiteLLM mutates its callback registries
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])  # mutable-ok: callback registry is mutable
    client: Final = AsyncHTTPHandler()
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(_upstream)) as transport:
        client.client = transport

        async def turn(index: int) -> None:
            request_kwargs: Final = _OBJECTS.validate_json(
                '{"litellm_metadata":{"user_api_key_hash":"test-caller-hash"},"litellm_session_id":"baseline-session"}'
            )
            Router._record_routing_decision(  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # exercise the production router stamp owner with its legacy kwargs contract
                request_kwargs,
                StandardLoggingRoutingDecision(
                    router_model_name="test-router",
                    router_type="complexity",
                    routed_model="sonnet",
                    cause="heuristic_scorer",
                    conversation_continuing=True,
                    savings_baseline_model="anthropic/claude-opus-5",
                    savings_baseline_deployment_id="baseline",
                ),
            )
            metadata: Final = _OBJECTS.validate_python(request_kwargs["litellm_metadata"])
            sent_metadata: Final = (
                _OBJECTS.validate_python(
                    MappingProxyType(
                        {
                            **metadata,
                            "_autorouter_baseline_route": _OBJECTS.validate_json(
                                '{"router_name":"test-router","baseline_model":"anthropic/claude-opus-5","baseline_deployment_id":"baseline"}'
                            ),
                        }
                    )
                )
                if not trusted_stamp
                else metadata
            )
            response: Final[object] = await litellm.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # the native SDK entrypoint has legacy untyped kwargs
                model="anthropic/claude-sonnet-5",
                api_key="test-selected",
                max_tokens=16,
                stream=stream,
                messages=_MESSAGES.validate_json(_MESSAGES_JSON),
                client=client,
                litellm_metadata=sent_metadata,
                litellm_session_id="baseline-session",
                litellm_call_id=f"{request_prefix}-{index}",
            )
            if stream:
                assert isinstance(response, _NativeStream)
                chunks: Final = await _consume_stream(response)
                assert chunks
            payload: Final = await asyncio.wait_for(capture.payloads.get(), timeout=20)
            estimate: Final = _OBJECTS.validate_python(payload.get("autorouter_savings_estimate"))
            if not trusted_stamp:
                assert estimate["status"] == "unknown", estimate
                assert payload["autorouter_savings"] is None
            elif index == 0:
                assert estimate["status"] == "unknown", estimate
                assert estimate["reason"] == "history_unavailable", estimate
                assert payload["autorouter_savings"] is None
            else:
                assert estimate["status"] == "estimated", estimate
                assert estimate["cache_read_input_tokens"] == 5000
                assert estimate["cache_creation_1h_input_tokens"] == 0
                saving: Final = payload["autorouter_savings"]
                assert isinstance(saving, float) and saving < 0

        await turn(0)
        await turn(1)


_RETRY_MODELS_JSON: Final = """
[
  {
    "model_name": "test-router",
    "litellm_params": {
      "model": "auto_router/complexity_router",
      "complexity_router_config": {
        "tiers": {"SIMPLE": "sonnet", "MEDIUM": "sonnet", "COMPLEX": "sonnet", "REASONING": "opus"},
        "session_affinity": false
      }
    }
  },
  {
    "model_name": "sonnet",
    "litellm_params": {"model": "anthropic/claude-sonnet-5", "api_key": "test-selected"},
    "model_info": {"id": "selected"}
  },
  {
    "model_name": "opus",
    "litellm_params": {"model": "anthropic/claude-opus-5", "api_key": "test-baseline"},
    "model_info": {"id": "baseline"}
  }
]
"""


class _AttemptCapture(_Capture):
    def __init__(self, call_ids: frozenset[str]) -> None:
        super().__init__(call_ids)
        self.attempts: tuple[Logging, ...] = ()

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if (
            isinstance(logging_obj, Logging)
            and logging_obj.litellm_call_id in self.call_ids
            and call_type == CallTypes.anthropic_messages
        ):
            self.attempts = (*self.attempts, logging_obj)  # rebind-ok: record the native dispatch attempts


def _logging(request_id: str, stream: bool = False) -> Logging:
    return Logging(  # pyright: ignore[reportUnknownMemberType]  # legacy constructor owns request logging state
        model="anthropic/claude-sonnet-5",
        messages=_MESSAGES.validate_json(_MESSAGES_JSON),
        stream=stream,
        call_type=CallTypes.anthropic_messages.value,
        start_time=datetime.now(),  # noqa: DTZ005  # native Logging uses naive timestamps throughout request timing
        litellm_call_id=request_id,
        function_id=request_id,
        kwargs=_OBJECTS.validate_json('{"litellm_session_id":"baseline-session"}'),
    )


def _native_metadata() -> Mapping[str, object]:
    kwargs: Final = _OBJECTS.validate_json('{"litellm_metadata":{"user_api_key_hash":"test-caller-hash"}}')
    Router._record_routing_decision(  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # production owner creates the trusted route stamp
        kwargs,
        StandardLoggingRoutingDecision(
            router_model_name="test-router",
            router_type="complexity",
            routed_model="sonnet",
            cause="heuristic_scorer",
            conversation_continuing=True,
            savings_baseline_model="anthropic/claude-opus-5",
            savings_baseline_deployment_id="baseline",
        ),
    )
    return _OBJECTS.validate_python(kwargs["litellm_metadata"])


def _install_callbacks(monkeypatch: pytest.MonkeyPatch, hook: AutoRouterBaselineCache, capture: _Capture) -> None:
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(litellm, "callbacks", [hook, capture])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "success_callback", [])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "failure_callback", [])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "_async_failure_callback", [capture])  # mutable-ok: callback registry is mutable


async def _native_call(client: AsyncHTTPHandler, logging_obj: Logging) -> object:
    return await litellm.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # exercise the actual decorated native SDK entry point
        model="anthropic/claude-sonnet-5",
        api_key="test-selected",
        max_tokens=16,
        num_retries=0,
        messages=_MESSAGES.validate_json(_MESSAGES_JSON),
        client=client,
        litellm_metadata=_native_metadata(),
        litellm_session_id="baseline-session",
        litellm_logging_obj=logging_obj,
        litellm_call_id=logging_obj.litellm_call_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_native_router_retry_with_shared_logging_is_unknown_and_cannot_warm_baseline(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    router: Final = Router(
        model_list=_MESSAGES.validate_json(_RETRY_MODELS_JSON),
        num_retries=1,
        retry_policy=RetryPolicy(RateLimitErrorRetries=1),
        disable_cooldowns=True,
    )

    def get_router() -> Router:
        return router

    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    request_id: Final = f"native-router-retry-{stream}"
    following_id: Final = f"native-after-retry-{stream}"
    capture: Final = _AttemptCapture(frozenset((request_id, following_id)))
    monkeypatch.setattr(litellm, "callbacks", [hook, capture])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "success_callback", [])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])  # mutable-ok: callback registry is mutable
    requests: Final[list[httpx.Request]] = []  # mutable-ok: transport records the wire attempts in order

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                429,
                text='{"type":"error","error":{"type":"rate_limit_error","message":"retry"}}',
                headers=MappingProxyType({"retry-after": "0"}),
                request=request,
            )
        return _upstream(request)

    shared_logging: Final = _logging(request_id, stream)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as transport:
        client.client = transport

        async def call(logging_obj: Logging) -> Mapping[str, object]:
            response: Final[object] = await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # dynamic router endpoint exercises the real retry loop
                model="test-router",
                max_tokens=16,
                stream=stream,
                messages=_MESSAGES.validate_json(_MESSAGES_JSON),
                litellm_logging_obj=logging_obj,
                litellm_metadata=_OBJECTS.validate_json('{"user_api_key_hash":"test-caller-hash"}'),
                litellm_session_id="baseline-session",
            )
            if stream:
                assert isinstance(response, _NativeStream)
                assert await _consume_stream(response)
            return await asyncio.wait_for(capture.payloads.get(), timeout=20)

        retried: Final = await call(shared_logging)
        assert len(requests) == 2
        assert capture.attempts == (shared_logging, shared_logging)
        assert retried["autorouter_savings"] is None
        assert _OBJECTS.validate_python(retried["autorouter_savings_estimate"])["reason"] == "retried_request"
        assert shared_logging.baseline_cache_context is None
        following: Final = await call(_logging(following_id, stream))
        assert len(requests) == 3
        assert following["autorouter_savings"] is None
        assert _OBJECTS.validate_python(following["autorouter_savings_estimate"])["reason"] == "history_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidation_fails", (False, True))
async def test_native_thinking_repair_retry_invalidates_baseline_without_changing_logging_attempt(
    monkeypatch: pytest.MonkeyPatch, invalidation_fails: bool
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    router: Final = Router(model_list=_MESSAGES.validate_json(_RETRY_MODELS_JSON), num_retries=0)

    def get_router() -> Router:
        return router

    cache: Final = InternalUsageCache(DualCache())
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count, clock=clock)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    capture: Final = _AttemptCapture(frozenset(("native-thinking-repair",)))
    monkeypatch.setattr(litellm, "callbacks", [hook, capture])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "success_callback", [])  # mutable-ok: callback registry is mutable
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])  # mutable-ok: callback registry is mutable
    requests: Final[list[httpx.Request]] = []  # mutable-ok: compare the original and repaired wire bodies

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            clock.failures_remaining = 2 if invalidation_fails else 0
            return httpx.Response(
                400,
                text='{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.content.0: Invalid `signature` in `thinking` block"}}',
                request=request,
            )
        return _upstream(request)

    shared_logging: Final = _logging("native-thinking-repair")
    messages: Final = _MESSAGES.validate_json("""
[
  {"role":"user","content":[{"type":"text","text":"stable","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
  {"role":"assistant","content":[{"type":"thinking","thinking":"reasoning","signature":"invalid"},{"type":"text","text":"answer"}]},
  {"role":"user","content":"question"}
]
""")
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as transport:
        client.client = transport
        await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # native HTTP owner performs the repair retry
            model="test-router",
            max_tokens=16,
            messages=messages,
            litellm_logging_obj=shared_logging,
            litellm_metadata=_OBJECTS.validate_json('{"user_api_key_hash":"test-caller-hash"}'),
            litellm_session_id="baseline-session",
        )
        payload: Final = await asyncio.wait_for(capture.payloads.get(), timeout=20)
    assert len(requests) == 2
    assert b'"signature": "invalid"' in requests[0].content
    assert b'"signature"' not in requests[1].content
    assert capture.attempts == (shared_logging,)
    assert payload["autorouter_savings"] is None
    assert _OBJECTS.validate_python(payload["autorouter_savings_estimate"])["reason"] == (
        "estimator_unavailable" if invalidation_fails else "retried_request"
    )
    assert shared_logging.baseline_cache_context is None


async def _reservation(estimator: BaselineCacheEstimator, request_id: str) -> BaselineReservation:
    reservation: Final = estimator.prepare(
        caller_key_hash="test-caller-hash",
        session_id="baseline-session",
        router_id="test-router",
        baseline_deployment_id="baseline",
        target=NativePredictionTarget("claude-opus-5", "test-baseline", "https://api.anthropic.com"),
        request_id=request_id,
    )
    assert isinstance(reservation, BaselineReservation)
    assert await estimator.reserve(reservation) is None
    return reservation


class _Clock:
    def __init__(self) -> None:
        self.now: float = 1000.0
        self.failures_remaining = 0
        self.failure_at: int | None = None
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls == self.failure_at:
            raise ValueError("injected estimator clock failure")
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise ValueError("injected estimator clock failure")
        return self.now


class _FailingHistoryStore(_HistoryStore):
    def __init__(self) -> None:
        super().__init__(DualCache())
        self.fail_next_exchange: Literal["before", "after"] | None = None

    async def exchange(self, scope: str, before: _Snapshot, after: _State, now: float) -> bool | _StoreFailure:
        fault: Final = self.fail_next_exchange
        self.fail_next_exchange = None
        if fault == "before":
            return _StoreFailure()
        applied: Final = await super().exchange(scope, before, after, now)
        return _StoreFailure() if fault == "after" else applied


async def _native_logging(
    estimator: BaselineCacheEstimator, clock: _Clock, request_id: str
) -> Logging:
    logging_obj: Final = _logging(request_id)
    logging_obj.baseline_cache_context = BaselineCacheContext(estimator, await _reservation(estimator, request_id))
    _stamp_native_wire(logging_obj, clock)
    return logging_obj


def _stamp_native_wire(logging_obj: Logging, clock: _Clock) -> None:
    timestamp: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # native Logging uses naive request timestamps
    logging_obj.completion_start_time = timestamp  # rebind-ok: inject native request timing on the real Logging object
    wire: Final = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json=_JSON_OBJECT.validate_python(
            MappingProxyType({"model": "claude-sonnet-5", "messages": _MESSAGES.validate_json(_MESSAGES_JSON)})
        ),
    )
    logging_obj.model_call_details.update(  # pyright: ignore[reportUnknownMemberType]  # native response evidence for real Logging
        httpx_response=_upstream(wire),
        api_call_start_time=timestamp,
        completion_start_time=timestamp,
        custom_llm_provider="anthropic",
        response_cost=0.125,
        stream=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("defer_clock_fails", (False, True))
async def test_finalize_failure_invalidates_pending_reservation_and_preserves_spend_logging(
    monkeypatch: pytest.MonkeyPatch, defer_clock_fails: bool
) -> None:
    clock: Final = _Clock()
    clock.now = round(time.time(), 6)
    initial_time: Final = clock.now
    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=_count, clock=clock)
    logging_obj: Final = await _native_logging(estimator, clock, "finalizer-fault")
    capture: Final = _Capture(frozenset(("finalizer-fault",)))
    monkeypatch.setattr(litellm, "callbacks", (capture,))
    monkeypatch.setattr(litellm, "success_callback", ())
    monkeypatch.setattr(litellm, "_async_success_callback", (capture,))
    clock.failures_remaining = 2 if defer_clock_fails else 1
    await logging_obj.async_success_handler(result=ModelResponse(model="claude-sonnet-5"))
    payload: Final = await asyncio.wait_for(capture.payloads.get(), timeout=5)
    assert payload["response_cost"] == 0.125
    assert logging_obj.baseline_cache_estimate == unknown_estimate("estimator_unavailable")
    assert logging_obj.baseline_cache_context is None
    recovery_boundary: Final = max((clock.now, *(repair.observed_at for repair in estimator.repairs.values())))
    assert recovery_boundary - initial_time < 60
    clock.now = round(recovery_boundary + 1.0, 6)
    following: Final = await _native_logging(estimator, clock, "after-finalizer-fault")
    await following.async_success_handler(result=ModelResponse(model="claude-sonnet-5"))
    assert following.baseline_cache_estimate == unknown_estimate("history_unavailable")
    clock.now += 1.0
    matching: Final = await _native_logging(estimator, clock, "matching-after-finalizer-fault")
    await matching.async_success_handler(result=ModelResponse(model="claude-sonnet-5"))
    estimate: Final = matching.baseline_cache_estimate
    assert estimate is not None and estimate.status == "estimated"
    assert estimate.cache_read_input_tokens == 5000


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_owns_reservation", (False, True))
@pytest.mark.parametrize("fault", ("exception", "before", "after"))
async def test_late_finalize_failure_cleans_original_reservation_without_overwriting_replacement(
    retry_owns_reservation: bool, fault: Literal["exception", "before", "after"]
) -> None:
    clock: Final = _Clock()
    counting: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def delayed_count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        counting.set()
        await release.wait()
        return await _count(model, api_key, body)

    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=delayed_count, clock=clock)
    store: Final = _FailingHistoryStore()
    estimator.store = store
    logging_obj: Final = await _native_logging(estimator, clock, "late-finalizer-fault")
    prepared_replacement: Final = (
        None
        if retry_owns_reservation
        else BaselineCacheContext(estimator, await _reservation(estimator, "replacement-after-fault"))
    )
    sentinel: Final = unknown_estimate("replacement_estimate")
    finalizing: Final = asyncio.create_task(
        logging_obj._prepare_baseline_cache_estimate(ModelResponse())  # pyright: ignore[reportPrivateUsage]  # inject a delayed error through the real Logging owner
    )
    await asyncio.wait_for(counting.wait(), timeout=5)
    if retry_owns_reservation:
        await invalidate_baseline_cache(logging_obj, "retried_request")
    replacement: Final = logging_obj.baseline_cache_context if retry_owns_reservation else prepared_replacement
    assert replacement is not None
    logging_obj.baseline_cache_context = replacement  # rebind-ok: new context takes ownership while the old finalizer waits
    logging_obj.baseline_cache_estimate = sentinel
    clock.failures_remaining = 1 if fault == "exception" else 0
    store.fail_next_exchange = fault if fault != "exception" else None
    release.set()
    await asyncio.wait_for(finalizing, timeout=5)
    assert logging_obj.baseline_cache_context is replacement
    assert logging_obj.baseline_cache_estimate is sentinel
    clock.now = 1001.0
    if retry_owns_reservation:
        overlapping: Final = await _native_logging(estimator, clock, "after-stale-finalizer-fault")
        await finalize_baseline_cache(overlapping, ModelResponse())
        assert overlapping.baseline_cache_estimate == unknown_estimate("pending_request")
        await invalidate_baseline_cache(logging_obj, "retried_request", completed=True)
        assert logging_obj.baseline_cache_context is None
        return
    wire: Final = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json=_JSON_OBJECT.validate_python(
            MappingProxyType({"model": "claude-sonnet-5", "messages": _MESSAGES.validate_json(_MESSAGES_JSON)})
        ),
    )
    recovered: Final = await estimator.finalize(
        replacement.reservation, wire=wire, request_started_at=clock.now, available_at=clock.now
    )
    assert recovered == unknown_estimate("history_unavailable")


@pytest.mark.asyncio
async def test_late_finalizer_cannot_overwrite_retry_invalidation() -> None:
    counting: Final = asyncio.Event()
    release_count: Final = asyncio.Event()
    clock: Final = _Clock()

    async def delayed_count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int:
        counting.set()
        await release_count.wait()
        return await _count(model, api_key, body)

    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=delayed_count, clock=clock)
    reservation: Final = await _reservation(estimator, "late-finalizer")
    logging_obj: Final = _logging("late-finalizer")
    logging_obj.baseline_cache_context = BaselineCacheContext(estimator, reservation)
    timestamp: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # match native request timing evidence
    wire: Final = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json=_JSON_OBJECT.validate_python(
            MappingProxyType({"model": "claude-sonnet-5", "messages": _MESSAGES.validate_json(_MESSAGES_JSON)})
        ),
    )
    logging_obj.model_call_details.update(  # pyright: ignore[reportUnknownMemberType]  # populate the real native response evidence
        httpx_response=_upstream(wire),
        api_call_start_time=timestamp,
        completion_start_time=timestamp,
        custom_llm_provider="anthropic",
        stream=False,
    )
    finalizing: Final = asyncio.create_task(finalize_baseline_cache(logging_obj, ModelResponse()))
    counting_started: Final = asyncio.create_task(counting.wait())
    await asyncio.wait((finalizing, counting_started), timeout=5, return_when=asyncio.FIRST_COMPLETED)
    if finalizing.done():
        counting_started.cancel()
        await finalizing
    assert counting.is_set(), logging_obj.baseline_cache_estimate
    await logging_obj.invalidate_baseline_cache_estimate("retried_request")
    invalidated: Final = logging_obj.baseline_cache_context
    assert invalidated is not None and invalidated.invalidated
    release_count.set()
    await asyncio.wait_for(finalizing, timeout=5)
    assert logging_obj.baseline_cache_context is invalidated
    assert logging_obj.baseline_cache_estimate == unknown_estimate("retried_request")
    clock.now = 1200.0
    await finalize_baseline_cache(logging_obj, ModelResponse())
    assert logging_obj.baseline_cache_context is None
    assert logging_obj.baseline_cache_estimate == unknown_estimate("retried_request")
    clock.now = 4601.0
    following: Final = await _reservation(estimator, "after-retry-start-ttl")
    estimate: Final = await estimator.finalize(
        following, wire=wire, request_started_at=clock.now, available_at=clock.now
    )
    assert estimate.status == "unknown"
    assert estimate.reason == "history_unavailable"


@pytest.mark.asyncio
async def test_native_provider_error_and_failure_telemetry_survive_invalidation_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router: Final = Router(model_list=_MESSAGES.validate_json(_MODELS_JSON))

    def get_router() -> Router:
        return router

    clock: Final = _Clock()
    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count, clock=clock)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    logging_obj: Final = _logging("native-failure-cleanup")
    capture: Final = _Capture(frozenset((logging_obj.litellm_call_id,)))
    _install_callbacks(monkeypatch, hook, capture)

    def upstream(request: httpx.Request) -> httpx.Response:
        clock.failures_remaining = 2
        return httpx.Response(
            401,
            json=_JSON_OBJECT.validate_json(
                '{"type":"error","error":{"type":"authentication_error","message":"original upstream authentication failure"}}'
            ),
            request=request,
        )

    client: Final = AsyncHTTPHandler()
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as transport:
        client.client = transport
        with pytest.raises(litellm.AuthenticationError, match="original upstream authentication failure") as caught:
            await _native_call(client, logging_obj)
    payload, logged_exception = await asyncio.wait_for(capture.failures.get(), timeout=5)
    assert caught.value.status_code == 401
    assert logged_exception is caught.value
    assert payload["status"] == "failure"
    assert capture.failures.empty()
    assert logging_obj.baseline_cache_estimate == unknown_estimate("estimator_unavailable")
    assert logging_obj.baseline_cache_context is not None and logging_obj.baseline_cache_context.invalidated


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("setup", "prepare", "reserve", "register_storage"))
async def test_native_generation_and_success_telemetry_survive_predispatch_estimator_fault(
    monkeypatch: pytest.MonkeyPatch, phase: Literal["setup", "prepare", "reserve", "register_storage"]
) -> None:
    router: Final = Router(model_list=_MESSAGES.validate_json(_MODELS_JSON))

    def get_router() -> Router:
        if phase == "setup":
            raise ValueError("injected baseline deployment lookup failure")
        return router

    clock: Final = _Clock()
    clock.failure_at = 1 if phase == "prepare" else 2 if phase == "reserve" else None
    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count, clock=clock)
    store: Final = _FailingHistoryStore()
    estimator.store = store
    store.fail_next_exchange = "before" if phase == "register_storage" else None
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    logging_obj: Final = _logging(f"native-{phase}-fault")
    capture: Final = _Capture(frozenset((logging_obj.litellm_call_id,)))
    _install_callbacks(monkeypatch, hook, capture)
    requests: Final[list[httpx.Request]] = []  # mutable-ok: record actual native dispatch despite estimator faults

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if phase in ("reserve", "register_storage"):
            assert logging_obj.baseline_cache_context is not None
            assert logging_obj.baseline_cache_context.invalidated
        return _upstream(request)

    client: Final = AsyncHTTPHandler()
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as transport:
        client.client = transport
        response: Final = await _native_call(client, logging_obj)
        assert response is not None
        payload: Final = await asyncio.wait_for(capture.payloads.get(), timeout=5)
    assert len(requests) == 1
    assert payload["status"] == "success"
    cost: Final = payload["response_cost"]
    assert isinstance(cost, float) and cost > 0
    assert logging_obj.baseline_cache_estimate == unknown_estimate(
        "state_unavailable" if phase == "register_storage" else "estimator_unavailable"
    )
    assert logging_obj.baseline_cache_context is None
    assert capture.failures.empty()


async def _wait_for_context(logging_obj: Logging, *, invalidated: bool = False) -> BaselineCacheContext:
    while logging_obj.baseline_cache_context is None or logging_obj.baseline_cache_context.invalidated is not invalidated:
        await asyncio.sleep(0)
    return logging_obj.baseline_cache_context


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_registration", (False, True))
async def test_late_registration_cannot_cancel_retry_owned_reservation(cancel_registration: bool) -> None:
    router: Final = Router(model_list=_MESSAGES.validate_json(_MODELS_JSON))

    def get_router() -> Router:
        return router

    clock: Final = _Clock()
    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count, clock=clock)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    logging_obj: Final = _logging(f"register-retry-race-{cancel_registration}")
    kwargs: Final = MappingProxyType(
        {"litellm_logging_obj": logging_obj, "litellm_metadata": _native_metadata(), "litellm_session_id": "baseline-session"}
    )
    await estimator.store.lock.acquire()
    registering: Final = asyncio.create_task(hook.async_pre_call_deployment_hook(kwargs, CallTypes.anthropic_messages))
    original: Final = await asyncio.wait_for(_wait_for_context(logging_obj), timeout=5)
    retrying: Final = asyncio.create_task(hook.async_pre_call_deployment_hook(kwargs, CallTypes.anthropic_messages))
    retry_owned: Final = await asyncio.wait_for(_wait_for_context(logging_obj, invalidated=True), timeout=5)
    assert retry_owned.reservation is original.reservation
    if cancel_registration:
        registering.cancel()
        with pytest.raises(asyncio.CancelledError):
            await registering
    estimator.store.lock.release()
    if not cancel_registration:
        await asyncio.wait_for(registering, timeout=5)
    await asyncio.wait_for(retrying, timeout=5)
    assert logging_obj.baseline_cache_context is retry_owned
    assert logging_obj.baseline_cache_estimate == unknown_estimate("retried_request")
    overlapping: Final = await _native_logging(estimator, clock, "overlapping-retry")
    await finalize_baseline_cache(overlapping, ModelResponse())
    assert overlapping.baseline_cache_estimate == unknown_estimate("pending_request")
    clock.now = 1001.0
    _stamp_native_wire(logging_obj, clock)
    await finalize_baseline_cache(logging_obj, ModelResponse())
    assert logging_obj.baseline_cache_context is None
    following: Final = await _native_logging(estimator, clock, "after-register-retry")
    await finalize_baseline_cache(following, ModelResponse())
    assert following.baseline_cache_estimate == unknown_estimate("history_unavailable"), following.baseline_cache_estimate


@pytest.mark.asyncio
async def test_cancelled_registration_retires_predispatch_reservation_and_propagates() -> None:
    router: Final = Router(model_list=_MESSAGES.validate_json(_MODELS_JSON))

    def get_router() -> Router:
        return router

    clock: Final = _Clock()
    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count, clock=clock)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    logging_obj: Final = _logging("cancelled-register")
    kwargs: Final = MappingProxyType(
        {"litellm_logging_obj": logging_obj, "litellm_metadata": _native_metadata(), "litellm_session_id": "baseline-session"}
    )
    await estimator.store.lock.acquire()
    registering: Final = asyncio.create_task(hook.async_pre_call_deployment_hook(kwargs, CallTypes.anthropic_messages))
    await asyncio.wait_for(_wait_for_context(logging_obj), timeout=5)
    registering.cancel()
    with pytest.raises(asyncio.CancelledError):
        await registering
    estimator.store.lock.release()
    assert logging_obj.baseline_cache_context is None
    following: Final = await _native_logging(estimator, clock, "after-cancelled-register")
    await finalize_baseline_cache(following, ModelResponse())
    assert following.baseline_cache_estimate == unknown_estimate("history_unavailable")


@pytest.mark.asyncio
async def test_cancelled_background_invalidation_preserves_active_retry_ownership() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=_count, clock=clock)
    logging_obj: Final = await _native_logging(estimator, clock, "cancelled-background-invalidation")
    await estimator.store.lock.acquire()
    invalidating: Final = asyncio.create_task(invalidate_baseline_cache(logging_obj, "retried_request"))
    await asyncio.wait_for(_wait_for_context(logging_obj, invalidated=True), timeout=5)
    invalidating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invalidating
    estimator.store.lock.release()
    retained: Final = logging_obj.baseline_cache_context
    assert retained is not None and retained.invalidated
    overlapping: Final = await _native_logging(estimator, clock, "after-cancelled-background-invalidation")
    await finalize_baseline_cache(overlapping, ModelResponse())
    assert overlapping.baseline_cache_estimate == unknown_estimate("pending_request")
    await finalize_baseline_cache(logging_obj, ModelResponse())
    assert logging_obj.baseline_cache_context is None


@pytest.mark.asyncio
async def test_partial_failure_spend_and_callbacks_survive_estimator_cleanup_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=_count, clock=clock)
    logging_obj: Final = await _native_logging(estimator, clock, "partial-failure-cleanup")
    logging_obj.record_partial_usage_for_failure(Usage(prompt_tokens=6000, completion_tokens=10), response_cost=0.125)
    capture: Final = _Capture(frozenset((logging_obj.litellm_call_id,)))
    monkeypatch.setattr(litellm, "callbacks", (capture,))
    monkeypatch.setattr(litellm, "failure_callback", ())
    monkeypatch.setattr(litellm, "_async_failure_callback", (capture,))
    original: Final = httpx.ReadError("original interrupted stream")
    clock.failures_remaining = 2
    await logging_obj.dispatch_failure_handlers(original, "original interrupted stream", prefer_async_handlers=True)
    payload, logged_exception = await asyncio.wait_for(capture.failures.get(), timeout=5)
    assert logged_exception is original
    assert payload["status"] == "failure"
    assert payload["response_cost"] == 0.125
    assert payload["prompt_tokens"] == 6000
    assert payload["completion_tokens"] == 10
    assert capture.failures.empty()
    assert logging_obj.baseline_cache_estimate == unknown_estimate("estimator_unavailable")


@pytest.mark.asyncio
@pytest.mark.parametrize("completed", (False, True))
async def test_native_stream_logging_without_wire_retains_scope_until_terminal_event(
    monkeypatch: pytest.MonkeyPatch, completed: bool
) -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=_count, clock=clock)
    logging_obj: Final = _logging(f"missing-wire-stream-{completed}", stream=True)
    logging_obj.baseline_cache_context = BaselineCacheContext(
        estimator, await _reservation(estimator, logging_obj.litellm_call_id)
    )
    capture: Final = _Capture(frozenset((logging_obj.litellm_call_id,)))
    monkeypatch.setattr(litellm, "callbacks", (capture,))
    monkeypatch.setattr(litellm, "success_callback", ())
    monkeypatch.setattr(litellm, "_async_success_callback", (capture,))
    timestamp: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # native callback contract uses naive times
    events: Final = _EVENTS if completed else _EVENTS[:-1]
    await PassThroughStreamingHandler._route_streaming_logging_to_handler(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # production route reconstructs both complete and disconnected native streams
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=PassThroughEndpointLogging(),
        url_route="/v1/messages",
        request_body=_JSON_OBJECT.validate_json('{"model":"claude-sonnet-5","stream":true}'),
        endpoint_type=EndpointType.ANTHROPIC,
        start_time=timestamp,
        raw_bytes=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
        end_time=timestamp,
    )
    payload: Final = await asyncio.wait_for(capture.payloads.get(), timeout=5)
    assert payload["status"] == "success"
    assert logging_obj.baseline_cache_estimate == unknown_estimate("missing_final_wire")
    if completed:
        assert logging_obj.baseline_cache_context is None
    else:
        retained: Final = logging_obj.baseline_cache_context
        assert retained is not None and retained.invalidated
        overlapping: Final = await _native_logging(estimator, clock, "during-missing-wire-stream")
        await finalize_baseline_cache(overlapping, ModelResponse())
        assert overlapping.baseline_cache_estimate == unknown_estimate("pending_request")


class _TerminalCapture(_AttemptCapture):
    def __init__(self, call_ids: frozenset[str]) -> None:
        super().__init__(call_ids)
        self.terminal_errors: asyncio.Queue[Exception] = asyncio.Queue()
        self.replacement_error = HTTPException(status_code=429, detail="transformed terminal provider error")

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> HTTPException:
        assert "litellm_logging_obj" not in request_data
        self.terminal_errors.put_nowait(original_exception)
        return self.replacement_error


@pytest.mark.asyncio
@pytest.mark.parametrize("retries", (0, 1))
@pytest.mark.parametrize("cleanup_fails", (False, True))
async def test_terminal_proxy_failure_retires_reservation_after_router_exhaustion(
    monkeypatch: pytest.MonkeyPatch, retries: int, cleanup_fails: bool
) -> None:
    router: Final = Router(
        model_list=_MESSAGES.validate_json(_RETRY_MODELS_JSON),
        num_retries=retries,
        retry_policy=RetryPolicy(RateLimitErrorRetries=retries),
        disable_cooldowns=True,
    )

    def get_router() -> Router:
        return router

    clock: Final = _Clock()
    clock.now = round(time.time(), 6)
    initial_time: Final = clock.now
    cache: Final = InternalUsageCache(DualCache())
    estimator: Final = BaselineCacheEstimator(cache.dual_cache, token_counter=_count, clock=clock)
    hook: Final = AutoRouterBaselineCache(cache, router=get_router, estimator=estimator)
    call_id: Final = f"terminal-proxy-failure-{retries}-{cleanup_fails}"
    capture: Final = _TerminalCapture(frozenset((call_id,)))
    _install_callbacks(monkeypatch, hook, capture)
    shared_logging: Final = _logging(call_id)
    requests: Final[list[httpx.Request]] = []  # mutable-ok: verify Router's actual exhausted attempt count

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            429,
            json=_JSON_OBJECT.validate_json(
                '{"type":"error","error":{"type":"rate_limit_error","message":"terminal provider failure"}}'
            ),
            headers=MappingProxyType({"retry-after": "0"}),
            request=request,
        )

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)
    await client.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as transport:
        client.client = transport
        with pytest.raises(litellm.RateLimitError, match="terminal provider failure") as caught:
            await router.anthropic_messages(  # pyright: ignore[reportUnknownMemberType]  # native Router owns the retry loop
                model="test-router",
                max_tokens=16,
                messages=_MESSAGES.validate_json(_MESSAGES_JSON),
                litellm_logging_obj=shared_logging,
                litellm_metadata=_OBJECTS.validate_json('{"user_api_key_hash":"test-caller-hash"}'),
                litellm_session_id="baseline-session",
            )
    assert len(requests) == retries + 1
    assert shared_logging.baseline_cache_context is not None
    proxy_logging: Final = ProxyLogging(UserApiKeyCache())
    proxy_logging.alert_types = []  # mutable-ok: disable optional alert sinks for the isolated boundary test
    request_data: Final = _OBJECTS.validate_python(
        MappingProxyType({"model": "test-router", "litellm_call_id": call_id, "litellm_logging_obj": shared_logging})
    )
    clock.failures_remaining = 2 if cleanup_fails else 0
    transformed: Final = await proxy_logging.post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # exercise the existing proxy terminal owner with its legacy request dictionary contract
        request_data=request_data,
        original_exception=caught.value,
        user_api_key_dict=UserAPIKeyAuth(request_route="/v1/messages"),
    )
    assert transformed is capture.replacement_error
    assert await asyncio.wait_for(capture.terminal_errors.get(), timeout=5) is caught.value
    assert "litellm_logging_obj" not in request_data
    assert shared_logging.baseline_cache_context is None
    recovery_boundary: Final = max((clock.now, *(repair.observed_at for repair in estimator.repairs.values())))
    assert recovery_boundary - initial_time < 60
    clock.now = round(recovery_boundary + 1.0, 6)
    following: Final = await _native_logging(estimator, clock, "after-terminal-proxy-failure")
    await finalize_baseline_cache(following, ModelResponse())
    assert following.baseline_cache_estimate == unknown_estimate("history_unavailable")
    clock.now += 1.0
    matching: Final = await _native_logging(estimator, clock, "matching-after-terminal-proxy-failure")
    await finalize_baseline_cache(matching, ModelResponse())
    assert matching.baseline_cache_estimate is not None
    assert matching.baseline_cache_estimate.cache_read_input_tokens == 5000


class _DelayedCancelEstimator(BaselineCacheEstimator):
    def __init__(self, clock: _Clock) -> None:
        super().__init__(DualCache(), token_counter=_count, clock=clock)
        self.cancel_started = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def cancel(self, reservation: BaselineReservation) -> None:
        self.cancel_started.set()
        await self.release_cancel.wait()
        await super().cancel(reservation)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_fails", (False, True))
async def test_late_cancel_cannot_clear_replacement_context(cancel_fails: bool) -> None:
    clock: Final = _Clock()
    estimator: Final = _DelayedCancelEstimator(clock)
    original: Final = BaselineCacheContext(estimator, await _reservation(estimator, "original"))
    replacement: Final = BaselineCacheContext(estimator, await _reservation(estimator, "replacement"))
    logging_obj: Final = _logging("late-cancel")
    logging_obj.baseline_cache_context = original
    cancelling: Final = asyncio.create_task(cancel_baseline_cache(logging_obj))
    await asyncio.wait_for(estimator.cancel_started.wait(), timeout=5)
    logging_obj.baseline_cache_context = replacement  # rebind-ok: simulate newer work while old cancellation waits
    clock.failures_remaining = 2 if cancel_fails else 0
    estimator.release_cancel.set()
    assert await asyncio.wait_for(cancelling, timeout=5) is False
    assert logging_obj.baseline_cache_context is replacement
    assert logging_obj.baseline_cache_estimate is None
    _stamp_native_wire(logging_obj, clock)
    await finalize_baseline_cache(logging_obj, ModelResponse())
    assert logging_obj.baseline_cache_estimate == unknown_estimate("history_unavailable")


@pytest.mark.asyncio
async def test_partial_stream_callback_keeps_retry_scope_until_completed_stream() -> None:
    clock: Final = _Clock()
    estimator: Final = BaselineCacheEstimator(DualCache(), token_counter=_count, clock=clock)
    reservation: Final = await _reservation(estimator, "partial-native-stream")
    logging_obj: Final = _logging("partial-native-stream", stream=True)
    logging_obj.baseline_cache_context = BaselineCacheContext(estimator, reservation)
    wire: Final = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        json=_JSON_OBJECT.validate_python(
            MappingProxyType(
                {"model": "claude-sonnet-5", "stream": True, "messages": _MESSAGES.validate_json(_MESSAGES_JSON)}
            )
        ),
    )
    initial: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # match native request timing evidence
    logging_obj.model_call_details.update(  # pyright: ignore[reportUnknownMemberType]  # simulate the native partial-stream callback evidence
        httpx_response=_upstream(wire),
        api_call_start_time=initial,
        completion_start_time=initial,
        custom_llm_provider="anthropic",
        stream=True,
        prompt_cache_response_complete=False,
    )
    await finalize_baseline_cache(logging_obj, ModelResponse())
    assert logging_obj.baseline_cache_estimate == unknown_estimate("incomplete_response")
    retained: Final = logging_obj.baseline_cache_context
    assert retained is not None and retained.invalidated
    await logging_obj.invalidate_baseline_cache_estimate("retried_request")
    clock.now = 1200.0
    completed: Final = datetime.fromtimestamp(clock.now)  # noqa: DTZ006  # retry completion observed later
    logging_obj.model_call_details.update(  # pyright: ignore[reportUnknownMemberType]  # retry's terminal native SSE event
        api_call_start_time=completed,
        completion_start_time=completed,
        prompt_cache_response_complete=True,
    )
    await finalize_baseline_cache(logging_obj, ModelResponse())
    assert logging_obj.baseline_cache_context is None
    assert logging_obj.baseline_cache_estimate == unknown_estimate("retried_request")
    clock.now = 4601.0
    following: Final = await _reservation(estimator, "after-partial-stream-start-ttl")
    estimate: Final = await estimator.finalize(
        following, wire=wire, request_started_at=clock.now, available_at=clock.now
    )
    assert estimate.status == "unknown"
    assert estimate.reason == "history_unavailable"
