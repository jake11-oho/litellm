"""The Rust spend log callback leaves the same queued spend log rows as the Python cost callback on the ledger
path. Each path writes into its own Rust spend log writer over an unreachable database, so the rows stay queued
and the test reads them back."""

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Final

import pytest
import pytest_asyncio

import litellm
import litellm.proxy.proxy_server as ps
from litellm.integrations.custom_logger_v2.enums import Hook
from litellm.integrations.custom_logger_v2.rust import RustLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.native_database import ServiceLoggingEvents
from litellm.proxy.db.rust_spend_log_writer import install_rust_spend_log_writer, uninstall_rust_spend_log_writer
from litellm.proxy.hooks.proxy_track_cost_callback import track_batch_accounting
from litellm.proxy.hooks.rust_spend_logger import rust_spend_loggers, spend_logger
from litellm.proxy.utils import (
    FAILURE_ROW_ALREADY_WRITTEN,
    FAILURE_ROW_ALREADY_WRITTEN_KEY,
    GUARDRAIL_ENDPOINT_FAILURE,
    GUARDRAIL_ENDPOINT_FAILURE_KEY,
    BilledFailedAttempt,
    PrismaClient,
    ProxyUpdateSpend,
)
from litellm.rust_bridge._native import DatabaseEndpoint, NativeDatabase, NativeSpendCounters, RustSpendLogWriter
from tests.unit.proxy.db.test_rust_spend_log_writer import FakePrismaClient
from tests.unit.proxy.hooks.rust_spend_logger_twin import (
    Completion,
    completion,
    legacy_failure,
    legacy_success,
    rust_failure,
    rust_success,
    without,
)

_UNREACHABLE: Final = "postgresql://litellm:secret@db.invalid:5432/proxy?pool_timeout=1"
_LEDGER_ON: Final[Mapping[str, object]] = {"use_spend_logs_ledger": True}
_VOLATILE: Final = ("accounting_delivery_id",)
_FAILURE_VOLATILE: Final = (
    "accounting_delivery_id",
    "endTime",
    "completionStartTime",
    "request_duration_ms",
    "session_id",
    "accounting_payload",
)
_KEYED: Final[Mapping[str, object]] = {
    "user_api_key": "hashed-key",
    "user_api_key_alias": "alias",
    "user_api_key_user_id": "user-1",
    "user_api_key_team_id": "team-1",
    "user_api_key_org_id": "org-1",
}
_STARTED: Final = datetime(2026, 11, 17, 9, 0, tzinfo=timezone.utc)


class Paths:
    def __init__(self, legacy_client: FakePrismaClient, legacy: RustSpendLogWriter, rust: RustSpendLogWriter) -> None:
        self.legacy_client: Final = legacy_client
        self.legacy: Final = legacy
        self.rust: Final = rust
        handle: Final = ps.spend_counter_store.handle
        assert isinstance(handle, NativeSpendCounters)
        self.logger: Final = spend_logger(rust, handle)

    def legacy_rows(self) -> Sequence[Mapping[str, object]]:
        return [json.loads(row) for row in self.legacy.dequeue(100)]

    def rust_rows(self) -> Sequence[Mapping[str, object]]:
        return [json.loads(row) for row in self.rust.dequeue(100)]


@pytest_asyncio.fixture
async def paths(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Paths]:
    monkeypatch.setattr(PrismaClient, "spend_log_queue_bytes", 0)
    database: Final = await NativeDatabase.connect(
        ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: None),
        await DatabaseEndpoint.resolve(_UNREACHABLE, 1),
        None,
    )
    legacy_client: Final = FakePrismaClient()
    rust_client: Final = FakePrismaClient()
    legacy: Final = await install_rust_spend_log_writer(legacy_client, database, _LEDGER_ON, None)  # pyright: ignore[reportArgumentType]  # the fake stands in for the Prisma client
    rust: Final = await install_rust_spend_log_writer(rust_client, database, _LEDGER_ON, None)  # pyright: ignore[reportArgumentType]  # the fake stands in for the Prisma client
    assert legacy is not None and rust is not None
    monkeypatch.setattr(ps, "prisma_client", legacy_client)
    monkeypatch.setattr(ps, "general_settings", {"use_spend_logs_ledger": True})
    monkeypatch.setattr(ps.proxy_logging_obj.slack_alerting_instance, "alerting", [])
    try:
        yield Paths(legacy_client, legacy, rust)
    finally:
        uninstall_rust_spend_log_writer(legacy_client)
        uninstall_rust_spend_log_writer(rust_client)
        ps.spend_counter_store.flush_local()


async def _both_succeed(
    paths: Paths, done: Completion
) -> tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]:
    await legacy_success(done)
    await rust_success(paths.logger, done)
    return paths.legacy_rows(), paths.rust_rows()


@pytest.mark.asyncio
async def test_a_priced_completion_queues_the_same_ledger_row(paths: Paths) -> None:
    legacy, rust = await _both_succeed(paths, await completion(_KEYED))

    assert len(rust) == 1
    assert [without(row, _VOLATILE) for row in rust] == [without(row, _VOLATILE) for row in legacy]
    assert rust[0]["spend"] == legacy[0]["spend"]
    assert rust[0]["accounting_delivery_id"]


@pytest.mark.asyncio
async def test_request_logs_off_queues_the_same_accounting_only_row(
    paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ps, "disable_spend_logs", True)

    legacy, rust = await _both_succeed(paths, await completion(_KEYED))

    assert len(rust) == 1
    assert [without(row, _VOLATILE) for row in rust] == [without(row, _VOLATILE) for row in legacy]


@pytest.mark.asyncio
async def test_an_end_user_and_tags_land_on_the_same_row(paths: Paths) -> None:
    done: Final = await completion(
        {**_KEYED, "tags": ["blue", "green"], "user_api_key_end_user_id": "end-user-1"}, user="end-user-1"
    )

    legacy, rust = await _both_succeed(paths, done)

    assert len(rust) == 1
    assert [without(row, _VOLATILE) for row in rust] == [without(row, _VOLATILE) for row in legacy]
    assert json.loads(str(rust[0]["request_tags"])) == ["blue", "green"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [pytest.param({}, id="nobody_to_charge"), pytest.param(_KEYED, id="spend_updates_disabled")],
)
async def test_an_untracked_completion_queues_no_row_on_either_path(
    paths: Paths, monkeypatch: pytest.MonkeyPatch, metadata: Mapping[str, object]
) -> None:
    if metadata:
        monkeypatch.setattr(ProxyUpdateSpend, "disable_spend_updates", staticmethod(lambda: True))

    legacy, rust = await _both_succeed(paths, await completion(metadata))

    assert (legacy, rust) == ([], [])


def _auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed-key",
        key_alias="alias",
        user_id="user-1",
        team_id="team-1",
        org_id="org-1",
        team_alias="team alias",
    )


async def _both_fail(
    paths: Paths, request_data: Mapping[str, object], error: Exception
) -> tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]:
    await legacy_failure(request_data, error, _auth())
    await rust_failure(paths.logger, request_data, error, _auth())
    return paths.legacy_rows(), paths.rust_rows()


@pytest.mark.asyncio
async def test_a_failed_request_queues_the_same_failure_row(paths: Paths) -> None:
    request_data: Final = {
        "model": "gpt-4o",
        "litellm_call_id": "failed-call",
        "start_time": _STARTED,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"caller": "value"},
        "proxy_server_request": {"body": {"model": "gpt-4o"}},
    }

    legacy, rust = await _both_fail(paths, request_data, ValueError("the provider refused"))

    assert len(rust) == 1
    assert rust[0]["status"] == "failure"
    assert [without(row, _FAILURE_VOLATILE) for row in rust] == [without(row, _FAILURE_VOLATILE) for row in legacy]


@pytest.mark.asyncio
async def test_each_billed_attempt_queues_the_same_attempt_row(paths: Paths) -> None:
    usage: Final = litellm.Usage(prompt_tokens=24, completion_tokens=0, total_tokens=24)
    logged: Final = (await completion(_KEYED)).kwargs["standard_logging_object"]
    assert isinstance(logged, Mapping)
    request_data: Final = {
        "model": "gpt-4.1",
        "litellm_call_id": "call-1",
        "start_time": _STARTED,
        "standard_logging_object": {**logged, "id": "call-1", "model_id": "dep-final"},
        "metadata": {
            "attempted_retries": 1,
            "model_info": {"id": "dep-final", "input_cost_per_token": 0.5},
            "previous_models": ({"exception_type": "Timeout"},),
        },
        "combined_usage_object": usage,
        "response_cost": 0.5,
        "billed_failed_attempts": (
            BilledFailedAttempt(attempted_retries=0, deployment_id="dep-a", usage=usage, response_cost=0.25),
        ),
    }

    legacy, rust = await _both_fail(
        paths, request_data, litellm.Timeout("Request timed out.", model="gpt-4.1", llm_provider="openai")
    )

    assert sorted(str(row["request_id"]) for row in rust) == ["call-1", "call-1-attempt-0"]

    def by_id(rows: Sequence[Mapping[str, object]]) -> Sequence[Mapping[str, object]]:
        return sorted((without(row, _FAILURE_VOLATILE) for row in rows), key=lambda row: str(row["request_id"]))

    assert by_id(rust) == by_id(legacy)


@pytest.mark.asyncio
async def test_an_untracked_failure_route_queues_no_row(paths: Paths) -> None:
    auth: Final = _auth()
    auth.request_route = "/key/generate"

    await legacy_failure({"model": "gpt-4o"}, ValueError("no"), auth)
    await rust_failure(paths.logger, {"model": "gpt-4o"}, ValueError("no"), auth)

    assert (paths.legacy_rows(), paths.rust_rows()) == ([], [])


@pytest.mark.asyncio
async def test_a_failure_whose_row_is_already_written_queues_no_row(paths: Paths) -> None:
    request_data: Final = {"model": "gpt-4o", FAILURE_ROW_ALREADY_WRITTEN_KEY: FAILURE_ROW_ALREADY_WRITTEN}

    await legacy_failure(request_data, ValueError("no"), _auth())
    await rust_failure(paths.logger, request_data, ValueError("no"), _auth())

    assert (paths.legacy_rows(), paths.rust_rows()) == ([], [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("marker", "rows"),
    [pytest.param(GUARDRAIL_ENDPOINT_FAILURE, 1, id="marked"), pytest.param(True, 0, id="forged")],
)
async def test_a_guardrail_endpoint_failure_is_tracked_only_when_marked(
    paths: Paths, marker: object, rows: int
) -> None:
    auth: Final = _auth()
    auth.request_route = "/guardrails/apply_guardrail"
    request_data: Final = {
        "guardrail_name": "bedrock-guard",
        "litellm_call_id": "guardrail-call",
        "start_time": _STARTED,
        GUARDRAIL_ENDPOINT_FAILURE_KEY: marker,
    }

    await legacy_failure(request_data, ValueError("blocked"), auth)
    await rust_failure(paths.logger, request_data, ValueError("blocked"), auth)

    legacy: Final = paths.legacy_rows()
    rust: Final = paths.rust_rows()
    assert len(rust) == rows
    assert [without(row, _FAILURE_VOLATILE) for row in rust] == [without(row, _FAILURE_VOLATILE) for row in legacy]


@pytest.mark.asyncio
async def test_the_cached_spend_bump_adds_the_same_charge_on_both_paths(
    paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    added: Final[list[tuple[object, ...]]] = []
    monkeypatch.setattr(ps, "add_spend_to_cached_spend_objects", lambda *args: added.append(args[1:]))
    done: Final = await completion({**_KEYED, "tags": ["blue"]})

    await legacy_success(done)
    await _settle_background_tasks()
    legacy: Final = list(added)
    added.clear()
    await rust_success(paths.logger, done)
    await _settle_background_tasks()

    assert len(legacy) == 1
    assert added == legacy


async def _settle_background_tasks() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


class _AllowanceCounter:
    def __init__(self, used: float) -> None:
        self.used = used
        self.increments: list[tuple[str, float, int | None]] = []

    async def async_increment(self, key: str, value: float, ttl: int | None = None) -> float:
        self.increments.append((key, value, ttl))
        self.used += value
        return self.used


def _grounded(done: Completion) -> Completion:
    kwargs: Final = copy.deepcopy(dict(done.kwargs))
    payload: Final = kwargs["standard_logging_object"]
    assert isinstance(payload, dict)
    payload["custom_llm_provider"] = "gemini"
    payload["model_map_information"] = {
        "model_map_key": "gemini-3-pro",
        "model_map_value": {
            "litellm_provider": "vertex_ai",
            "web_search_billing_unit": "per_query",
            "search_context_cost_per_query": {"search_context_size_medium": 1e-6},
            "search_free_requests": {"count": 5, "period": "day", "shared_group": "gemini-3"},
        },
    }
    response: Final = copy.deepcopy(done.response)
    assert isinstance(response, litellm.ModelResponse)
    response.usage.prompt_tokens_details = litellm.types.utils.PromptTokensDetailsWrapper(web_search_requests=3)  # pyright: ignore[reportAttributeAccessIssue]  # the mock usage gains a grounding count
    return Completion(kwargs, response, done.start_time, done.end_time)


@pytest.mark.asyncio
async def test_grounding_under_the_free_allowance_is_discounted_the_same_on_both_paths(
    paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "apply_search_free_tier", True)
    dual_cache: Final = ps.proxy_logging_obj.internal_usage_cache.dual_cache
    done: Final = _grounded(await completion(_KEYED))
    legacy_counter: Final = _AllowanceCounter(used=3)
    rust_counter: Final = _AllowanceCounter(used=3)

    monkeypatch.setattr(dual_cache, "redis_cache", legacy_counter)
    await legacy_success(done)
    monkeypatch.setattr(dual_cache, "redis_cache", rust_counter)
    await rust_success(paths.logger, done)
    legacy: Final = paths.legacy_rows()
    rust: Final = paths.rust_rows()

    assert [without(row, _VOLATILE) for row in rust] == [without(row, _VOLATILE) for row in legacy]
    assert float(str(rust[0]["spend"])) < float(str(done.kwargs["response_cost"]))
    ((legacy_key, legacy_units, legacy_ttl),) = legacy_counter.increments
    ((rust_key, rust_units, rust_ttl),) = rust_counter.increments
    assert (rust_key, rust_units) == (legacy_key, legacy_units)
    assert legacy_ttl is not None and rust_ttl is not None and abs(rust_ttl - legacy_ttl) <= 2


def _unpriced(done: Completion) -> Completion:
    kwargs: Final = copy.deepcopy(dict(done.kwargs))
    payload: Final = kwargs["standard_logging_object"]
    assert isinstance(payload, dict)
    payload["response_cost"] = None
    payload["response_cost_failure_debug_info"] = {"error_str": "no price for this model"}
    kwargs["response_cost"] = None
    return Completion(kwargs, done.response, done.start_time, done.end_time)


def _without_traceback(message: str) -> tuple[str, str]:
    head, _, rest = message.partition("\n Traceback:")
    return head, rest[rest.index("\n Args to") :]


@pytest.mark.asyncio
async def test_an_unpriced_completion_raises_the_same_failed_tracking_alert(
    paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    alerts: Final[list[tuple[str, str]]] = []

    async def failed_tracking_alert(error_message: str, failing_model: str) -> None:
        alerts.append((error_message, failing_model))

    monkeypatch.setattr(ps.proxy_logging_obj, "failed_tracking_alert", failed_tracking_alert)
    done: Final = _unpriced(await completion(_KEYED))

    await legacy_success(done)
    await rust_success(paths.logger, done)
    await _settle_background_tasks()

    ((legacy_message, legacy_model), (rust_message, rust_model)) = alerts
    assert rust_model == legacy_model == "gpt-4o"
    assert _without_traceback(rust_message) == _without_traceback(legacy_message)
    assert (paths.legacy_rows(), paths.rust_rows()) == ([], [])


@pytest.mark.asyncio
@pytest.mark.parametrize("priced", [pytest.param(True, id="priced"), pytest.param(False, id="unpriced")])
async def test_a_batch_costing_hears_the_same_outcome_from_both_paths(paths: Paths, priced: bool) -> None:
    done: Final = await completion(_KEYED)
    request: Final = done if priced else _unpriced(done)
    call_id: Final = str(request.kwargs["litellm_call_id"])

    with track_batch_accounting(call_id) as legacy:
        await legacy_success(request)
    with track_batch_accounting(call_id) as rust:
        await rust_success(paths.logger, request)

    assert (rust.write_started, rust.failed) == (legacy.write_started, legacy.failed) == (priced, not priced)


@pytest.mark.asyncio
async def test_the_proxy_registers_the_rust_callback_only_over_a_rust_writer(paths: Paths) -> None:
    registered: Final = rust_spend_loggers(paths.legacy_client)
    missing: Final = rust_spend_loggers(object())

    assert isinstance(registered, RustLogger) and missing is None
    assert registered.selection is not None and registered.selection.name == "spend_log"
    assert hasattr(registered, f"on_{Hook.attempt_ended.value}")
    assert hasattr(registered, f"on_{Hook.post_call_failure.value}")
