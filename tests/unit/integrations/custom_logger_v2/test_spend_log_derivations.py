import asyncio
import copy
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import SimpleNamespace
from typing import Final, cast
from unittest.mock import patch

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import CallContext, CustomLoggerV2, Field, Hook, on
from litellm.integrations.custom_logger_v2 import derivations
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.litellm_core_utils.core_helpers import (
    budget_reservation_from_metadata,
    get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # the writer's untyped metadata read
)
from litellm.litellm_core_utils.internal_call_metadata import BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX
from litellm.litellm_core_utils.batch_utils import BATCH_COST_REQUEST_ID_SUFFIX
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
from litellm.proxy.spend_tracking.spend_tracking_utils import (
    get_logging_payload,  # pyright: ignore[reportUnknownVariableType]  # the writer's row builder takes an untyped dict
)
from litellm.proxy.utils import BilledFailedAttempt
from litellm.types.integrations.custom_logger_v2 import BudgetReservation, Cost, SpendLogId
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    SETTLES_BACKGROUND_RESPONSE_KWARG,
    BackgroundResponseSettlement,
)
from litellm.types.utils import Usage

_RESERVATION: Final[Mapping[str, object]] = {
    "reserved_cost": 0.75,
    "entries": [{"counter_key": "spend:key:k", "entity_type": "key", "entity_id": "k", "reserved_cost": 0.75}],
    "finalized": False,
}


class _WriterInputs(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger takes untyped kwargs
        self.seen: list[tuple[Mapping[str, object], object]] = []  # mutable-ok: what the spend writer is handed

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.seen.append((dict(kwargs), response_obj))


class _SpendReader(CustomLoggerV2):
    def __init__(self) -> None:
        self.seen: list[tuple[object, object, object]] = []  # mutable-ok: what attempt_ended hands the port
        self.done = threading.Event()

    @on(Hook.attempt_ended, reads={Field.spend_log_id, Field.cost, Field.budget_reservation}, awaited=True)
    async def attempt_ended(self, view: object, ctx: CallContext[object]) -> None:
        self.seen.append(
            (
                getattr(view, "spend_log_id", None),
                getattr(view, "cost", None),
                getattr(view, "budget_reservation", None),
            )
        )
        self.done.set()


def _only(callbacks: Sequence[object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", list(callbacks))
    for name in ("input_callback", "success_callback", "failure_callback", "_async_success_callback"):
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])


async def _logged(
    monkeypatch: pytest.MonkeyPatch, **call: object
) -> tuple[tuple[Mapping[str, object], object], tuple[object, object, object]]:
    writer: Final = _WriterInputs()
    reader: Final = _SpendReader()
    _only((writer, reader), monkeypatch)
    response: Final = await litellm.acompletion(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], mock_response="ok", **call
    )
    if call.get("stream") is True:
        assert isinstance(response, litellm.CustomStreamWrapper)
        _ = [chunk async for chunk in response]  # pyright: ignore[reportUnknownVariableType]  # the stream yields untyped chunks
    assert await asyncio.to_thread(reader.done.wait, 30)
    for _ in range(300):
        if writer.seen:
            break
        await asyncio.sleep(0.01)
    return writer.seen[0], reader.seen[0]


def _row_id(kwargs: Mapping[str, object], response: object) -> object:
    now: Final = datetime.now()
    return get_logging_payload(dict(kwargs), response, now, now)["request_id"]


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "stream"])
async def test_the_spend_log_id_is_the_request_id_the_writer_builds_from_what_it_is_handed(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    (kwargs, response), (spend_log_id, _, _) = await _logged(monkeypatch, stream=stream)

    assert isinstance(spend_log_id, SpendLogId)
    assert spend_log_id.cache_hit is False
    assert spend_log_id.id == _row_id(kwargs, response)


async def test_the_response_id_wins_over_the_logged_ids_like_the_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    (kwargs, response), _ = await _logged(monkeypatch)
    assert isinstance(response, litellm.ModelResponse)

    for answer in (response.model_copy(update={"id": "resp-other"}), {"id": "dict-id"}, "plain text"):
        derived = derivations.spend_log_id(HookArgs(request=kwargs, result=answer))
        assert derived is not None
        assert derived.id == _row_id(kwargs, answer)


@pytest.mark.parametrize(
    ("extra", "suffix"),
    [
        ({"call_type": "aretrieve_batch"}, BATCH_COST_REQUEST_ID_SUFFIX),
        (
            {SETTLES_BACKGROUND_RESPONSE_KWARG: BackgroundResponseSettlement.SETTLES},
            BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX,
        ),
    ],
    ids=["batch", "background"],
)
async def test_the_spend_log_id_carries_the_settle_once_suffix_the_writer_appends(
    monkeypatch: pytest.MonkeyPatch, extra: Mapping[str, object], suffix: str
) -> None:
    (kwargs, response), _ = await _logged(monkeypatch)
    suffixed: Final = {**kwargs, **extra}

    derived: Final = derivations.spend_log_id(HookArgs(request=suffixed, result=response))

    assert derived is not None
    assert derived.id == _row_id(suffixed, response)
    assert isinstance(derived.id, str) and derived.id.endswith(suffix)


async def test_a_cache_hit_keeps_the_base_id_the_writer_suffixes_with_the_time(monkeypatch: pytest.MonkeyPatch) -> None:
    (kwargs, response), _ = await _logged(monkeypatch)
    hit: Final = {**kwargs, "cache_hit": True}

    derived: Final = derivations.spend_log_id(HookArgs(request=hit, result=response))
    written: Final = _row_id(hit, response)

    assert derived is not None and derived.cache_hit is True
    assert isinstance(written, str) and written.startswith(f"{derived.id}_cache_hit")
    assert written.removeprefix(f"{derived.id}_cache_hit").replace(".", "").isdigit()


async def test_the_logged_reservation_is_the_one_the_writer_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    (kwargs, _), (_, _, reservation) = await _logged(
        monkeypatch, metadata={"user_api_key_budget_reservation": copy.deepcopy(_RESERVATION)}
    )
    settled: Final = budget_reservation_from_metadata(get_litellm_metadata_from_kwargs(dict(kwargs)))  # pyright: ignore[reportUnknownArgumentType]  # the writer's untyped metadata read

    assert settled is not None
    assert reservation == derivations.budget_reservation(HookArgs(auth=SimpleNamespace(budget_reservation=settled)))
    assert isinstance(reservation, BudgetReservation) and reservation.reserved_cost == 0.75


async def test_a_request_without_a_reservation_reads_none_like_the_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    (kwargs, _), (_, cost, reservation) = await _logged(monkeypatch)

    assert budget_reservation_from_metadata(get_litellm_metadata_from_kwargs(dict(kwargs))) is None  # pyright: ignore[reportUnknownArgumentType]  # the writer's untyped metadata read
    assert reservation is None
    assert isinstance(cost, Cost) and cost.response_cost_failure_debug_info_present is True


@pytest.mark.parametrize(
    ("payload", "finalized", "debug_info", "present"),
    [
        ({"cost_finalized": True, "response_cost_failure_debug_info": None}, True, None, True),
        ({"response_cost_failure_debug_info": {"error": "no price"}}, None, "{'error': 'no price'}", True),
        ({"cost_finalized": 1}, 1, None, False),
    ],
    ids=["finalized", "debug_info_dict", "debug_info_missing"],
)
def test_cost_carries_the_finalized_flag_and_debug_info_the_writer_reads(
    payload: Mapping[str, object], finalized: object, debug_info: object, present: bool
) -> None:
    cost: Final = derivations.cost(HookArgs(request={"standard_logging_object": payload}))

    assert cost is not None
    assert cost.cost_finalized == finalized
    text: Final = getattr(cost.response_cost_failure_debug_info, "text", cost.response_cost_failure_debug_info)
    assert text == debug_info
    assert cost.response_cost_failure_debug_info_present is present


class _RecordingSpendWriter:
    def __init__(self) -> None:
        self.rows: list[Mapping[str, object]] = []  # mutable-ok: the spend rows the failure hook writes

    async def update_database(self, **kwargs: object) -> None:
        self.rows.append(kwargs)


def _failed_request(call_id: str, payload: Mapping[str, object] | None) -> dict[str, object]:
    usage: Final = Usage(prompt_tokens=24, completion_tokens=0, total_tokens=24)
    return {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"tags": ["batch-jobs"]},
        "litellm_call_id": call_id,
        "call_type": "acompletion",
        "combined_usage_object": usage,
        "response_cost": 0.5,
        "billed_failed_attempts": (
            BilledFailedAttempt(attempted_retries=0, deployment_id="dep-a", usage=usage, response_cost=0.25),
            BilledFailedAttempt(attempted_retries=1, deployment_id=None, usage=usage, response_cost=0.125),
        ),
        **({} if payload is None else {"standard_logging_object": dict(payload)}),
    }


async def _written_rows(request: dict[str, object], error: Exception) -> list[Mapping[str, object]]:
    writer: Final = _RecordingSpendWriter()
    with patch("litellm.proxy.proxy_server.general_settings", dict[str, object]()):
        await ProxyDBLogger(spend_writer=lambda: cast(DBSpendUpdateWriter, writer)).async_post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # the hook takes the untyped request dict
            request_data=request,
            original_exception=error,
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key", user_id="u", team_id="t"),
        )
    return writer.rows


def _failure_row_id(row: Mapping[str, object]) -> object:
    kwargs: Final = cast("Mapping[str, object]", row["kwargs"])
    now: Final = datetime.now()
    return get_logging_payload(dict(kwargs), row["completion_response"], now, now)["request_id"]


async def _logged_payload(monkeypatch: pytest.MonkeyPatch) -> Mapping[str, object]:
    (kwargs, _), _ = await _logged(monkeypatch)
    return cast("Mapping[str, object]", kwargs["standard_logging_object"])


@pytest.mark.parametrize("with_payload", [False, True], ids=["call_id", "payload_id"])
async def test_failure_ids_and_billed_attempts_are_the_rows_the_failure_writer_writes(
    monkeypatch: pytest.MonkeyPatch, with_payload: bool
) -> None:
    payload: Final = await _logged_payload(monkeypatch) if with_payload else None
    request: Final = _failed_request("call-1", payload)
    error: Final = Exception("upstream broke")
    args: Final = HookArgs(auth=UserAPIKeyAuth(api_key="hashed-key"), request=copy.deepcopy(request), result=error)

    final_id: Final = derivations.failure_spend_log_id(args)
    attempts: Final = derivations.billed_retry_attempts(args)
    cost: Final = derivations.failure_cost(args)
    rows: Final = await _written_rows(request, error)

    assert final_id is not None and attempts is not None and cost is not None
    assert [final_id.id, *(attempt.spend_log_id.id for attempt in attempts if attempt.spend_log_id)] == [
        _failure_row_id(row) for row in rows
    ]
    assert [cost.failure_row_response_cost, *(attempt.response_cost for attempt in attempts)] == [
        row["response_cost"] for row in rows
    ]
    assert [(attempt.position, attempt.attempted_retries, attempt.deployment_id) for attempt in attempts] == [
        (0, 0, "dep-a"),
        (1, 1, None),
    ]
    assert [(attempt.litellm_call_id, attempt.parent_request_id) for attempt in attempts] == [
        ("call-1-attempt-0", "call-1"),
        ("call-1-attempt-1", "call-1"),
    ]
    assert [attempt.usage.prompt_tokens for attempt in attempts if attempt.usage] == [24, 24]


async def test_a_failure_without_billed_attempts_has_none_and_keeps_the_finalized_flag() -> None:
    request: Final = {"litellm_call_id": "call-2", "standard_logging_object": {"cost_finalized": True}}
    args: Final = HookArgs(request=request, result=Exception("no"))

    cost: Final = derivations.failure_cost(args)

    assert derivations.billed_retry_attempts(args) is None
    assert cost is not None and cost.cost_finalized is True
    assert derivations.failure_spend_log_id(args) == SpendLogId(id="call-2", cache_hit=False)


async def test_the_failure_reservation_is_the_auth_reservation_the_failure_writer_releases() -> None:
    auth: Final = UserAPIKeyAuth(api_key="hashed-key", budget_reservation=dict(copy.deepcopy(_RESERVATION)))

    reservation: Final = derivations.DERIVED_POST_CALL_FAILURE_FIELDS[Field.budget_reservation].read(
        HookArgs(auth=auth, request={})
    )

    assert reservation == derivations.budget_reservation(HookArgs(auth=auth))
    assert isinstance(reservation, BudgetReservation) and reservation.reserved_cost == 0.75
