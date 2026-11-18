"""Drives one request through the Python cost callback and the Rust spend log callback, each writing into
its own Rust spend log writer, so a test can compare the rows the two paths leave."""

from __future__ import annotations

import asyncio
import copy
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Final

import litellm
from litellm._logging import redact_string
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2 import derivations
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.integrations.custom_logger_v2.rust import RustLogger
from litellm.litellm_core_utils.litellm_logging import V2_ASYNC_LOGGING_LINE
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
from litellm.proxy.utils import FAILURE_ROW_ALREADY_WRITTEN, GUARDRAIL_ENDPOINT_FAILURE, REJECTED_BEFORE_REQUEST_SETUP
from litellm.types.integrations.custom_logger_v2 import Outcome

_SUCCESS_READS: Final = frozenset({Field.spend_log_input, Field.budget_reservation, Field.outcome})
_FAILURE_READS: Final = frozenset({Field.spend_log_input, Field.budget_reservation})


@dataclass(frozen=True, slots=True)
class Completion:
    kwargs: Mapping[str, object]
    response: object
    start_time: datetime
    end_time: datetime


class _Capture(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.completions: list[Completion] = []

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.completions.append(Completion(kwargs, response_obj, start_time, end_time))


async def completion(metadata: Mapping[str, object], **request: object) -> Completion:
    capture: Final = _Capture()
    _ = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello there",
        metadata=dict(metadata),
        callbacks=[capture],
        **request,  # pyright: ignore[reportArgumentType]  # the scenario's extra completion params
    )
    for _ in range(200):
        if capture.completions:
            return capture.completions[0]
        await asyncio.sleep(0.01)
    raise AssertionError("the completion never reached its success callback")


async def legacy_success(done: Completion) -> None:
    await ProxyDBLogger()._PROXY_track_cost_callback(  # pyright: ignore[reportPrivateUsage]  # the cost callback's own entry
        copy.deepcopy(dict(done.kwargs)), done.response, done.start_time, done.end_time
    )


def _view(**fields: object) -> SimpleNamespace:  # kwargs-ok: one keyword per field the view carries
    return SimpleNamespace(**{name: value for name, value in fields.items() if value is not None})


async def rust_success(logger: RustLogger, done: Completion) -> None:
    kwargs: Final = copy.deepcopy(dict(done.kwargs))
    args: Final = HookArgs(
        request=kwargs,
        result=done.response,
        start_time=done.start_time,
        end_time=done.end_time,
        logging_obj=kwargs.get("litellm_logging_obj"),
    )
    view: Final = _view(
        spend_log_input=derivations.spend_log_success_input(args),
        budget_reservation=derivations.logged_budget_reservation(args),
        outcome=Outcome.success,
    )
    line: Final = V2_ASYNC_LOGGING_LINE.set(True)
    try:
        _ = await logger.run_rust(Hook.attempt_ended, _SUCCESS_READS, view, CallContext(Hook.attempt_ended, None, None))
    finally:
        V2_ASYNC_LOGGING_LINE.reset(line)


def traceback_of(error: BaseException) -> str:
    return redact_string("".join(traceback.format_exception(error)))


def _copied(request_data: Mapping[str, object]) -> dict[str, object]:
    """A deep copy that keeps the proxy's request sentinels, which are matched by identity."""
    sentinels: Final = (FAILURE_ROW_ALREADY_WRITTEN, GUARDRAIL_ENDPOINT_FAILURE, REJECTED_BEFORE_REQUEST_SETUP)
    return copy.deepcopy(dict(request_data), {id(sentinel): sentinel for sentinel in sentinels})


async def legacy_failure(request_data: Mapping[str, object], error: Exception, auth: UserAPIKeyAuth) -> None:
    await ProxyDBLogger().async_post_call_failure_hook(
        request_data=_copied(request_data),
        original_exception=error,
        user_api_key_dict=auth,
        traceback_str=traceback_of(error),
    )


async def rust_failure(
    logger: RustLogger, request_data: Mapping[str, object], error: Exception, auth: UserAPIKeyAuth
) -> None:
    args: Final = HookArgs(auth=auth, request=_copied(request_data), result=error)
    view: Final = _view(
        spend_log_input=derivations.spend_log_failure_input(args),
        budget_reservation=derivations.budget_reservation(args),
    )
    _ = await logger.run_rust(
        Hook.post_call_failure, _FAILURE_READS, view, CallContext(Hook.post_call_failure, None, None)
    )


def without(row: Mapping[str, object], volatile: Sequence[str]) -> Mapping[str, object]:
    return {
        key: datetime.fromisoformat(value)
        if key in {"startTime", "endTime", "completionStartTime"} and isinstance(value, str)
        else value
        for key, value in row.items()
        if key not in volatile
    }
