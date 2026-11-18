import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypeIs  # noqa: TID251  # TypeIs reaches typing only on 3.13

import litellm
from litellm._uuid import uuid
from litellm.constants import (
    CALLER_SESSION_ID_METADATA_KEY,
    MINIMUM_CUSTOM_KEY_LENGTH,
    SESSION_ID_GENERATED_METADATA_KEY,
)
from litellm.litellm_core_utils.core_helpers import (
    get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # the helper's dict parameter is untyped upstream
)
from litellm.litellm_core_utils.hidden_params import served_from_cache
from litellm.litellm_core_utils.internal_call_metadata import is_unbilled_non_inference_call
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.litellm_core_utils.ptu_pricing import AZURE_SPILLOVER_FROM_HEADER, AZURE_SPILLOVER_HEADER
from litellm.litellm_core_utils.redact_messages import MESSAGE_REDACTION_ENABLE_HEADERS
from litellm.litellm_core_utils.safe_json_dumps import safe_json_structure
from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
from litellm.proxy.route_llm_request import ProxyModelNotFoundError
from litellm.rust_bridge import _native
from litellm.secret_managers.main import get_secret_bool
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    SETTLES_BACKGROUND_RESPONSE_KWARG,
    BackgroundResponseSettlement,
)
from litellm.types.utils import LiteLLMBatch, batch_chat_usage
from litellm.types.videos.utils import POLLED_VIDEO_STATUS_KWARG

if TYPE_CHECKING:
    from litellm.router import Router

SPEND_LOG_ROW_INPUT_VERSION: Final = 2
_COMPLETION_START_TIME: Final[TypeAdapter[datetime | None]] = TypeAdapter(datetime | None)
_ROW_METADATA: Final = TypeAdapter(dict[str, object])
_METADATA_KEYS: Final = frozenset(
    {
        *SpendLogsMetadata.__annotations__,
        "model_info",
        "model_group",
        "tags",
        "user_api_key_hash",
        "user_api_key_end_user_id",
        "status",
        "session_id",
        "litellm_session_id_omitted",
        "agent_id",
        "deployment",
        "user_api_key_metadata",
        "user_api_key_team_metadata",
        "standard_logging_guardrail_information",
        CALLER_SESSION_ID_METADATA_KEY,
    }
)
_REQUEST_METADATA_KEYS: Final = frozenset(
    {
        "used_client_oauth_token",
        "kubernetes_pod_routing",
        "session_id",
        "trace_id",
        SESSION_ID_GENERATED_METADATA_KEY,
    }
)
_MESSAGE_REDACTION_HEADERS: Final = frozenset({"litellm-disable-message-redaction", *MESSAGE_REDACTION_ENABLE_HEADERS})
_PAYLOAD_KEYS: Final = frozenset(
    {
        "id",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "request_tags",
        "model_id",
        "model_group",
        "api_base",
        "custom_llm_provider",
        "model",
        "model_map_information",
        "guardrail_information",
        "cost_breakdown",
        "autorouter_savings",
        "autorouter_savings_estimate",
        "autorouter_baseline_observation",
        "call_type",
        "response",
        "response_cost",
        "trace_id",
        "classifier_input",
        "originating_request_masked",
    }
)
_PAYLOAD_METADATA_KEYS: Final = frozenset(
    {
        "user_api_key_hash",
        "user_api_key_alias",
        "user_api_key_end_user_id",
        "is_silent_experiment",
        "applied_guardrails",
        "mcp_tool_call_metadata",
        "vector_store_request_metadata",
        "usage_object",
        "cold_storage_object_key",
    }
)
_HIDDEN_PARAMS_KEYS: Final = frozenset(
    {
        "litellm_overhead_time_ms",
        "upstream_provider",
        "batch_models",
        "batch_successful_requests",
        "batch_failed_requests",
    }
)
_LITELLM_PARAMS_KEYS: Final = frozenset(
    {
        "api_base",
        "litellm_call_id",
        "proxy_server_request",
        "video_id",
        "litellm_session_id",
        "user_api_key_end_user_id",
    }
)
_KWARGS_KEYS: Final = frozenset(
    {
        "model",
        "custom_llm_provider",
        "response_cost",
        "agent_id",
        "litellm_call_id",
        "litellm_trace_id",
        "realtime_tools",
        "realtime_tool_calls",
        POLLED_VIDEO_STATUS_KWARG,
    }
)
_SPILLOVER_HEADERS: Final = (AZURE_SPILLOVER_HEADER, AZURE_SPILLOVER_FROM_HEADER)
_PROVIDER_SPILLOVER_HEADERS: Final = tuple(f"llm_provider-{header}" for header in _SPILLOVER_HEADERS)


def _is_str_mapping(value: object) -> TypeIs[Mapping[str, object]]:  # guard-ok: logging mappings are str-keyed
    return isinstance(value, Mapping)


def _is_dict(value: object) -> TypeIs[dict[str, object]]:  # guard-ok: str-keyed  # mutable-ok: main's dict
    return isinstance(value, dict)


def _is_array(value: object) -> TypeIs[Sequence[object]]:  # guard-ok: isinstance decides it
    return isinstance(value, list | tuple | set | frozenset)


def _str_mapping(value: object) -> Mapping[str, object] | None:
    return value if _is_str_mapping(value) else None


def _text(value: str) -> str:
    plain: Final = str.__str__(value)
    try:
        plain.encode()
    except UnicodeEncodeError:
        return plain.encode("utf-8", "replace").decode("utf-8")
    return plain


def _plain(value: object) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, BaseModel):
        return _plain(value.model_dump())
    if _is_str_mapping(value):
        return {key if type(key) is str else str(key): _plain(item) for key, item in value.items()}
    if _is_array(value):
        return [_plain(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return _text(str(value))


def _projected(source: Mapping[str, object], keys: frozenset[str]) -> Mapping[str, object]:
    return {key: _plain(value) for key, value in source.items() if key in keys}


def _projected_headers(headers: object, names: Sequence[str] | frozenset[str]) -> Mapping[str, object] | None:
    found: Final = _str_mapping(headers)
    if found is None:
        return None
    return {name: _plain(value) for name in names if (value := found.get(name)) is not None}


def _request_metadata(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: {
            **_projected(metadata, _REQUEST_METADATA_KEYS),
            **(
                {"headers": _projected_headers(metadata.get("headers"), _MESSAGE_REDACTION_HEADERS)}
                if "headers" in metadata
                else {}
            ),
        }
        if (metadata := _str_mapping(litellm_params[key])) is not None
        else None
        for key in ("metadata", "litellm_metadata")
        if key in litellm_params
    }


def _timestamp(value: datetime) -> Mapping[str, object]:
    return {
        "raw": value.isoformat(),
        "utc": value.astimezone(timezone.utc).isoformat(),
        "aware": value.tzinfo is not None,
    }


def _usage(
    call_type: object,
    metadata: Mapping[str, object],
    response_obj: object,
    response_obj_dict: dict[str, object],  # mutable-ok: forwarded to the OCR usage extractor, which takes a dict
) -> Mapping[str, object]:
    from litellm.proxy.spend_tracking.spend_tracking_utils import (
        _extract_usage_for_ocr_call,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # the row input reuses the logging payload's own OCR usage extraction
    )

    if call_type in ["ocr", "aocr"]:
        return _str_mapping(_extract_usage_for_ocr_call(response_obj, response_obj_dict)) or {}  # pyright: ignore[reportUnknownArgumentType]  # the OCR extractor returns an untyped dict
    if is_unbilled_non_inference_call(call_type, metadata, response_obj_dict):  # pyright: ignore[reportArgumentType]  # call_type is the logged kwargs value, passed as main passes it
        return {}
    chat_usage_of_batch: Final = batch_chat_usage(response_obj) if isinstance(response_obj, LiteLLMBatch) else None
    usage: Final = (
        chat_usage_of_batch.model_dump()
        if chat_usage_of_batch is not None
        else StandardLoggingPayloadSetup.get_usage_as_dict(response_obj_dict)
    )
    if isinstance(usage, litellm.Usage):
        return dict(usage)
    return usage


def _final_usage(kwargs: Mapping[str, object], usage: Mapping[str, object]) -> Mapping[str, object]:
    combined: Final = kwargs.get("combined_usage_object")
    replaced: Final = not usage or "continued_partial_usage" in kwargs or "failed_attempts_billed_on_success" in kwargs
    if replaced and isinstance(combined, litellm.Usage):
        return combined.model_dump()
    return usage


def spend_log_row_config() -> Mapping[str, object]:
    from litellm.proxy.proxy_server import general_settings_view

    settings: Final = general_settings_view()
    return {
        "version": SPEND_LOG_ROW_INPUT_VERSION,
        "store_prompts_in_spend_logs": _plain(settings.get("store_prompts_in_spend_logs")),
        "store_prompts_in_spend_logs_env": get_secret_bool("STORE_PROMPTS_IN_SPEND_LOGS") is True,
        "spend_logs_prompt_storage": _plain(settings.get("spend_logs_prompt_storage")),
        "redact_tokens_in_spend_logs": settings.get("redact_tokens_in_spend_logs") is True,
        "disable_requester_ip_address_logging": _plain(settings.get("disable_requester_ip_address_logging")),
        "missing_session_id": _plain(settings.get("missing_session_id")),
        "max_string_length_prompt_in_db": os.getenv("MAX_STRING_LENGTH_PROMPT_IN_DB"),
        "disable_end_user_cost_tracking": bool(litellm.disable_end_user_cost_tracking),
        "request_correlation_in_logs": bool(litellm.request_correlation_in_logs),
        "turn_off_message_logging": litellm.turn_off_message_logging is True,
    }


@lru_cache(maxsize=1)
def token_word_ranges() -> tuple[tuple[int, int], ...]:
    characters: Final = "".join(map(chr, range(128, sys.maxunicode + 1)))
    return tuple((match.start() + 128, match.end() + 127) for match in re.finditer(r"\w+", characters))


def spend_log_row_input(
    kwargs: dict[str, object],  # mutable-ok: the logging kwargs as litellm hands them over, only read here
    response_obj: object,
    start_time: datetime,
    end_time: datetime,
    llm_router: "Router | None",
    now: float,
    fallback_session_id: str,
) -> Mapping[str, object]:
    from litellm.proxy.spend_tracking.spend_tracking_utils import (
        _is_configured_model_group,  # pyright: ignore[reportPrivateUsage]  # router registry lookups stay with the router
        _model_group_provider,  # pyright: ignore[reportPrivateUsage]  # router registry lookups stay with the router
        _string_or_empty,  # pyright: ignore[reportPrivateUsage]  # the router lookup key is read the logging payload's way
    )

    response_obj_dict: Final[dict[str, object]] = (  # mutable-ok: the response dict main's extractors take
        {}
        if response_obj is None
        else response_obj
        if _is_dict(response_obj)
        else response_obj.model_dump()
        if isinstance(response_obj, BaseModel)
        else {"result": str(response_obj)}
    )
    response: Final = response_obj if isinstance(response_obj, BaseModel) else response_obj_dict
    litellm_params: Final = _str_mapping(kwargs.get("litellm_params", {})) or {}
    metadata: Final = _str_mapping(get_litellm_metadata_from_kwargs(kwargs)) or {}  # pyright: ignore[reportUnknownArgumentType]  # the metadata helper returns an untyped dict
    completion_start_time: Final = _COMPLETION_START_TIME.validate_python(kwargs.get("completion_start_time", end_time))
    call_type: Final = kwargs.get("call_type")
    usage: Final = _final_usage(kwargs, _usage(call_type, metadata, response, response_obj_dict))
    payload: Final = _str_mapping(kwargs.get("standard_logging_object", None))
    payload_metadata: Final = _str_mapping(payload.get("metadata")) if payload is not None else None
    hidden_params: Final = _str_mapping(payload.get("hidden_params")) if payload is not None else None
    dynamic_params: Final = _str_mapping(kwargs.get("standard_callback_dynamic_params"))

    model_group: Final = _string_or_empty(metadata.get("model_group")) or (
        _string_or_empty(payload.get("model_group")) if payload is not None else ""
    )
    logged_provider: Final = kwargs.get("custom_llm_provider") or (
        _string_or_empty(payload.get("custom_llm_provider")) if payload is not None else ""
    )
    cache_key: Final = (
        "Cache OFF"
        if litellm.cache is None
        else litellm_params["preset_cache_key"]
        if litellm_params.get("preset_cache_key") is not None
        else litellm.cache.get_cache_key(**kwargs)
    )

    return {
        "version": SPEND_LOG_ROW_INPUT_VERSION,
        "call_type": _plain(call_type),
        "cache_hit": _plain(kwargs.get("cache_hit", False)),
        "served_from_cache": served_from_cache(response_obj),
        "rejected_as_unknown_model": isinstance(response_obj, ProxyModelNotFoundError),
        "unknown_model_error_message": _plain(response_obj.spend_log_error_message)
        if isinstance(response_obj, ProxyModelNotFoundError)
        else None,
        "response_id": _plain(response_obj_dict.get("id")),
        "now": now,
        "fallback_session_id": fallback_session_id,
        "start_time": _timestamp(start_time),
        "end_time": _timestamp(end_time),
        "completion_start_time": _timestamp(completion_start_time) if completion_start_time is not None else None,
        "usage": _plain(usage),
        "metadata": _projected(metadata, _METADATA_KEYS),
        "litellm_params": {**_projected(litellm_params, _LITELLM_PARAMS_KEYS), **_request_metadata(litellm_params)},
        "kwargs": {
            **_projected(kwargs, _KWARGS_KEYS),
            "response_headers": _projected_headers(kwargs.get("response_headers"), _SPILLOVER_HEADERS),
            SETTLES_BACKGROUND_RESPONSE_KWARG: kwargs.get(SETTLES_BACKGROUND_RESPONSE_KWARG)
            is BackgroundResponseSettlement.SETTLES,
            "turn_off_message_logging": _plain(dynamic_params.get("turn_off_message_logging"))
            if dynamic_params
            else None,
        },
        "standard_logging_object": None
        if payload is None
        else {
            **_projected(payload, _PAYLOAD_KEYS),
            **(
                {"messages": _plain(payload.get("messages"))}
                if "messages" in payload and payload.get("call_type") == "_arealtime"
                else {}
            ),
            **(
                {"metadata": _projected(payload_metadata, _PAYLOAD_METADATA_KEYS)}
                if payload_metadata is not None
                else {}
            ),
            **(
                {
                    "hidden_params": {
                        **_projected(hidden_params, _HIDDEN_PARAMS_KEYS),
                        "additional_headers": _projected_headers(
                            hidden_params.get("additional_headers"), _PROVIDER_SPILLOVER_HEADERS
                        ),
                    }
                }
                if hidden_params is not None
                else {}
            ),
        },
        "cache_key": _plain(cache_key),
        "model_group_provider": None if logged_provider else _model_group_provider(model_group, llm_router),
        "is_configured_model_group": _is_configured_model_group(model_group, llm_router)
        if metadata.get("status") == "failure"
        else False,
    }


def current_spend_log_row_input(
    kwargs: dict[str, object],  # mutable-ok: the logging kwargs as litellm hands them over, only read here
    response_obj: object,
    start_time: datetime,
    end_time: datetime,
    llm_router: "Router | None",
) -> Mapping[str, object]:
    return spend_log_row_input(kwargs, response_obj, start_time, end_time, llm_router, time.time(), str(uuid.uuid4()))


def decode_row(row_json: str) -> SpendLogsPayload:
    row: Final[SpendLogsPayload] = json.loads(row_json)  # pyright: ignore[reportAny]  # the Rust builder returns the row as a JSON object
    start_time: Final = row["startTime"]
    end_time: Final = row["endTime"]
    completion_start_time: Final = row["completionStartTime"]
    if isinstance(start_time, str):
        row["startTime"] = datetime.fromisoformat(start_time)
    if isinstance(end_time, str):
        row["endTime"] = datetime.fromisoformat(end_time)
    if isinstance(completion_start_time, str):
        row["completionStartTime"] = datetime.fromisoformat(completion_start_time)
    return row


@dataclass(frozen=True, slots=True)
class LedgerRowInput:
    billable_cost: float | None
    org_id: str | None
    team_id: str | None
    delivery_id: str
    entities: Mapping[str, str | None]
    model_access_groups: Sequence[str]
    windows: Sequence[Mapping[str, object]]
    router: "Callable[[], Router | None]"


def uses_spend_logs_ledger() -> bool:
    from litellm.proxy.proxy_server import general_settings_view

    return general_settings_view().get("use_spend_logs_ledger", True) is not False


@lru_cache(maxsize=8)
def _row_builder(settings_json: str) -> _native.RustSpendLogRowBuilder:
    return _native.RustSpendLogRowBuilder(settings_json, MINIMUM_CUSTOM_KEY_LENGTH, token_word_ranges())


def rust_logging_payload(
    kwargs: dict[str, object],  # mutable-ok: the logging kwargs as litellm hands them over, only read here
    response_obj: object,
    start_time: datetime,
    end_time: datetime,
    llm_router: "Router | None",
) -> SpendLogsPayload:
    row_input: Final = current_spend_log_row_input(kwargs, response_obj, start_time, end_time, llm_router)
    return decode_row(_row_builder(json.dumps(spend_log_row_config())).build_row(json.dumps(row_input)))


def _json_structure(value: object) -> object:
    return json.loads(json.dumps(safe_json_structure(value)))  # pyright: ignore[reportAny]  # a JSON value read back as object


def ledger_spend_log_row(payload: SpendLogsPayload, ledger: LedgerRowInput) -> SpendLogsPayload:
    from litellm.accounting.snapshot import (
        _savings_or_zero,  # pyright: ignore[reportPrivateUsage]  # savings pricing stays in Python and feeds the Rust row
    )

    row: Final = {
        key: value.isoformat() if isinstance(value, datetime) else _plain(value) for key, value in payload.items()
    }
    ledger_input: Final = {
        "response_cost": ledger.billable_cost,
        "org_id": ledger.org_id,
        "team_id": ledger.team_id,
        "delivery_id": ledger.delivery_id,
        "entities": _json_structure(dict(ledger.entities)),
        "model_access_groups": _json_structure(list(ledger.model_access_groups)),
        "windows": _json_structure(list(ledger.windows)),
        "savings": _savings_or_zero(payload, _ROW_METADATA.validate_json(payload["metadata"]), ledger.router)._asdict(),
    }
    finalized: Final[SpendLogsPayload] = json.loads(  # pyright: ignore[reportAny]  # the Rust builder returns the row as a JSON object
        _native.RustSpendLogRowBuilder.finalize_ledger_row(json.dumps(row), json.dumps(ledger_input))
    )
    if "completionStartTime" in payload:
        finalized["completionStartTime"] = payload["completionStartTime"]
    return finalized
