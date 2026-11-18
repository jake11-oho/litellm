import base64
import datetime
import json
from datetime import timezone
from typing import Final

import pytest

import litellm
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

_NOW: Final = datetime.datetime(2026, 1, 1, tzinfo=timezone.utc)
_LONG_SUMMARY: Final = "summary-" + "s" * 70000
_AFFINITY: Final = "litellm_enc:" + base64.b64encode(b"model-id-1").decode().rstrip("=") + ";"


def _compaction_content(summary: str, extra: dict[str, object] | None = None) -> str:
    return json.dumps({"type": "compaction", "content": summary, **(extra or {})})


_CHAT_RESPONSE: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "token_ids": [1, 2],
            "provider_specific_fields": {"citations": ["c"], "reasoning": "why", "other": 1},
            "message": {
                "role": "assistant",
                "content": "secret answer",
                "reasoning_content": "secret thought",
                "thinking_blocks": [{"type": "thinking", "thinking": "t"}],
                "audio": {"data": "a"},
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "f", "arguments": '{"q": 1}'}},
                    {"id": "t2", "type": "function", "function": {"name": "g"}},
                    "not-a-call",
                ],
                "function_call": {"name": "legacy", "arguments": "{}"},
                "provider_specific_fields": {
                    "reasoning_content": "r",
                    "server_tool_calls": [{"function": {"name": "s", "arguments": "x"}}, {"other": 1}],
                    "web_search_calls": [
                        {"action": {"query": "q", "url": None, "sources": ["s"], "type": "search"}},
                        {"id": "no-action"},
                    ],
                    "token_ids": [3],
                },
            },
        },
        {"index": 1, "text": "legacy completion text", "provider_specific_fields": None},
        {"index": 2, "delta": {"content": "streamed", "tool_calls": None}},
    ],
    "vertex_ai_grounding_metadata": [{"q": "x"}],
    "_hidden_params": {"vertex_ai_safety_ratings": [1], "kept": True},
    "trace": {"guardrail": {"findings": [{"match": "4111", "regex": "\\d+", "type": "CARD"}]}},
    "amazon-bedrock-trace": ["not-a-mapping"],
}

_RESPONSES_RESPONSE: Final = {
    "id": "resp_1",
    "object": "response",
    "instructions": "be brief",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "secret", "annotations": []},
                {"type": "refusal", "refusal": "no"},
                "plain",
            ],
        },
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}], "text": "raw"},
        {"type": "compaction", "id": "cmp_1", "encrypted_content": _compaction_content("conversation summary")},
        {"type": "function_call", "name": "f", "arguments": "{}"},
        {"type": "custom_tool_call", "name": "c", "input": "payload"},
        {"type": "web_search_call", "action": {"type": "search", "query": "q", "queries": ["a"], "pattern": None}},
    ],
}

_REQUEST_BODY: Final = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "secret prompt"}],
    "instructions": "system secret",
    "retrieval_query": None,
    "extra_body": {"instructions": "nested", "complete_input_dict": {"query_image": "img"}},
    "optional_params": {"instructions": "opt"},
    "served_output_texts": ["served"],
    "api_key": "sk-request-body-key",
    "litellm_params": {
        "instructions": "inner",
        "metadata": {
            "hidden_params": {
                "provider_specific_fields": {"citations": [1], "reasoning_text": "x"},
                "vertex_ai_citation_metadata": [1],
            }
        },
        "litellm_metadata": {"hidden_params": {"vertex_ai_grounding_metadata": [2]}},
        "proxy_server_request": {
            "classifier_input": {"x": 1},
            "body": {"messages": [{"role": "user", "content": "nested secret"}], "prompt": "p"},
        },
    },
    "standard_logging_object": {
        "messages": [{"role": "user", "content": "logged"}],
        "classifier_input": {"y": 2},
        "model_parameters": {"instructions": "mp"},
        "response": {"output": [{"type": "message", "text": "x"}], "vertex_ai_safety_results": [1]},
    },
}

_RESPONSES_REQUEST_BODY: Final = {
    "model": "gpt-5",
    "input": [
        {"role": "user", "content": "hello"},
        {"type": "compaction", "id": "cmp_1", "encrypted_content": _AFFINITY + _compaction_content(_LONG_SUMMARY)},
        {"type": "compaction", "id": "cmp_2", "encrypted_content": "opaque-provider-blob"},
        {"type": "compaction", "encrypted_content": _compaction_content("short", {"extra": True})},
    ],
}


def _kwargs(
    body: dict[str, object],
    response: object,
    call_type: str,
    *,
    headers: dict[str, object] | None = None,
    dynamic: object = None,
    bucket: str = "metadata",
) -> dict[str, object]:
    metadata: Final[dict[str, object]] = {"user_api_key": "sk-scenario"}
    if headers is not None:
        metadata["headers"] = headers
    return {
        "model": "gpt-4o",
        "call_type": call_type,
        "litellm_call_id": f"call-{call_type}",
        "litellm_params": {bucket: metadata, "proxy_server_request": {"body": body}},
        "standard_callback_dynamic_params": {} if dynamic is None else {"turn_off_message_logging": dynamic},
        "standard_logging_object": {
            "call_type": call_type,
            "model_map_information": None,
            "response": response,
            "metadata": {},
        },
    }


_SCENARIOS: Final = {
    "chat_global_redaction": (_REQUEST_BODY, _CHAT_RESPONSE, "acompletion", None, None, "metadata", True),
    "responses_enable_header": (
        _RESPONSES_REQUEST_BODY,
        _RESPONSES_RESPONSE,
        "aresponses",
        {"x-litellm-enable-message-redaction": "true"},
        None,
        "litellm_metadata",
        False,
    ),
    "disable_header_wins": (
        _REQUEST_BODY,
        _CHAT_RESPONSE,
        "acompletion",
        {"litellm-disable-message-redaction": True, "litellm-enable-message-redaction": True},
        True,
        "metadata",
        True,
    ),
    "dynamic_false_beats_header": (
        _REQUEST_BODY,
        _RESPONSES_RESPONSE,
        "aresponses",
        {"litellm-enable-message-redaction": "yes"},
        "false",
        "metadata",
        True,
    ),
    "dynamic_true_string": (_REQUEST_BODY, "plain text response", "acompletion", None, "True", "metadata", False),
    "unrecognized_response_shape": (_REQUEST_BODY, {"data": [1]}, "aembedding", None, True, "metadata", False),
    "compaction_kept": (
        _RESPONSES_REQUEST_BODY,
        {
            "id": "resp_2",
            "output": [
                {"type": "compaction", "encrypted_content": _AFFINITY + _compaction_content(_LONG_SUMMARY)},
                {"type": "compaction", "content": _LONG_SUMMARY},
                {"type": "compaction", "encrypted_content": "opaque"},
                {"type": "message", "content": [{"type": "output_text", "text": _LONG_SUMMARY}]},
            ],
        },
        "aresponses",
        None,
        None,
        "metadata",
        False,
    ),
    "compaction_redacted": (
        _RESPONSES_REQUEST_BODY,
        {"id": "resp_3", "output": [{"type": "compaction", "encrypted_content": _compaction_content("x")}]},
        "aresponses",
        None,
        True,
        "metadata",
        False,
    ),
}


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_a_golden_scenario_row(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    from litellm.proxy import proxy_server

    body, response, call_type, headers, dynamic, bucket, global_off = _SCENARIOS[name]
    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    monkeypatch.setattr(litellm, "turn_off_message_logging", global_off)
    kwargs: Final = _kwargs(body, response, call_type, headers=headers, dynamic=dynamic, bucket=bucket)

    row: Final = get_logging_payload(kwargs=kwargs, response_obj={"id": f"resp-{name}"}, start_time=_NOW, end_time=_NOW)

    assert row["request_id"] == f"resp-{name}"
