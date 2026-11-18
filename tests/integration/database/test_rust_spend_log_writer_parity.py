import asyncio
import os
import re
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest
from psycopg.rows import dict_row
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.db.native_database import ServiceLoggingEvents
from litellm.proxy.db.rust_spend_log_writer import install_rust_spend_log_writer, uninstall_rust_spend_log_writer
from litellm.proxy.utils import PrismaClient, ProxyUpdateSpend, enqueue_spend_logs, jsonify_object
from litellm.rust_bridge._native import DatabaseEndpoint, NativeDatabase

_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])
_PLACEHOLDER: Final = re.compile(r"\$(\d+)")
_STARTED: Final = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)


def _database_url() -> str:
    url: Final = os.environ.get("DATABASE_URL")
    if url is None:
        pytest.skip("Set DATABASE_URL to a Prisma-migrated PostgreSQL database.")
    return url


class PsycopgTransaction:
    def __init__(self, connection: psycopg.AsyncConnection[dict[str, object]]) -> None:
        self._connection = connection

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        cursor: Final = await self._connection.execute(_PLACEHOLDER.sub(r"%(p\1)s", query), _named(args))
        return await cursor.fetchall() if cursor.description is not None else []

    async def execute_raw(self, query: str, *args: object) -> int:
        cursor: Final = await self._connection.execute(_PLACEHOLDER.sub(r"%(p\1)s", query), _named(args))
        return cursor.rowcount


def _named(args: Sequence[object]) -> Mapping[str, object]:
    return {f"p{index}": value for index, value in enumerate(args, start=1)}


class PsycopgDatabase:
    def __init__(self, url: str) -> None:
        self._url = url

    @asynccontextmanager
    async def tx(self, timeout: timedelta) -> AsyncIterator[PsycopgTransaction]:
        async with await psycopg.AsyncConnection.connect(self._url, row_factory=dict_row) as connection:
            async with connection.transaction():
                yield PsycopgTransaction(connection)


class FakePrismaClient:
    def __init__(self, url: str) -> None:
        self.writer_db = PsycopgDatabase(url)
        self.db = MagicMock()
        self.spend_log_transactions: list[Mapping[str, object]] = []
        self._spend_log_transactions_lock = asyncio.Lock()

    def jsonify_object(self, data: Mapping[str, object]) -> Mapping[str, object]:
        return jsonify_object(dict(data))


def _proxy_logging() -> MagicMock:
    proxy_logging_obj: Final = MagicMock()
    proxy_logging_obj.failure_handler = AsyncMock()
    return proxy_logging_obj


def _rows(prefix: str) -> Sequence[Mapping[str, object]]:
    full: Final = {
        "request_id": f"{prefix}-full",
        "call_type": "acompletion",
        "api_key": "hashed",
        "spend": 0.25,
        "total_tokens": 3,
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "startTime": _STARTED,
        "endTime": _STARTED + timedelta(seconds=1),
        "completionStartTime": _STARTED + timedelta(milliseconds=500),
        "request_duration_ms": 1000,
        "model": "gpt",
        "model_id": "deployment",
        "model_group": "gpt",
        "custom_llm_provider": "openai",
        "api_base": "https://example.test",
        "user": "user",
        "metadata": '{"status": "success", "tenant": "a", "nested": {"x": [1, 2.5, null]}}',
        "cache_hit": "False",
        "cache_key": "Cache OFF",
        "request_tags": '["tag", "other"]',
        "team_id": "team",
        "organization_id": None,
        "end_user": "",
        "requester_ip_address": None,
        "messages": [{"role": "user", "content": "héllo ☃"}],
        "response": {"id": "resp", "choices": [{"text": 'a "quoted" \\ value'}]},
        "proxy_server_request": "{}",
        "session_id": None,
        "status": "success",
        "mcp_namespaced_tool_name": None,
        "agent_id": None,
        "litellm_call_id": "call",
        "accounting_delivery_id": f"{prefix}-full-delivery",
        "accounting_payload": {"request_id": f"{prefix}-full", "spend": 0.25},
        "accounting_log_disabled": False,
    }
    return (
        full,
        {**full, "spend": 9.0, "accounting_payload": {"request_id": f"{prefix}-full", "spend": 9.0}},
        {
            "request_id": f"{prefix}-sparse",
            "call_type": "aembedding",
            "spend": 1.0,
            "startTime": _STARTED,
            "endTime": _STARTED,
            "accounting_delivery_id": f"{prefix}-sparse-delivery",
            "accounting_payload": {"spend": 1.0},
        },
        {
            "request_id": f"{prefix}-batch",
            "call_type": "aretrieve_batch",
            "spend": 4.0,
            "status": "success",
            "startTime": _STARTED,
            "endTime": _STARTED,
            "accounting_payload": '{"spend": 4.0}',
        },
        {
            "request_id": f"{prefix}-disabled",
            "call_type": "acompletion",
            "accounting_delivery_id": f"{prefix}-disabled-delivery",
            "accounting_payload": {"spend": 0.5},
            "accounting_log_disabled": True,
        },
    )


async def _stored(url: str, prefix: str) -> tuple[Sequence[Mapping[str, JsonValue]], Sequence[Mapping[str, JsonValue]]]:
    async with await psycopg.AsyncConnection.connect(url, row_factory=dict_row) as connection:
        logs: Final = await (
            await connection.execute(
                """SELECT to_jsonb(s) - 'created_at' - 'updated_at' AS row FROM "LiteLLM_SpendLogs" s """
                "WHERE request_id LIKE %s ORDER BY request_id",
                (f"{prefix}%",),
            )
        ).fetchall()
        deliveries: Final = await (
            await connection.execute(
                """SELECT to_jsonb(d) - 'created_at' - 'next_attempt_at' AS row FROM "LiteLLM_SpendLogDelivery" d """
                "WHERE request_id LIKE %s ORDER BY delivery_id",
                (f"{prefix}%",),
            )
        ).fetchall()
        await connection.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', (f"{prefix}%",))
        await connection.execute('DELETE FROM "LiteLLM_SpendLogDelivery" WHERE request_id LIKE %s', (f"{prefix}%",))
    return (
        _ROWS.validate_python([row["row"] for row in logs]),
        _ROWS.validate_python([row["row"] for row in deliveries]),
    )


async def _flush(client: FakePrismaClient, rows: Sequence[Mapping[str, object]]) -> None:
    await enqueue_spend_logs(client, rows)
    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=3, prisma_client=client, db_writer_client=None, proxy_logging_obj=_proxy_logging()
    )


@pytest.mark.asyncio
async def test_the_rust_writer_stores_what_the_python_writer_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    url: Final = _database_url()
    monkeypatch.delenv("SPEND_LOGS_URL", raising=False)
    monkeypatch.setattr(PrismaClient, "spend_log_queue_bytes", 0)
    prefix: Final = f"rust-writer-{uuid.uuid4().hex}"

    python_client: Final = FakePrismaClient(url)
    await _flush(python_client, _rows(prefix))
    await _flush(python_client, _rows(prefix))
    python_stored: Final = await _stored(url, prefix)

    rust_client: Final = FakePrismaClient(url)
    database: Final = await NativeDatabase.connect(
        ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: None),
        await DatabaseEndpoint.resolve(url, 2),
        None,
    )
    writer: Final = await install_rust_spend_log_writer(rust_client, database, {}, None)
    try:
        await _flush(rust_client, _rows(prefix))
        await _flush(rust_client, _rows(prefix))
    finally:
        uninstall_rust_spend_log_writer(rust_client)
    rust_stored: Final = await _stored(url, prefix)

    assert writer is not None
    assert len(writer) == 0
    assert [len(python_stored[0]), len(python_stored[1])] == [3, 4]
    assert rust_stored == python_stored
