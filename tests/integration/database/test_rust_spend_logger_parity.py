"""A completion and a failed request charged through the Rust spend log callback land the same SpendLogs and
delivery rows in Postgres as through the Python cost callback. Needs DATABASE_URL on a Prisma-migrated
database."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Final

import psycopg
import pytest
import pytest_asyncio
from psycopg.rows import dict_row
from pydantic import JsonValue, TypeAdapter

import litellm
import litellm.proxy.proxy_server as ps
from litellm.integrations.custom_logger_v2.rust import RustLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.native_database import ServiceLoggingEvents
from litellm.proxy.db.rust_spend_log_writer import (
    install_rust_spend_log_writer,
    rust_spend_log_writer,
    uninstall_rust_spend_log_writer,
)
from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
from litellm.proxy.hooks.rust_spend_logger import SpendLogHost, spend_logger
from litellm.proxy.spend_tracking.spend_logs_ledger_setting import capture_spend_logs_ledger_setting
from litellm.proxy.utils import PrismaClient, ProxyUpdateSpend
from litellm.rust_bridge._native import DatabaseEndpoint, NativeDatabase, NativeSpendCounters, RustSpendLogWriter
from tests.integration.database.test_rust_spend_log_writer_parity import FakePrismaClient, _database_url, _proxy_logging
from tests.unit.proxy.hooks.rust_spend_logger_twin import (
    completion,
    legacy_failure,
    legacy_success,
    rust_failure,
    rust_success,
)

_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])
_STARTED: Final = datetime(2026, 11, 17, 9, 0, tzinfo=timezone.utc)
_LOG_VOLATILE: Final = ("accounting_delivery_id",)
_FAILURE_LOG_VOLATILE: Final = (
    "accounting_delivery_id",
    "endTime",
    "completionStartTime",
    "request_duration_ms",
    "session_id",
)
_DELIVERY_VOLATILE: Final = ("delivery_id",)


class Ledger:
    def __init__(self, url: str, client: FakePrismaClient, writer: RustSpendLogWriter, logger: RustLogger) -> None:
        self.url: Final = url
        self.client: Final = client
        self.writer: Final = writer
        self.logger: Final = logger

    async def flush_python(self) -> None:
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=3, prisma_client=self.client, db_writer_client=None, proxy_logging_obj=_proxy_logging()
        )

    async def stored(
        self, request_ids: Sequence[str]
    ) -> tuple[Sequence[Mapping[str, JsonValue]], Sequence[Mapping[str, JsonValue]]]:
        async with await psycopg.AsyncConnection.connect(self.url, row_factory=dict_row) as connection:
            logs: Final = await (
                await connection.execute(
                    """SELECT to_jsonb(s) - 'created_at' - 'updated_at' AS row FROM "LiteLLM_SpendLogs" s """
                    "WHERE request_id = ANY(%s) ORDER BY request_id",
                    (list(request_ids),),
                )
            ).fetchall()
            deliveries: Final = await (
                await connection.execute(
                    """SELECT to_jsonb(d) - 'created_at' - 'next_attempt_at' AS row FROM "LiteLLM_SpendLogDelivery" d """
                    "WHERE request_id = ANY(%s) ORDER BY request_id",
                    (list(request_ids),),
                )
            ).fetchall()
            await connection.execute('DELETE FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)', (list(request_ids),))
            await connection.execute(
                'DELETE FROM "LiteLLM_SpendLogDelivery" WHERE request_id = ANY(%s)', (list(request_ids),)
            )
        return (
            _ROWS.validate_python([row["row"] for row in logs]),
            _ROWS.validate_python([row["row"] for row in deliveries]),
        )


@pytest_asyncio.fixture
async def ledger(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Ledger]:
    url: Final = _database_url()
    monkeypatch.delenv("SPEND_LOGS_URL", raising=False)
    monkeypatch.setattr(PrismaClient, "spend_log_queue_bytes", 0)
    client: Final = FakePrismaClient(url)
    database: Final = await NativeDatabase.connect(
        ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: None),
        await DatabaseEndpoint.resolve(url, 2),
        None,
    )
    writer: Final = await install_rust_spend_log_writer(client, database, {"use_spend_logs_ledger": True}, None)  # pyright: ignore[reportArgumentType]  # the fake stands in for the Prisma client
    assert writer is not None
    handle: Final = ps.spend_counter_store.handle
    assert isinstance(handle, NativeSpendCounters)
    monkeypatch.setattr(ps, "prisma_client", client)
    monkeypatch.setattr(ps, "general_settings", {"use_spend_logs_ledger": True})
    try:
        yield Ledger(url, client, writer, spend_logger(writer, handle))
    finally:
        uninstall_rust_spend_log_writer(client)


def _masked(rows: Sequence[Mapping[str, JsonValue]], volatile: Sequence[str]) -> Sequence[Mapping[str, JsonValue]]:
    return [{key: value for key, value in row.items() if key not in volatile} for row in rows]


def _counters(keys: Sequence[str]) -> Sequence[float]:
    return [ps.spend_counter_store.get_local(key) or 0.0 for key in keys]


@pytest.mark.asyncio
async def test_a_completion_lands_the_same_spend_log_and_delivery_rows(ledger: Ledger) -> None:
    token: Final = f"hashed-{uuid.uuid4().hex}"
    team: Final = f"team-{uuid.uuid4().hex}"
    counters: Final = (f"spend:key:{token}", f"spend:team:{team}", f"spend:team_member:user-1:{team}")
    done: Final = await completion(
        {
            "user_api_key": token,
            "user_api_key_alias": "alias",
            "user_api_key_user_id": "user-1",
            "user_api_key_team_id": team,
            "user_api_key_org_id": "org-1",
            "tags": ["blue"],
        }
    )
    request_id: Final = str(getattr(done.response, "id", None))

    await legacy_success(done)
    await ledger.flush_python()
    python_logs, python_deliveries = await ledger.stored([request_id])
    after_python: Final = _counters(counters)
    await rust_success(ledger.logger, done)
    await ledger.logger.flush_rust()
    rust_logs, rust_deliveries = await ledger.stored([request_id])
    after_rust: Final = _counters(counters)

    assert len(ledger.writer) == 0
    assert [len(rust_logs), len(rust_deliveries)] == [1, 1]
    assert rust_logs[0]["spend"] == python_logs[0]["spend"]
    assert rust_deliveries[0]["request_id"] == request_id
    assert _masked(rust_logs, _LOG_VOLATILE) == _masked(python_logs, _LOG_VOLATILE)
    assert _masked(rust_deliveries, _DELIVERY_VOLATILE) == _masked(python_deliveries, _DELIVERY_VOLATILE)
    cost: Final = rust_logs[0]["spend"]
    assert isinstance(cost, float) and cost > 0
    assert after_python == [cost] * len(counters)
    assert after_rust == [2 * cost] * len(counters)


@pytest.mark.asyncio
async def test_a_failed_request_lands_the_same_failure_rows(ledger: Ledger) -> None:
    request_id: Final = f"failed-{uuid.uuid4().hex}"
    auth: Final = UserAPIKeyAuth(api_key=f"hashed-{uuid.uuid4().hex}", user_id="user-1", team_id="team-1")
    request_data: Final = {
        "model": "gpt-4o",
        "litellm_call_id": request_id,
        "start_time": _STARTED,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"caller": "value"},
    }
    error: Final = ValueError("the provider refused")

    await legacy_failure(request_data, error, auth)
    await ledger.flush_python()
    python_logs, python_deliveries = await ledger.stored([request_id])
    await rust_failure(ledger.logger, request_data, error, auth)
    await ledger.logger.flush_rust()
    rust_logs, rust_deliveries = await ledger.stored([request_id])

    assert len(rust_logs) == 1
    assert rust_logs[0]["status"] == "failure"
    assert _masked(rust_logs, _FAILURE_LOG_VOLATILE) == _masked(python_logs, _FAILURE_LOG_VOLATILE)
    assert _masked(rust_deliveries, _DELIVERY_VOLATILE) == _masked(python_deliveries, _DELIVERY_VOLATILE)


class _ClosableFakePrismaClient(FakePrismaClient):
    async def aclose(self) -> None:
        return None


async def test_default_startup_opens_the_rust_pool_and_registers_the_rust_spend_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url: Final = _database_url()
    for name in ("RUST_RESERVED_CONNECTIONS", "SPEND_LOGS_URL", "DATABASE_URL_READ_REPLICA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    forget: Final = capture_spend_logs_ledger_setting({})
    client: Final = _ClosableFakePrismaClient(url)
    try:
        async with AsyncExitStack() as resources:
            database: Final = await ps._own_database(resources, client)  # pyright: ignore[reportArgumentType]  # the fake stands in for the Prisma client
            assert database is not None
            assert database.native is not None
            await ps._install_rust_spend_log_writer(resources, database)
            assert rust_spend_log_writer(client) is not None
            monkeypatch.setattr(ps, "prisma_client", client)
            ps.cost_tracking()
            spend_loggers: Final = [
                callback
                for callback in litellm.callbacks
                if isinstance(callback, RustLogger) and isinstance(callback.host, SpendLogHost)
            ]
            assert len(spend_loggers) == 1
            assert spend_loggers[0] in litellm._async_success_callback
            assert not any(isinstance(callback, ProxyDBLogger) for callback in litellm.callbacks)
        assert rust_spend_log_writer(client) is None
    finally:
        forget()
