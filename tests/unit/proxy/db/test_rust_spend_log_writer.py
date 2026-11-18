import asyncio
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from litellm.proxy.db.native_database import ServiceLoggingEvents
from litellm.proxy.db.rust_spend_log_writer import (
    install_rust_spend_log_writer,
    rust_spend_log_writer,
    uninstall_rust_spend_log_writer,
)
from litellm.proxy.db.spend_log_batching import spend_log_row_bytes
from litellm.proxy.utils import (
    PrismaClient,
    _total_queued_spend_transactions,
    accounting_only_spend_log,
    dequeue_spend_logs,
    enqueue_spend_logs,
    update_spend,
)
from litellm.rust_bridge._native import DatabaseEndpoint, NativeDatabase, RustSpendLogWriteFailure, RustSpendLogWriter

_UNREACHABLE: Final = "postgresql://litellm:secret@db.invalid:5432/proxy?pool_timeout=1"
_REFUSED: Final = "postgresql://litellm:secret@127.0.0.1:1/proxy?pool_timeout=1"
_RETRY_SLEEPS_SECONDS: Final = 1 + 2 + 4
_LEDGER_ON: Final[Mapping[str, object]] = {"use_spend_logs_ledger": True}


class FakePrismaClient:
    def __init__(self) -> None:
        self.spend_log_transactions: list[Mapping[str, object]] = []
        self.tool_usage_transactions: list[object] = []
        self.model_usage_transactions: list[object] = []
        self.autorouter_turn_transactions: list[object] = []
        self.baseline_accounting_transactions: list[object] = []
        self._spend_log_transactions_lock = asyncio.Lock()
        self._tool_usage_transactions_lock = asyncio.Lock()
        self._model_usage_transactions_lock = asyncio.Lock()
        self._autorouter_turn_transactions_lock = asyncio.Lock()
        self.baseline_accounting_lock = asyncio.Lock()
        self.spend_log_write_lock = asyncio.Lock()
        self.spend_log_flush_requested = None

    def jsonify_object(self, obj: Mapping[str, object]) -> Mapping[str, object]:
        return obj


async def _unreachable_database(url: str = _UNREACHABLE) -> NativeDatabase:
    return await NativeDatabase.connect(
        ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: None),
        await DatabaseEndpoint.resolve(url, 1),
        None,
    )


def _proxy_logging() -> MagicMock:
    proxy_logging_obj: Final = MagicMock()
    proxy_logging_obj.failure_handler = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.redis_update_buffer.get_spend_logs_from_redis_buffer = AsyncMock(
        return_value=[]
    )
    proxy_logging_obj.db_spend_update_writer.redis_update_buffer.store_spend_logs_in_redis = AsyncMock(
        return_value=False
    )
    return proxy_logging_obj


def _row(request_id: str, **fields: object) -> Mapping[str, object]:
    return {"request_id": request_id, "spend": 1.5, "messages": "[]", **fields}


@pytest.fixture(autouse=True)
def _python_queue_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PrismaClient, "spend_log_queue_bytes", 0)


@pytest_asyncio.fixture
async def installed() -> AsyncIterator[tuple[FakePrismaClient, RustSpendLogWriter]]:
    client: Final = FakePrismaClient()
    writer: Final = await install_rust_spend_log_writer(client, await _unreachable_database(), _LEDGER_ON, None)
    assert writer is not None
    yield client, writer
    uninstall_rust_spend_log_writer(client)


def _request_ids(rows: Sequence[Mapping[str, object]]) -> Sequence[object]:
    return [row.get("request_id") for row in rows]


@pytest.mark.asyncio
async def test_rust_queue_keeps_the_python_queue_order(
    installed: tuple[FakePrismaClient, RustSpendLogWriter],
) -> None:
    client, writer = installed

    await enqueue_spend_logs(client, [_row("a"), _row("b")])
    await enqueue_spend_logs(client, [_row("c")], at_head=True)

    assert client.spend_log_transactions == []
    assert len(writer) == 3
    assert await _total_queued_spend_transactions(client) == 3
    assert _request_ids(await dequeue_spend_logs(client, 2)) == ["c", "a"]
    assert await dequeue_spend_logs(client, 10) == [_row("b")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        _row("ascii"),
        _row("unicode", messages="héllo ☃ 𝄞", metadata={"tags": ["é", 1, 2.5, None, True]}),
        _row("escapes", response={"text": 'quote " backslash \\ newline \n tab \t'}),
        _row("numbers", spend=0.1 + 0.2, total_tokens=12345678901234, prompt_tokens=-3),
        _row("charged", accounting_payload={"spend": 1.5, "nested": {"a": [1, {"b": "c"}]}}),
    ],
)
async def test_rust_queue_measures_rows_like_the_python_queue(
    installed: tuple[FakePrismaClient, RustSpendLogWriter], row: Mapping[str, object]
) -> None:
    client, writer = installed

    await enqueue_spend_logs(client, [row])

    assert writer.queued_bytes == spend_log_row_bytes(row)
    assert await dequeue_spend_logs(client, 1) == [row]


@pytest.mark.asyncio
async def test_overflow_drops_the_same_rows_and_keeps_the_same_charges_as_the_python_queue(
    installed: tuple[FakePrismaClient, RustSpendLogWriter],
) -> None:
    client, _ = installed
    python_client: Final = FakePrismaClient()
    rows: Final = [
        _row(str(index), messages="x" * 200, **({"accounting_payload": {"spend": index}} if index % 2 else {}))
        for index in range(10)
    ]
    budget: Final = spend_log_row_bytes(rows[0]) * 4

    for row in rows:
        await enqueue_spend_logs(client, [row], max_bytes=budget)
        await enqueue_spend_logs(python_client, [row], max_bytes=budget)

    rust_rows: Final = await dequeue_spend_logs(client, 100)
    assert any(row.get("accounting_log_disabled") is True for row in rust_rows)
    assert rust_rows == python_client.spend_log_transactions


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [_UNREACHABLE, _REFUSED], ids=["unresolvable_host", "refused_connect"])
async def test_an_outage_is_retried_three_times_then_requeued_at_the_head(url: str) -> None:
    client: Final = FakePrismaClient()
    writer: Final = await install_rust_spend_log_writer(client, await _unreachable_database(url), _LEDGER_ON, None)
    proxy_logging_obj: Final = _proxy_logging()
    rows: Final = [_row("a", accounting_payload={"spend": 1.5}), _row("b")]
    await enqueue_spend_logs(client, rows)
    started: Final = time.monotonic()

    with pytest.raises(RustSpendLogWriteFailure) as failure:
        await update_spend(client, None, proxy_logging_obj)
    elapsed: Final = time.monotonic() - started
    await asyncio.sleep(0)
    requeued: Final = await dequeue_spend_logs(client, 10)
    uninstall_rust_spend_log_writer(client)

    assert writer is not None
    assert failure.value.requeue is True
    assert failure.value.kind == "unreachable"
    assert elapsed >= _RETRY_SLEEPS_SECONDS
    assert requeued == rows
    assert proxy_logging_obj.failure_handler.await_count == 1


@pytest.mark.asyncio
async def test_a_row_the_rust_writer_cannot_encode_is_dropped_and_keeps_its_charge(
    installed: tuple[FakePrismaClient, RustSpendLogWriter],
) -> None:
    client, _ = installed
    unencodable: Final = _row("lone", messages="\ud800", accounting_payload={"spend": 2.0}, call_type="acompletion")

    await enqueue_spend_logs(client, [_row("kept"), unencodable])

    assert await dequeue_spend_logs(client, 10) == [_row("kept"), accounting_only_spend_log(unencodable)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("general_settings", "spend_logs_url"),
    [({"use_spend_logs_ledger": False}, None), (_LEDGER_ON, "https://spend-logs.example"), ({}, "https://x.example")],
)
async def test_the_python_queue_stays_when_the_ledger_is_off_or_spend_logs_are_exported(
    general_settings: Mapping[str, object], spend_logs_url: str | None
) -> None:
    client: Final = FakePrismaClient()
    await enqueue_spend_logs(client, [_row("a")])

    writer: Final = await install_rust_spend_log_writer(
        client, await _unreachable_database(), general_settings, spend_logs_url
    )

    assert writer is None
    assert rust_spend_log_writer(client) is None
    assert client.spend_log_transactions == [_row("a")]


@pytest.mark.asyncio
async def test_installing_moves_rows_already_queued_in_python_ahead_of_new_rows() -> None:
    client: Final = FakePrismaClient()
    await enqueue_spend_logs(client, [_row("early"), _row("earlier", accounting_payload={"spend": 1.0})])

    writer: Final = await install_rust_spend_log_writer(client, await _unreachable_database(), {}, None)
    await enqueue_spend_logs(client, [_row("late")])

    assert writer is not None
    assert client.spend_log_transactions == []
    assert PrismaClient.spend_log_queue_bytes == 0
    assert _request_ids(await dequeue_spend_logs(client, 10)) == ["early", "earlier", "late"]
    uninstall_rust_spend_log_writer(client)
    assert rust_spend_log_writer(client) is None
    assert writer.queued_bytes == 0
