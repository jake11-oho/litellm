import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    DEFAULT_MAX_RECURSE_DEPTH,
    SPEND_LOG_QUEUE_MAX_BACKOFF_SECONDS,
    SPEND_LOG_QUEUE_MAX_BYTES,
    SPEND_LOG_QUEUE_POLL_INTERVAL,
    SPEND_LOG_QUEUE_SIZE_THRESHOLD,
    SPEND_LOG_WRITE_BATCH_MAX_BYTES,
    SPEND_LOG_WRITE_BATCH_MAX_ROWS,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.spend_tracking.spend_log_error_logger import spend_log_error
from litellm.rust_bridge._native import NativeDatabase, RustSpendLogWriteFailure, RustSpendLogWriter

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient, ProxyLogging

_ROWS: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class _Installed:
    client: object
    writer: RustSpendLogWriter


_installed: Final[list[_Installed]] = []  # mutable-ok: registry rewritten in place by install and uninstall


def rust_spend_log_writer(prisma_client: object) -> RustSpendLogWriter | None:
    return next((installed.writer for installed in _installed if installed.client is prisma_client), None)


def rust_spend_log_queue_size(prisma_client: object) -> int:
    writer: Final = rust_spend_log_writer(prisma_client)
    return 0 if writer is None else len(writer)


def spend_log_writer_runs_in_rust(general_settings: Mapping[str, object], spend_logs_url: str | None) -> bool:
    return general_settings.get("use_spend_logs_ledger", True) is not False and spend_logs_url is None


def _encoded(row: Mapping[str, object]) -> str:
    try:
        return json.dumps(row, default=str)
    except (TypeError, ValueError):
        return safe_dumps(row)


def _enqueue_rows(
    writer: RustSpendLogWriter, rows: Sequence[Mapping[str, object]], at_head: bool, max_bytes: int
) -> tuple[int, Sequence[int]]:
    return writer.enqueue([_encoded(row) for row in rows], at_head, max_bytes)


def enqueue_in_rust(
    writer: RustSpendLogWriter,
    logs: Sequence[Mapping[str, object]],
    *,
    at_head: bool,
    max_bytes: int,
) -> None:
    from litellm.proxy.utils import accounting_only_spend_log

    dropped, refused = _enqueue_rows(writer, logs, at_head, max_bytes)
    for index in refused:
        spend_log_error(
            "Spend tracking - dropping spend log row Postgres rejected. request_id=%s error=%s",
            logs[index].get("request_id"),
            "the row holds text the Rust writer cannot encode",
        )
    charged: Final = tuple(
        accounting_only_spend_log(logs[index]) for index in refused if "accounting_payload" in logs[index]
    )
    rescue_dropped, unkept = _enqueue_rows(writer, charged, at_head, max_bytes) if charged else (0, ())
    if unkept:
        spend_log_error(
            "Spend tracking - could not keep the charges of %d dropped spend log rows. error=%s",
            len(unkept),
            "the charge holds text the Rust writer cannot encode",
        )
    if dropped + rescue_dropped > 0:
        verbose_proxy_logger.error(
            "Spend tracking - spend log queue is at its %d byte budget; dropped the %d oldest spend logs",
            max_bytes,
            dropped + rescue_dropped,
        )


def dequeue_from_rust(
    writer: RustSpendLogWriter, limit: int
) -> list[dict[str, object]]:  # mutable-ok: dequeue_spend_logs hands callers the rows the Python queue would
    return [_ROWS.validate_python(json.loads(text)) for text in writer.dequeue(limit)]


def _retention() -> tuple[tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...]] | None:
    from litellm.proxy._types import SPEND_LOGS_METADATA_ALWAYS_KEPT_FIELDS
    from litellm.proxy.spend_tracking.spend_tracking_utils import configured_spend_logs_metadata_fields

    fields: Final = configured_spend_logs_metadata_fields()
    if fields is None:
        return None
    return fields.include, fields.exclude, tuple(sorted(SPEND_LOGS_METADATA_ALWAYS_KEPT_FIELDS))


async def write_in_rust(
    writer: RustSpendLogWriter,
    prisma_client: "PrismaClient",
    proxy_logging_obj: "ProxyLogging",
    logs: Sequence[Mapping[str, object]],
) -> None:
    from litellm.proxy.utils import raise_failed_update_spend_exception, requeue_spend_logs

    start_time: Final = time.time()
    try:
        await writer.write([_encoded(row) for row in logs], _retention())
    except RustSpendLogWriteFailure as e:
        if e.kind == "invalid_metadata":
            raise
        if e.requeue:
            await requeue_spend_logs(prisma_client, proxy_logging_obj, logs)
        raise_failed_update_spend_exception(e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj)


async def install_rust_spend_log_writer(
    prisma_client: "PrismaClient",
    database: NativeDatabase,
    general_settings: Mapping[str, object],
    spend_logs_url: str | None,
) -> RustSpendLogWriter | None:
    if not spend_log_writer_runs_in_rust(general_settings, spend_logs_url):
        return None
    writer: Final = RustSpendLogWriter(
        database,
        queue_max_bytes=SPEND_LOG_QUEUE_MAX_BYTES,
        write_batch_max_bytes=SPEND_LOG_WRITE_BATCH_MAX_BYTES,
        write_batch_max_rows=SPEND_LOG_WRITE_BATCH_MAX_ROWS,
        queue_size_threshold=SPEND_LOG_QUEUE_SIZE_THRESHOLD,
        poll_interval_seconds=SPEND_LOG_QUEUE_POLL_INTERVAL,
        max_backoff_seconds=SPEND_LOG_QUEUE_MAX_BACKOFF_SECONDS,
        max_depth=DEFAULT_MAX_RECURSE_DEPTH,
    )
    from litellm.proxy.utils import dequeue_spend_logs

    uninstall_rust_spend_log_writer(prisma_client)
    pending: Final = await dequeue_spend_logs(prisma_client, sys.maxsize)
    _installed.append(_Installed(client=prisma_client, writer=writer))
    if pending:
        enqueue_in_rust(writer, pending, at_head=True, max_bytes=SPEND_LOG_QUEUE_MAX_BYTES)
    return writer


def uninstall_rust_spend_log_writer(prisma_client: object) -> None:
    _installed[:] = [installed for installed in _installed if installed.client is not prisma_client]
