import copy
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from unittest import mock

CRATE = Path(__file__).resolve().parents[1]
REPO = CRATE.parents[3]
GOLDENS = CRATE / "tests" / "goldens"
TEST_FILES = (
    "tests/unit/proxy/spend_tracking/test_spend_tracking_utils.py",
    "tests/unit/proxy/db/test_db_spend_update_writer.py",
    "tests/unit/accounting",
    "tests/unit/proxy/hooks/test_proxy_track_cost_callback.py",
)
FIXED_NOW = 1763337600.123456
FIXED_SESSION = "7d1c9f9e-1d61-4cf4-9a43-c1b5e5d6c0f1"

_records: dict[str, list[dict]] = {"rows": []}
_seen: set[str] = set()
_current_test = ["<collection>"]


class Unencodable(Exception):
    pass


def _jsonable(value: object, *, lenient: bool = False) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or type(value) in (bool, int, float):
        return value
    if type(value) is str:
        return value
    if lenient and isinstance(value, str):
        return str.__str__(value)
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise Unencodable(repr(value)[:200])
        return {key: _jsonable(item, lenient=lenient) for key, item in value.items()}
    if type(value) is list or (lenient and type(value) is tuple):
        return [_jsonable(item, lenient=lenient) for item in value]
    if lenient and isinstance(value, Mapping):
        return {str(key): _jsonable(item, lenient=lenient) for key, item in value.items()}
    raise Unencodable(f"{type(value).__name__}: {repr(value)[:200]}")


def _row_text(row: Mapping[str, object]) -> str:
    return json.dumps(_jsonable(dict(row), lenient=True))


def _add(kind: str, record: dict) -> None:
    digest = hashlib.sha256(json.dumps({**record, "case": None}, sort_keys=True).encode()).hexdigest()
    if (kind, digest) in _seen:
        return
    _seen.add((kind, digest))
    _records[kind].append({"case": _current_test[0], **record})


def _install() -> None:
    from litellm.proxy.spend_tracking import rust_spend_log_rows, spend_tracking_utils

    original_payload = spend_tracking_utils.get_logging_payload

    def recording_payload(kwargs, response_obj, start_time, end_time, llm_router=None):
        record: dict = {}
        try:
            config = rust_spend_log_rows.spend_log_row_config()
            row_input = rust_spend_log_rows.spend_log_row_input(
                copy.copy(kwargs) if kwargs is not None else {},
                response_obj,
                start_time,
                end_time,
                llm_router,
                FIXED_NOW,
                FIXED_SESSION,
            )
            record = {"config": config, "input": row_input}
        except Exception as error:  # noqa: BLE001
            record = {"input_error": f"{type(error).__name__}: {error}"}
        fixed_uuid = mock.Mock(return_value=FIXED_SESSION)
        with (
            mock.patch("time.time", return_value=FIXED_NOW),
            mock.patch.object(spend_tracking_utils_uuid(), "uuid4", fixed_uuid),
            mock.patch.object(spend_tracking_utils, "uses_spend_logs_ledger", return_value=False),
        ):
            result = original_payload(kwargs, response_obj, start_time, end_time, llm_router)
        if "input" in record:
            try:
                record["output"] = _row_text(result)
            except Unencodable as error:
                record = {"output_error": str(error)}
        _add("rows", record)
        return result

    spend_tracking_utils.get_logging_payload = recording_payload


def spend_tracking_utils_uuid():
    from litellm import _uuid

    return _uuid.uuid


def pytest_configure(config):
    _install()


def pytest_runtest_setup(item):
    _current_test[0] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    from litellm.proxy.spend_tracking.rust_spend_log_rows import token_word_ranges

    GOLDENS.mkdir(parents=True, exist_ok=True)
    for kind, records in _records.items():
        with open(GOLDENS / f"{kind}.jsonl", "w", encoding="ascii") as out:
            for line in sorted(json.dumps(record) for record in records):
                out.write(line + "\n")
        sys.stderr.write(f"{kind}: {len(records)} records\n")
    (GOLDENS / "token_word_ranges.json").write_text(json.dumps([list(pair) for pair in token_word_ranges()]) + "\n")


if __name__ == "__main__":
    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(
        pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "-p",
                "no:xdist",
                "-p",
                "record_goldens",
                str(Path(__file__).resolve().parent / "golden_scenarios.py"),
                *(str(REPO / p) for p in TEST_FILES),
            ]
        )
    )
