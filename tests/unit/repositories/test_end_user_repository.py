from pathlib import Path
from typing import Final

from litellm_proxy_extras.request_log_indexes import END_USER_RESTRICTED_PREDICATE

_RESTRICTED_IDS_SQL: Final = (
    Path(__file__).parents[3] / "litellm/rust/crates/repositories/src/end_user/queries/find_restricted_end_user_ids.sql"
)


def test_rust_restricted_ids_query_filters_by_the_partial_index_predicate() -> None:
    assert f"WHERE {END_USER_RESTRICTED_PREDICATE}\n" in _RESTRICTED_IDS_SQL.read_text()
