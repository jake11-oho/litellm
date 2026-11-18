//! Rust repositories for the auth objects the proxy reads from Postgres when its auth cache misses, and for the
//! spend logs and ledger deliveries it writes.
//!
//! Each repository is a folder that owns its queries, the shaping of their rows into what its caller consumes,
//! and any memory tier private to that object. Queries are private to their folder, so every read goes through a
//! repository and its `Database::run`. The shared auth cache in front of them stays with the host. See README.md
//! for the layout.

macro_rules_attribute::attribute_alias! {
    #[apply(db_row)] =
        #[derive(Clone, Debug, PartialEq, sqlx::FromRow)];
}

mod database;
mod jwt_key_mapping;
mod key;
mod model_access_group;
mod raw_json;
mod spend_log;
#[cfg(test)]
mod test_support;
mod utc_timestamp;

pub use database::{Database, Side};
pub use jwt_key_mapping::{JwtKeyMappingRepository, MappedClaim};
pub use key::{FetchedKey, KeyRepository};
pub use model_access_group::{ModelAccessGroupBudget, ModelAccessGroupRepository};
use raw_json::RawJson;
pub use spend_log::{
    SPEND_LOG_JSON_COLUMNS, SpendLogDelivery, SpendLogLedgerEntry, SpendLogRecord,
    SpendLogRepository, parse_spend_log_timestamp,
};
