//! The `LiteLLM_SpendLogs` row, and the accounting payload the spend logs ledger stores with it,
//! built in Rust for the proxy's ledger path.
//!
//! The Python glue (`litellm/proxy/spend_tracking/rust_spend_log_rows.py`) turns the logging
//! kwargs and response into plain JSON, plus the few values that need Python-only state (the
//! router registry, the cache instance, pricing for savings, provider usage normalization). This
//! crate parses that JSON into [`input::RowInput`] and applies the whole row policy:
//!
//! | Module | What it decides |
//! |---|---|
//! | [`row`] | the row's columns and the metadata JSON |
//! | [`request`] | the stored request body, response and realtime messages |
//! | [`redaction`] | message redaction when message logging is turned off |
//! | [`compaction`] | bounding the summaries inside compaction blocks |
//! | [`privacy`] | prompt storage, credential masking, truncation, guardrail and error redaction |
//! | [`text`] | prompt-leak, IP address and raw model scrubbing in error text |
//! | [`stamps`] | end user, caller session, oauth, pod routing, spillover and classifier fields |
//! | [`request_id`] | the row's request id and its cost-row suffixes |
//! | [`accounting`] | the ledger's spend, delivery id and accounting payload |
//!
//! Byte equality with Python is pinned by `tests/goldens.rs`, which replays rows recorded from
//! Python by `scripts/record_goldens.py`.

mod accounting;
pub mod callback;
mod compaction;
mod input;
mod privacy;
mod py;
mod redaction;
mod request;
mod request_id;
mod row;
mod stamps;
mod text;

use litellm_python_compat::{Value, json, number, pydantic};
use litellm_tracing::TokenRedactor;

use crate::py::get_or_none;

pub use accounting::{build_accounting_payload, finalize_ledger_row};
pub use callback::{SpendLogCallback, SpendLogParts};
pub use input::INPUT_VERSION;

const DEFAULT_MAX_STRING_LENGTH_PROMPT_IN_DB: i64 = 2048;

/// Why an input could not become a row: a version mismatch, or a field missing or of the wrong
/// type.
#[derive(Debug, thiserror::Error)]
pub enum RowError {
    #[error("spend log input version {0} is not {INPUT_VERSION}")]
    Version(String),
    #[error("{0} is missing")]
    Missing(&'static str),
    #[error("{0} is not a mapping")]
    NotAMapping(&'static str),
    #[error("{0} has an unexpected type")]
    Type(&'static str),
    #[error(transparent)]
    Json(#[from] litellm_python_compat::Error),
}

/// The proxy settings and environment the row policy reads, resolved once per settings change.
pub struct RowConfig {
    /// `store_prompts_in_spend_logs` from general settings or the environment.
    store_prompts: bool,
    /// `spend_logs_prompt_storage: latest_turn`.
    latest_turn_only: bool,
    redact_tokens: bool,
    drop_requester_ip_address: bool,
    omit_missing_session_id: bool,
    max_string_length_prompt_in_db: i64,
    disable_end_user_cost_tracking: bool,
    request_correlation_in_logs: bool,
    /// `litellm.turn_off_message_logging`.
    turn_off_message_logging: bool,
}

impl RowConfig {
    pub fn from_settings(settings: &Value) -> Result<Self, RowError> {
        input::check_version(settings)?;
        let flag = |key: &str| matches!(get_or_none(settings, key), Value::Bool(true));
        let drop_ip = get_or_none(settings, "disable_requester_ip_address_logging");
        let stores_by_setting = match get_or_none(settings, "store_prompts_in_spend_logs") {
            Value::Bool(true) => true,
            Value::Str(setting) => setting.to_lowercase() == "true",
            _ => false,
        };
        Ok(Self {
            store_prompts: stores_by_setting || flag("store_prompts_in_spend_logs_env"),
            latest_turn_only: matches!(
                get_or_none(settings, "spend_logs_prompt_storage"),
                Value::Str(storage) if storage == "latest_turn"
            ),
            redact_tokens: flag("redact_tokens_in_spend_logs"),
            drop_requester_ip_address: !matches!(drop_ip, Value::None)
                && pydantic::lax_bool(drop_ip) == Some(true),
            omit_missing_session_id: matches!(
                get_or_none(settings, "missing_session_id"),
                Value::Str(policy) if policy == "omit"
            ),
            max_string_length_prompt_in_db: match get_or_none(
                settings,
                "max_string_length_prompt_in_db",
            ) {
                Value::Str(raw) => number::int(raw)
                    .and_then(|length| i64::try_from(&length).ok())
                    .unwrap_or(DEFAULT_MAX_STRING_LENGTH_PROMPT_IN_DB),
                _ => DEFAULT_MAX_STRING_LENGTH_PROMPT_IN_DB,
            },
            disable_end_user_cost_tracking: flag("disable_end_user_cost_tracking"),
            request_correlation_in_logs: flag("request_correlation_in_logs"),
            turn_off_message_logging: flag("turn_off_message_logging"),
        })
    }
}

/// Builds spend log rows under one [`RowConfig`].
pub struct SpendLogRowBuilder {
    config: RowConfig,
    tokens: TokenRedactor,
}

impl SpendLogRowBuilder {
    pub fn new(config: RowConfig, tokens: TokenRedactor) -> Self {
        Self { config, tokens }
    }

    /// The row for one request, from the glue's input document.
    pub fn build_row(&self, input: &Value) -> Result<Value, RowError> {
        row::build(self, &input::RowInput::parse(input)?)
    }

    pub fn build_row_json(&self, input: &str) -> Result<String, RowError> {
        Ok(json::dumps(&self.build_row(&json::loads(input)?)?)?)
    }
}
