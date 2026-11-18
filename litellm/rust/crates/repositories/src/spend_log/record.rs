//! The request log and delivery types the spend log repository writes: one `LiteLLM_SpendLogs`
//! row parsed from the log the proxy queued, and the `LiteLLM_SpendLogDelivery` row that
//! carries its accounting work to the ledger worker.

use chrono::{DateTime, NaiveDateTime};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};

/// The `LiteLLM_SpendLogs` columns a request log stores as `jsonb`, which callers
/// hand over as JSON text.
pub const SPEND_LOG_JSON_COLUMNS: [&str; 5] = [
    "metadata",
    "request_tags",
    "messages",
    "response",
    "proxy_server_request",
];

fn empty_text() -> Option<String> {
    Some(String::new())
}

fn empty_object() -> Option<Value> {
    Some(json!({}))
}

fn empty_array() -> Option<Value> {
    Some(json!([]))
}

/// Reads a request log timestamp as the proxy writes it: RFC 3339 with an offset (made
/// UTC), or a naive `T`- or space-separated time taken as UTC.
pub fn parse_spend_log_timestamp(text: &str) -> Option<NaiveDateTime> {
    DateTime::parse_from_rfc3339(text)
        .map(|timestamp| timestamp.naive_utc())
        .or_else(|_| {
            DateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f%:z").map(|t| t.naive_utc())
        })
        .or_else(|_| NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f"))
        .or_else(|_| NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f"))
        .ok()
}

fn timestamp<'de, D: Deserializer<'de>>(deserializer: D) -> Result<NaiveDateTime, D::Error> {
    let text = String::deserialize(deserializer)?;
    parse_spend_log_timestamp(&text)
        .ok_or_else(|| serde::de::Error::custom(format!("invalid timestamp {text:?}")))
}

fn optional_timestamp<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<NaiveDateTime>, D::Error> {
    Option::<String>::deserialize(deserializer)?
        .map(|text| {
            parse_spend_log_timestamp(&text)
                .ok_or_else(|| serde::de::Error::custom(format!("invalid timestamp {text:?}")))
        })
        .transpose()
}

fn timestamp_text<S: Serializer>(value: &NaiveDateTime, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(&value.format("%Y-%m-%dT%H:%M:%S%.f"))
}

fn optional_timestamp_text<S: Serializer>(
    value: &Option<NaiveDateTime>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => timestamp_text(value, serializer),
        None => serializer.serialize_none(),
    }
}

/// One `LiteLLM_SpendLogs` row, parsed from the request log the proxy queued.
///
/// A column the log leaves out takes the schema default, as a Prisma insert would; an
/// explicit `null` stays `NULL`; a column the table does not have refuses the row.
/// `created_at` and `updated_at` default to the commit time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpendLogRecord {
    pub request_id: String,
    pub call_type: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub spend: f64,
    #[serde(default)]
    pub total_tokens: i32,
    #[serde(default)]
    pub prompt_tokens: i32,
    #[serde(default)]
    pub completion_tokens: i32,
    #[serde(
        rename = "startTime",
        deserialize_with = "timestamp",
        serialize_with = "timestamp_text"
    )]
    pub start_time: NaiveDateTime,
    #[serde(
        rename = "endTime",
        deserialize_with = "timestamp",
        serialize_with = "timestamp_text"
    )]
    pub end_time: NaiveDateTime,
    #[serde(default)]
    pub request_duration_ms: Option<i32>,
    #[serde(
        rename = "completionStartTime",
        default,
        deserialize_with = "optional_timestamp",
        serialize_with = "optional_timestamp_text"
    )]
    pub completion_start_time: Option<NaiveDateTime>,
    #[serde(default)]
    pub model: String,
    #[serde(default = "empty_text")]
    pub model_id: Option<String>,
    #[serde(default = "empty_text")]
    pub model_group: Option<String>,
    #[serde(default = "empty_text")]
    pub custom_llm_provider: Option<String>,
    #[serde(default = "empty_text")]
    pub api_base: Option<String>,
    #[serde(default = "empty_text")]
    pub user: Option<String>,
    #[serde(default = "empty_object")]
    pub metadata: Option<Value>,
    #[serde(default = "empty_text")]
    pub cache_hit: Option<String>,
    #[serde(default = "empty_text")]
    pub cache_key: Option<String>,
    #[serde(default = "empty_array")]
    pub request_tags: Option<Value>,
    #[serde(default)]
    pub team_id: Option<String>,
    #[serde(default)]
    pub organization_id: Option<String>,
    #[serde(default)]
    pub end_user: Option<String>,
    #[serde(default)]
    pub requester_ip_address: Option<String>,
    #[serde(default = "empty_object")]
    pub messages: Option<Value>,
    #[serde(default = "empty_object")]
    pub response: Option<Value>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub mcp_namespaced_tool_name: Option<String>,
    #[serde(default)]
    pub billing_agent_id: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default = "empty_object")]
    pub proxy_server_request: Option<Value>,
    #[serde(default)]
    pub litellm_call_id: Option<String>,
    #[serde(
        default,
        deserialize_with = "optional_timestamp",
        serialize_with = "optional_timestamp_text"
    )]
    pub created_at: Option<NaiveDateTime>,
    #[serde(
        default,
        deserialize_with = "optional_timestamp",
        serialize_with = "optional_timestamp_text"
    )]
    pub updated_at: Option<NaiveDateTime>,
}

impl SpendLogRecord {
    /// Parses one request log. A log the table cannot hold fails here, before it reaches
    /// Postgres, so the caller can drop it alone and keep the rest of its batch.
    pub fn from_columns(columns: Map<String, Value>) -> Result<Self, serde_json::Error> {
        serde_json::from_value(Value::Object(columns))
    }

    /// The row with `created_at` and `updated_at` set to `now` where the log left them out.
    pub(super) fn stamped(&self, now: NaiveDateTime) -> Self {
        Self {
            created_at: self.created_at.or(Some(now)),
            updated_at: self.updated_at.or(Some(now)),
            ..self.clone()
        }
    }
}

/// The accounting work one request leaves for the ledger worker. `accounting_payload` is
/// the worker's own JSON contract and is stored as given; `is_batch` marks a batch or
/// video job that settles once under its request id.
#[derive(Clone, Debug, PartialEq)]
pub struct SpendLogDelivery {
    pub delivery_id: String,
    pub request_id: String,
    pub accounting_payload: Map<String, Value>,
    pub is_batch: bool,
}

/// A delivery with the request log it brings, if the request is logged at all.
#[derive(Clone, Debug, PartialEq)]
pub struct SpendLogLedgerEntry {
    pub delivery: SpendLogDelivery,
    pub row: Option<SpendLogRecord>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::{SpendLogRecord, parse_spend_log_timestamp};

    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            _ => Map::new(),
        }
    }

    #[rstest]
    #[case::rfc3339_offset("2026-11-02T14:00:00+02:00", "2026-11-02T12:00:00.000")]
    #[case::python_str_of_an_aware_datetime(
        "2026-11-02 12:00:00.250000+00:00",
        "2026-11-02T12:00:00.250"
    )]
    #[case::naive_iso("2026-11-02T12:00:00.5", "2026-11-02T12:00:00.500")]
    #[case::naive_space("2026-11-02 12:00:00", "2026-11-02T12:00:00.000")]
    fn reads_the_timestamps_the_proxy_writes(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(
            parse_spend_log_timestamp(text)
                .map(|parsed| parsed.format("%Y-%m-%dT%H:%M:%S%.3f").to_string()),
            Some(expected.to_owned())
        );
    }

    #[rstest]
    #[case::unknown_column(json!({"request_id": "bad", "call_type": "x", "startTime": "2026-11-02 12:00:00", "endTime": "2026-11-02 12:00:00", "unexpected": 1}))]
    #[case::missing_start(json!({"request_id": "bad", "call_type": "x", "endTime": "2026-11-02 12:00:00"}))]
    #[case::not_a_number(json!({"request_id": "bad", "call_type": "x", "startTime": "2026-11-02 12:00:00", "endTime": "2026-11-02 12:00:00", "spend": "abc"}))]
    #[case::not_a_time(json!({"request_id": "bad", "call_type": "x", "startTime": "soon", "endTime": "2026-11-02 12:00:00"}))]
    #[case::null_in_a_required_column(json!({"request_id": "bad", "call_type": "x", "startTime": "2026-11-02 12:00:00", "endTime": "2026-11-02 12:00:00", "model": null}))]
    fn a_log_the_table_cannot_hold_is_refused(#[case] columns: Value) {
        assert!(SpendLogRecord::from_columns(object(columns)).is_err());
    }

    #[rstest]
    #[case::missing_columns_take_defaults(
        json!({"request_id": "r", "call_type": "x", "startTime": "2026-11-02 12:00:00", "endTime": "2026-11-02 12:00:00"}),
        json!({"api_key": "", "model_id": "", "metadata": {}, "request_tags": [], "team_id": null, "created_at": null})
    )]
    #[case::explicit_nulls_stay_null(
        json!({"request_id": "r", "call_type": "x", "startTime": "2026-11-02 12:00:00", "endTime": "2026-11-02 12:00:00", "model_id": null, "metadata": null}),
        json!({"api_key": "", "model_id": null, "metadata": null, "request_tags": [], "team_id": null, "created_at": null})
    )]
    fn parses_a_log_into_its_columns(#[case] columns: Value, #[case] expected: Value) {
        let stored = serde_json::to_value(
            SpendLogRecord::from_columns(object(columns)).expect("record parses"),
        )
        .expect("serializes");

        assert_eq!(
            object(expected)
                .into_iter()
                .filter(|(column, value)| stored.get(column) != Some(value))
                .collect::<Vec<_>>(),
            Vec::new()
        );
    }
}
