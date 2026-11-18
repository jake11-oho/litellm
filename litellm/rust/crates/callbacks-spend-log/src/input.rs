//! The typed inputs of the builder, parsed once from the JSON that the Python glue
//! (`litellm/proxy/spend_tracking/rust_spend_log_rows.py`) sends.
//!
//! Scalars and flags become Rust types here. Documents the row stores as JSON, and whose shape the
//! caller or provider defines (request metadata, the request body, the response, usage counts,
//! guardrail records), stay [`Value`]s, since the builder copies them through rather than reading
//! a fixed schema out of them.

use litellm_python_compat::{
    Value,
    datetime::{IsoDateTime, fromisoformat},
    flags,
};

use crate::RowError;
use crate::py::{NONE, get, get_or_none};

/// The version both sides of the boundary agree on.
pub const INPUT_VERSION: i64 = 2;

pub(crate) fn check_version(value: &Value) -> Result<(), RowError> {
    match get_or_none(value, "version") {
        Value::Int(version) if *version == INPUT_VERSION.into() => Ok(()),
        other => Err(RowError::Version(litellm_python_compat::repr::repr(other))),
    }
}

fn required<'a>(value: &'a Value, key: &'static str) -> Result<&'a Value, RowError> {
    get(value, key).ok_or(RowError::Missing(key))
}

fn owned(value: &Value, key: &str) -> Value {
    get_or_none(value, key).clone()
}

fn non_empty_text(value: &Value) -> Option<String> {
    match value {
        Value::Str(text) if !text.is_empty() => Some(text.clone()),
        _ => None,
    }
}

fn mapping(value: &Value) -> Option<Value> {
    matches!(value, Value::Dict(_)).then(|| value.clone())
}

/// One timestamp of the request, as the glue reads it from a Python `datetime`.
pub(crate) struct Timestamp {
    /// `value.astimezone(timezone.utc).isoformat()`, which the row stores.
    pub(crate) utc: Value,
    /// `value.isoformat()` parsed back, for the request duration.
    pub(crate) raw: Option<IsoDateTime>,
}

impl Timestamp {
    fn parse(value: &Value) -> Self {
        Self {
            utc: owned(value, "utc"),
            raw: match get_or_none(value, "raw") {
                Value::Str(raw) => fromisoformat(raw),
                _ => None,
            },
        }
    }
}

/// A metadata bucket of `litellm_params` (`metadata` or `litellm_metadata`), projected to the
/// keys the row reads.
pub(crate) enum RequestMetadata {
    Absent,
    NotAMapping,
    Mapping(Value),
}

impl RequestMetadata {
    fn parse(params: &Value, key: &str) -> Self {
        match get(params, key) {
            None => Self::Absent,
            Some(found @ Value::Dict(_)) => Self::Mapping(found.clone()),
            Some(_) => Self::NotAMapping,
        }
    }

    pub(crate) fn mapping(&self) -> Option<&Value> {
        match self {
            Self::Mapping(value) => Some(value),
            _ => None,
        }
    }
}

/// The request's `litellm_params`, projected.
pub(crate) struct LitellmParams {
    pub(crate) api_base: Option<Value>,
    pub(crate) litellm_call_id: Value,
    /// Absent, `None`, or the proxy's record of the client request, whose `body` the row stores.
    pub(crate) proxy_server_request: Option<Value>,
    pub(crate) video_id: Option<String>,
    pub(crate) litellm_session_id: Value,
    pub(crate) user_api_key_end_user_id: Value,
    pub(crate) metadata: RequestMetadata,
    pub(crate) litellm_metadata: RequestMetadata,
}

impl LitellmParams {
    fn parse(value: &Value) -> Self {
        Self {
            api_base: get(value, "api_base").cloned(),
            litellm_call_id: owned(value, "litellm_call_id"),
            proxy_server_request: get(value, "proxy_server_request").cloned(),
            video_id: non_empty_text(get_or_none(value, "video_id")),
            litellm_session_id: owned(value, "litellm_session_id"),
            user_api_key_end_user_id: owned(value, "user_api_key_end_user_id"),
            metadata: RequestMetadata::parse(value, "metadata"),
            litellm_metadata: RequestMetadata::parse(value, "litellm_metadata"),
        }
    }

    /// The buckets that are mappings, `metadata` first.
    pub(crate) fn metadata_buckets(&self) -> impl Iterator<Item = &Value> {
        [self.metadata.mapping(), self.litellm_metadata.mapping()]
            .into_iter()
            .flatten()
    }
}

/// The logging kwargs the row reads.
pub(crate) struct Kwargs {
    pub(crate) model: Value,
    pub(crate) custom_llm_provider: Value,
    pub(crate) response_cost: Option<Value>,
    pub(crate) agent_id: Value,
    pub(crate) litellm_call_id: Value,
    pub(crate) litellm_trace_id: Value,
    pub(crate) realtime_tools: Value,
    pub(crate) realtime_tool_calls: Value,
    pub(crate) polled_video_status: Value,
    /// Whether the kwarg marking a poll that settles a background response's cost is the
    /// proxy's own marker (the glue checks its identity, so a forged request field never counts).
    pub(crate) settles_background_response: bool,
    /// The Azure spillover headers among the provider's response headers.
    pub(crate) response_headers: Option<Value>,
    /// `turn_off_message_logging` from the request's dynamic callback params, when it is set.
    pub(crate) turn_off_message_logging: Option<bool>,
}

impl Kwargs {
    fn parse(value: &Value) -> Self {
        Self {
            model: owned(value, "model"),
            custom_llm_provider: owned(value, "custom_llm_provider"),
            response_cost: get(value, "response_cost").cloned(),
            agent_id: owned(value, "agent_id"),
            litellm_call_id: owned(value, "litellm_call_id"),
            litellm_trace_id: owned(value, "litellm_trace_id"),
            realtime_tools: owned(value, "realtime_tools"),
            realtime_tool_calls: owned(value, "realtime_tool_calls"),
            polled_video_status: owned(value, "polled_video_status"),
            settles_background_response: matches!(
                get_or_none(value, "settles_background_response"),
                Value::Bool(true)
            ),
            response_headers: mapping(get_or_none(value, "response_headers")),
            turn_off_message_logging: match get_or_none(value, "turn_off_message_logging") {
                Value::Bool(flag) => Some(*flag),
                Value::Str(setting) => flags::str_to_bool(setting),
                _ => None,
            },
        }
    }
}

/// The request's `StandardLoggingPayload`, projected. Its fields are read the way
/// `get_logging_payload` reads them, so a missing field and a `None` one stay distinct.
pub(crate) struct LoggingPayload {
    pub(crate) fields: Value,
    pub(crate) metadata: Value,
    pub(crate) hidden_params: Value,
}

impl LoggingPayload {
    fn parse(value: &Value) -> Result<Self, RowError> {
        Ok(Self {
            fields: value.clone(),
            metadata: get(value, "metadata")
                .cloned()
                .ok_or(RowError::Missing("standard_logging_object.metadata"))?,
            hidden_params: owned(value, "hidden_params"),
        })
    }

    /// `payload.get(field)`.
    pub(crate) fn get(&self, field: &str) -> &Value {
        get_or_none(&self.fields, field)
    }

    /// `payload[field]` when present.
    pub(crate) fn field(&self, field: &str) -> Option<&Value> {
        get(&self.fields, field)
    }

    /// `payload["metadata"].get(field)`.
    pub(crate) fn metadata(&self, field: &str) -> &Value {
        get_or_none(&self.metadata, field)
    }

    /// `payload.get("hidden_params", {}).get(field)`.
    pub(crate) fn hidden(&self, field: &str) -> &Value {
        get(&self.hidden_params, field).unwrap_or(&NONE)
    }
}

/// Everything one spend log row is built from.
pub(crate) struct RowInput {
    pub(crate) call_type: Option<String>,
    /// The `cache_hit` kwarg, `None` when it is not a bool.
    pub(crate) cache_hit: Option<bool>,
    pub(crate) served_from_cache: bool,
    pub(crate) rejected_as_unknown_model: bool,
    pub(crate) unknown_model_error_message: Value,
    /// The response's `id`, when it is a non-empty string.
    pub(crate) response_id: Option<String>,
    /// `time.time()` when the row is built, for the cache-hit id suffix.
    pub(crate) now: f64,
    /// A fresh `uuid4`, the session of a request that carries none.
    pub(crate) fallback_session_id: String,
    pub(crate) start_time: Timestamp,
    pub(crate) end_time: Timestamp,
    pub(crate) completion_start_time: Option<Timestamp>,
    /// Token and unit counts as the response reported them, in their reported order.
    pub(crate) usage: Value,
    /// The request's merged LiteLLM metadata, projected to the keys the row reads.
    pub(crate) metadata: Value,
    pub(crate) litellm_params: LitellmParams,
    pub(crate) kwargs: Kwargs,
    pub(crate) payload: Option<LoggingPayload>,
    /// The cache key Python's cache computed for the request.
    pub(crate) cache_key: Value,
    /// The provider the router registers for the model group, looked up in Python.
    pub(crate) model_group_provider: Value,
    /// Whether the router knows the model group, looked up in Python for failed requests.
    pub(crate) is_configured_model_group: bool,
}

impl RowInput {
    pub(crate) fn parse(value: &Value) -> Result<Self, RowError> {
        check_version(value)?;
        let usage = required(value, "usage")?;
        if !matches!(usage, Value::Dict(_)) {
            return Err(RowError::NotAMapping("usage"));
        }
        let metadata = required(value, "metadata")?;
        if !matches!(metadata, Value::Dict(_)) {
            return Err(RowError::NotAMapping("metadata"));
        }
        Ok(Self {
            call_type: match get_or_none(value, "call_type") {
                Value::Str(kind) => Some(kind.clone()),
                _ => None,
            },
            cache_hit: match get_or_none(value, "cache_hit") {
                Value::Bool(flag) => Some(*flag),
                _ => None,
            },
            served_from_cache: matches!(get_or_none(value, "served_from_cache"), Value::Bool(true)),
            rejected_as_unknown_model: matches!(
                get_or_none(value, "rejected_as_unknown_model"),
                Value::Bool(true)
            ),
            unknown_model_error_message: owned(value, "unknown_model_error_message"),
            response_id: non_empty_text(get_or_none(value, "response_id")),
            now: match get_or_none(value, "now") {
                Value::Float(now) => *now,
                _ => return Err(RowError::Type("now")),
            },
            fallback_session_id: match get_or_none(value, "fallback_session_id") {
                Value::Str(session) => session.clone(),
                _ => return Err(RowError::Type("fallback_session_id")),
            },
            start_time: Timestamp::parse(required(value, "start_time")?),
            end_time: Timestamp::parse(required(value, "end_time")?),
            completion_start_time: match get_or_none(value, "completion_start_time") {
                Value::None => None,
                found => Some(Timestamp::parse(found)),
            },
            usage: usage.clone(),
            metadata: metadata.clone(),
            litellm_params: LitellmParams::parse(required(value, "litellm_params")?),
            kwargs: Kwargs::parse(required(value, "kwargs")?),
            payload: match get_or_none(value, "standard_logging_object") {
                Value::None => None,
                found => Some(LoggingPayload::parse(found)?),
            },
            cache_key: owned(value, "cache_key"),
            model_group_provider: owned(value, "model_group_provider"),
            is_configured_model_group: matches!(
                get_or_none(value, "is_configured_model_group"),
                Value::Bool(true)
            ),
        })
    }

    /// `payload.get(field)`, or `None` without a payload.
    pub(crate) fn payload_get(&self, field: &str) -> &Value {
        self.payload
            .as_ref()
            .map_or(&NONE, |payload| payload.get(field))
    }
}
