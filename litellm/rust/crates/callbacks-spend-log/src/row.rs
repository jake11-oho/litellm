//! The `LiteLLM_SpendLogs` row: `get_logging_payload` from
//! `litellm/proxy/spend_tracking/spend_tracking_utils.py`, applied to the typed [`RowInput`].
//! It resolves the row's identity columns (request id, api key hash, model, model group,
//! provider, session), the cleaned metadata JSON, token counts and timings, and delegates the
//! prompt-bearing columns to [`crate::request`].

use std::sync::LazyLock;

use litellm_python_compat::{Value, datetime::IsoDateTime, repr, truthy::truthy, unicode};
use regex::Regex;
use sha2::{Digest, Sha256};

use crate::input::{LoggingPayload, RowInput, Timestamp};
use crate::privacy::SCRUBBED_ERROR_TEXT_FIELDS;
use crate::py::{
    Dict, NONE, as_str, dict_get, entries, get, get_or_none, has, int, is_true, or, safe_dumps,
    string_or_empty, text, to_str,
};
use crate::request_id::spend_logs_id;
use crate::stamps;
use crate::text::{UNKNOWN_MODEL, raw_model_spellings, scrub_raw_model_from_error_text};
use crate::{RowError, SpendLogRowBuilder};

const MCP_SPEND_LOG_MODEL_PREFIX: &str = "MCP: ";
const MAX_SPEND_LOG_MODEL_NAME_LENGTH: usize = 256;
const BATCH_COST_REQUEST_ID_SUFFIX: &str = "_batch_cost";
const BATCH_TRACE_CALL_TYPES: [&str; 4] = [
    "create_batch",
    "acreate_batch",
    "retrieve_batch",
    "aretrieve_batch",
];
const NON_SECRET_KEY_ALIASES: [&str; 3] = [
    "litellm_proxy_master_key",
    "litellm-internal-health-check",
    "litellm-cli",
];
const CLI_SESSION_KEY_PREFIX: &str = "cli-session";
const SPEND_LOGS_METADATA_FIELDS: [&str; 61] = [
    "user_api_key_customer_org_id",
    "user_api_key_customer_org_alias",
    "user_api_key_customer_team_id",
    "user_api_key_customer_team_alias",
    "user_api_key_customer_member_id",
    "user_api_key_customer_member_alias",
    "user_api_key_customer_usage",
    "actor_agent_id",
    "target_agent_id",
    "billing_agent_id",
    "agent_execution_mode",
    "verified_human_user_id",
    "autorouter_baseline_observation",
    "additional_usage_values",
    "user_api_key",
    "user_api_key_alias",
    "user_api_key_team_id",
    "user_api_key_project_id",
    "user_api_key_project_alias",
    "user_api_key_org_id",
    "user_api_key_user_id",
    "user_api_key_user_email",
    "user_api_key_team_alias",
    "spend_logs_metadata",
    "requester_ip_address",
    "user_agent",
    "litellm_call_id",
    "applied_guardrails",
    "mcp_tool_call_metadata",
    "vector_store_request_metadata",
    "routing_decision",
    "kubernetes_pod_routing",
    "internal_call_origin",
    "billed_retry_attempt",
    "upstream_provider",
    "litellm_roi_estimator",
    "is_silent_experiment",
    "guardrail_information",
    "eval_information",
    "status",
    "proxy_server_request",
    "batch_models",
    "batch_successful_requests",
    "batch_failed_requests",
    "error_information",
    "usage_object",
    "model_map_information",
    "cold_storage_object_key",
    "litellm_overhead_time_ms",
    "attempted_retries",
    "max_retries",
    "attempted_fallbacks",
    "original_model_group",
    "cost_breakdown",
    "compression_savings",
    "autorouter_savings",
    "autorouter_savings_estimate",
    "litellm_gateway_injected_cache",
    "router_metadata",
    "azure_spillover",
    "used_client_oauth_token",
];
const STAMPED_METADATA_KEYS: [&str; 18] = [
    "router_metadata",
    "azure_spillover",
    "autorouter_savings",
    "autorouter_savings_estimate",
    "autorouter_baseline_observation",
    "used_client_oauth_token",
    "upstream_provider",
    "kubernetes_pod_routing",
    "litellm_roi_estimator",
    "is_silent_experiment",
    "billed_retry_attempt",
    "user_api_key_user_email",
    "user_api_key_customer_org_id",
    "user_api_key_customer_org_alias",
    "user_api_key_customer_team_id",
    "user_api_key_customer_team_alias",
    "user_api_key_customer_member_id",
    "user_api_key_customer_member_alias",
];
const CUSTOMER_IDENTITY_FIELDS: [&str; 6] = [
    "user_api_key_customer_org_id",
    "user_api_key_customer_org_alias",
    "user_api_key_customer_team_id",
    "user_api_key_customer_team_alias",
    "user_api_key_customer_member_id",
    "user_api_key_customer_member_alias",
];
const SPECIAL_USAGE_FIELDS: [&str; 3] = ["completion_tokens", "prompt_tokens", "total_tokens"];

static SHA256_HEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A[a-fA-F0-9]{64}\z").expect("sha256 pattern compiles"));
static HASHED_JWT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\Ahashed-jwt-[a-fA-F0-9]{64}\z").expect("hashed jwt pattern compiles")
});

fn strip_bearer(value: &str) -> &str {
    match value.get(..7) {
        Some(prefix) if prefix.eq_ignore_ascii_case("bearer ") => &value[7..],
        _ => value,
    }
}

fn hash_token(token: &str) -> String {
    Sha256::digest(token.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_non_secret_key_value(value: &str, key_alias: &Value) -> bool {
    NON_SECRET_KEY_ALIASES.contains(&value)
        || SHA256_HEX.is_match(value)
        || HASHED_JWT.is_match(value)
        || (value.starts_with(&format!("{CLI_SESSION_KEY_PREFIX}-"))
            && matches!(key_alias, Value::Str(alias) if alias == value))
}

fn redact_logged_api_key(value: &Value, already_redacted: bool, key_alias: &Value) -> Value {
    let Some(raw) = as_str(value).filter(|raw| !raw.is_empty()) else {
        return Value::None;
    };
    let stripped = strip_bearer(raw);
    if stripped.is_empty() {
        return Value::None;
    }
    if already_redacted && is_non_secret_key_value(stripped, key_alias) {
        return text(stripped);
    }
    Value::Str(hash_token(stripped))
}

fn key_already_redacted(metadata: &Value, api_key: &Value) -> bool {
    let trusted = get_or_none(metadata, "user_api_key_hash");
    match trusted {
        Value::Str(hash) => {
            is_non_secret_key_value(hash, get_or_none(metadata, "user_api_key_alias"))
                && trusted == api_key
        }
        _ => false,
    }
}

fn looks_like_model_name(model: &str) -> bool {
    let candidate = model
        .strip_prefix(MCP_SPEND_LOG_MODEL_PREFIX)
        .unwrap_or(model);
    candidate.chars().count() <= MAX_SPEND_LOG_MODEL_NAME_LENGTH
        && !candidate.chars().any(unicode::isspace)
}

fn reconstruct_model_name(
    model_name: &str,
    custom_llm_provider: &Value,
    metadata: &Value,
) -> String {
    if let Value::Str(deployment) = get_or_none(metadata, "deployment")
        && deployment.contains('/')
    {
        return deployment.clone();
    }
    if truthy(custom_llm_provider)
        && !model_name.is_empty()
        && !model_name.contains('/')
        && matches!(custom_llm_provider, Value::Str(provider) if provider == "bedrock")
    {
        return format!("bedrock/{model_name}");
    }
    model_name.to_owned()
}

fn scrub_raw_model_from_error_information(information: &Value, raw_model: &str) -> Value {
    if matches!(information, Value::None) || raw_model.is_empty() {
        return information.clone();
    }
    let Some(items) = entries(information) else {
        return information.clone();
    };
    let spellings = raw_model_spellings(raw_model);
    Value::Dict(
        items
            .iter()
            .map(|(key, value)| match (key, value) {
                (Value::Str(name), Value::Str(content))
                    if SCRUBBED_ERROR_TEXT_FIELDS.contains(&name.as_str()) =>
                {
                    (
                        key.clone(),
                        Value::Str(scrub_raw_model_from_error_text(content, &spellings)),
                    )
                }
                _ => (key.clone(), value.clone()),
            })
            .collect(),
    )
}

/// `_placeholder_stored_request_body`: when the row's model is a placeholder, the stored request
/// body's model, model group and error text must not leak the raw model either.
fn placeholder_stored_request(
    proxy_server_request: Option<&Value>,
    persisted_model_group: &Value,
    raw_model: &str,
) -> Option<Value> {
    let Some(Value::Dict(request_items)) = proxy_server_request else {
        return proxy_server_request.cloned();
    };
    let Some(body @ Value::Dict(body_items)) =
        proxy_server_request.and_then(|found| get(found, "body"))
    else {
        return proxy_server_request.cloned();
    };
    let mut placeholdered = Dict::from_entries(body_items);
    if has(body, "model") {
        placeholdered.set("model", text(UNKNOWN_MODEL));
    }
    if let Some(Value::Dict(metadata_items)) = get(body, "metadata") {
        let stored_metadata = metadata_items
            .iter()
            .map(|(key, value)| match as_str(key) {
                Some("model_group") => (key.clone(), persisted_model_group.clone()),
                Some("error_information") => (
                    key.clone(),
                    match value {
                        Value::Dict(_) => scrub_raw_model_from_error_information(value, raw_model),
                        other => other.clone(),
                    },
                ),
                _ => (key.clone(), value.clone()),
            })
            .collect();
        placeholdered.set("metadata", Value::Dict(stored_metadata));
    }
    let mut request = Dict::from_entries(request_items);
    request.set("body", placeholdered.into_value());
    Some(request.into_value())
}

fn router_metadata(
    metadata: &Value,
    requested_model: &Value,
    selected_model: &str,
    selected_provider: &Value,
    router_correlation_id: &Value,
) -> Value {
    let model_info = get_or_none(metadata, "model_info");
    if !matches!(model_info, Value::Dict(_))
        || !is_true(get_or_none(model_info, "internal_router_model"))
    {
        return Value::None;
    }
    let present = |value: &Value| {
        if truthy(value) {
            value.clone()
        } else {
            Value::None
        }
    };
    Value::Dict(vec![
        (text("requested_model"), present(requested_model)),
        (text("selected_model"), present(&text(selected_model))),
        (text("selected_provider"), present(selected_provider)),
        (text("router_correlation_id"), router_correlation_id.clone()),
    ])
}

fn stored_user_email(metadata: &Value) -> Value {
    match get_or_none(metadata, "user_api_key_user_email") {
        email @ Value::Str(content) if !content.trim_matches(unicode::isspace).is_empty() => {
            email.clone()
        }
        _ => Value::None,
    }
}

/// `request_duration_ms`: whole milliseconds between start and end, when both are naive or both
/// are aware.
fn duration_ms(start: &Timestamp, end: &Timestamp) -> Value {
    match (start.raw, end.raw) {
        (Some(IsoDateTime::Aware(start)), Some(IsoDateTime::Aware(end)))
        | (Some(IsoDateTime::Naive(start)), Some(IsoDateTime::Naive(end))) => {
            int(((end - start) as f64 / 1e6 * 1000.0) as i64)
        }
        _ => Value::None,
    }
}

/// The logging payload's attribution fields, read the way `get_logging_payload` falls back to them.
struct Attribution<'a> {
    payload: Option<&'a LoggingPayload>,
}

impl Attribution<'_> {
    fn fallback(&self, field: &str) -> Value {
        self.payload
            .map_or_else(|| text(""), |payload| string_or_empty(payload.get(field)))
    }

    fn payload_get(&self, field: &str) -> &Value {
        self.payload.map_or(&NONE, |payload| payload.get(field))
    }

    fn metadata(&self, field: &str) -> &Value {
        self.payload
            .map_or(&NONE, |payload| payload.metadata(field))
    }

    fn hidden(&self, field: &str) -> &Value {
        self.payload.map_or(&NONE, |payload| payload.hidden(field))
    }
}

/// `get_logging_payload(kwargs, response_obj, start_time, end_time)`: the whole row.
pub(crate) fn build(builder: &SpendLogRowBuilder, input: &RowInput) -> Result<Value, RowError> {
    let config = &builder.config;
    let metadata = &input.metadata;
    let kwargs = &input.kwargs;
    let litellm_params = &input.litellm_params;
    let usage = &input.usage;
    let payload = input.payload.as_ref();
    let attribution = Attribution { payload };
    let rejected_as_unknown_model = input.rejected_as_unknown_model;

    let token_count = |field: &str| match payload {
        Some(payload) => payload.field(field).cloned().unwrap_or_else(|| int(0)),
        None => int(0),
    };

    let api_key_value = get(metadata, "user_api_key")
        .cloned()
        .unwrap_or_else(|| text(""));
    let already_redacted = key_already_redacted(metadata, &api_key_value);
    let mut api_key = redact_logged_api_key(
        &api_key_value,
        already_redacted,
        get_or_none(metadata, "user_api_key_alias"),
    );
    if matches!(api_key, Value::None) {
        api_key = text("");
    }
    let mut end_user_id = stamps::end_user_id(input, config.disable_end_user_cost_tracking);
    if payload.is_some() {
        if !truthy(&api_key) {
            let from_payload = redact_logged_api_key(
                attribution.metadata("user_api_key_hash"),
                true,
                attribution.metadata("user_api_key_alias"),
            );
            api_key = if truthy(&from_payload) {
                from_payload
            } else {
                text("")
            };
        }
        if !config.disable_end_user_cost_tracking && !truthy(&end_user_id) {
            end_user_id = attribution.metadata("user_api_key_end_user_id").clone();
        }
    }
    let tags = get(metadata, "tags")
        .cloned()
        .unwrap_or_else(|| Value::List(Vec::new()));
    let mut request_tags = if matches!(tags, Value::List(_)) {
        safe_dumps(&tags, false)?
    } else {
        "[]".to_owned()
    };
    if let Some(payload) = payload
        && !matches!(payload.get("request_tags"), Value::None)
    {
        request_tags = safe_dumps(payload.get("request_tags"), false)?;
    }

    let model_info = match get(metadata, "model_info") {
        Some(info) => info.clone(),
        None => Value::Dict(Vec::new()),
    };
    let model_id_value = get(&model_info, "id").cloned().unwrap_or_else(|| text(""));
    if !matches!(model_info, Value::Dict(_)) {
        return Err(RowError::NotAMapping("metadata.model_info"));
    }
    let model_id = or(&model_id_value, &attribution.fallback("model_id")).clone();
    let model_group = or(
        &string_or_empty(get_or_none(metadata, "model_group")),
        &attribution.fallback("model_group"),
    )
    .clone();
    let api_base_value = litellm_params.api_base.clone().unwrap_or_else(|| text(""));
    let api_base = or(&api_base_value, &attribution.fallback("api_base")).clone();

    let litellm_overhead_time_ms = attribution.hidden("litellm_overhead_time_ms").clone();
    let upstream_provider = match attribution.hidden("upstream_provider") {
        provider @ Value::Str(_) => provider.clone(),
        _ => Value::None,
    };

    let fallback_provider = attribution.fallback("custom_llm_provider");
    let logged_provider = or(&kwargs.custom_llm_provider, &fallback_provider);
    let logged_provider = if truthy(logged_provider) {
        logged_provider.clone()
    } else {
        Value::None
    };
    let custom_llm_provider = or(&logged_provider, &input.model_group_provider).clone();
    let requested_model = &kwargs.model;
    let raw_model = as_str(requested_model).unwrap_or_default();
    let model_is_malformed = !matches!(requested_model, Value::None | Value::Str(_));
    let resolved_model = match attribution.payload_get("model") {
        Value::Str(model) if !model.is_empty() => model.clone(),
        _ => reconstruct_model_name(raw_model, &logged_provider, metadata),
    };
    let failed =
        matches!(get_or_none(metadata, "status"), Value::Str(status) if status == "failure");
    let failed_with_prompt_shaped_model = failed
        && !truthy(&model_id)
        && !looks_like_model_name(&resolved_model)
        && !input.is_configured_model_group;
    let model_name =
        if rejected_as_unknown_model || failed_with_prompt_shaped_model || model_is_malformed {
            UNKNOWN_MODEL.to_owned()
        } else {
            resolved_model
        };
    let model_is_placeholdered = model_name == UNKNOWN_MODEL;
    let persisted_model_group = if model_is_placeholdered
        && matches!(&model_group, Value::Str(group) if group == raw_model)
        && !looks_like_model_name(raw_model)
    {
        text("")
    } else {
        model_group.clone()
    };
    let persisted_metadata = if model_is_placeholdered {
        let mut scrubbed = Dict::from_entries(entries(metadata).unwrap_or_default());
        scrubbed.set(
            "error_information",
            scrub_raw_model_from_error_information(
                get_or_none(metadata, "error_information"),
                raw_model,
            ),
        );
        scrubbed.into_value()
    } else {
        metadata.clone()
    };
    let litellm_call_id = or(&kwargs.litellm_call_id, &litellm_params.litellm_call_id).clone();

    let silent_experiment = truthy(attribution.metadata("is_silent_experiment"));
    let guardrail_information = match payload {
        Some(payload) => payload.get("guardrail_information").clone(),
        None => get_or_none(metadata, "standard_logging_guardrail_information").clone(),
    };
    let model_map_information = match payload {
        Some(payload) => payload
            .field("model_map_information")
            .ok_or(RowError::Missing(
                "standard_logging_object.model_map_information",
            ))?
            .clone(),
        None => Value::None,
    };

    let mut clean = Dict::default();
    for field in SPEND_LOGS_METADATA_FIELDS {
        if !STAMPED_METADATA_KEYS.contains(&field) {
            clean.set(field, get_or_none(&persisted_metadata, field).clone());
        }
    }
    for field in CUSTOMER_IDENTITY_FIELDS {
        if let value @ Value::Str(content) = get_or_none(&persisted_metadata, field)
            && !content.trim_matches(unicode::isspace).is_empty()
        {
            clean.set(field, value.clone());
        }
    }
    if !matches!(upstream_provider, Value::None) {
        clean.set("upstream_provider", upstream_provider);
    }
    for field in [
        "autorouter_savings",
        "autorouter_savings_estimate",
        "autorouter_baseline_observation",
    ] {
        clean.set(field, attribution.payload_get(field).clone());
    }
    clean.set(
        "router_metadata",
        router_metadata(
            metadata,
            &persisted_model_group,
            &model_name,
            &custom_llm_provider,
            &litellm_call_id,
        ),
    );
    clean.set("azure_spillover", stamps::azure_spillover(input));
    clean.set(
        "used_client_oauth_token",
        stamps::used_client_oauth_token(litellm_params, &custom_llm_provider),
    );
    clean.set(
        "kubernetes_pod_routing",
        stamps::kubernetes_pod_routing(litellm_params),
    );
    clean.set(
        "litellm_roi_estimator",
        Value::Bool(is_true(get_or_none(
            &persisted_metadata,
            "litellm_roi_estimator",
        ))),
    );
    if silent_experiment {
        clean.set("is_silent_experiment", Value::Bool(true));
    }
    clean.set(
        "billed_retry_attempt",
        Value::Bool(is_true(get_or_none(
            &persisted_metadata,
            "billed_retry_attempt",
        ))),
    );
    clean.set(
        "user_api_key_user_email",
        stored_user_email(&persisted_metadata),
    );

    let raw_key = clean.get("user_api_key").cloned().unwrap_or(Value::None);
    let clean_key_redacted = key_already_redacted(&persisted_metadata, &raw_key);
    clean.set(
        "user_api_key",
        redact_logged_api_key(
            &raw_key,
            clean_key_redacted,
            get_or_none(&persisted_metadata, "user_api_key_alias"),
        ),
    );
    clean.set(
        "applied_guardrails",
        attribution.metadata("applied_guardrails").clone(),
    );
    for field in [
        "batch_models",
        "batch_successful_requests",
        "batch_failed_requests",
    ] {
        clean.set(field, attribution.hidden(field).clone());
    }
    clean.set(
        "mcp_tool_call_metadata",
        attribution.metadata("mcp_tool_call_metadata").clone(),
    );
    clean.set(
        "vector_store_request_metadata",
        builder.vector_store_request_metadata(
            attribution.metadata("vector_store_request_metadata"),
            metadata,
        ),
    );
    clean.set(
        "guardrail_information",
        builder.sanitize_guardrail_information(&guardrail_information, metadata),
    );
    clean.set("usage_object", attribution.metadata("usage_object").clone());
    clean.set("model_map_information", model_map_information);
    clean.set(
        "cold_storage_object_key",
        attribution.metadata("cold_storage_object_key").clone(),
    );
    clean.set("litellm_overhead_time_ms", litellm_overhead_time_ms);
    clean.set(
        "cost_breakdown",
        attribution.payload_get("cost_breakdown").clone(),
    );
    clean.set("litellm_call_id", litellm_call_id.clone());

    let usage_items = entries(usage).ok_or(RowError::NotAMapping("usage"))?;
    let mut additional_usage_values = Dict::default();
    for (key, value) in usage_items {
        if !as_str(key).is_some_and(|name| SPECIAL_USAGE_FIELDS.contains(&name)) {
            additional_usage_values.set_value(key.clone(), value.clone());
        }
    }
    let mut prompt_tokens_details = additional_usage_values
        .get("prompt_tokens_details")
        .cloned();
    if !matches!(prompt_tokens_details, Some(Value::Dict(_)))
        && let Some(usage_object @ Value::Dict(_)) = clean.get("usage_object")
    {
        prompt_tokens_details = get(usage_object, "prompt_tokens_details").cloned();
    }
    if let Some(details @ Value::Dict(_)) = prompt_tokens_details {
        if additional_usage_values
            .get("cache_read_input_tokens")
            .is_none()
            && let cached @ (Value::Int(_) | Value::Bool(_)) =
                get_or_none(&details, "cached_tokens")
            && positive(cached)
        {
            additional_usage_values.set("cache_read_input_tokens", cached.clone());
        }
        if additional_usage_values
            .get("cache_creation_input_tokens")
            .is_none()
        {
            let written = or(
                get_or_none(&details, "cache_write_tokens"),
                get_or_none(&details, "cache_creation_tokens"),
            );
            if matches!(written, Value::Int(_) | Value::Bool(_)) && positive(written) {
                additional_usage_values.set("cache_creation_input_tokens", written.clone());
            }
        }
    }
    clean.set(
        "additional_usage_values",
        additional_usage_values.into_value(),
    );
    if config.drop_requester_ip_address {
        clean.set("requester_ip_address", Value::None);
    }
    let error_information = clean
        .get("error_information")
        .cloned()
        .unwrap_or(Value::None);
    clean.set(
        "error_information",
        builder.sanitize_error_information(
            &error_information,
            rejected_as_unknown_model.then_some(&input.unknown_model_error_message),
            metadata,
        ),
    );

    let base_request_id = spend_logs_id(input);
    let request_id = if input.cache_hit == Some(true) || input.served_from_cache {
        let base = base_request_id.as_deref().unwrap_or("None");
        Some(format!("{base}_cache_hit{}", repr::float_repr(input.now)))
    } else {
        base_request_id
    };

    let mcp_namespaced_tool_name = match clean.get("mcp_tool_call_metadata") {
        Some(Value::None) | None => Value::None,
        Some(found) => dict_get(found, "namespaced_tool_name", "mcp_tool_call_metadata")?.clone(),
    };
    let agent_id = or(&kwargs.agent_id, get_or_none(metadata, "agent_id")).clone();
    let billing_agent_id = clean
        .get("billing_agent_id")
        .cloned()
        .unwrap_or(Value::None);
    let requester_ip_address = clean
        .get("requester_ip_address")
        .cloned()
        .unwrap_or(Value::None);
    let metadata_json = safe_dumps(&clean.into_value(), true)?;

    let placeholdered_request = model_is_placeholdered
        .then(|| {
            placeholder_stored_request(
                litellm_params.proxy_server_request.as_ref(),
                &persisted_model_group,
                raw_model,
            )
        })
        .flatten();
    let stored_request = if model_is_placeholdered {
        placeholdered_request.as_ref()
    } else {
        litellm_params.proxy_server_request.as_ref()
    };
    let session_id = session_id(builder, input, request_id.as_deref());
    let status = if failed { "failure" } else { "success" };
    let completion_start_time = input
        .completion_start_time
        .as_ref()
        .map_or(Value::None, |time| time.utc.clone());
    let cache_hit = match input.cache_hit {
        Some(true) => "True",
        Some(false) => "False",
        None => "None",
    };
    let field = |name: &str, value: Value| (text(name), value);
    let text_or_empty = |value: Option<&Value>| or(value.unwrap_or(&text("")), &text("")).clone();
    Ok(Value::Dict(vec![
        field(
            "request_id",
            Value::Str(request_id.unwrap_or_else(|| "None".to_owned())),
        ),
        field(
            "call_type",
            text(input.call_type.as_deref().unwrap_or_default()),
        ),
        field("api_key", Value::Str(to_str(&api_key))),
        field("cache_hit", text(cache_hit)),
        field("startTime", input.start_time.utc.clone()),
        field("endTime", input.end_time.utc.clone()),
        field("completionStartTime", completion_start_time),
        field("model", Value::Str(model_name)),
        field("user", text_or_empty(get(metadata, "user_api_key_user_id"))),
        field(
            "team_id",
            text_or_empty(get(metadata, "user_api_key_team_id")),
        ),
        field(
            "organization_id",
            or(get_or_none(metadata, "user_api_key_org_id"), &text("")).clone(),
        ),
        field("metadata", Value::Str(metadata_json)),
        field("cache_key", input.cache_key.clone()),
        field(
            "spend",
            kwargs.response_cost.clone().unwrap_or_else(|| int(0)),
        ),
        field(
            "total_tokens",
            get(usage, "total_tokens")
                .cloned()
                .unwrap_or_else(|| token_count("total_tokens")),
        ),
        field(
            "prompt_tokens",
            get(usage, "prompt_tokens")
                .cloned()
                .unwrap_or_else(|| token_count("prompt_tokens")),
        ),
        field(
            "completion_tokens",
            get(usage, "completion_tokens")
                .cloned()
                .unwrap_or_else(|| token_count("completion_tokens")),
        ),
        field("request_tags", Value::Str(request_tags)),
        field("end_user", or(&end_user_id, &text("")).clone()),
        field("api_base", api_base),
        field("model_group", persisted_model_group),
        field("model_id", model_id),
        field("mcp_namespaced_tool_name", mcp_namespaced_tool_name),
        field("agent_id", agent_id),
        field("billing_agent_id", billing_agent_id),
        field("requester_ip_address", requester_ip_address),
        field(
            "custom_llm_provider",
            or(&custom_llm_provider, &text("")).clone(),
        ),
        field("messages", Value::Str(builder.messages_json(input)?)),
        field("response", Value::Str(builder.response_json(input)?)),
        field(
            "proxy_server_request",
            Value::Str(builder.proxy_server_request_json(input, stored_request)?),
        ),
        field("session_id", session_id),
        field(
            "request_duration_ms",
            duration_ms(&input.start_time, &input.end_time),
        ),
        field("status", text(status)),
        field("litellm_call_id", litellm_call_id),
    ]))
}

fn positive(value: &Value) -> bool {
    match value {
        Value::Int(number) => *number > litellm_python_compat::BigInt::from(0),
        Value::Bool(flag) => *flag,
        _ => false,
    }
}

/// The row's `session_id`: the stamped session when missing sessions are omitted, the batch id
/// for batch calls, the caller's session under request correlation, else the trace id.
fn session_id(builder: &SpendLogRowBuilder, input: &RowInput, request_id: Option<&str>) -> Value {
    let metadata = &input.metadata;
    let omit = truthy(get_or_none(metadata, "litellm_session_id_omitted"))
        || builder.config.omit_missing_session_id;
    if omit {
        let session = get_or_none(metadata, "session_id");
        return if truthy(session) {
            Value::Str(to_str(session))
        } else {
            Value::None
        };
    }
    if let (Some(kind), Some(id)) = (input.call_type.as_deref(), request_id)
        && BATCH_TRACE_CALL_TYPES.contains(&kind)
        && !id.is_empty()
    {
        return text(id.strip_suffix(BATCH_COST_REQUEST_ID_SUFFIX).unwrap_or(id));
    }
    if builder.config.request_correlation_in_logs
        && let Some(caller) = stamps::caller_session(input)
    {
        return Value::Str(caller);
    }
    if let Some(payload) = &input.payload
        && let Some(trace_id) = payload.field("trace_id")
        && !matches!(trace_id, Value::None)
    {
        return Value::Str(to_str(trace_id));
    }
    match &input.kwargs.litellm_trace_id {
        Value::None => text(&input.fallback_session_id),
        trace_id => Value::Str(to_str(trace_id)),
    }
}
