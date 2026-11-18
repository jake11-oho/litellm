//! What the row may keep of prompts and credentials: the prompt storage decision, credential
//! masking and truncation of stored request and response bodies, token redaction, and the
//! redaction of guardrail records, vector store results and error text when prompts are not
//! stored in full.

use litellm_python_compat::{Value, flags};

use crate::SpendLogRowBuilder;
use crate::compaction::{compaction_summary, is_compaction_item, with_compaction_summary};
use crate::py::{Dict, as_str, char_len, entries, get_or_none, has, slice, text, truthy};
use crate::text::{masked_ip_text, redact_prompt_leaks_in_error_string};

pub(crate) const REDACTED: &str = "REDACTED_BY_LITELM";
pub(crate) const TRUNCATED_FIELD: &str = "litellm_truncated";
pub(crate) const TRUNCATION_NOTE: &str = "Truncation is a DB storage safeguard. Full, untruncated data is logged to logging callbacks (OTEL, Datadog, etc.). To increase the truncation limit, set `MAX_STRING_LENGTH_PROMPT_IN_DB` in your env.";
const MINIMUM_TOKEN_LENGTH: usize = 15;
const MAX_STRING_LENGTH_COMPACTION_SUMMARY_IN_DB: i64 = 64 * 1024;
const TENANT_PROMPT_STORAGE_FIELDS: [&str; 2] =
    ["user_api_key_metadata", "user_api_key_team_metadata"];
const SENSITIVE_PATTERNS: [&str; 15] = [
    "password",
    "secret",
    "key",
    "token",
    "auth",
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "access",
    "private",
    "certificate",
    "fingerprint",
    "tenancy",
    "apikey",
];
const TOOL_INPUT_BLOCK_TYPES: [&str; 3] = ["tool_use", "server_tool_use", "mcp_tool_use"];
pub(crate) const TOOL_OUTPUT_BLOCK_TYPES: [&str; 3] =
    ["tool_result", "mcp_tool_result", "function_call_output"];
const PROMPT_CARRYING_GUARDRAIL_FIELDS: [&str; 4] = [
    "match_details",
    "classification",
    "guardrail_response",
    "guardrail_request",
];
const NUMERIC_COMPRESSION_STAT_KEYS: [&str; 4] = [
    "tokens_before",
    "tokens_after",
    "tokens_saved",
    "compression_ratio",
];
pub(crate) const SCRUBBED_ERROR_TEXT_FIELDS: [&str; 2] = ["error_message", "traceback"];

fn disables_prompt_storage(tenant_metadata: &Value) -> bool {
    match get_or_none(tenant_metadata, "store_prompts_in_spend_logs") {
        Value::Bool(false) => true,
        Value::Str(setting) => flags::str_to_bool(setting) == Some(false),
        _ => false,
    }
}

impl SpendLogRowBuilder {
    pub(crate) fn stores_prompts(&self, metadata: &Value) -> bool {
        if TENANT_PROMPT_STORAGE_FIELDS
            .iter()
            .any(|field| disables_prompt_storage(get_or_none(metadata, field)))
        {
            return false;
        }
        self.config.store_prompts
    }

    pub(crate) fn stores_full_prompts(&self, metadata: &Value) -> bool {
        self.stores_prompts(metadata) && !self.config.latest_turn_only
    }

    pub(crate) fn without_tokens(&self, value: &str) -> String {
        if !self.config.redact_tokens {
            return value.to_owned();
        }
        let stripped = value.replace('\0', "");
        if char_len(&stripped) < MINIMUM_TOKEN_LENGTH {
            return stripped;
        }
        self.tokens.redact(&stripped, REDACTED).unwrap_or(stripped)
    }

    pub(crate) fn truncated_for_db(&self, value: &str) -> String {
        truncated(value, self.config.max_string_length_prompt_in_db)
    }

    /// `_compaction_summary_for_db(summary)`.
    pub(crate) fn compaction_summary_for_db(&self, summary: &str) -> String {
        truncated(
            &self.without_tokens(summary),
            self.config
                .max_string_length_prompt_in_db
                .max(MAX_STRING_LENGTH_COMPACTION_SUMMARY_IN_DB),
        )
    }

    /// `_bounded_compaction_summary(parent, key, value, keep_compaction_blocks)`.
    fn bounded_compaction_summary(
        &self,
        parent: &Value,
        key: &str,
        value: &Value,
    ) -> Option<String> {
        let Value::Str(content) = value else {
            return None;
        };
        if !is_compaction_item(parent) {
            return None;
        }
        match key {
            "content" => Some(self.compaction_summary_for_db(content)),
            "encrypted_content" => compaction_summary(parent).map(|summary| {
                with_compaction_summary(content, &self.compaction_summary_for_db(&summary))
            }),
            _ => None,
        }
    }

    /// `_sanitize_request_body_for_spend_logs_payload(body, mask_credentials=...,
    /// keep_compaction_blocks=...)`: drops `secret_fields`, masks credential-named string values
    /// outside tool payloads, redacts tokens and truncates every string.
    pub(crate) fn sanitize_body(
        &self,
        body: &[(Value, Value)],
        mask_credentials: bool,
        keep_compaction_blocks: bool,
    ) -> Value {
        let parent = Value::Dict(body.to_vec());
        let mut result = Dict::default();
        for (key, value) in body {
            if matches!(key, Value::Str(name) if name == "secret_fields") {
                continue;
            }
            let name = as_str(key).unwrap_or_default();
            let stored =
                if mask_credentials && matches!(value, Value::Str(_)) && is_sensitive_key(name) {
                    text(REDACTED)
                } else if let Some(bounded) = keep_compaction_blocks
                    .then(|| self.bounded_compaction_summary(&parent, name, value))
                    .flatten()
                {
                    Value::Str(bounded)
                } else {
                    self.sanitize_value(
                        value,
                        mask_credentials && !is_spend_log_content(body, name),
                        keep_compaction_blocks,
                    )
                };
            result.set_value(key.clone(), stored);
        }
        result.into_value()
    }

    fn sanitize_value(
        &self,
        value: &Value,
        mask_credentials: bool,
        keep_compaction_blocks: bool,
    ) -> Value {
        match value {
            Value::Dict(items) => {
                self.sanitize_body(items, mask_credentials, keep_compaction_blocks)
            }
            Value::List(items) => Value::List(
                items
                    .iter()
                    .map(|item| self.sanitize_value(item, mask_credentials, keep_compaction_blocks))
                    .collect(),
            ),
            Value::Str(content) => Value::Str(self.truncated_for_db(&self.without_tokens(content))),
            other => other.clone(),
        }
    }

    pub(crate) fn sanitize_guardrail_information(
        &self,
        information: &Value,
        metadata: &Value,
    ) -> Value {
        if matches!(information, Value::None) || self.stores_full_prompts(metadata) {
            return information.clone();
        }
        let listed = match information {
            Value::Dict(_) => vec![information.clone()],
            Value::List(items) | Value::Tuple(items) => items.clone(),
            Value::Str(_) => Vec::new(),
            other => return other.clone(),
        };
        Value::List(
            listed
                .iter()
                .filter_map(|entry| entries(entry).map(redact_guardrail_entry))
                .collect(),
        )
    }

    pub(crate) fn vector_store_request_metadata(
        &self,
        requests: &Value,
        metadata: &Value,
    ) -> Value {
        if self.stores_full_prompts(metadata) {
            return requests.clone();
        }
        match requests {
            Value::None => Value::None,
            Value::List(items) => {
                Value::List(items.iter().map(redacted_vector_store_request).collect())
            }
            other => other.clone(),
        }
    }

    pub(crate) fn sanitize_error_information(
        &self,
        information: &Value,
        unknown_model_message: Option<&Value>,
        metadata: &Value,
    ) -> Value {
        let Some(items) = entries(information) else {
            return information.clone();
        };
        let mut sanitized = Dict::from_entries(items);
        if let Some(message) = unknown_model_message {
            sanitized.set("error_message", message.clone());
        }
        if !self.stores_full_prompts(metadata) {
            for field in SCRUBBED_ERROR_TEXT_FIELDS {
                if let Some(Value::Str(content)) = sanitized.get(field) {
                    let redacted = redact_prompt_leaks_in_error_string(content);
                    sanitized.set(field, Value::Str(redacted));
                }
            }
        }
        let sanitized = self.sanitize_body(&dict_entries(sanitized.into_value()), true, false);
        if self.config.drop_requester_ip_address {
            error_information_without_ip_addresses(&sanitized)
        } else {
            sanitized
        }
    }
}

fn dict_entries(value: Value) -> Vec<(Value, Value)> {
    match value {
        Value::Dict(items) => items,
        _ => Vec::new(),
    }
}

pub(crate) fn truncated(value: &str, max_length: i64) -> String {
    let length = char_len(value) as i64;
    if length <= max_length {
        return value.to_owned();
    }
    let start_chars = (max_length as f64 * 0.35) as i64;
    let end_chars = (max_length as f64 * 0.65) as i64;
    let tail = end_chars.min(max_length - start_chars);
    format!(
        "{}... ({TRUNCATED_FIELD} skipped {} chars. {TRUNCATION_NOTE}) ...{}",
        slice(value, None, Some(start_chars)),
        length - start_chars - end_chars,
        slice(value, Some(-tail), None),
    )
}

fn is_sensitive_key(key: &str) -> bool {
    let lowered = key.to_lowercase().replace('-', "_");
    let segments: Vec<&str> = lowered.split('_').collect();
    if segments.contains(&"cost") {
        return false;
    }
    SENSITIVE_PATTERNS
        .iter()
        .any(|pattern| segments.contains(pattern))
}

fn is_spend_log_content(parent: &[(Value, Value)], key: &str) -> bool {
    let parent_value = Value::Dict(parent.to_vec());
    let block_type = as_str(get_or_none(&parent_value, "type"));
    key == "arguments"
        || key == "logprobs"
        || (key == "input" && block_type.is_some_and(|kind| TOOL_INPUT_BLOCK_TYPES.contains(&kind)))
        || ((key == "content" || key == "output")
            && (block_type.is_some_and(|kind| TOOL_OUTPUT_BLOCK_TYPES.contains(&kind))
                || matches!(get_or_none(&parent_value, "role"), Value::Str(role) if role == "tool")))
}

fn is_number(value: &Value) -> bool {
    matches!(value, Value::Int(_) | Value::Float(_))
}

fn redact_guardrail_entry(entry: &[(Value, Value)]) -> Value {
    let entry_value = Value::Dict(entry.to_vec());
    let preserved: Option<Vec<(Value, Value)>> =
        entries(get_or_none(&entry_value, "guardrail_response"))
            .map(|response| {
                response
                    .iter()
                    .filter(|(key, value)| {
                        as_str(key)
                            .is_some_and(|name| NUMERIC_COMPRESSION_STAT_KEYS.contains(&name))
                            && is_number(value)
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .filter(|stats| !stats.is_empty());
    let mut redacted = Dict::from_entries(entry);
    for field in PROMPT_CARRYING_GUARDRAIL_FIELDS {
        if has(&entry_value, field) {
            redacted.set(field, text(REDACTED));
        }
    }
    if let Some(stats) = preserved {
        redacted.set("guardrail_response", Value::Dict(stats));
    }
    redacted.into_value()
}

fn redacted_vector_store_request(request: &Value) -> Value {
    let Some(items) = entries(request) else {
        return request.clone();
    };
    let found = get_or_none(request, "vector_store_search_response");
    let response = if truthy(found) {
        found.clone()
    } else {
        Value::Dict(Vec::new())
    };
    let data = sequence(get_or_none(&response, "data"))
        .iter()
        .map(redacted_vector_store_result)
        .collect();
    let mut stored_response = Dict::from_entries(entries(&response).unwrap_or_default());
    stored_response.set("data", Value::List(data));
    let mut stored = Dict::from_entries(items);
    stored.set("vector_store_search_response", stored_response.into_value());
    stored.into_value()
}

fn sequence(value: &Value) -> Vec<Value> {
    match value {
        Value::List(items) | Value::Tuple(items) if !items.is_empty() => items.clone(),
        _ => Vec::new(),
    }
}

fn redacted_vector_store_result(result: &Value) -> Value {
    let contents = sequence(get_or_none(result, "content"))
        .iter()
        .map(|content| {
            if has(content, "text") {
                let mut stored = Dict::from_entries(entries(content).unwrap_or_default());
                stored.set("text", text(REDACTED));
                stored.into_value()
            } else {
                content.clone()
            }
        })
        .collect();
    let mut stored = Dict::from_entries(entries(result).unwrap_or_default());
    stored.set("content", Value::List(contents));
    stored.into_value()
}

pub(crate) fn error_information_without_ip_addresses(information: &Value) -> Value {
    let Some(items) = entries(information) else {
        return information.clone();
    };
    Value::Dict(
        items
            .iter()
            .map(|(key, value)| match (key, value) {
                (Value::Str(name), Value::Str(content))
                    if SCRUBBED_ERROR_TEXT_FIELDS.contains(&name.as_str()) =>
                {
                    (key.clone(), Value::Str(masked_ip_text(content)))
                }
                _ => (key.clone(), value.clone()),
            })
            .collect(),
    )
}
