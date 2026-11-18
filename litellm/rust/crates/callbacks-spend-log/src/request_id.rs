//! The row's request id: `get_spend_logs_id`, which takes the response's id and marks the rows that
//! carry a job's or a background response's cost, so each cost lands under one id.

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use litellm_python_compat::Value;

use crate::accounting::{finite_cost, guardrail_information_cost};
use crate::input::RowInput;
use crate::py::get_or_none;

const BATCH_COST_REQUEST_ID_SUFFIX: &str = "_batch_cost";
const BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX: &str = "_background_cost";
const JOB_POLL_GUARDRAIL_COST_REQUEST_ID_SUFFIX: &str = "_guardrail_cost";
const VIDEO_COST_REQUEST_ID_SUFFIX: &str = "_video_cost";
const VIDEO_ID_PREFIX: &str = "video_";
const VIDEO_STATUS_CALL_TYPES: [&str; 2] = ["avideo_status", "video_status"];
const INTERNAL_CALL_ORIGIN_METADATA_KEY: &str = "internal_call_origin";
const BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN: &str = "background_response_cost_poll";
const NON_INFERENCE_CALL_TYPES: [&str; 30] = [
    "get_responses",
    "aget_responses",
    "delete_responses",
    "adelete_responses",
    "cancel_responses",
    "acancel_responses",
    "list_input_items",
    "alist_input_items",
    "vector_store_create",
    "avector_store_create",
    "vector_store_retrieve",
    "avector_store_retrieve",
    "vector_store_list",
    "avector_store_list",
    "vector_store_update",
    "avector_store_update",
    "vector_store_delete",
    "avector_store_delete",
    "vector_store_file_create",
    "avector_store_file_create",
    "vector_store_file_list",
    "avector_store_file_list",
    "vector_store_file_retrieve",
    "avector_store_file_retrieve",
    "vector_store_file_content",
    "avector_store_file_content",
    "vector_store_file_update",
    "avector_store_file_update",
    "vector_store_file_delete",
    "avector_store_file_delete",
];

fn is_video_status(call_type: &str) -> bool {
    VIDEO_STATUS_CALL_TYPES.contains(&call_type)
}

fn is_non_inference(call_type: &str) -> bool {
    NON_INFERENCE_CALL_TYPES.contains(&call_type)
}

/// `charged_poll_cost(kwargs)`: the poll's whole cost, guardrails included.
fn charged_poll_cost(input: &RowInput) -> Value {
    match input.payload_get("response_cost") {
        Value::None => input.kwargs.response_cost.clone().unwrap_or(Value::None),
        logged => logged.clone(),
    }
}

/// `job_cost_of_poll(response_cost, payload)`: what the polled job itself cost.
fn job_cost_of_poll(input: &RowInput, response_cost: &Value) -> f64 {
    finite_cost(response_cost).unwrap_or(0.0)
        - guardrail_information_cost(input.payload_get("guardrail_information")).value()
}

/// `canonical_video_id(video_id)`: one spelling per LiteLLM video id, padded or not.
fn canonical_video_id(video_id: &str) -> String {
    let Some(encoded) = video_id.strip_prefix(VIDEO_ID_PREFIX) else {
        return video_id.to_owned();
    };
    let alphabet: String = encoded
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || "+/".contains(*character))
        .collect();
    let padded = format!("{alphabet}{}", "=".repeat((4 - alphabet.len() % 4) % 4));
    match STANDARD
        .decode(padded)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
    {
        Some(payload) if payload.contains(';') => {
            format!("{VIDEO_ID_PREFIX}{}", STANDARD.encode(payload))
        }
        _ => video_id.to_owned(),
    }
}

/// `settles_background_response(call_type, metadata, kwargs)`.
fn settles_background_response(input: &RowInput, call_type: &str) -> bool {
    input.kwargs.settles_background_response
        || (is_non_inference(call_type)
            && matches!(
                get_or_none(&input.metadata, INTERNAL_CALL_ORIGIN_METADATA_KEY),
                Value::Str(origin) if origin == BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN
            ))
}

/// `get_spend_logs_id(call_type or "acompletion", response, kwargs, metadata)`.
pub(crate) fn spend_logs_id(input: &RowInput) -> Option<String> {
    let call_type = input
        .call_type
        .as_deref()
        .filter(|kind| !kind.is_empty())
        .unwrap_or("acompletion");
    let payload_id = match input.payload_get("id") {
        Value::Str(id) if !id.is_empty() => Some(id.clone()),
        _ => None,
    };
    let call_id = match &input.kwargs.litellm_call_id {
        Value::Str(id) if !id.is_empty() => Some(id.clone()),
        _ => None,
    };
    let resolved = input
        .response_id
        .clone()
        .or(payload_id)
        .or_else(|| call_id.clone())?;
    let response_cost = charged_poll_cost(input);
    if is_video_status(call_type) {
        let job_cost = job_cost_of_poll(input, &response_cost);
        let completed = matches!(&input.kwargs.polled_video_status, Value::Str(status) if status == "completed");
        if completed && job_cost > 0.0 {
            let video_id = input
                .litellm_params
                .video_id
                .as_deref()
                .unwrap_or(&resolved);
            return Some(format!(
                "{}{VIDEO_COST_REQUEST_ID_SUFFIX}",
                canonical_video_id(video_id)
            ));
        }
        if job_cost <= 0.0
            && 0.0 < finite_cost(&response_cost).unwrap_or(0.0)
            && let Some(call_id) = call_id
        {
            return Some(format!(
                "{call_id}{JOB_POLL_GUARDRAIL_COST_REQUEST_ID_SUFFIX}"
            ));
        }
    }
    let suffix = if call_type == "aretrieve_batch" {
        BATCH_COST_REQUEST_ID_SUFFIX
    } else if settles_background_response(input, call_type) {
        BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX
    } else {
        ""
    };
    Some(format!("{resolved}{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::canonical_video_id;

    // Expected values are litellm.types.videos.utils.canonical_video_id
    #[test]
    fn video_ids_get_their_padding_back() {
        let padded = format!(
            "video_{}",
            base64::Engine::encode(&super::STANDARD, "openai;id1")
        );
        let unpadded = padded.trim_end_matches('=').to_owned();
        assert_eq!(canonical_video_id(&unpadded), padded);
        assert_eq!(canonical_video_id("video_plain"), "video_plain");
        assert_eq!(canonical_video_id("vid_123"), "vid_123");
    }
}
