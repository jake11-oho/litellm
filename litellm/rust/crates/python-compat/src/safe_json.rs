//! LiteLLM's sanitizers for values it writes to Postgres as JSON.
//!
//! | Function | Python |
//! |---|---|
//! | [`safe_json`] | the value `litellm_core_utils.safe_json_dumps.safe_dumps(value, non_finite_as_null=True)` encodes |
//! | [`jsonify_object`] | `PrismaClient.jsonify_object(row)` |
//!
//! `safe_dumps` turns NaN and infinities into `null`, strips NUL characters Postgres
//! cannot store, drops keys that are not strings, renders other objects with `str()`,
//! and replaces anything nested deeper than `max_depth` with `"MaxDepthExceeded"`.

use serde_json::Number;

use crate::{Value, json::dumps, repr::to_str};

/// The marker `safe_dumps` writes in place of a value nested past its depth limit.
pub const MAX_DEPTH_EXCEEDED: &str = "MaxDepthExceeded";

/// The text `jsonify_object` stores when a dict cannot be dumped.
pub const FAILED_TO_SERIALIZE_JSON: &str = "failed-to-serialize-json";

fn strip_nul(text: &str) -> String {
    text.replace('\0', "")
}

/// The JSON `safe_dumps(value, non_finite_as_null=True)` writes for `value`, which sits
/// `depth` levels below the value `safe_dumps` was called with.
pub fn safe_json(value: &Value, depth: usize, max_depth: usize) -> serde_json::Value {
    if depth > max_depth {
        return serde_json::Value::String(MAX_DEPTH_EXCEEDED.to_owned());
    }
    match value {
        Value::None => serde_json::Value::Null,
        Value::Bool(value) => serde_json::Value::Bool(*value),
        Value::Int(value) => {
            let text = value.to_string();
            text.parse::<i64>()
                .map(Number::from)
                .or_else(|_| text.parse::<u64>().map(Number::from))
                .ok()
                .or_else(|| text.parse::<f64>().ok().and_then(Number::from_f64))
                .map_or(serde_json::Value::Null, serde_json::Value::Number)
        }
        Value::Float(value) => {
            Number::from_f64(*value).map_or(serde_json::Value::Null, serde_json::Value::Number)
        }
        Value::Str(text) => serde_json::Value::String(strip_nul(text)),
        Value::List(values) | Value::Tuple(values) | Value::Set(values) => {
            serde_json::Value::Array(
                values
                    .iter()
                    .map(|value| safe_json(value, depth + 1, max_depth))
                    .collect(),
            )
        }
        Value::Dict(entries) => serde_json::Value::Object(
            entries
                .iter()
                .filter_map(|(key, value)| match key {
                    Value::Str(key) => {
                        Some((strip_nul(key), safe_json(value, depth + 1, max_depth)))
                    }
                    _ => None,
                })
                .collect(),
        ),
        other => serde_json::Value::String(strip_nul(&to_str(other))),
    }
}

/// `jsonify_object(row)`: every top-level dict becomes its `json.dumps` text, and every
/// other value is kept as it is.
pub fn jsonify_object(entries: &[(String, Value)]) -> Vec<(String, Value)> {
    entries
        .iter()
        .map(|(key, value)| match value {
            Value::Dict(_) => (
                key.clone(),
                Value::Str(dumps(value).unwrap_or_else(|_| FAILED_TO_SERIALIZE_JSON.to_owned())),
            ),
            _ => (key.clone(), value.clone()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::{jsonify_object, safe_json};
    use crate::{Value, json::loads};

    #[rstest]
    #[case::non_finite_as_null(r#"{"spend": NaN, "cap": Infinity}"#, 100, json!({"spend": null, "cap": null}))]
    #[case::nul_stripped(r#"{"a\u0000": "x\u0000y"}"#, 100, json!({"a": "xy"}))]
    #[case::depth_capped(r#"{"a": {"b": [1]}}"#, 1, json!({"a": {"b": "MaxDepthExceeded"}}))]
    #[case::integers_stay_integers(r#"{"n": 12345678901234}"#, 100, json!({"n": 12345678901234_i64}))]
    fn writes_what_safe_dumps_writes(
        #[case] text: &str,
        #[case] max_depth: usize,
        #[case] expected: serde_json::Value,
    ) {
        assert_eq!(
            safe_json(&loads(text).expect("parses"), 0, max_depth),
            expected
        );
    }

    #[rstest]
    #[case::non_string_keys_dropped(Value::Dict(vec![(Value::Int(1.into()), Value::None), (Value::Str("k".into()), Value::Bool(true))]), json!({"k": true}))]
    fn drops_keys_json_cannot_hold(#[case] value: Value, #[case] expected: serde_json::Value) {
        assert_eq!(safe_json(&value, 0, 100), expected);
    }

    #[rstest]
    #[case::dict_becomes_text(
        r#"{"payload": {"spend": 1.0}}"#,
        r#"{"payload": "{\"spend\": 1.0}"}"#
    )]
    #[case::list_stays(r#"{"messages": [1]}"#, r#"{"messages": [1]}"#)]
    fn jsonify_dumps_top_level_dicts(#[case] text: &str, #[case] expected: &str) {
        let Value::Dict(pairs) = loads(text).expect("parses") else {
            panic!("not a dict");
        };
        let entries: Vec<(String, Value)> = pairs
            .into_iter()
            .filter_map(|(key, value)| match key {
                Value::Str(key) => Some((key, value)),
                _ => None,
            })
            .collect();
        let jsonified = jsonify_object(&entries)
            .into_iter()
            .map(|(key, value)| (Value::Str(key), value))
            .collect();

        assert_eq!(
            crate::json::dumps(&Value::Dict(jsonified)).expect("dumps"),
            expected
        );
    }
}
