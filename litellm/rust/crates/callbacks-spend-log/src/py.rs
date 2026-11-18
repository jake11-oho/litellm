//! Value helpers shared by the row builder: the Python dict and str operations from
//! `litellm-python-compat`, and a port of LiteLLM's `safe_json_structure`, which decides what
//! every JSON column of the row holds.

use litellm_python_compat::{BigInt, Value, repr};
use num_traits::ToPrimitive;

pub(crate) use litellm_python_compat::dict::{Dict, entries, get, get_or_none, has};
pub(crate) use litellm_python_compat::pystr::{len as char_len, slice};
pub(crate) use litellm_python_compat::truthy::{or, truthy};

use crate::RowError;

pub(crate) const NONE: Value = Value::None;

/// `value[key]` where Python requires `value` to be a dict, failing with the field's name otherwise.
pub(crate) fn dict_get<'a>(
    value: &'a Value,
    key: &str,
    field: &'static str,
) -> Result<&'a Value, RowError> {
    match value {
        Value::Dict(_) => Ok(get_or_none(value, key)),
        _ => Err(RowError::NotAMapping(field)),
    }
}

pub(crate) fn as_str(value: &Value) -> Option<&str> {
    match value {
        Value::Str(text) => Some(text),
        _ => None,
    }
}

/// `value is True`.
pub(crate) fn is_true(value: &Value) -> bool {
    matches!(value, Value::Bool(true))
}

pub(crate) fn text(value: &str) -> Value {
    Value::Str(value.to_owned())
}

/// `str(value)`.
pub(crate) fn to_str(value: &Value) -> String {
    repr::to_str(value)
}

/// `value if isinstance(value, str) else ""`.
pub(crate) fn string_or_empty(value: &Value) -> Value {
    match value {
        Value::Str(_) => value.clone(),
        _ => text(""),
    }
}

pub(crate) fn int(value: i64) -> Value {
    Value::Int(BigInt::from(value))
}

pub(crate) fn big_to_f64(value: &BigInt) -> Option<f64> {
    value.to_f64().filter(|number| number.is_finite())
}

/// Options of LiteLLM's `safe_json_structure`.
pub(crate) struct Structure {
    pub(crate) non_finite_as_null: bool,
    pub(crate) integer_limit: Option<i64>,
}

const MAX_DEPTH: usize = 100;

/// LiteLLM's `safe_json_structure`: NUL bytes stripped from every string and key, non-string keys
/// dropped, containers past a depth of 100 replaced by a marker, and optionally non-finite floats
/// nulled and integers clamped.
pub(crate) fn safe_structure(value: &Value, options: &Structure) -> Value {
    structure_at(value, 0, options)
}

fn structure_at(value: &Value, depth: usize, options: &Structure) -> Value {
    if depth > MAX_DEPTH {
        return text("MaxDepthExceeded");
    }
    match value {
        Value::Str(content) => Value::Str(content.replace('\0', "")),
        Value::Int(number) => match options.integer_limit {
            Some(limit) => Value::Int(
                number
                    .clone()
                    .clamp(BigInt::from(-limit), BigInt::from(limit)),
            ),
            None => value.clone(),
        },
        Value::Float(number) if options.non_finite_as_null && !number.is_finite() => Value::None,
        Value::None | Value::Bool(_) | Value::Float(_) => value.clone(),
        Value::Dict(items) => {
            let mut result = Dict::default();
            for (key, item) in items {
                if let Value::Str(key) = key {
                    let cleaned = key.replace('\0', "");
                    let stored = structure_at(item, depth + 1, options);
                    result.set_value(Value::Str(cleaned), stored);
                }
            }
            result.into_value()
        }
        Value::List(items) | Value::Tuple(items) | Value::Set(items) => Value::List(
            items
                .iter()
                .map(|item| structure_at(item, depth + 1, options))
                .collect(),
        ),
        other => Value::Str(repr::to_str(other).replace('\0', "")),
    }
}

/// LiteLLM's `safe_dumps`: `json.dumps(safe_json_structure(value))`.
pub(crate) fn safe_dumps(value: &Value, non_finite_as_null: bool) -> Result<String, RowError> {
    let structure = safe_structure(
        value,
        &Structure {
            non_finite_as_null,
            integer_limit: None,
        },
    );
    Ok(litellm_python_compat::json::dumps(&structure)?)
}
