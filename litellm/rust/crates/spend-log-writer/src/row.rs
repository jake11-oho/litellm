//! A queued request log as the proxy handed it over: an ordered JSON object whose size
//! is measured the way Python's `json.dumps` writes it.

use litellm_python_compat::{
    Value,
    json::{dumps, loads},
};

pub const ACCOUNTING_PAYLOAD: &str = "accounting_payload";
pub const ACCOUNTING_LOG_DISABLED: &str = "accounting_log_disabled";
pub const ACCOUNTING_DELIVERY_ID: &str = "accounting_delivery_id";

const ACCOUNTING_ONLY_KEYS: [&str; 4] = [
    "request_id",
    "call_type",
    ACCOUNTING_PAYLOAD,
    ACCOUNTING_DELIVERY_ID,
];

#[derive(Debug, thiserror::Error)]
pub enum RowError {
    #[error("spend log row is not valid JSON")]
    Decode(#[source] litellm_python_compat::Error),
    #[error("spend log row is not a JSON object")]
    NotAnObject,
}

/// One queued request log, kept in the key order the proxy built it, with its size as
/// `json.dumps` writes it so the queue budget matches the Python queue's.
#[derive(Clone, Debug, PartialEq)]
pub struct SpendLogRow {
    entries: Vec<(String, Value)>,
    bytes: usize,
}

fn measured_bytes(entries: &[(String, Value)]) -> usize {
    dumps(&dict(entries)).map_or(0, |text| text.len())
}

fn dict(entries: &[(String, Value)]) -> Value {
    Value::Dict(
        entries
            .iter()
            .map(|(key, value)| (Value::Str(key.clone()), value.clone()))
            .collect(),
    )
}

impl SpendLogRow {
    pub fn new(entries: Vec<(String, Value)>) -> Self {
        let bytes = measured_bytes(&entries);
        Self { entries, bytes }
    }

    pub fn parse(text: &str) -> Result<Self, RowError> {
        let Value::Dict(pairs) = loads(text).map_err(RowError::Decode)? else {
            return Err(RowError::NotAnObject);
        };
        let entries = pairs
            .into_iter()
            .map(|(key, value)| match key {
                Value::Str(key) => Ok((key, value)),
                _ => Err(RowError::NotAnObject),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(entries))
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn entries(&self) -> &[(String, Value)] {
        &self.entries
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    pub fn carries_a_charge(&self) -> bool {
        self.get(ACCOUNTING_PAYLOAD).is_some()
    }

    pub fn log_disabled(&self) -> bool {
        matches!(self.get(ACCOUNTING_LOG_DISABLED), Some(Value::Bool(true)))
    }

    /// The row reduced to what charges the request, marked as not logged.
    pub fn accounting_only(&self) -> Self {
        let mut entries: Vec<(String, Value)> = self
            .entries
            .iter()
            .filter(|(key, _)| ACCOUNTING_ONLY_KEYS.contains(&key.as_str()))
            .cloned()
            .collect();
        entries.push((ACCOUNTING_LOG_DISABLED.to_owned(), Value::Bool(true)));
        Self::new(entries)
    }

    pub fn with(&self, key: &str, value: Value) -> Self {
        let mut entries = self.entries.clone();
        match entries.iter_mut().find(|(name, _)| name == key) {
            Some(entry) => entry.1 = value,
            None => entries.push((key.to_owned(), value)),
        }
        Self::new(entries)
    }

    pub fn to_json(&self) -> String {
        dumps(&dict(&self.entries)).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::SpendLogRow;

    #[rstest]
    #[case::python_separators(r#"{"a":1,"b":[1,2]}"#, r#"{"a": 1, "b": [1, 2]}"#)]
    #[case::ensure_ascii("{\"m\": \"caf\u{e9}\"}", "{\"m\": \"caf\\u00e9\"}")]
    #[case::python_float_repr(r#"{"spend": 3.5e-05}"#, r#"{"spend": 3.5e-05}"#)]
    #[case::nan_like_json_loads(r#"{"spend": NaN}"#, r#"{"spend": NaN}"#)]
    fn measures_what_json_dumps_writes(#[case] text: &str, #[case] dumped: &str) {
        let row = SpendLogRow::parse(text).expect("row parses");

        assert_eq!(row.to_json(), dumped);
        assert_eq!(row.bytes(), dumped.len());
    }

    #[rstest]
    #[case::charged(
        r#"{"request_id": "r", "call_type": "c", "messages": "[]", "accounting_payload": {"spend": 1.0}, "accounting_delivery_id": "d", "accounting_log_disabled": false}"#,
        r#"{"request_id": "r", "call_type": "c", "accounting_payload": {"spend": 1.0}, "accounting_delivery_id": "d", "accounting_log_disabled": true}"#
    )]
    fn accounting_only_keeps_the_charge_in_row_order(#[case] text: &str, #[case] expected: &str) {
        let row = SpendLogRow::parse(text).expect("row parses");

        assert_eq!(row.accounting_only().to_json(), expected);
        assert!(row.accounting_only().log_disabled());
    }

    #[rstest]
    #[case::array("[1]")]
    #[case::lone_surrogate(r#"{"m": "\ud800"}"#)]
    fn refuses_what_it_cannot_hold(#[case] text: &str) {
        assert!(SpendLogRow::parse(text).is_err());
    }
}
