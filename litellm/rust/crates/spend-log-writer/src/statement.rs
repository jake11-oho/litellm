//! Turning queued rows into the ledger transaction's input: metadata retention, the
//! proxy's JSON clean-up, and parsing each row into a typed `SpendLogRecord`.

use litellm_python_compat::{
    Value,
    json::{dumps, loads},
    repr::to_str,
    safe_json::{jsonify_object, safe_json},
};
use litellm_repositories::{
    SPEND_LOG_JSON_COLUMNS, SpendLogDelivery, SpendLogLedgerEntry, SpendLogRecord,
};
use serde_json::Map;

use crate::{
    Error, SpendLogRow,
    row::{ACCOUNTING_DELIVERY_ID, ACCOUNTING_LOG_DISABLED, ACCOUNTING_PAYLOAD},
};

/// Which metadata fields a logged row keeps (`spend_logs_metadata` settings).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Retention {
    pub include: Option<Vec<String>>,
    pub exclude: Option<Vec<String>>,
    pub always_kept: Vec<String>,
}

impl Retention {
    pub fn keeps(&self, name: &str) -> bool {
        if self.always_kept.iter().any(|kept| kept == name) {
            return true;
        }
        match (&self.include, &self.exclude) {
            (Some(include), _) => include.iter().any(|field| field == name),
            (None, exclude) => !exclude.iter().flatten().any(|field| field == name),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("spend log metadata is not a JSON object")]
pub struct InvalidMetadata;

/// The row with its metadata cut down to the retained fields.
pub fn retain_metadata(
    row: &SpendLogRow,
    retention: Option<&Retention>,
    max_depth: usize,
) -> Result<SpendLogRow, InvalidMetadata> {
    let (Some(retention), Some(Value::Str(metadata))) = (retention, row.get("metadata")) else {
        return Ok(row.clone());
    };
    let Ok(Value::Dict(entries)) = loads(metadata) else {
        return Err(InvalidMetadata);
    };
    let retained = entries
        .into_iter()
        .map(|(key, value)| match key {
            Value::Str(key) => Ok((key, value)),
            _ => Err(InvalidMetadata),
        })
        .filter(|entry| match entry {
            Ok((key, _)) => retention.keeps(key),
            Err(_) => true,
        })
        .map(|entry| entry.map(|(key, value)| (Value::Str(key), value)))
        .collect::<Result<Vec<_>, _>>()?;
    let text = safe_json(&Value::Dict(retained), 0, max_depth).to_string();
    let reparsed = loads(&text).map_err(|_| InvalidMetadata)?;
    Ok(row.with(
        "metadata",
        Value::Str(dumps(&reparsed).map_err(|_| InvalidMetadata)?),
    ))
}

pub fn jsonified(row: &SpendLogRow) -> SpendLogRow {
    SpendLogRow::new(jsonify_object(row.entries()))
}

fn column_value(column: &str, value: &Value, max_depth: usize) -> Result<serde_json::Value, Error> {
    match value {
        Value::Str(text) if SPEND_LOG_JSON_COLUMNS.contains(&column) => loads(text)
            .map(|parsed| safe_json(&parsed, 2, max_depth))
            .map_err(|_| Error::InvalidJsonColumn {
                column: column.to_owned(),
            }),
        value => Ok(safe_json(value, 2, max_depth)),
    }
}

fn spend_log_record(row: &SpendLogRow, max_depth: usize) -> Result<SpendLogRecord, Error> {
    let columns = row
        .entries()
        .iter()
        .filter(|(key, _)| {
            ![
                ACCOUNTING_PAYLOAD,
                ACCOUNTING_LOG_DISABLED,
                ACCOUNTING_DELIVERY_ID,
            ]
            .contains(&key.as_str())
        })
        .map(|(key, value)| Ok((key.clone(), column_value(key, value, max_depth)?)))
        .collect::<Result<Map<_, _>, Error>>()?;
    SpendLogRecord::from_columns(columns).map_err(Error::RejectedRow)
}

fn accounting_payload(value: &Value) -> Result<Value, Error> {
    let payload = match value {
        Value::Str(text) => loads(text).map_err(|_| Error::InvalidDelivery)?,
        value => value.clone(),
    };
    match &payload {
        Value::Dict(entries) if entries.iter().all(|(key, _)| matches!(key, Value::Str(_))) => {
            Ok(payload)
        }
        _ => Err(Error::InvalidDelivery),
    }
}

/// The id a delivery settles under: the row's delivery id, or its request id for a
/// batch or video job that settles once.
pub fn delivery_id(row: &SpendLogRow) -> String {
    match row
        .get(ACCOUNTING_DELIVERY_ID)
        .or_else(|| row.get("request_id"))
    {
        Some(value) => to_str(value),
        None => "None".to_owned(),
    }
}

/// One statement's rows split the way the ledger writes them: logs with no charge,
/// and deliveries with the logs they bring.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LedgerRequest {
    pub legacy: Vec<SpendLogRecord>,
    pub ledger: Vec<SpendLogLedgerEntry>,
}

/// Parses each row at the boundary into the typed records the ledger writes. A row
/// the table cannot hold fails the statement as a rejected row, so the flusher
/// isolates it.
pub fn ledger_request(rows: &[SpendLogRow], max_depth: usize) -> Result<LedgerRequest, Error> {
    let mut request = LedgerRequest::default();
    for row in rows {
        let Some(payload) = row.get(ACCOUNTING_PAYLOAD) else {
            request.legacy.push(spend_log_record(row, max_depth)?);
            continue;
        };
        let serde_json::Value::Object(accounting_payload) =
            safe_json(&accounting_payload(payload)?, 2, max_depth)
        else {
            return Err(Error::InvalidDelivery);
        };
        request.ledger.push(SpendLogLedgerEntry {
            delivery: SpendLogDelivery {
                delivery_id: delivery_id(row),
                request_id: row
                    .get("request_id")
                    .map(to_str)
                    .ok_or(Error::MissingRequestId)?,
                accounting_payload,
                is_batch: row.get(ACCOUNTING_DELIVERY_ID).is_none(),
            },
            row: if row.log_disabled() {
                None
            } else {
                Some(spend_log_record(row, max_depth)?)
            },
        });
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use litellm_db::ErrorKind;
    use rstest::rstest;
    use serde_json::json;

    use super::{Retention, jsonified, ledger_request, retain_metadata};
    use crate::SpendLogRow;

    fn row(text: &str) -> SpendLogRow {
        SpendLogRow::parse(text).expect("row parses")
    }

    #[rstest]
    #[case::include(Retention { include: Some(vec!["a".into()]), exclude: None, always_kept: vec!["k".into()] }, r#"{"a": 1, "k": 3}"#)]
    #[case::exclude(Retention { include: None, exclude: Some(vec!["a".into()]), always_kept: vec![] }, r#"{"b": 2, "k": 3}"#)]
    fn retains_configured_metadata_fields(#[case] retention: Retention, #[case] metadata: &str) {
        let retained = retain_metadata(
            &row(r#"{"request_id": "r", "metadata": "{\"a\": 1, \"b\": 2, \"k\": 3}"}"#),
            Some(&retention),
            100,
        )
        .expect("metadata is an object");

        assert_eq!(
            retained.get("metadata"),
            Some(&litellm_python_compat::Value::Str(metadata.into()))
        );
    }

    #[rstest]
    #[case::not_json(r#"{"metadata": "nope"}"#)]
    #[case::not_an_object(r#"{"metadata": "[1]"}"#)]
    fn invalid_metadata_fails_retention(#[case] text: &str) {
        assert!(retain_metadata(&row(text), Some(&Retention::default()), 100).is_err());
    }

    #[rstest]
    #[case::ledger_row(
        r#"{"request_id": "r", "call_type": "c", "startTime": "2026-11-02 12:00:00+00:00", "endTime": "2026-11-02 12:00:01+00:00", "accounting_delivery_id": "d", "accounting_payload": "{\"spend\": NaN}", "accounting_log_disabled": false, "metadata": "{\"a\": \"x\\u0000y\"}"}"#,
        json!({"legacy": [], "ledger": [["d", "r", {"spend": null}, false, ["r", {"a": "xy"}]]]})
    )]
    #[case::batch_job_without_log(
        r#"{"request_id": "job", "accounting_payload": {"spend": 2}, "accounting_log_disabled": true}"#,
        json!({"legacy": [], "ledger": [["job", "job", {"spend": 2}, true, null]]})
    )]
    #[case::legacy_row(
        r#"{"request_id": "old", "call_type": "c", "startTime": "2026-11-02 12:00:00", "endTime": "2026-11-02 12:00:00", "request_tags": "[\"t\"]"}"#,
        json!({"legacy": [["old", ["t"]]], "ledger": []})
    )]
    fn builds_the_ledger_request(#[case] text: &str, #[case] expected: serde_json::Value) {
        let request = ledger_request(&[jsonified(&row(text))], 100).expect("request builds");
        let ledger: Vec<serde_json::Value> = request
            .ledger
            .iter()
            .map(|entry| {
                json!([
                    entry.delivery.delivery_id,
                    entry.delivery.request_id,
                    entry.delivery.accounting_payload,
                    entry.delivery.is_batch,
                    entry
                        .row
                        .as_ref()
                        .map(|row| json!([row.request_id, row.metadata])),
                ])
            })
            .collect();
        let legacy: Vec<serde_json::Value> = request
            .legacy
            .iter()
            .map(|row| json!([row.request_id, row.request_tags]))
            .collect();

        assert_eq!(json!({"legacy": legacy, "ledger": ledger}), expected);
    }

    #[rstest]
    #[case::invalid_json_column(
        r#"{"request_id": "r", "accounting_payload": {}, "messages": "not json"}"#,
        ErrorKind::DataRejected
    )]
    #[case::missing_required_column(
        r#"{"request_id": "r", "accounting_payload": {}, "messages": "[]"}"#,
        ErrorKind::DataRejected
    )]
    #[case::payload_not_an_object(
        r#"{"request_id": "r", "accounting_payload": "[1]"}"#,
        ErrorKind::Other
    )]
    fn refuses_rows_it_cannot_encode(#[case] text: &str, #[case] kind: ErrorKind) {
        let error = ledger_request(&[row(text)], 100).expect_err("row is refused");

        assert_eq!(error.kind(), kind);
    }
}
