use std::path::PathBuf;

use litellm_callbacks_spend_log::{
    RowConfig, RowError, SpendLogRowBuilder, build_accounting_payload, finalize_ledger_row,
};
use litellm_python_compat::{Value, json};
use litellm_tracing::TokenRedactor;
use rstest::rstest;

const MINIMUM_CUSTOM_KEY_LENGTH: usize = 16;

fn goldens(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn field<'a>(record: &'a Value, key: &str) -> Option<&'a Value> {
    match record {
        Value::Dict(entries) => entries
            .iter()
            .find(|(name, _)| matches!(name, Value::Str(name) if name == key))
            .map(|(_, value)| value),
        _ => None,
    }
}

fn word_ranges() -> Vec<(u32, u32)> {
    match json::loads(&goldens("token_word_ranges.json")).expect("word ranges parse") {
        Value::List(pairs) => pairs
            .iter()
            .map(|pair| match pair {
                Value::List(bounds) => match bounds.as_slice() {
                    [Value::Int(start), Value::Int(end)] => (
                        u32::try_from(start).expect("range start fits"),
                        u32::try_from(end).expect("range end fits"),
                    ),
                    _ => panic!("malformed word range"),
                },
                _ => panic!("malformed word range"),
            })
            .collect(),
        _ => panic!("word ranges are a list"),
    }
}

fn build_row(record: &Value) -> Result<Value, RowError> {
    let config = RowConfig::from_settings(field(record, "config").expect("config"))?;
    let builder = SpendLogRowBuilder::new(
        config,
        TokenRedactor::new(MINIMUM_CUSTOM_KEY_LENGTH, word_ranges()),
    );
    builder.build_row(field(record, "input").expect("input"))
}

fn build_accounting(record: &Value) -> Result<Value, RowError> {
    let input = field(record, "input").expect("input");
    build_accounting_payload(field(input, "row").expect("row"), input)
}

fn build_ledger_row(record: &Value) -> Result<Value, RowError> {
    let input = field(record, "input").expect("input");
    finalize_ledger_row(field(input, "row").expect("row"), input)
}

#[rstest]
#[case::rows("rows.jsonl", build_row)]
#[case::accounting("accounting.jsonl", build_accounting)]
#[case::ledger_rows("ledger_rows.jsonl", build_ledger_row)]
fn rust_matches_every_python_golden(
    #[case] file: &str,
    #[case] build: fn(&Value) -> Result<Value, RowError>,
) {
    let lines = goldens(file);
    let records: Vec<Value> = lines
        .lines()
        .map(|line| json::loads(line).expect("golden parses"))
        .filter(|record| field(record, "output").is_some())
        .collect();
    assert!(!records.is_empty(), "{file} has no goldens");
    let mismatches: Vec<String> = records
        .iter()
        .filter_map(|record| {
            let Some(Value::Str(expected)) = field(record, "output") else {
                return Some("output is not text".to_owned());
            };
            let case = field(record, "case")
                .map(litellm_python_compat::repr::to_str)
                .unwrap_or_default();
            match build(record).map(|row| json::dumps(&row)) {
                Ok(Ok(actual)) if &actual == expected => None,
                Ok(Ok(actual)) => Some(format!("{case}\n  python: {expected}\n  rust:   {actual}")),
                Ok(Err(error)) => Some(format!("{case}: {error}")),
                Err(error) => Some(format!("{case}: {error}")),
            }
        })
        .collect();
    assert!(
        mismatches.is_empty(),
        "{} of {} {file} goldens differ:\n{}",
        mismatches.len(),
        records.len(),
        mismatches.join("\n")
    );
}
