//! The spend logs ledger's half of the row: the billable spend, the delivery id that dedupes the
//! accounting worker's writes, and the accounting payload (`build_accounting_payload`), which
//! carries the guardrail share of spend and the entities, windows and savings the worker debits.

use litellm_python_compat::{Value, json, number, pydantic};

use crate::RowError;
use crate::py::{
    Dict, Structure, as_str, big_to_f64, entries, get, get_or_none, safe_structure, text, truthy,
};

const JSON_SAFE_INTEGER: i64 = (1 << 53) - 1;
const METADATA_FIELDS: [&str; 8] = [
    "usage_object",
    "status",
    "guardrail_information",
    "standard_logging_guardrail_information",
    "compression_savings",
    "internal_call_origin",
    "billed_retry_attempt",
    "user_api_key_customer_usage",
];
const GUARDRAIL_METADATA_FIELDS: [&str; 2] = [
    "guardrail_information",
    "standard_logging_guardrail_information",
];
const ACCOUNTING_FIELDS: [&str; 22] = [
    "request_id",
    "call_type",
    "api_key",
    "spend",
    "prompt_tokens",
    "completion_tokens",
    "startTime",
    "model",
    "model_group",
    "custom_llm_provider",
    "user",
    "team_id",
    "organization_id",
    "end_user",
    "customer_member_id",
    "customer_team_id",
    "customer_org_id",
    "agent_id",
    "request_tags",
    "status",
    "request_duration_ms",
    "mcp_namespaced_tool_name",
];
const GUARDRAIL_ACCOUNTING_FIELDS: [&str; 6] = [
    "guardrail_id",
    "guardrail_name",
    "guardrail_provider",
    "guardrail_usage",
    "guardrail_cost_by_unit",
    "guardrail_cost_in_spend",
];
const VIDEO_STATUS_CALL_TYPES: [&str; 2] = ["video_status", "avideo_status"];

/// A Python number result that keeps `int` and `float` apart, since `json.dumps` writes them
/// differently.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Number {
    Int(i64),
    Float(f64),
}

impl Number {
    pub(crate) fn value(self) -> f64 {
        match self {
            Number::Int(number) => number as f64,
            Number::Float(number) => number,
        }
    }

    fn into_value(self) -> Value {
        match self {
            Number::Int(number) => Value::Int(number.into()),
            Number::Float(number) => Value::Float(number),
        }
    }
}

fn entry_cost(entry: &Value) -> f64 {
    let Value::Dict(_) = entry else {
        return 0.0;
    };
    let cost = match get_or_none(entry, "guardrail_cost") {
        Value::None => None,
        found => match pydantic::lax_float(found) {
            Some(cost) => Some(cost),
            None => return 0.0,
        },
    };
    let in_spend = match get(entry, "guardrail_cost_in_spend") {
        None => Some(true),
        Some(Value::None) => None,
        Some(found) => match pydantic::lax_bool(found) {
            Some(flag) => Some(flag),
            None => return 0.0,
        },
    };
    if in_spend == Some(false) {
        return 0.0;
    }
    match cost {
        Some(cost) if cost.is_finite() && cost > 0.0 => cost,
        _ => 0.0,
    }
}

/// `guardrail_information_cost(guardrail_information)`: the billable cost of the guardrail records.
pub(crate) fn guardrail_information_cost(information: &Value) -> Number {
    match information {
        Value::None => Number::Float(0.0),
        Value::List(items) | Value::Tuple(items) if items.is_empty() => Number::Int(0),
        Value::List(items) | Value::Tuple(items) => {
            Number::Float(number::sum(items.iter().map(entry_cost)))
        }
        other => Number::Float(entry_cost(other)),
    }
}

fn spend_number(spend: &Value) -> Result<Number, RowError> {
    match spend {
        Value::Float(number) => Ok(Number::Float(*number)),
        Value::Int(number) => i64::try_from(number)
            .map(Number::Int)
            .map_err(|_| RowError::Type("spend")),
        Value::Bool(flag) => Ok(Number::Int(i64::from(*flag))),
        _ => Err(RowError::Type("spend")),
    }
}

fn guardrail_share_of_spend(metadata: &Value, spend: &Value) -> Result<Number, RowError> {
    let spend = spend_number(spend)?;
    let floor = Number::Float(0.0);
    let bounded_spend = if floor.value() > spend.value() {
        floor
    } else {
        spend
    };
    let cost = guardrail_information_cost(get_or_none(metadata, "guardrail_information"));
    Ok(if bounded_spend.value() < cost.value() {
        bounded_spend
    } else {
        cost
    })
}

fn accounting_guardrail_entry(entry: &[(Value, Value)]) -> Value {
    let mut kept = Dict::default();
    for (key, value) in entry {
        if as_str(key).is_some_and(|name| GUARDRAIL_ACCOUNTING_FIELDS.contains(&name)) {
            kept.set_value(key.clone(), value.clone());
        }
    }
    let entry_value = Value::Dict(entry.to_vec());
    if let Some(response @ Value::Dict(_)) = get(&entry_value, "guardrail_response")
        && let Some(tokens_saved) = get(response, "tokens_saved")
    {
        kept.set(
            "guardrail_response",
            Value::Dict(vec![(text("tokens_saved"), tokens_saved.clone())]),
        );
    }
    kept.into_value()
}

fn accounting_guardrail_entries(value: &Value) -> Value {
    let listed: &[Value] = match value {
        Value::Dict(_) => std::slice::from_ref(value),
        Value::List(items) | Value::Tuple(items) => items,
        _ => return Value::None,
    };
    let Some(validated) = listed.iter().map(entries).collect::<Option<Vec<_>>>() else {
        return Value::None;
    };
    Value::List(
        validated
            .into_iter()
            .map(accounting_guardrail_entry)
            .collect(),
    )
}

/// What the accounting worker debits besides the row itself, resolved in Python from the
/// request's key, team and customer state.
struct AccountingInput {
    entities: Value,
    savings: Value,
    model_access_groups: Value,
    windows: Value,
}

impl AccountingInput {
    fn parse(input: &Value) -> Result<Self, RowError> {
        let field = |name: &'static str| get(input, name).cloned().ok_or(RowError::Missing(name));
        Ok(Self {
            entities: field("entities")?,
            savings: field("savings")?,
            model_access_groups: field("model_access_groups")?,
            windows: field("windows")?,
        })
    }
}

/// `build_accounting_payload(row, entities=..., model_access_groups=..., windows=..., router=...)`,
/// with the savings already priced in Python.
pub fn build_accounting_payload(row: &Value, input: &Value) -> Result<Value, RowError> {
    accounting_payload(row, &AccountingInput::parse(input)?)
}

fn accounting_payload(row: &Value, input: &AccountingInput) -> Result<Value, RowError> {
    let metadata = match get(row, "metadata") {
        Some(Value::Str(metadata)) => json::loads(metadata)?,
        _ => return Err(RowError::Type("metadata")),
    };
    let metadata_items = entries(&metadata).ok_or(RowError::NotAMapping("metadata"))?;
    let mut accounting_metadata = Dict::default();
    for (key, value) in metadata_items {
        let Some(name) = as_str(key).filter(|name| METADATA_FIELDS.contains(name)) else {
            continue;
        };
        let stored = if GUARDRAIL_METADATA_FIELDS.contains(&name) {
            accounting_guardrail_entries(value)
        } else {
            value.clone()
        };
        accounting_metadata.set_value(key.clone(), stored);
    }
    accounting_metadata.set("accounting_entities", input.entities.clone());
    accounting_metadata.set("accounting_savings", input.savings.clone());
    accounting_metadata.set(
        "accounting_guardrail_spend",
        guardrail_share_of_spend(&metadata, get_or_none(row, "spend"))?.into_value(),
    );
    accounting_metadata.set(
        "accounting_model_access_groups",
        input.model_access_groups.clone(),
    );
    accounting_metadata.set("accounting_windows", input.windows.clone());
    let mut payload = Dict::default();
    for (key, value) in entries(row).ok_or(RowError::NotAMapping("row"))? {
        if as_str(key).is_some_and(|name| ACCOUNTING_FIELDS.contains(&name)) {
            payload.set_value(key.clone(), value.clone());
        }
    }
    payload.set("metadata", accounting_metadata.into_value());
    Ok(safe_structure(
        &payload.into_value(),
        &Structure {
            non_finite_as_null: true,
            integer_limit: Some(JSON_SAFE_INTEGER),
        },
    ))
}

/// `finite_cost(value)`: a finite number as a float, else `None`.
pub(crate) fn finite_cost(value: &Value) -> Option<f64> {
    match value {
        Value::Float(number) if number.is_finite() => Some(*number),
        Value::Int(number) => big_to_f64(number),
        _ => None,
    }
}

fn settles_once(row: &Value) -> bool {
    if !matches!(get_or_none(row, "status"), Value::Str(status) if status == "success") {
        return false;
    }
    let Some(request_id) = as_str(get_or_none(row, "request_id")) else {
        return false;
    };
    match as_str(get_or_none(row, "call_type")) {
        Some("aretrieve_batch") => request_id.ends_with("_batch_cost"),
        Some(kind) if VIDEO_STATUS_CALL_TYPES.contains(&kind) => {
            request_id.ends_with("_video_cost")
        }
        _ => false,
    }
}

/// The writer's ledger inputs for one row.
struct LedgerInput {
    billable_cost: Option<f64>,
    org_id: Value,
    team_id: Value,
    delivery_id: Value,
    accounting: AccountingInput,
}

impl LedgerInput {
    fn parse(input: &Value) -> Result<Self, RowError> {
        Ok(Self {
            billable_cost: finite_cost(get_or_none(input, "response_cost"))
                .filter(|cost| *cost != 0.0),
            org_id: get_or_none(input, "org_id").clone(),
            team_id: get_or_none(input, "team_id").clone(),
            delivery_id: get(input, "delivery_id")
                .cloned()
                .ok_or(RowError::Missing("delivery_id"))?,
            accounting: AccountingInput::parse(input)?,
        })
    }
}

/// The row as the writer stores it on the ledger path: spend set to the billable cost, the
/// org and team the request resolved to, a delivery id unless a settle-once poll carries it,
/// and the accounting payload.
pub fn finalize_ledger_row(row: &Value, input: &Value) -> Result<Value, RowError> {
    let input = LedgerInput::parse(input)?;
    let mut finalized = Dict::from_entries(entries(row).ok_or(RowError::NotAMapping("row"))?);
    finalized.set("spend", Value::Float(input.billable_cost.unwrap_or(0.0)));
    for (field, value) in [
        ("organization_id", &input.org_id),
        ("team_id", &input.team_id),
    ] {
        if truthy(value) {
            finalized.set(field, value.clone());
        }
    }
    if !settles_once(&finalized.clone().into_value()) {
        finalized.set("accounting_delivery_id", input.delivery_id.clone());
    }
    let accounting = accounting_payload(&finalized.clone().into_value(), &input.accounting)?;
    finalized.set("accounting_payload", accounting);
    Ok(finalized.into_value())
}
