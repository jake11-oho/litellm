use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_callbacks::{
    BudgetReservation, Callback, Field, FieldSet, FieldValues, Hook, HookInput, HostRequest,
    ReservationEntry, hosted,
};
use litellm_callbacks_budgets::Budgets;
use litellm_callbacks_spend_log::{SpendLogCallback, SpendLogParts};
use litellm_coroutine::CoroutineState;
use litellm_python_compat::{Value, json as pyjson};
use litellm_redis_store::{
    Number,
    spend_counters::{SpendCounter, SpendCounters},
};
use litellm_spend_log_writer::{
    LedgerStore, SpendLogFlusher, SpendLogRow, SpendLogWriter, WriterSettings,
};
use rstest::{fixture, rstest};
use serde_json::{Value as Json, json};

const KEY_COUNTER: &str = "spend:key:hashed";

#[derive(Clone, Default)]
struct Store {
    written: Arc<Mutex<Vec<SpendLogRow>>>,
    accepts: Option<usize>,
}

impl LedgerStore for Store {
    async fn insert_with_deliveries(
        &self,
        rows: &[SpendLogRow],
    ) -> Result<usize, litellm_spend_log_writer::Error> {
        self.written.lock().unwrap().extend_from_slice(rows);
        Ok(self.accepts.unwrap_or(rows.len()))
    }
}

struct Harness {
    callback: Arc<SpendLogCallback<Store>>,
    counters: Arc<SpendCounters>,
    store: Store,
}

fn harness(store: Store) -> Harness {
    harness_with(store, WriterSettings::default())
}

fn harness_with(store: Store, settings: WriterSettings) -> Harness {
    let counters = Arc::new(SpendCounters::new(None));
    counters
        .set_local(
            &SpendCounter::parse(KEY_COUNTER).unwrap(),
            Number::Float(5.0),
            None,
        )
        .unwrap();
    let writer = Arc::new(SpendLogWriter::new(SpendLogFlusher::new(
        store.clone(),
        settings,
    )));
    Harness {
        callback: Arc::new(SpendLogCallback::new(SpendLogParts {
            writer,
            budgets: Arc::new(Budgets::new(Arc::clone(&counters))),
            minimum_custom_key_length: 16,
            word_ranges: Vec::new(),
            retention: None,
        })),
        counters,
        store,
    }
}

#[fixture]
fn golden() -> (Json, Json) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/rows.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let record: Json = text
        .lines()
        .filter_map(|line| serde_json::from_str::<Json>(line).ok())
        .find(|record| {
            record["input"]["call_type"] == "acompletion"
                && record["output"]
                    .as_str()
                    .is_some_and(|output| output.contains("\"status\": \"success\""))
                && record["config"]["turn_off_message_logging"] == false
        })
        .unwrap();
    (record["config"].clone(), record["input"].clone())
}

fn settings(config: &Json) -> Json {
    json!({
        "disable_spend_updates": false,
        "disable_spend_logs": false,
        "disable_error_logs": false,
        "disable_responses_id_security": false,
        "row_config": config,
        "proxy_budget_name": "litellm-proxy-budget",
    })
}

fn identity() -> Json {
    json!({"token": "sk-raw-key-value", "user_id": "user-1", "team_id": "team-1", "end_user_id": null, "key_alias": "alias"})
}

fn reservation() -> BudgetReservation {
    BudgetReservation {
        reserved_cost: Some(2.0),
        entries: Some(vec![Some(ReservationEntry {
            counter_key: Some(KEY_COUNTER.to_owned()),
            entity_type: Some("key".to_owned()),
            entity_id: Some("hashed".to_owned()),
            reserved_cost: Some(2.0),
            applied_adjustment: Some(0.0),
            ..ReservationEntry::default()
        })]),
        ..BudgetReservation::default()
    }
}

fn input(hook: Hook, spend_log_input: Json, reservation: Option<BudgetReservation>) -> HookInput {
    let mut fields = FieldValues::new(FieldSet::of(&[
        Field::SpendLogInput,
        Field::BudgetReservation,
        Field::Outcome,
    ]))
    .with(Field::SpendLogInput, spend_log_input);
    if let Some(reservation) = reservation {
        fields.insert(
            Field::BudgetReservation,
            serde_json::to_value(reservation).unwrap(),
        );
    }
    HookInput {
        hook,
        fields,
        call: None,
    }
}

/// Runs one hook with a host that answers every call and records them.
/// One `null` per batch item, the proxy's answer when nothing is cached or stored.
fn nulls(request: &Json, items: &str) -> Json {
    Json::Array(vec![
        Json::Null;
        request[items].as_array().map_or(0, Vec::len)
    ])
}

async fn run(harness: &Harness, input: HookInput) -> Vec<(String, Json)> {
    let callback = Arc::clone(&harness.callback);
    let mut body = hosted(|host| async move {
        let _ = callback.handle_hosted(&input, None, host).await;
    });
    let mut calls = Vec::new();
    loop {
        match body.resume().await.unwrap() {
            CoroutineState::Complete(()) => return calls,
            CoroutineState::Yielded(HostRequest {
                call,
                request,
                reply,
            }) => {
                let answer = match call {
                    "accounting_context" => json!({
                        "windows": [],
                        "savings": {"compression": 0.0, "prompt_caching": 0.0},
                        "model_access_groups": [],
                        "chains_through_db": request["responses_turn"],
                    }),
                    "increment_search_free_tier" => json!(SEARCH_COUNTER_TOTAL_AFTER),
                    "enrich_key_identity" => {
                        json!({"user_id": "enriched-user", "team_alias": "enriched-team"})
                    }
                    "budget_windows" => nulls(&request, "entities"),
                    "stored_spend" => nulls(&request, "counters"),
                    "cached_spend" => {
                        Json::Array(vec![
                            json!(0.0);
                            request["sources"].as_array().map_or(0, Vec::len)
                        ])
                    }
                    _ => Json::Null,
                };
                calls.push((call.to_owned(), request));
                reply.send(Ok(answer));
            }
        }
    }
}

const SEARCH_COUNTER_TOTAL_AFTER: f64 = 6.0;

fn success(config: &Json, row: Json, cost: Option<f64>) -> Json {
    json!({
        "kind": "success",
        "settings": settings(config),
        "identity": identity(),
        "call_type": "acompletion",
        "row": {"row": row, "response_cost": cost},
    })
}

fn queued(harness: &Harness) -> Vec<Value> {
    harness
        .callback
        .writer()
        .queue()
        .dequeue(100)
        .iter()
        .map(|row| pyjson::loads(&row.to_json()).unwrap())
        .collect()
}

fn get<'a>(row: &'a Value, key: &str) -> Option<&'a Value> {
    litellm_python_compat::dict::get(row, key)
}

fn counter(harness: &Harness) -> Option<f64> {
    harness
        .counters
        .get_local(&SpendCounter::parse(KEY_COUNTER).unwrap())
        .unwrap()
        .map(Number::as_f64)
}

fn names(calls: &[(String, Json)]) -> Vec<&str> {
    calls.iter().map(|(name, _)| name.as_str()).collect()
}

#[rstest]
#[tokio::test]
async fn a_priced_success_queues_one_ledger_row_and_calls_the_host_in_order(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    let calls = run(
        &harness,
        input(Hook::AttemptEnded, success(&config, row, Some(0.25)), None),
    )
    .await;

    assert_eq!(
        names(&calls),
        [
            "accounting_context",
            "budget_windows",
            "stored_spend",
            "cached_spend",
            "budget_windows",
            "soft_budget_alert",
            "cached_spend_bump",
            "increment_proxy_spend",
            "customer_spend_alert"
        ]
    );
    let user = harness
        .counters
        .get_local(&SpendCounter::parse("spend:user:user-1").unwrap())
        .unwrap()
        .map(Number::as_f64);
    assert_eq!(user, Some(0.25));
    let rows = queued(&harness);
    assert_eq!(rows.len(), 1);
    assert_eq!(get(&rows[0], "spend"), Some(&Value::Float(0.25)));
    assert_eq!(
        get(&rows[0], "accounting_log_disabled"),
        Some(&Value::Bool(false))
    );
    assert!(get(&rows[0], "accounting_payload").is_some());
    assert!(get(&rows[0], "accounting_delivery_id").is_some());
}

#[rstest]
#[tokio::test]
async fn spend_logs_off_queue_only_the_charge(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row, Some(0.25));
    request["settings"]["disable_spend_logs"] = json!(true);

    run(&harness, input(Hook::AttemptEnded, request, None)).await;

    let rows = queued(&harness);
    assert_eq!(
        get(&rows[0], "accounting_log_disabled"),
        Some(&Value::Bool(true))
    );
    assert!(get(&rows[0], "metadata").is_none());
}

#[rstest]
#[tokio::test]
async fn disabled_spend_updates_skip_the_row_and_release_the_reservation(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row, Some(0.25));
    request["settings"]["disable_spend_updates"] = json!(true);

    let calls = run(
        &harness,
        input(Hook::AttemptEnded, request, Some(reservation())),
    )
    .await;

    assert!(calls.is_empty());
    assert!(queued(&harness).is_empty());
    assert_eq!(counter(&harness), Some(3.0));
}

#[rstest]
#[case::deferred_batch(Some(0.25), json!({"defers_batch_cost": true}), Some(3.0))]
#[case::unbilled_interaction(None, json!({"unbilled_interaction": true}), Some(3.0))]
#[case::interaction_still_running(None, json!({"unbilled_interaction": true, "interaction_in_progress": true}), Some(5.0))]
#[case::unpriced(None, json!({}), Some(3.0))]
#[tokio::test]
async fn an_uncharged_success_releases_or_keeps_the_reservation(
    golden: (Json, Json),
    #[case] cost: Option<f64>,
    #[case] response: Json,
    #[case] counter_after: Option<f64>,
) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row, cost);
    request["response"] = response;
    request["settings"]["background_interaction_cost_polling"] = json!(true);

    run(
        &harness,
        input(Hook::AttemptEnded, request, Some(reservation())),
    )
    .await;

    assert!(queued(&harness).is_empty());
    assert_eq!(counter(&harness), counter_after);
}

#[rstest]
#[tokio::test]
async fn an_unpriced_success_raises_the_failed_tracking_alert_and_fails_the_batch(
    golden: (Json, Json),
) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row.clone(), None);
    request["payload"] = json!({"cost_failure_debug_info": "no price for the model"});
    request["batch_accounting"] = json!(true);

    let calls = run(
        &harness,
        input(Hook::AttemptEnded, request, Some(reservation())),
    )
    .await;

    assert_eq!(names(&calls), ["failed_tracking_alert", "batch_accounting"]);
    let alert = &calls[0].1;
    let model = row["kwargs"]["model"].as_str().unwrap();
    assert_eq!(alert["failing_model"], json!(model));
    let message = alert["error_message"].as_str().unwrap();
    assert!(
        message.starts_with(&format!(
            "Error in tracking cost callback - Cost tracking failed for model={model}.\nDebug info - no price for the model\n"
        )),
        "{message}"
    );
    assert!(message.ends_with("call_type: acompletion\n"), "{message}");
    assert_eq!(
        calls[1].1,
        json!({"call_id": row["kwargs"]["litellm_call_id"], "write_started": true, "failed": true})
    );
}

#[rstest]
#[tokio::test]
async fn a_charged_success_without_a_reservation_starts_the_batch_write(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row.clone(), Some(0.25));
    request["batch_accounting"] = json!(true);

    let calls = run(&harness, input(Hook::AttemptEnded, request, None)).await;

    let (_, outcome) = calls
        .iter()
        .find(|(name, _)| name == "batch_accounting")
        .unwrap();
    assert_eq!(
        *outcome,
        json!({"call_id": row["kwargs"]["litellm_call_id"], "write_started": true, "failed": false})
    );
}

fn grounded(row: &Json, provider: &str) -> Json {
    let mut row = row.clone();
    row["usage"]["prompt_tokens_details"] = json!({"web_search_requests": 3});
    row["standard_logging_object"]["custom_llm_provider"] = json!(provider);
    row["standard_logging_object"]["cost_breakdown"] = json!({"input_cost": 0.5});
    row["standard_logging_object"]["model_map_information"] = json!({"model_map_value": {
        "litellm_provider": provider,
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {"search_context_size_medium": 0.1},
        "search_free_requests": {"count": 5, "period": "day", "shared_group": "g3"},
    }});
    row
}

#[rstest]
#[tokio::test]
async fn grounding_under_the_free_allowance_is_not_charged(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, grounded(&row, "gemini"), Some(1.0));
    request["settings"]["apply_search_free_tier"] = json!(true);

    let calls = run(&harness, input(Hook::AttemptEnded, request, None)).await;

    let (_, counted) = calls
        .iter()
        .find(|(name, _)| name == "increment_search_free_tier")
        .unwrap();
    let day = counted["key"]
        .as_str()
        .unwrap()
        .strip_prefix("litellm:search_free_tier:gemini:g3:")
        .unwrap();
    assert!(
        chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d").is_ok(),
        "{day}"
    );
    assert_eq!(counted["units"], json!(3));
    let units = 3.0;
    let freed = (5.0 - (SEARCH_COUNTER_TOTAL_AFTER - units)).min(units);
    let charged = 1.0 - 0.1 * units * freed / units;
    let user = harness
        .counters
        .get_local(&SpendCounter::parse("spend:user:user-1").unwrap())
        .unwrap()
        .map(Number::as_f64)
        .unwrap();
    assert!((user - charged).abs() < 1e-12, "{user} != {charged}");
    let rows = queued(&harness);
    assert_eq!(get(&rows[0], "spend"), Some(&Value::Float(charged)));
    let (_, bump) = calls
        .iter()
        .find(|(name, _)| name == "cached_spend_bump")
        .unwrap();
    assert_eq!(bump["response_cost"], json!(charged));
}

#[rstest]
#[tokio::test]
async fn a_finalized_cost_does_not_draw_from_the_allowance_again(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, grounded(&row, "gemini"), Some(1.0));
    request["settings"]["apply_search_free_tier"] = json!(true);
    request["payload"] = json!({"cost_finalized": true});

    let calls = run(&harness, input(Hook::AttemptEnded, request, None)).await;

    assert!(!names(&calls).contains(&"increment_search_free_tier"));
    assert_eq!(get(&queued(&harness)[0], "spend"), Some(&Value::Float(1.0)));
}

#[rstest]
#[tokio::test]
async fn a_responses_turn_wakes_the_writer_at_once(golden: (Json, Json)) {
    let (config, mut row) = golden;
    row["call_type"] = json!("aresponses");
    let store = Store::default();
    let harness = harness_with(
        store.clone(),
        WriterSettings {
            poll_interval: Duration::from_secs(600),
            max_backoff: Duration::from_secs(600),
            ..WriterSettings::default()
        },
    );
    let writer = Arc::clone(harness.callback.writer());
    let monitor = tokio::spawn(async move { writer.run(|| None).await });
    tokio::task::yield_now().await;

    run(
        &harness,
        input(Hook::AttemptEnded, success(&config, row, Some(0.25)), None),
    )
    .await;

    let written = tokio::time::timeout(Duration::from_secs(10), async {
        while store.written.lock().unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    monitor.abort();
    assert!(written.is_ok(), "the Responses turn waited for the poll");
}

#[rstest]
#[tokio::test]
async fn a_request_with_no_identity_is_not_charged(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row, Some(0.25));
    request["identity"] = json!({});

    let calls = run(&harness, input(Hook::AttemptEnded, request, None)).await;

    assert!(calls.is_empty());
    assert!(queued(&harness).is_empty());
}

#[rstest]
#[tokio::test]
async fn a_failed_attempt_is_left_to_the_fallback(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut hook = input(Hook::AttemptEnded, success(&config, row, Some(0.25)), None);
    hook.fields.insert(Field::Outcome, json!("failure"));

    run(&harness, hook).await;

    assert!(queued(&harness).is_empty());
}

#[rstest]
#[tokio::test]
async fn enrichment_fills_the_row_metadata_the_key_left_empty(golden: (Json, Json)) {
    let (config, mut row) = golden;
    row["metadata"]["user_api_key_user_id"] = Json::Null;
    let harness = harness(Store::default());
    let mut request = success(&config, row, Some(0.25));
    request["identity"]["user_id"] = Json::Null;

    let calls = run(&harness, input(Hook::AttemptEnded, request, None)).await;

    assert_eq!(names(&calls)[0], "enrich_key_identity");
    assert_eq!(calls[1].1["user_id"], json!("enriched-user"));
    let rows = queued(&harness);
    let Some(Value::Str(metadata)) = get(&rows[0], "metadata") else {
        panic!("metadata is a JSON string")
    };
    assert!(metadata.contains("enriched-user"), "{metadata}");
}

fn job_row(row: &Json) -> Json {
    let mut job = row.clone();
    job["call_type"] = json!("aretrieve_batch");
    job["response_id"] = json!("batch_abc");
    job
}

#[rstest]
#[case::first_poll_owns_the_charge(Some(1), 1)]
#[case::a_later_poll_charges_nothing(Some(0), 0)]
#[tokio::test]
async fn a_settle_once_job_row_is_claimed_through_the_ledger_store(
    golden: (Json, Json),
    #[case] accepts: Option<usize>,
    #[case] alerts: usize,
) {
    let (config, row) = golden;
    let harness = harness(Store {
        accepts,
        ..Store::default()
    });
    let mut request = success(&config, job_row(&row), Some(0.5));
    request["call_type"] = json!("aretrieve_batch");

    let calls = run(&harness, input(Hook::AttemptEnded, request, None)).await;

    assert_eq!(harness.store.written.lock().unwrap().len(), 1);
    assert!(queued(&harness).is_empty());
    assert_eq!(
        names(&calls)
            .iter()
            .filter(|name| **name == "customer_spend_alert")
            .count(),
        alerts
    );
}

fn failure(config: &Json, rows: Vec<Json>, completed: bool) -> Json {
    json!({
        "kind": "failure",
        "settings": settings(config),
        "identity": identity(),
        "completed_cost": completed,
        "track": true,
        "rows": rows.into_iter().map(|row| json!({"row": row, "response_cost": 0.0})).collect::<Vec<_>>(),
    })
}

#[rstest]
#[tokio::test]
async fn a_failure_releases_the_reservation_and_writes_every_billed_row(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut attempt = row.clone();
    attempt["response_id"] = json!("chatcmpl-attempt-1");

    let calls = run(
        &harness,
        input(
            Hook::PostCallFailure,
            failure(&config, vec![attempt, row], false),
            Some(reservation()),
        ),
    )
    .await;

    assert_eq!(counter(&harness), Some(3.0));
    assert_eq!(queued(&harness).len(), 2);
    assert_eq!(names(&calls), ["accounting_context", "accounting_context"]);
}

#[rstest]
#[tokio::test]
async fn an_untracked_failure_only_releases(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = failure(&config, vec![row], false);
    request["track"] = json!(false);

    run(
        &harness,
        input(Hook::PostCallFailure, request, Some(reservation())),
    )
    .await;

    assert_eq!(counter(&harness), Some(3.0));
    assert!(queued(&harness).is_empty());
}

#[rstest]
#[tokio::test]
async fn flush_drains_the_queue_into_the_store(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    run(
        &harness,
        input(Hook::AttemptEnded, success(&config, row, Some(0.25)), None),
    )
    .await;

    harness.callback.flush().await;

    assert!(harness.callback.writer().queue().is_empty());
    assert_eq!(harness.store.written.lock().unwrap().len(), 1);
}
