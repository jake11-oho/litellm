//! The success and failure flows and `update_database`'s ledger half: gate, search free tier,
//! reconcile, build, claim or enqueue, then the counters and the host calls, in the cost
//! callback's order.

use std::sync::atomic::{AtomicBool, Ordering};

use litellm_callbacks::{BudgetReservation, Host};
use litellm_callbacks_budgets::{CounterStep, CustomerWindow, Failure as Failed, post_call};
use litellm_python_compat::{
    Value,
    dict::{Dict, get, get_or_none},
    json,
};
use litellm_spend_log_writer::{LedgerStore, SpendLogRow, jsonified, retain_metadata};
use serde_json::{Map, Value as Json, json};
use sha2::{Digest, Sha256};

use super::{
    SpendLogCallback,
    hook_input::{Failure, Identity, RowRequest, Settings, Success},
    host::{
        AccountingContext, BatchAccounting, CachedSpendBump, CustomerSpendAlert, EnrichKeyIdentity,
        EnrichedKeyIdentity, FailedTrackingAlert, IncrementProxySpend, IncrementSearchFreeTier,
        SoftBudgetAlert,
    },
    policy::{self, Disposition, RESPONSES_CALL_TYPES},
    search_free_tier,
};
use crate::finalize_ledger_row;

const ACCOUNTING_ONLY_KEYS: [&str; 4] = [
    "request_id",
    "call_type",
    "accounting_payload",
    "accounting_delivery_id",
];
const VIDEO_STATUS_CALL_TYPES: [&str; 2] = ["avideo_status", "video_status"];
const RESPONSES_ID_CALL_TYPES: [&str; 2] = ["aget_responses", "acancel_responses"];

/// What `update_database` did with one row: whether the request is charged, and what the
/// accounting context resolved for the counter step (the model access groups the served model
/// debits, the customer window counters and the committed twins).
#[derive(Default)]
struct Written {
    charged: bool,
    model_access_groups: Vec<String>,
    customer_windows: Vec<CustomerWindow>,
    committed_tracked: Vec<String>,
}

impl Written {
    fn charged() -> Self {
        Self {
            charged: true,
            ..Self::default()
        }
    }
}

/// Who and what one charged row's counter step charges.
struct Charge<'a> {
    identity: &'a Identity,
    row: &'a RowRequest,
    cost: f64,
    customer_hierarchy: bool,
}

/// What a success tells CheckBatchCost (`BatchAccountingOutcome`): whether the spend write began,
/// and whether tracking failed.
#[derive(Default)]
struct BatchOutcome {
    write_started: AtomicBool,
    failed: AtomicBool,
}

impl BatchOutcome {
    async fn report(&self, row: &Json, host: &Host) {
        let write_started = self.write_started.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        let Some(call_id) = row
            .get("kwargs")
            .and_then(|kwargs| kwargs.get("litellm_call_id"))
            .and_then(Json::as_str)
        else {
            return;
        };
        if !(write_started || failed) {
            return;
        }
        if let Err(error) = host
            .call(BatchAccounting {
                call_id: call_id.to_owned(),
                write_started,
                failed,
            })
            .await
        {
            tracing::debug!(%error, "the batch accounting host call failed");
        }
    }
}

fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_sha256_hex(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// `hash_token` for a raw `sk-` key; anything else is already the logged form.
fn hashed_token(token: Option<&str>) -> Option<String> {
    token.map(|token| {
        if token.starts_with("sk-") {
            sha256_hex(token)
        } else {
            token.to_owned()
        }
    })
}

/// `_persistable_api_key`: a key the row did not log as is, and that is no hash, is stored
/// hashed with any bearer prefix dropped.
fn persistable_api_key(token: Option<&str>, logged: &str) -> Option<String> {
    let token = token?;
    if token == logged || is_sha256_hex(token) {
        return Some(token.to_owned());
    }
    let stripped = token
        .get(..7)
        .filter(|prefix| prefix.eq_ignore_ascii_case("bearer "))
        .map_or(token, |_| &token[7..]);
    (!stripped.is_empty()).then(|| sha256_hex(stripped))
}

fn text<'a>(row: &'a Value, key: &str) -> Option<&'a str> {
    match get(row, key) {
        Some(Value::Str(value)) => Some(value),
        _ => None,
    }
}

fn settles_once(row: &Value) -> bool {
    if text(row, "status") != Some("success") {
        return false;
    }
    let request_id = text(row, "request_id").unwrap_or_default();
    match text(row, "call_type") {
        Some("aretrieve_batch") => request_id.ends_with("_batch_cost"),
        Some(kind) if VIDEO_STATUS_CALL_TYPES.contains(&kind) => {
            request_id.ends_with("_video_cost")
        }
        _ => false,
    }
}

fn spend(row: &Value) -> f64 {
    match get_or_none(row, "spend") {
        Value::Float(value) => *value,
        _ => 0.0,
    }
}

fn accounting_only(row: &Value) -> Value {
    let mut kept = Dict::default();
    for key in ACCOUNTING_ONLY_KEYS {
        if let Some(value) = get(row, key) {
            kept.set(key, value.clone());
        }
    }
    kept.set("accounting_log_disabled", Value::Bool(true));
    kept.into_value()
}

fn with_entry(row: &Value, key: &str, value: Value) -> Value {
    let Value::Dict(entries) = row else {
        return row.clone();
    };
    let mut dict = Dict::from_entries(entries);
    dict.set(key, value);
    dict.into_value()
}

fn queued_row(row: Value) -> Option<SpendLogRow> {
    let Value::Dict(entries) = row else {
        return None;
    };
    Some(SpendLogRow::new(
        entries
            .into_iter()
            .filter_map(|(key, value)| match key {
                Value::Str(key) => Some((key, value)),
                _ => None,
            })
            .collect(),
    ))
}

/// `_write_spend_metadata_to_kwargs` on the row input: the enriched `user_api_key_*` values fill
/// the metadata the row reads wherever it holds none.
fn enrich_row(row: &mut RowRequest, enriched: &EnrichedKeyIdentity) {
    let Some(metadata) = row.row.get_mut("metadata").and_then(Json::as_object_mut) else {
        return;
    };
    for (key, value) in [
        ("user_api_key_alias", &enriched.key_alias),
        ("user_api_key_user_id", &enriched.user_id),
        ("user_api_key_team_id", &enriched.team_id),
        ("user_api_key_org_id", &enriched.org_id),
        ("user_api_key_project_id", &enriched.project_id),
        ("user_api_key_team_alias", &enriched.team_alias),
    ] {
        let Some(value) = value else { continue };
        if metadata.get(key).is_none_or(Json::is_null) {
            metadata.insert(key.to_owned(), Json::String(value.clone()));
        }
    }
}

fn fill(slot: &mut Option<String>, value: &Option<String>) {
    if slot.is_none() {
        slot.clone_from(value);
    }
}

/// `_charge_request_cost`'s second charge: the row of a repeat poll carries only what the poll
/// still pays for.
fn repriced(row: &RowRequest, cost: f64) -> RowRequest {
    let mut repriced = row.clone();
    repriced.response_cost = Some(cost);
    for key in ["kwargs", "standard_logging_object"] {
        if let Some(section) = repriced.row.get_mut(key).and_then(Json::as_object_mut)
            && section.contains_key("response_cost")
        {
            section.insert("response_cost".to_owned(), json!(cost));
        }
    }
    repriced
}

impl<S: LedgerStore + 'static> SpendLogCallback<S> {
    async fn enrich(
        &self,
        host: &Host,
        identity: &Identity,
        resolve_missing_key_identity: bool,
    ) -> Option<EnrichedKeyIdentity> {
        let api_key = identity.token.clone()?;
        match host
            .call(EnrichKeyIdentity {
                api_key,
                resolve_missing_key_identity,
                key_alias: identity.key_alias.clone(),
                user_id: identity.user_id.clone(),
                team_id: identity.team_id.clone(),
                org_id: identity.org_id.clone(),
                project_id: identity.project_id.clone(),
                team_alias: None,
            })
            .await
        {
            Ok(enriched) => Some(enriched),
            Err(error) => {
                tracing::debug!(%error, "Failed to enrich spend log metadata with key info");
                None
            }
        }
    }

    async fn release(
        &self,
        reservation: &mut Option<BudgetReservation>,
        window_resets: &[Option<String>],
    ) {
        if let Some(reservation) = reservation.as_mut() {
            post_call::release_or_invalidate(&self.budgets, reservation, window_resets).await;
        }
    }

    /// `_PROXY_track_cost_callback`: whatever it could not track raises the failed tracking
    /// alert, and a batch's costing learns whether it may be retried.
    pub(super) async fn on_success(
        &self,
        success: Success,
        reservation: Option<BudgetReservation>,
        host: &Host,
    ) {
        let mut success = success;
        let mut reservation = reservation;
        if let Some(resolve) = policy::enrich(&success)
            && let Some(enriched) = self.enrich(host, &success.identity, resolve).await
        {
            enrich_row(&mut success.row, &enriched);
            let identity = &mut success.identity;
            fill(&mut identity.key_alias, &enriched.key_alias);
            fill(&mut identity.user_id, &enriched.user_id);
            fill(&mut identity.team_id, &enriched.team_id);
            fill(&mut identity.org_id, &enriched.org_id);
            fill(&mut identity.project_id, &enriched.project_id);
        }
        let outcome = BatchOutcome::default();
        if reservation.is_some() || success.settings.apply_search_free_tier {
            outcome.write_started.store(true, Ordering::Relaxed);
        }
        if let Err(error) = self
            .track(&mut success, &mut reservation, &outcome, host)
            .await
        {
            outcome.failed.store(true, Ordering::Relaxed);
            failed_tracking_alert(&success, &error, host).await;
        }
        if success.batch_accounting {
            outcome.report(&success.row.row, host).await;
        }
    }

    async fn track(
        &self,
        success: &mut Success,
        reservation: &mut Option<BudgetReservation>,
        outcome: &BatchOutcome,
        host: &Host,
    ) -> Result<(), String> {
        let resets = success.window_resets.clone();
        match policy::disposition(success) {
            Disposition::InteractionInProgress => Ok(()),
            Disposition::Charge(cost) => {
                let cost = if success.settings.apply_search_free_tier
                    && !success.payload.cost_finalized
                    && success
                        .row
                        .row
                        .get("standard_logging_object")
                        .is_some_and(|payload| !payload.is_null())
                {
                    self.search_free_tier(success, cost, host).await
                } else {
                    cost
                };
                if !tracked(success) {
                    self.release(reservation, &resets).await;
                    return Ok(());
                }
                let charged = self
                    .charge_request(success, cost, reservation, outcome, host)
                    .await
                    .map_err(|error| error.to_string())?;
                if let Some(charged) = charged {
                    after_charge(success, charged, host).await;
                }
                Ok(())
            }
            Disposition::Unpriced => {
                self.release(reservation, &resets).await;
                Err(unpriced(success))
            }
            Disposition::DeferBatch | Disposition::InteractionUnbilled | Disposition::Released => {
                self.release(reservation, &resets).await;
                Ok(())
            }
        }
    }

    /// `apply_search_free_tier`: the cost after the grounding the free allowance covers. Any
    /// counter failure bills the grounding at full price.
    async fn search_free_tier(&self, success: &mut Success, cost: f64, host: &Host) -> f64 {
        let now = chrono::Utc::now();
        let Some(draw) = search_free_tier::draw(&success.row.row, success.payload.cache_hit, now)
        else {
            return cost;
        };
        let total_after = match host
            .call(IncrementSearchFreeTier {
                key: draw.key.clone(),
                units: draw.units,
                ttl: draw.ttl,
            })
            .await
        {
            Ok(Some(total_after)) => total_after,
            Ok(None) => {
                tracing::warn!(
                    "apply_search_free_tier is on but the proxy has no Redis, grounding is billed at full price"
                );
                return cost;
            }
            Err(error) => {
                tracing::warn!(%error, key = draw.key, "search free tier counter is unavailable, grounding is billed at full price");
                return cost;
            }
        };
        let discount = search_free_tier::discount(&draw, cost, total_after);
        search_free_tier::apply(&mut success.row, cost, discount)
    }

    /// `_charge_request_cost`: what the request charged, trying the repeat poll's own cost when
    /// an earlier poll already charged the job.
    async fn charge_request(
        &self,
        success: &Success,
        cost: f64,
        reservation: &mut Option<BudgetReservation>,
        outcome: &BatchOutcome,
        host: &Host,
    ) -> Result<Option<f64>, Failed> {
        if self
            .update_and_count(
                &success.settings,
                &success.identity,
                &success.row,
                cost,
                reservation,
                &success.window_resets,
                outcome,
                host,
            )
            .await?
        {
            return Ok(Some(cost));
        }
        let repeat = policy::repeat_poll_cost(success.call_type.as_deref(), cost, &success.row.row);
        if repeat <= 0.0 {
            return Ok(None);
        }
        let charged = self
            .update_and_count(
                &success.settings,
                &success.identity,
                &repriced(&success.row, repeat),
                repeat,
                &mut None,
                &[],
                outcome,
                host,
            )
            .await?;
        Ok(charged.then_some(repeat))
    }

    /// `_update_database_and_spend_counters`: reconcile the reservation, write the row, release
    /// the reservation when nothing was charged, then the counters, whose failure invalidates
    /// the reserved counters and is raised.
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per value the Python function takes"
    )]
    async fn update_and_count(
        &self,
        settings: &Settings,
        identity: &Identity,
        row: &RowRequest,
        cost: f64,
        reservation: &mut Option<BudgetReservation>,
        window_resets: &[Option<String>],
        outcome: &BatchOutcome,
        host: &Host,
    ) -> Result<bool, Failed> {
        if let Some(held) = reservation.as_mut()
            && let Err(error) =
                post_call::reconcile(&self.budgets, held, window_resets, cost, false, false).await
        {
            tracing::warn!(%error, "Failed to reconcile budget reservation before persisting spend; invalidating reserved counters");
            post_call::invalidate(self.budgets.store(), held).await;
            held.finalized = Some(true);
        }
        outcome.write_started.store(true, Ordering::Relaxed);
        let written = self
            .update_database(settings, identity, row, Some(cost), host)
            .await;
        if !written.charged {
            self.release(reservation, window_resets).await;
            return Ok(false);
        }
        let charge = Charge {
            identity,
            row,
            cost,
            customer_hierarchy: true,
        };
        if let Err(error) = self
            .count(host, &charge, written, reservation.as_mut(), window_resets)
            .await
        {
            if let Some(held) = reservation.as_mut() {
                post_call::invalidate(self.budgets.store(), held).await;
                held.finalized = Some(true);
            }
            return Err(error);
        }
        Ok(true)
    }

    /// `increment_spend_counters` on the ledger path: the whole post-call counter step in
    /// `Budgets::charge_counters`, its stamps and finalization copied onto the held reservation.
    async fn count(
        &self,
        host: &Host,
        charge: &Charge<'_>,
        written: Written,
        reservation: Option<&mut BudgetReservation>,
        window_resets: &[Option<String>],
    ) -> Result<(), Failed> {
        let identity = charge.identity;
        let step = CounterStep {
            token: identity.token.clone(),
            team_id: identity.team_id.clone(),
            user_id: identity.user_id.clone(),
            org_id: identity.org_id.clone(),
            end_user_id: identity.end_user_id.clone(),
            project_id: identity.project_id.clone(),
            tags: policy::tags(&charge.row.row).map(|tags| tags.into_iter().map(Some).collect()),
            model_access_groups: Some(written.model_access_groups.into_iter().map(Some).collect()),
            response_cost: Some(charge.cost),
            committed_tracked: written.committed_tracked,
            customer_windows: written.customer_windows,
            customer_scopes: if charge.customer_hierarchy {
                charge.row.customer_hierarchy.clone()
            } else {
                Vec::new()
            },
            window_resets: window_resets.to_vec(),
        };
        let outcome = self
            .budgets
            .charge_counters(host, None, &step, reservation.as_deref())
            .await;
        if let Some(held) = reservation {
            if let Some(entries) = held.entries.as_mut() {
                for (entry, stamp) in entries.iter_mut().zip(&outcome.stamps) {
                    if let (Some(entry), Some(stamp)) = (entry.as_mut(), stamp) {
                        entry.applied_adjustment = Some(*stamp);
                    }
                }
            }
            if outcome.finalized {
                held.finalized = Some(true);
            }
        }
        outcome.error.map_or(Ok(()), Err)
    }

    /// `DBSpendUpdateWriter.update_database` on the ledger path. A failure is logged and still
    /// counts as charged, as the Python writer does.
    async fn update_database(
        &self,
        settings: &Settings,
        identity: &Identity,
        row: &RowRequest,
        response_cost: Option<f64>,
        host: &Host,
    ) -> Written {
        if settings.disable_spend_updates {
            return Written::charged();
        }
        match self
            .write_row(settings, identity, row, response_cost, host)
            .await
        {
            Ok(written) => written,
            Err(error) => {
                tracing::error!(
                    %error,
                    response_cost,
                    user_id = identity.user_id,
                    team_id = identity.team_id,
                    "Spend tracking - update_database failed. Spend log insertion may not have completed for this request."
                );
                Written::charged()
            }
        }
    }

    async fn write_row(
        &self,
        settings: &Settings,
        identity: &Identity,
        request: &RowRequest,
        response_cost: Option<f64>,
        host: &Host,
    ) -> Result<Written, Box<dyn std::error::Error + Send + Sync>> {
        let builder = self.builders.for_settings(&settings.row_config)?;
        let row = builder.build_row(&json::from_json(request.row.clone()))?;
        let billable = response_cost.filter(|cost| cost.is_finite());
        if response_cost.is_some() && billable.is_none() {
            tracing::warn!(
                ?response_cost,
                "Spend tracking - recording a non-finite response_cost as 0"
            );
        }
        if settings.disable_responses_id_security
            && text(&row, "call_type").is_some_and(|kind| RESPONSES_ID_CALL_TYPES.contains(&kind))
            && text(&row, "status") == Some("success")
            && billable.is_none_or(|cost| cost == 0.0)
        {
            return Ok(Written::charged());
        }
        let token = hashed_token(identity.token.as_deref());
        let charges = billable.is_some_and(|cost| cost != 0.0);
        let responses_turn =
            text(&row, "call_type").is_some_and(|kind| RESPONSES_CALL_TYPES.contains(&kind));
        let accounting = host
            .call(AccountingContext {
                row: json::dumps(&row)?,
                token: token.clone(),
                team_id: identity.team_id.clone(),
                user_id: identity.user_id.clone(),
                windows: charges,
                customer: request.customer_entities.clone(),
                customer_scopes: request.customer_scopes.clone(),
                model_access_groups: request.model_access_groups.clone(),
                org_id: identity.org_id.clone(),
                end_user_id: identity.end_user_id.clone(),
                warm_committed: request.committed_tracked.is_none()
                    && billable.is_some_and(|cost| cost > 0.0),
                responses_turn,
            })
            .await?;
        let mut entities = Map::new();
        entities.insert(
            "api_key".to_owned(),
            json!(persistable_api_key(
                token.as_deref(),
                text(&row, "api_key").unwrap_or_default()
            )),
        );
        for (key, value) in [
            ("user", &identity.user_id),
            ("end_user", &identity.end_user_id),
            ("team_id", &identity.team_id),
            ("organization_id", &identity.org_id),
            ("project_id", &identity.project_id),
        ] {
            entities.insert(key.to_owned(), json!(value));
        }
        if let Some(customer) = &request.customer_entities {
            entities.extend(customer.clone());
        }
        entities.insert(
            "agent_id".to_owned(),
            json::to_json(get_or_none(&row, "agent_id"))?,
        );
        entities.insert(
            "proxy_budget_name".to_owned(),
            json!(settings.proxy_budget_name),
        );
        let ledger = json!({
            "response_cost": billable,
            "org_id": identity.org_id,
            "team_id": identity.team_id,
            "delivery_id": uuid::Uuid::new_v4().simple().to_string(),
            "entities": entities,
            "model_access_groups": accounting.model_access_groups,
            "windows": if charges { accounting.windows } else { Vec::new() },
            "savings": accounting.savings,
        });
        let finalized = finalize_ledger_row(&row, &json::from_json(ledger))?;
        let keep = !settings.disable_spend_logs
            && !(text(&finalized, "status") == Some("failure") && settings.disable_error_logs);
        Ok(Written {
            charged: self
                .record(finalized, keep, accounting.chains_through_db)
                .await,
            model_access_groups: accounting.model_access_groups,
            customer_windows: accounting.customer_windows,
            committed_tracked: request
                .committed_tracked
                .clone()
                .unwrap_or(accounting.committed_tracked),
        })
    }

    /// `_record_spend_log`: a settle-once job row claims its charge now, every other row is
    /// queued, as an accounting-only row when request logs are off. A Responses turn the next
    /// turn replays from the database wakes the writer at once.
    async fn record(&self, row: Value, keep: bool, chains_through_db: bool) -> bool {
        if settles_once(&row) {
            return self.claim(row, keep).await;
        }
        let stored = if keep {
            with_entry(&row, "accounting_log_disabled", Value::Bool(false))
        } else {
            accounting_only(&row)
        };
        self.enqueue(stored);
        if chains_through_db {
            self.writer.request_flush();
        }
        true
    }

    fn enqueue(&self, row: Value) {
        let Some(row) = queued_row(row) else { return };
        let budget = self.writer.settings().queue_max_bytes;
        self.writer.enqueue(vec![row], false, budget);
    }

    /// `_claim_settle_once_cost_row` for a ledger row: the job's delivery row is the claim, so
    /// the poll whose insert lands owns the charge. A $0 poll records nothing.
    async fn claim(&self, row: Value, keep: bool) -> bool {
        let charged = spend(&row) > 0.0;
        let stored = if keep { row } else { accounting_only(&row) };
        if !charged {
            return true;
        }
        let Some(queued) = queued_row(stored) else {
            return true;
        };
        let max_depth = self.writer.settings().max_depth;
        let retained = match retain_metadata(&queued, self.retention.as_ref(), max_depth) {
            Ok(retained) => retained,
            Err(error) => {
                tracing::warn!(%error, "Could not claim a job's cost row, queueing it");
                self.writer
                    .enqueue(vec![queued], false, self.writer.settings().queue_max_bytes);
                return true;
            }
        };
        match self
            .writer
            .flusher()
            .store()
            .insert_with_deliveries(&[jsonified(&retained)])
            .await
        {
            Ok(accepted) => accepted == 1,
            Err(error) => {
                tracing::warn!(%error, "Could not claim a job's cost row, queueing it");
                self.writer.enqueue(
                    vec![retained],
                    false,
                    self.writer.settings().queue_max_bytes,
                );
                true
            }
        }
    }

    pub(super) async fn on_failure(
        &self,
        failure: Failure,
        reservation: Option<BudgetReservation>,
        host: &Host,
    ) {
        let mut failure = failure;
        let mut reservation = reservation;
        let disabled = failure.settings.disable_spend_updates;
        let resets = failure.window_resets.clone();
        if !failure.completed_cost || disabled {
            self.release(&mut reservation, &resets).await;
        }
        if (failure.completed_cost && disabled) || !failure.track {
            return;
        }
        if failure.enrich
            && let Some(enriched) = self.enrich(host, &failure.identity, true).await
        {
            for row in &mut failure.rows {
                enrich_row(row, &enriched);
            }
        }
        if failure.completed_cost {
            let outcome = BatchOutcome::default();
            for row in &failure.rows {
                let cost = row.response_cost.unwrap_or(0.0);
                if let Err(error) = self
                    .update_and_count(
                        &failure.settings,
                        &failure.identity,
                        row,
                        cost,
                        &mut reservation,
                        &resets,
                        &outcome,
                        host,
                    )
                    .await
                {
                    tracing::error!(%error, "Failed to write the spend log for a failure with a completed cost");
                }
            }
            return;
        }
        let mut counted = Vec::new();
        for row in &failure.rows {
            let written = self
                .update_database(
                    &failure.settings,
                    &failure.identity,
                    row,
                    row.response_cost,
                    host,
                )
                .await;
            let cost = row.response_cost.unwrap_or(0.0);
            if written.charged && cost > 0.0 {
                counted.push((row, cost, written));
            }
        }
        if disabled {
            return;
        }
        for (row, cost, written) in counted {
            let charge = Charge {
                identity: &failure.identity,
                row,
                cost,
                customer_hierarchy: false,
            };
            if let Err(error) = self.count(host, &charge, written, None, &[]).await {
                tracing::error!(%error, "Failed to raise spend counters for a failure spend row");
            }
        }
    }
}

/// `_should_track_cost_callback`.
fn tracked(success: &Success) -> bool {
    if success.settings.disable_spend_updates {
        return false;
    }
    let identity = &success.identity;
    identity.agent_id.is_some()
        || identity.token.is_some()
        || identity.user_id.is_some()
        || identity.team_id.is_some()
        || identity.end_user_id.is_some()
        || policy::tracks_unattributed(success.call_type.as_deref())
}

fn logged_model(row: &Json) -> String {
    match policy::model(row) {
        None => String::new(),
        Some(Json::Null) => "None".to_owned(),
        Some(Json::String(model)) => model.clone(),
        Some(other) => other.to_string(),
    }
}

/// The exception the cost callback raises for a priced call that lost its cost.
fn unpriced(success: &Success) -> String {
    let model = match policy::model(&success.row.row) {
        Some(Json::String(model)) => model.clone(),
        Some(Json::Null) | None => "None".to_owned(),
        Some(other) => other.to_string(),
    };
    let has_payload = success
        .row
        .row
        .get("standard_logging_object")
        .is_some_and(|payload| !payload.is_null());
    let debug_info = match (&success.payload.cost_failure_debug_info, has_payload) {
        (_, false) => "standard_logging_object not found",
        (Some(info), true) => info.as_str(),
        (None, true) => "response_cost_failure_debug_info is None in standard_logging_object",
    };
    format!(
        "Cost tracking failed for model={model}.\nDebug info - {debug_info}\nAdd custom pricing - https://docs.litellm.ai/docs/proxy/custom_pricing"
    )
}

/// The cost callback's `except`: the failed tracking alert, fired and forgotten.
async fn failed_tracking_alert(success: &Success, error: &str, host: &Host) {
    tracing::error!(error, "Error in tracking cost callback");
    let failing_model = logged_model(&success.row.row);
    let call_type = success.call_type.as_deref().unwrap_or_default();
    let error_message = format!(
        "Error in tracking cost callback - {error}\n Traceback:raised in the Rust spend log callback\n Args to _PROXY_track_cost_callback\n model: {failing_model}\n call_type: {call_type}\n"
    );
    if let Err(error) = host
        .call(FailedTrackingAlert {
            error_message,
            failing_model,
        })
        .await
    {
        tracing::debug!(%error, "the failed tracking alert host call failed");
    }
}

/// The work `update_cache` and the cost callback leave to Python once a request is charged.
async fn after_charge(success: &Success, charged: f64, host: &Host) {
    let identity = &success.identity;
    if let Some(token) = &identity.token
        && let Err(error) = host
            .call(SoftBudgetAlert {
                token: token.clone(),
                response_cost: charged,
            })
            .await
    {
        tracing::debug!(%error, "the soft budget alert host call failed");
    }
    if let Err(error) = host
        .call(CachedSpendBump {
            user_id: identity.user_id.clone(),
            end_user_id: identity.end_user_id.clone(),
            team_id: identity.team_id.clone(),
            tags: policy::tags(&success.row.row),
            response_cost: charged,
        })
        .await
    {
        tracing::debug!(%error, "the cached spend bump host call failed");
    }
    if success.settings.proxy_budget_name.is_some()
        && let Err(error) = host
            .call(IncrementProxySpend {
                response_cost: charged,
            })
            .await
    {
        tracing::warn!(%error, "Spend tracking - failed to add spend to the global proxy spend");
    }
    if let Err(error) = host
        .call(CustomerSpendAlert {
            token: identity.token.clone(),
            key_alias: identity.key_alias.clone(),
            end_user_id: identity.end_user_id.clone(),
            response_cost: charged,
            max_budget: success.end_user_max_budget,
        })
        .await
    {
        tracing::debug!(%error, "the customer spend alert host call failed");
    }
}
