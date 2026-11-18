//! The whole post-call spend counter step for one charged request, I/O included.
//!
//! [`charge_counters`] settles the customer hierarchy counters, reconciles the budget reservation,
//! warms every counter [`plan`](crate::plan) names with one batch read, seeds the cold ones, and
//! sends every increment in one pipeline. What only the proxy knows comes back through three host
//! calls, each answering a whole batch so the proxy can run its part concurrently:
//! [`BudgetWindows`] (an entity's budget windows from the auth cache or the stored row),
//! [`StoredSpend`] (a counter's spend from the database) and [`CachedSpend`] (the auth cache's spend
//! for a cold counter the database cannot answer).
//!
//! Scopes keep the fan-out's failure rules: a sequential scope drops everything it prepared at its
//! first failure, a concurrent scope keeps every success and each failure separately. All failures
//! are reported after the increments went out and the reservation was stamped.

use std::{collections::HashMap, sync::Arc};

use futures_util::future::join_all;
use litellm_callbacks::{BudgetReservation, Host, HostCall, HostCallError};
use litellm_redis_store::{
    ErrorKind, Number, RequestRedis, Send,
    spend_counters::{
        LostSeed, NotASpendCounter, PostCallSettlement, Reading, SpendCounter, SpendCounters,
    },
};
use serde::{Deserialize, Serialize};

use crate::{
    counters::{
        Charge, CustomerScopeSpend, CustomerWindow, Task, WindowedEntity, charged,
        committed_counter_keys, customer_settlement, plan, window_counters,
    },
    engine::{self, Settle},
    window::{self, Moment},
};

/// One charged request's counter inputs, as the cost callback has them after the spend write.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CounterStep {
    pub token: Option<String>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub org_id: Option<String>,
    pub end_user_id: Option<String>,
    pub project_id: Option<String>,
    pub tags: Option<Vec<Option<String>>>,
    pub model_access_groups: Option<Vec<Option<String>>>,
    pub response_cost: Option<f64>,
    /// The entity counters whose committed twin exists for this charge.
    pub committed_tracked: Vec<String>,
    pub customer_windows: Vec<CustomerWindow>,
    /// The customer hierarchy counters the request was admitted under, settled before the rest.
    pub customer_scopes: Vec<CustomerScopeSpend>,
    /// Each reservation entry's `window_reset_at`, in entry order.
    pub window_resets: Vec<Option<String>>,
}

/// Why a counter could not be charged.
#[derive(Debug)]
pub enum CounterError {
    Store(litellm_redis_store::Error),
    Host(HostCallError),
    Counter(NotASpendCounter),
}

impl std::fmt::Display for CounterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(error) => write!(f, "{error}"),
            Self::Host(error) => write!(f, "{error}"),
            Self::Counter(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for CounterError {}

/// A failure shared by every counter one failed call left unwarmed.
pub type Failure = Arc<CounterError>;

fn store_failure(error: litellm_redis_store::Error) -> Failure {
    Arc::new(CounterError::Store(error))
}

fn host_failure(error: HostCallError) -> Failure {
    Arc::new(CounterError::Host(error))
}

/// What the step did, for the caller to record on its reservation.
#[derive(Debug, Default)]
pub struct CounterOutcome {
    /// The `applied_adjustment` each reservation entry now carries, `None` to leave it as it is.
    pub stamps: Vec<Option<f64>>,
    /// Whether the reservation is settled for good.
    pub finalized: bool,
    /// The post-call pipeline that carried committed twins, which later readers wait on.
    pub committed_settlement: Option<PostCallSettlement>,
    /// The first failure, reported after everything else ran.
    pub error: Option<Failure>,
}

/// One budget window as the proxy resolved it: its duration and the start of the window the
/// request falls in, `None` when the window is not valid.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct BudgetWindow {
    pub budget_duration: String,
    pub window_start: Option<String>,
}

/// The entity whose budget windows a [`BudgetWindows`] call reads.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WindowSource {
    pub entity_type: WindowedEntity,
    pub entity_id: String,
    pub cache_key: String,
}

/// Host call: each entity's budget windows, from the auth cache (`None` when the entity is not
/// cached) or, with `stored`, from its database row.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BudgetWindows {
    pub entities: Vec<WindowSource>,
    pub stored: bool,
}

impl HostCall for BudgetWindows {
    const NAME: &'static str = "budget_windows";
    type Reply = Vec<Option<Vec<BudgetWindow>>>;
}

/// A counter whose spend the database holds: an entity row named by the counter, or a budget
/// window's spend from its maintained row or the spend logs.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum StoredCounter {
    Entity {
        counter_key: String,
    },
    Window {
        entity_type: String,
        entity_id: String,
        window_duration: Option<String>,
        window_start: String,
    },
}

/// Host call: each counter's spend from the database, `None` when it has no readable row.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StoredSpend {
    pub counters: Vec<StoredCounter>,
}

impl HostCall for StoredSpend {
    const NAME: &'static str = "stored_spend";
    type Reply = Vec<Option<f64>>;
}

/// Host call: for each cold counter, the spend on the first auth cache entry of its sources that
/// is cached, `0.0` when none is.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CachedSpend {
    pub sources: Vec<Vec<String>>,
}

impl HostCall for CachedSpend {
    const NAME: &'static str = "cached_spend";
    type Reply = Vec<f64>;
}

/// Whether the database row a counter is named after holds its spend. End user, tag, access group
/// and key, team and user window counters have none.
fn has_stored_spend(counter: &SpendCounter) -> bool {
    matches!(
        counter,
        SpendCounter::Key { window: None, .. }
            | SpendCounter::Team { window: None, .. }
            | SpendCounter::TeamMember { .. }
            | SpendCounter::User(_)
            | SpendCounter::Org(_)
            | SpendCounter::Project(_)
            | SpendCounter::CustomerOrg(_)
            | SpendCounter::CustomerTeam(_)
            | SpendCounter::CustomerMember(_)
            | SpendCounter::CustomerWindow { .. }
    )
}

/// One batch read's answer for a counter: its value, and whether Redis gave it.
#[derive(Clone, Copy, Debug, Default)]
struct Read {
    value: Option<f64>,
    authoritative: bool,
}

/// Reads every counter with one `MGET`, or one read each when there is no Redis or the `MGET`
/// failed. A value Redis returns is mirrored into memory, as the warm check does.
async fn read_all(store: &SpendCounters, counters: &[SpendCounter]) -> HashMap<String, Read> {
    if counters.is_empty() {
        return HashMap::new();
    }
    let batched = match store.read_many(Send::Now, counters) {
        Some(pending) => pending.await.ok().flatten(),
        None => None,
    };
    let reads: Vec<Read> = match batched {
        Some(values) => values
            .into_iter()
            .map(|value| Read {
                value: value.map(Number::as_f64),
                authoritative: true,
            })
            .collect(),
        None => join_all(counters.iter().map(|counter| store.read(counter)))
            .await
            .into_iter()
            .map(|reading| match reading {
                Ok(Reading::Redis(value)) => Read {
                    value: value.map(Number::as_f64),
                    authoritative: true,
                },
                Ok(Reading::Memory(value)) => Read {
                    value: value.map(Number::as_f64),
                    authoritative: false,
                },
                Err(_) => Read::default(),
            })
            .collect(),
    };
    counters
        .iter()
        .zip(reads)
        .map(|(counter, read)| {
            if let (true, Some(value)) = (read.authoritative, read.value)
                && let Err(error) = store.set_local(counter, Number::Float(value), None)
            {
                tracing::debug!(%error, "mirroring a spend counter into memory failed");
            }
            (counter.key(), read)
        })
        .collect()
}

/// `increment_spend_counter_cache`: a counter's increment on its own, dropped everywhere when it
/// may not have landed.
async fn increment_one(
    store: &SpendCounters,
    counter: &SpendCounter,
    amount: f64,
) -> Result<(), Failure> {
    if store.has_redis() {
        store
            .increment(counter, Number::Float(amount))
            .await
            .map_err(store_failure)?;
    } else {
        store
            .increment_local(counter, Number::Float(amount))
            .map_err(store_failure)?;
    }
    Ok(())
}

fn is_unavailable(error: &litellm_redis_store::Error) -> bool {
    matches!(error.kind(), ErrorKind::BreakerOpen | ErrorKind::Timeout)
}

/// `_apply_spend_counter_increments`: the request's post-call pipeline when it has one, else one
/// pipeline now that drops every counter when it fails. A timed out or breaker-blocked pipeline is
/// dropped silently. Answers with the post-call settlement when it carried a committed twin.
async fn apply(
    store: &Arc<SpendCounters>,
    request: Option<&RequestRedis>,
    pending: &[(SpendCounter, Number)],
) -> Result<Option<PostCallSettlement>, Failure> {
    if pending.is_empty() {
        return Ok(None);
    }
    if let Some(request) = request
        && let Some(settlement) = store.increment_many_post_call(request, pending)
    {
        let committed = pending
            .iter()
            .any(|(counter, _)| matches!(counter, SpendCounter::Committed(_)));
        return Ok(committed.then_some(settlement));
    }
    if !store.has_redis() {
        for (counter, amount) in pending {
            store
                .increment_local(counter, *amount)
                .map_err(store_failure)?;
        }
        return Ok(None);
    }
    if let Err(error) = store.increment_many(pending).await {
        let counters: Vec<SpendCounter> = pending.iter().map(|(c, _)| c.clone()).collect();
        store.invalidate(&counters).await;
        if !is_unavailable(&error) {
            return Err(store_failure(error));
        }
    }
    Ok(None)
}

/// Charges each customer hierarchy counter, seeding the cold ones with their admitted spend first.
pub async fn settle_customer_spend(
    store: &SpendCounters,
    scopes: &[CustomerScopeSpend],
    cost: f64,
) -> Result<(), Failure> {
    let counters = scopes
        .iter()
        .map(|scope| SpendCounter::parse(&scope.counter_key))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| Arc::new(CounterError::Counter(error)))?;
    let readings = join_all(counters.iter().map(|counter| store.read(counter)))
        .await
        .into_iter()
        .map(|reading| reading.map(|reading| reading.value().map(Number::as_f64)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(store_failure)?;
    let settlement = customer_settlement(scopes, &readings, cost);
    let seeds = settlement
        .seeds
        .iter()
        .map(|(key, spend)| (SpendCounter::parse(key), *spend));
    for seeded in join_all(seeds.map(|(counter, spend)| async move {
        match counter {
            Ok(counter) => store
                .seed(&counter, Number::Float(spend), LostSeed::Answer)
                .await
                .map(drop)
                .map_err(store_failure),
            Err(error) => Err(Arc::new(CounterError::Counter(error))),
        }
    }))
    .await
    {
        seeded?;
    }
    let increments: Vec<(SpendCounter, Number)> = counters
        .into_iter()
        .zip(&settlement.increments)
        .map(|(counter, (_, cost))| (counter, Number::Float(*cost)))
        .collect();
    store
        .increment_many(&increments)
        .await
        .map(drop)
        .map_err(store_failure)
}

/// The `applied_adjustment` every entry with a counter carries once the reserved counters hold the
/// actual cost: nothing for a window that reset before settlement, the cost otherwise, less what
/// the entry reserved.
fn settled_stamps(
    reservation: &BudgetReservation,
    step: &CounterStep,
    now: Moment,
) -> Vec<Option<f64>> {
    let actual = step.response_cost.unwrap_or(0.0);
    let default_reserved = reservation.reserved_cost.unwrap_or(0.0);
    reservation
        .entries
        .as_deref()
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let entry = entry.as_ref()?;
            entry.counter_key.as_ref()?;
            let reset = step
                .window_resets
                .get(index)
                .and_then(Option::as_deref)
                .and_then(|reset| window::parse_reset(reset).ok().flatten());
            let settled = match reset {
                Some(reset) if reset <= now => 0.0,
                _ => actual,
            };
            Some(settled - entry.reserved_cost.unwrap_or(default_reserved))
        })
        .collect()
}

/// The reservation's adjustments still to be written, and the counters it holds; both empty when
/// the reservation could not be reconciled and its counters were dropped.
#[derive(Default)]
struct Reconciled {
    reserved: Vec<String>,
    pending: Vec<(SpendCounter, Number)>,
}

fn reserved_keys(reservation: &BudgetReservation) -> Vec<String> {
    reservation
        .entries
        .as_deref()
        .unwrap_or_default()
        .iter()
        .flatten()
        .filter_map(|entry| entry.counter_key.clone())
        .collect()
}

async fn drop_reserved(store: &SpendCounters, reserved: &[String]) {
    let counters: Vec<SpendCounter> = reserved
        .iter()
        .filter_map(|key| SpendCounter::parse(key).ok())
        .collect();
    store.invalidate(&counters).await;
}

/// `reseed_reserved_counter`: raise a counter the reconcile found inconsistent to its stored spend,
/// then charge it the settled cost.
async fn reseed_reserved(
    store: &SpendCounters,
    host: &Host,
    reseeds: &[engine::Reseed],
) -> Result<(), Failure> {
    let counters = reseeds
        .iter()
        .map(|reseed| SpendCounter::parse(&reseed.counter_key))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| Arc::new(CounterError::Counter(error)))?;
    let asked: Vec<usize> = (0..counters.len())
        .filter(|index| has_stored_spend(&counters[*index]))
        .collect();
    if asked.is_empty() {
        return Ok(());
    }
    let stored = host
        .call(StoredSpend {
            counters: asked
                .iter()
                .map(|index| StoredCounter::Entity {
                    counter_key: reseeds[*index].counter_key.clone(),
                })
                .collect(),
        })
        .await
        .map_err(host_failure)?;
    for (index, spend) in asked.into_iter().zip(stored) {
        let Some(spend) = spend else { continue };
        let counter = &counters[index];
        if let Err(error) = store.set_max(counter, Number::Float(spend)).await {
            tracing::debug!(%error, counter = %counter.key(), "Unable to repair stale spend counter in Redis");
        }
        if reseeds[index].actual_cost > 0.0 {
            increment_one(store, counter, reseeds[index].actual_cost).await?;
        }
    }
    Ok(())
}

/// `_reconcile_budget_reservation_for_counter_update`: settle the open reservation against the
/// counters it holds, with its adjustments left for the pipeline. On failure the reserved counters
/// are dropped and the charge falls back to incrementing them directly.
async fn reconcile(
    store: &SpendCounters,
    host: &Host,
    reservation: &BudgetReservation,
    step: &CounterStep,
    reads: &HashMap<String, Read>,
    outcome: &mut CounterOutcome,
) -> Reconciled {
    let reserved = reserved_keys(reservation);
    let observed: Vec<Option<f64>> = reservation
        .entries
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|entry| {
            entry
                .as_ref()
                .and_then(|entry| entry.counter_key.as_ref())
                .and_then(|key| reads.get(key))
                .and_then(|read| read.value)
        })
        .collect();
    let settled = engine::settle(
        store,
        &Settle {
            reservation,
            window_resets: &step.window_resets,
            observed: &observed,
            actual: step.response_cost.unwrap_or(0.0),
            apply_consistent: false,
        },
        &Moment::now(),
    )
    .await;
    let settled = match settled {
        Ok(settlement) => match reseed_reserved(store, host, &settlement.reseed).await {
            Ok(()) => Ok(settlement),
            Err(error) => Err(error.to_string()),
        },
        Err(error) => Err(error.to_string()),
    };
    match settled {
        Ok(settlement) => {
            for (stamp, adjustment) in outcome.stamps.iter_mut().zip(settlement.adjustments) {
                if adjustment.is_some() {
                    *stamp = adjustment;
                }
            }
            let pending = settlement
                .pending
                .into_iter()
                .filter_map(|pending| {
                    SpendCounter::parse(&pending.counter_key)
                        .ok()
                        .map(|counter| (counter, Number::Float(pending.increment)))
                })
                .collect();
            Reconciled { reserved, pending }
        }
        Err(error) => {
            tracing::warn!(
                %error,
                "Failed to reconcile budget reservation after persisted spend; invalidating reserved counters and falling back to direct increment"
            );
            drop_reserved(store, &reserved).await;
            Reconciled::default()
        }
    }
}

/// One counter a task warms and charges.
#[derive(Clone, Debug)]
enum Unit {
    /// A lifetime counter, seeded from its row or, without one, from the auth cache.
    Counter {
        counter: SpendCounter,
        sources: Vec<String>,
    },
    /// A budget window, seeded from its window spend. A strict (customer) window is skipped when
    /// only this pod's memory could answer for it.
    Window {
        counter: SpendCounter,
        label: String,
        entity_id: String,
        duration: Option<String>,
        start: String,
        strict: bool,
    },
    /// A committed twin, charged only once it holds the stored spend of the counter it twins.
    Committed {
        base: SpendCounter,
        twin: SpendCounter,
    },
}

impl Unit {
    fn warmed(&self) -> &SpendCounter {
        match self {
            Self::Counter { counter, .. } | Self::Window { counter, .. } => counter,
            Self::Committed { twin, .. } => twin,
        }
    }

    /// A failed window inside an entity's windows is kept beside its siblings; any other failure
    /// fails the task.
    fn fails_task(&self) -> bool {
        !matches!(self, Self::Window { strict: false, .. })
    }
}

/// What one unit came to: charge it, skip it, or a failure.
enum Warmed {
    Charge,
    Skip,
    Failed(Failure),
}

/// A task's units, or the failure that kept it from naming them. `uncached` names an entity whose
/// windows wait for its stored row.
#[derive(Default)]
struct TaskUnits {
    units: Vec<usize>,
    uncached: Vec<WindowSource>,
    failure: Option<Failure>,
}

fn window_unit(
    entity: WindowedEntity,
    entity_id: &str,
    window: &BudgetWindow,
    reserved: &[String],
) -> Result<Option<Unit>, Failure> {
    let counter = window_counters(
        entity,
        entity_id,
        std::slice::from_ref(&window.budget_duration),
        reserved,
    )
    .remove(0);
    if counter.reserved {
        return Ok(None);
    }
    let Some(start) = &window.window_start else {
        tracing::warn!(
            counter = counter.counter_key,
            "Skipping spend counter increment for invalid budget window"
        );
        return Ok(None);
    };
    Ok(Some(Unit::Window {
        counter: SpendCounter::parse(&counter.counter_key)
            .map_err(|error| Arc::new(CounterError::Counter(error)))?,
        label: counter.entity_label.to_owned(),
        entity_id: entity_id.to_owned(),
        duration: Some(window.budget_duration.clone()),
        start: start.clone(),
        strict: false,
    }))
}

/// Warms every unit: one batch read, one stored spend call for the cold ones, their seeds run
/// together, then one cached spend call for the counters the database could not answer.
async fn warm(
    store: &SpendCounters,
    host: &Host,
    units: &[Unit],
    reads: &HashMap<String, Read>,
) -> Vec<Warmed> {
    let mut warmed: Vec<Option<Warmed>> = units
        .iter()
        .map(|unit| {
            reads
                .get(&unit.warmed().key())
                .and_then(|read| read.value)
                .map(|_| Warmed::Charge)
        })
        .collect();
    let mut asked = Vec::new();
    let mut fallback = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        if warmed[index].is_some() {
            continue;
        }
        let stored = match unit {
            Unit::Counter { counter, .. } if has_stored_spend(counter) => {
                Some(StoredCounter::Entity {
                    counter_key: counter.key(),
                })
            }
            Unit::Counter { .. } => {
                fallback.push(index);
                None
            }
            Unit::Window {
                counter,
                label,
                entity_id,
                duration,
                start,
                strict,
            } => {
                let authoritative = reads
                    .get(&counter.key())
                    .is_some_and(|read| read.authoritative);
                if *strict && store.has_redis() && !authoritative {
                    tracing::warn!(
                        counter = counter.key(),
                        "Skipping cold spend counter seed because window spend could not be loaded"
                    );
                    warmed[index] = Some(Warmed::Skip);
                    None
                } else {
                    Some(StoredCounter::Window {
                        entity_type: label.clone(),
                        entity_id: entity_id.clone(),
                        window_duration: duration.clone(),
                        window_start: start.clone(),
                    })
                }
            }
            Unit::Committed { base, .. }
                if has_stored_spend(base) || matches!(base, SpendCounter::EndUser(_)) =>
            {
                Some(StoredCounter::Entity {
                    counter_key: base.key(),
                })
            }
            Unit::Committed { .. } => {
                warmed[index] = Some(Warmed::Skip);
                None
            }
        };
        if let Some(stored) = stored {
            asked.push((index, stored));
        }
    }
    if !asked.is_empty() {
        let request = StoredSpend {
            counters: asked.iter().map(|(_, stored)| stored.clone()).collect(),
        };
        match host.call(request).await {
            Ok(spends) => {
                let seeds = asked
                    .iter()
                    .zip(spends.into_iter().chain(std::iter::repeat(None)))
                    .map(|((index, _), spend)| seed(store, &units[*index], spend));
                let seeded = join_all(seeds).await;
                for ((index, _), seeded) in asked.iter().zip(seeded) {
                    match seeded {
                        Seeded::Done(result) => warmed[*index] = Some(result),
                        Seeded::Fallback => fallback.push(*index),
                    }
                }
            }
            Err(error) => {
                let failure = host_failure(error);
                for (index, _) in &asked {
                    warmed[*index] = Some(Warmed::Failed(failure.clone()));
                }
            }
        }
    }
    if !fallback.is_empty() {
        fallback.sort_unstable();
        let sources = fallback
            .iter()
            .map(|index| match &units[*index] {
                Unit::Counter { sources, .. } => sources.clone(),
                _ => Vec::new(),
            })
            .collect();
        match host.call(CachedSpend { sources }).await {
            Ok(spends) => {
                let increments = fallback
                    .iter()
                    .zip(spends.into_iter().chain(std::iter::repeat(0.0)))
                    .map(|(index, spend)| async move {
                        let counter = units[*index].warmed();
                        if spend > 0.0 {
                            increment_one(store, counter, spend).await?;
                        }
                        Ok::<(), Failure>(())
                    });
                for (index, result) in fallback.iter().zip(join_all(increments).await) {
                    warmed[*index] = Some(match result {
                        Ok(()) => Warmed::Charge,
                        Err(failure) => Warmed::Failed(failure),
                    });
                }
            }
            Err(error) => {
                let failure = host_failure(error);
                for index in &fallback {
                    warmed[*index] = Some(Warmed::Failed(failure.clone()));
                }
            }
        }
    }
    warmed
        .into_iter()
        .map(|warmed| warmed.unwrap_or(Warmed::Skip))
        .collect()
}

enum Seeded {
    Done(Warmed),
    Fallback,
}

/// `SpendCounterReseed.coalesced(_window)` and `seed_committed_spend` past their DB read: seed the
/// cold counter with `SET NX` so one pod wins.
async fn seed(store: &SpendCounters, unit: &Unit, spend: Option<f64>) -> Seeded {
    match (unit, spend) {
        (Unit::Counter { .. }, None) => Seeded::Fallback,
        (Unit::Counter { counter, .. }, Some(spend)) => Seeded::Done(
            match store
                .seed(counter, Number::Float(spend), LostSeed::Answer)
                .await
            {
                Ok(_) => Warmed::Charge,
                Err(error) => Warmed::Failed(store_failure(error)),
            },
        ),
        (Unit::Window { counter, .. }, None) => {
            tracing::warn!(
                counter = counter.key(),
                "Skipping cold spend counter seed because window spend could not be loaded"
            );
            Seeded::Done(Warmed::Skip)
        }
        (
            Unit::Window {
                counter, strict, ..
            },
            Some(spend),
        ) => Seeded::Done(
            match store
                .seed(counter, Number::Float(spend), LostSeed::Increment)
                .await
            {
                Ok(_) => Warmed::Charge,
                Err(_) if *strict => Warmed::Skip,
                Err(error) => Warmed::Failed(store_failure(error)),
            },
        ),
        (Unit::Committed { .. }, None) => Seeded::Done(Warmed::Skip),
        (Unit::Committed { twin, .. }, Some(spend)) => Seeded::Done(
            match store
                .seed(twin, Number::Float(spend), LostSeed::Answer)
                .await
            {
                Ok(_) => Warmed::Charge,
                Err(error) => {
                    tracing::debug!(%error, twin = twin.key(), "committed spend seed failed");
                    Warmed::Skip
                }
            },
        ),
    }
}

/// The step's state while it turns the plan into increments.
struct Run<'a> {
    store: &'a Arc<SpendCounters>,
    host: &'a Host,
    step: &'a CounterStep,
    cost: f64,
    reserved: Vec<String>,
}

/// What a set of scopes came to: the increments, the entities whose windows wait for their stored
/// row, and every failure.
#[derive(Default)]
struct Charged {
    pending: Vec<(SpendCounter, Number)>,
    uncached: Vec<WindowSource>,
    failures: Vec<Failure>,
}

impl Run<'_> {
    fn charge(&self) -> Charge {
        Charge {
            token: self.step.token.clone(),
            team_id: self.step.team_id.clone(),
            user_id: self.step.user_id.clone(),
            org_id: self.step.org_id.clone(),
            end_user_id: self.step.end_user_id.clone(),
            project_id: self.step.project_id.clone(),
            tags: self.step.tags.clone(),
            model_access_groups: self.step.model_access_groups.clone(),
            response_cost: self.step.response_cost,
            reserved_counter_keys: self.reserved.clone(),
            committed_counter_keys: committed_counter_keys(
                &self.step.committed_tracked,
                self.step.response_cost,
            ),
            customer_windows: self.step.customer_windows.clone(),
        }
    }

    /// The units of one task, appended to `units`.
    fn task_units(
        &self,
        task: &Task,
        windows: &HashMap<String, Result<Option<Vec<BudgetWindow>>, Failure>>,
        units: &mut Vec<Unit>,
    ) -> TaskUnits {
        let mut named = TaskUnits::default();
        let mut push = |unit: Unit, named: &mut TaskUnits| {
            named.units.push(units.len());
            units.push(unit);
        };
        let parsed = |key: &str| {
            SpendCounter::parse(key).map_err(|error| Arc::new(CounterError::Counter(error)))
        };
        match task {
            Task::Counter {
                counter_key,
                source_cache_keys,
            } => match parsed(counter_key) {
                Ok(counter) => push(
                    Unit::Counter {
                        counter,
                        sources: source_cache_keys.clone(),
                    },
                    &mut named,
                ),
                Err(failure) => named.failure = Some(failure),
            },
            Task::CachedWindows {
                entity_type,
                entity_id,
                cache_key,
            } => match windows.get(cache_key) {
                Some(Ok(Some(windows))) => {
                    for window in windows {
                        match window_unit(*entity_type, entity_id, window, &self.reserved) {
                            Ok(Some(unit)) => push(unit, &mut named),
                            Ok(None) => {}
                            Err(failure) => named.failure = Some(failure),
                        }
                    }
                }
                Some(Ok(None)) | None => named.uncached.push(WindowSource {
                    entity_type: *entity_type,
                    entity_id: entity_id.clone(),
                    cache_key: cache_key.clone(),
                }),
                Some(Err(failure)) => named.failure = Some(failure.clone()),
            },
            Task::CustomerWindow {
                index,
                counter_key,
                entity_label,
                reserved,
            } => {
                let window = &self.step.customer_windows[*index];
                if *reserved {
                    return named;
                }
                if let Some(start) = &window.window_start {
                    match parsed(counter_key) {
                        Ok(counter) => push(
                            Unit::Window {
                                counter,
                                label: entity_label.clone(),
                                entity_id: window.entity_id.clone(),
                                duration: window.window_duration.clone(),
                                start: start.clone(),
                                strict: entity_label.starts_with("customer_"),
                            },
                            &mut named,
                        ),
                        Err(failure) => named.failure = Some(failure),
                    }
                } else {
                    tracing::warn!(
                        counter = counter_key,
                        "Skipping spend counter increment for invalid budget window"
                    );
                }
            }
            Task::Committed { counter_keys } => {
                for base in counter_keys {
                    match parsed(base) {
                        Ok(base) => push(
                            Unit::Committed {
                                twin: SpendCounter::Committed(Box::new(base.clone())),
                                base,
                            },
                            &mut named,
                        ),
                        Err(failure) => named.failure = Some(failure),
                    }
                }
            }
        }
        named
    }

    /// Reads the auth cache's budget windows for every entity that has them.
    async fn cached_windows(
        &self,
        sources: Vec<WindowSource>,
        stored: bool,
    ) -> HashMap<String, Result<Option<Vec<BudgetWindow>>, Failure>> {
        if sources.is_empty() {
            return HashMap::new();
        }
        let keys: Vec<String> = sources.iter().map(|s| s.cache_key.clone()).collect();
        match self
            .host
            .call(BudgetWindows {
                entities: sources,
                stored,
            })
            .await
        {
            Ok(windows) => keys
                .into_iter()
                .zip(windows.into_iter().chain(std::iter::repeat(None)))
                .map(|(key, windows)| (key, Ok(windows)))
                .collect(),
            Err(error) => {
                let failure = host_failure(error);
                keys.into_iter()
                    .map(|key| (key, Err(failure.clone())))
                    .collect()
            }
        }
    }

    /// Charges the scopes: names every unit, warms them, and folds each scope's results by its
    /// failure rule.
    async fn scopes(
        &self,
        scopes: &[(bool, Vec<Task>)],
        windows: &HashMap<String, Result<Option<Vec<BudgetWindow>>, Failure>>,
        reads: Option<&HashMap<String, Read>>,
    ) -> Charged {
        let mut units = Vec::new();
        let named: Vec<(bool, Vec<TaskUnits>)> = scopes
            .iter()
            .map(|(concurrent, tasks)| {
                (
                    *concurrent,
                    tasks
                        .iter()
                        .map(|task| self.task_units(task, windows, &mut units))
                        .collect(),
                )
            })
            .collect();
        let reads = match reads {
            Some(reads) => reads.clone(),
            None => {
                let counters: Vec<SpendCounter> =
                    units.iter().map(|unit| unit.warmed().clone()).collect();
                read_all(self.store, &counters).await
            }
        };
        let warmed = warm(self.store, self.host, &units, &reads).await;
        let mut charged = Charged::default();
        for (concurrent, tasks) in named {
            let mut scope = Charged::default();
            let mut scope_failure = None;
            for task in tasks {
                let mut task_failure = task.failure;
                let mut task_pending = Vec::new();
                for index in task.units {
                    match &warmed[index] {
                        Warmed::Charge => task_pending
                            .push((units[index].warmed().clone(), Number::Float(self.cost))),
                        Warmed::Skip => {}
                        Warmed::Failed(failure) if units[index].fails_task() => {
                            task_failure.get_or_insert_with(|| failure.clone());
                        }
                        Warmed::Failed(failure) => scope.failures.push(failure.clone()),
                    }
                }
                match task_failure {
                    Some(failure) if concurrent => scope.failures.push(failure),
                    Some(failure) => {
                        scope_failure = Some(failure);
                        break;
                    }
                    None => {
                        scope.pending.extend(task_pending);
                        scope.uncached.extend(task.uncached);
                    }
                }
            }
            if let Some(failure) = scope_failure {
                charged.failures.push(failure);
                continue;
            }
            charged.pending.extend(scope.pending);
            charged.uncached.extend(scope.uncached);
            charged.failures.extend(scope.failures);
        }
        charged
    }
}

/// Runs the post-call counter step. Every store write that can run together does, and the proxy is
/// asked for what only it knows through `host`. The increments ride the request's post-call
/// pipeline when `request` is given and there is Redis.
pub async fn charge_counters(
    store: &Arc<SpendCounters>,
    host: &Host,
    request: Option<&RequestRedis>,
    step: &CounterStep,
    reservation: Option<&BudgetReservation>,
) -> CounterOutcome {
    let entries = reservation
        .and_then(|reservation| reservation.entries.as_ref())
        .map_or(0, Vec::len);
    let mut outcome = CounterOutcome {
        stamps: vec![None; entries],
        ..CounterOutcome::default()
    };
    let cost = charged(step.response_cost);
    if let Some(cost) = cost
        && !step.customer_scopes.is_empty()
        && let Err(failure) = settle_customer_spend(store, &step.customer_scopes, cost).await
    {
        outcome.error = Some(failure);
        return outcome;
    }
    let open = reservation.filter(|reservation| reservation.finalized != Some(true));
    let mut run = Run {
        store,
        host,
        step,
        cost: cost.unwrap_or(0.0),
        reserved: Vec::new(),
    };
    let plan_all = plan(&run.charge());
    let sources: Vec<WindowSource> = match cost {
        Some(_) => plan_all
            .scopes
            .iter()
            .flat_map(|scope| &scope.tasks)
            .filter_map(|task| match task {
                Task::CachedWindows {
                    entity_type,
                    entity_id,
                    cache_key,
                } => Some(WindowSource {
                    entity_type: *entity_type,
                    entity_id: entity_id.clone(),
                    cache_key: cache_key.clone(),
                }),
                _ => None,
            })
            .collect(),
        None => Vec::new(),
    };
    let windows = run.cached_windows(sources, false).await;
    let scopes: Vec<(bool, Vec<Task>)> = plan_all
        .scopes
        .iter()
        .map(|scope| (scope.concurrent, scope.tasks.clone()))
        .collect();
    let mut every = Vec::new();
    if cost.is_some() {
        for (_, tasks) in &scopes {
            for task in tasks {
                run.task_units(task, &windows, &mut every);
            }
        }
    }
    let mut counters: Vec<SpendCounter> = open
        .map(reserved_keys)
        .unwrap_or_default()
        .iter()
        .filter_map(|key| SpendCounter::parse(key).ok())
        .collect();
    counters.extend(every.iter().map(|unit| unit.warmed().clone()));
    let mut seen = std::collections::HashSet::new();
    counters.retain(|counter| seen.insert(counter.key()));
    let reads = read_all(store, &counters).await;
    let reconciled = match open {
        Some(open) => reconcile(store, host, open, step, &reads, &mut outcome).await,
        None => Reconciled::default(),
    };
    run.reserved = reconciled.reserved;
    if cost.is_none() {
        if let Err(failure) = apply(store, request, &reconciled.pending).await {
            outcome.error = Some(failure);
            return outcome;
        }
        return finish(outcome, reservation, step, Vec::new());
    }
    let planned: Vec<(bool, Vec<Task>)> = plan(&run.charge())
        .scopes
        .into_iter()
        .map(|scope| (scope.concurrent, scope.tasks))
        .collect();
    let charged = run.scopes(&planned, &windows, Some(&reads)).await;
    let mut pending = reconciled.pending;
    pending.extend(charged.pending);
    match apply(store, request, &pending).await {
        Ok(settlement) => outcome.committed_settlement = settlement,
        Err(failure) => {
            outcome.error = Some(failure);
            return outcome;
        }
    }
    let mut failures = charged.failures;
    if !charged.uncached.is_empty() {
        let stored = run.cached_windows(charged.uncached.clone(), true).await;
        let deferred: Vec<(bool, Vec<Task>)> = vec![(
            true,
            charged
                .uncached
                .iter()
                .map(|source| Task::CachedWindows {
                    entity_type: source.entity_type,
                    entity_id: source.entity_id.clone(),
                    cache_key: source.cache_key.clone(),
                })
                .collect(),
        )];
        let stored_windows: HashMap<String, Result<Option<Vec<BudgetWindow>>, Failure>> = stored
            .into_iter()
            .map(|(key, windows)| {
                (
                    key,
                    windows.map(|windows| Some(windows.unwrap_or_default())),
                )
            })
            .collect();
        let later = run.scopes(&deferred, &stored_windows, None).await;
        failures.extend(later.failures);
        if let Err(failure) = apply(store, request, &later.pending).await {
            outcome.error = Some(failure);
            return outcome;
        }
    }
    finish(outcome, reservation, step, failures)
}

fn finish(
    mut outcome: CounterOutcome,
    reservation: Option<&BudgetReservation>,
    step: &CounterStep,
    failures: Vec<Failure>,
) -> CounterOutcome {
    if let Some(reservation) = reservation {
        outcome.stamps = settled_stamps(reservation, step, Moment::now());
    }
    match failures.into_iter().next() {
        Some(failure) => outcome.error = Some(failure),
        None => outcome.finalized = reservation.is_some(),
    }
    outcome
}
