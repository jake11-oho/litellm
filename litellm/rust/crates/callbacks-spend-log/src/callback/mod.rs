//! The proxy's spend log callback on the ledger path, the Rust port of `_ProxyDBLogger`
//! (`litellm/proxy/hooks/proxy_track_cost_callback.py`).
//!
//! It subscribes, awaited, to `attempt_ended` (successful attempts only, since a failed attempt
//! a fallback recovers is charged by the success) and to `post_call_failure`. Per request it
//! keeps the cost callback's order (`_update_database_and_spend_counters`):
//!
//! 1. the gate (`_should_track_cost_callback`) and the deferrals that release or keep the
//!    budget reservation ([`policy`]);
//! 2. the search free tier discount ([`search_free_tier`]) and the reservation reconcile through
//!    `callbacks-budgets`;
//! 3. the row from [`crate::SpendLogRowBuilder`] and its ledger columns;
//! 4. the settle-once claim of a batch or video job's cost row, straight to the ledger store;
//! 5. the enqueue into the Rust spend log writer, woken at once for a Responses turn the next
//!    turn reads back from the database;
//! 6. the post-call counters, through `Budgets::charge_counters`;
//! 7. the host calls for the key soft budget alert, the cached spend, the proxy spend and the
//!    customer alert.
//!
//! A cost it cannot track raises the failed tracking alert, and a batch's costing hears whether
//! the spend write began and whether tracking failed.
//!
//! Python hands over what it alone holds through the `spend_log_input` field
//! ([`hook_input::SpendLogInput`]) and answers the host calls in [`host`].

mod charge;
pub mod hook_input;
pub mod host;
mod policy;
mod search_free_tier;

use std::sync::{Arc, Mutex, PoisonError};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, BudgetReservation, Callback, Field, FieldSet, Hook, HookInput, HookOutput, Host,
    Manifest, Outcome, Selection, Subscription, typed,
};
use litellm_callbacks_budgets::Budgets;
use litellm_spend_log_writer::{LedgerStore, Retention, SpendLogWriter};
use litellm_tracing::TokenRedactor;

use crate::{RowConfig, RowError, SpendLogRowBuilder};

pub use hook_input::SpendLogInput;

pub const NAME: &str = "spend_log";

fn manifest() -> Manifest {
    let reads = FieldSet::of(&[
        Field::SpendLogInput,
        Field::BudgetReservation,
        Field::Outcome,
    ]);
    Manifest {
        selection: Selection {
            activation: Activation::Required,
            opt_out_by: Vec::new(),
            honors_no_log: false,
            ..Selection::named(NAME)
        },
        subscriptions: vec![
            Subscription {
                reads,
                awaited: true,
                ..Subscription::on(Hook::AttemptEnded)
            },
            Subscription {
                reads: FieldSet::of(&[Field::SpendLogInput, Field::BudgetReservation]),
                awaited: true,
                ..Subscription::on(Hook::PostCallFailure)
            },
        ],
        has_request_state: false,
    }
}

/// What the bridge builds the callback from: the writer the proxy flushes, the budget engine
/// over the shared spend counters, the token redaction tables and the metadata retention.
pub struct SpendLogParts<S> {
    pub writer: Arc<SpendLogWriter<S>>,
    pub budgets: Arc<Budgets>,
    pub minimum_custom_key_length: usize,
    pub word_ranges: Vec<(u32, u32)>,
    pub retention: Option<Retention>,
}

/// The row builder for the settings it was last asked to build under.
struct Builders {
    minimum_custom_key_length: usize,
    word_ranges: Vec<(u32, u32)>,
    current: Mutex<Option<(String, Arc<SpendLogRowBuilder>)>>,
}

impl Builders {
    fn for_settings(
        &self,
        settings: &serde_json::Value,
    ) -> Result<Arc<SpendLogRowBuilder>, RowError> {
        let key = settings.to_string();
        let mut current = self.current.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((cached, builder)) = current.as_ref()
            && *cached == key
        {
            return Ok(Arc::clone(builder));
        }
        let config =
            RowConfig::from_settings(&litellm_python_compat::json::from_json(settings.clone()))?;
        let builder = Arc::new(SpendLogRowBuilder::new(
            config,
            TokenRedactor::new(self.minimum_custom_key_length, self.word_ranges.clone()),
        ));
        *current = Some((key, Arc::clone(&builder)));
        Ok(builder)
    }
}

/// The spend log callback. `S` is the writer's ledger store: Postgres in the proxy, a fake in
/// tests.
pub struct SpendLogCallback<S> {
    manifest: Manifest,
    writer: Arc<SpendLogWriter<S>>,
    budgets: Arc<Budgets>,
    builders: Builders,
    retention: Option<Retention>,
}

impl<S: LedgerStore + 'static> SpendLogCallback<S> {
    pub fn new(parts: SpendLogParts<S>) -> Self {
        Self {
            manifest: manifest(),
            writer: parts.writer,
            budgets: parts.budgets,
            builders: Builders {
                minimum_custom_key_length: parts.minimum_custom_key_length,
                word_ranges: parts.word_ranges,
                current: Mutex::new(None),
            },
            retention: parts.retention,
        }
    }

    pub fn writer(&self) -> &Arc<SpendLogWriter<S>> {
        &self.writer
    }

    async fn run(&self, input: &HookInput, host: Host) -> HookOutput {
        let Some(raw) = input.fields.get::<typed::SpendLogInput>()? else {
            return Ok(None);
        };
        let reservation: Option<BudgetReservation> =
            input.fields.get::<typed::BudgetReservation>()?;
        let parsed: SpendLogInput = match serde_json::from_value(raw) {
            Ok(parsed) => parsed,
            Err(error) => {
                tracing::error!(%error, "Error in tracking cost callback - the spend log input does not parse");
                return Ok(None);
            }
        };
        match (input.hook, parsed) {
            (Hook::AttemptEnded, SpendLogInput::Success(success)) => {
                if matches!(
                    input.fields.get::<typed::Outcome>()?,
                    Some(Outcome::Failure | Outcome::FallbackFailure)
                ) {
                    return Ok(None);
                }
                self.on_success(*success, reservation, &host).await;
            }
            (Hook::PostCallFailure, SpendLogInput::Failure(failure)) => {
                self.on_failure(*failure, reservation, &host).await;
            }
            _ => {}
        }
        Ok(None)
    }
}

impl<S: LedgerStore + 'static> Callback for SpendLogCallback<S> {
    type State = ();

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a ()>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(self.run(input, Host::detached()))
    }

    fn handle_hosted<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a ()>,
        host: Host,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(self.run(input, host))
    }

    /// The shutdown drain: write the queue in up to the writer's `drain_passes` passes.
    fn flush(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            match self.writer.drain(self.retention.as_ref()).await {
                Ok(0) => {}
                Ok(remaining) => {
                    tracing::error!(
                        remaining,
                        "Spend tracking - spend logs left queued at shutdown"
                    );
                }
                Err(failure) => {
                    tracing::error!(%failure, "Spend tracking - the shutdown spend log drain failed");
                }
            }
        })
    }
}
