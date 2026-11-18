//! The budget work a spend log does after its call: reconciling the reservation admission held,
//! releasing it when the call is not charged, and the post-call counter step.
//!
//! `reconcile` and `release` port `reconcile_budget_reservation` (with `settle_native_reservation`):
//! the reserved counters are read, [`crate::settle`] decides each entry's adjustment, the
//! targets it settles are stamped on the reservation, and a finalized reservation is left alone.
//! The reservation is the caller's own copy, so Rust callers carry it from reconcile to the
//! counter step themselves.
//!
//! The counter step itself is [`crate::Budgets::charge_counters`]. One seam stays a placeholder:
//! [`reseed_reserved_counters`] (`reseed_reserved_counter`, a DB read before the spend write),
//! which only logs; the counter step reseeds a reserved counter it finds cold.

use litellm_callbacks::BudgetReservation;
use litellm_redis_store::{
    Number,
    spend_counters::{SpendCounter, SpendCounters},
};

use crate::{
    Budgets, EngineError, PendingIncrement, Reseed, Settle,
    engine::{self},
    window::Moment,
};

/// What one reconcile left to apply: the counter adjustments the caller writes with its own
/// increments when it did not apply them here.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reconciled {
    pub pending: Vec<PendingIncrement>,
}

fn reserved_counter_keys(reservation: &BudgetReservation) -> Vec<Option<String>> {
    reservation
        .entries
        .iter()
        .flatten()
        .map(|entry| entry.as_ref().and_then(|entry| entry.counter_key.clone()))
        .collect()
}

async fn observed(store: &SpendCounters, keys: &[Option<String>]) -> Vec<Option<f64>> {
    let mut values = Vec::with_capacity(keys.len());
    for key in keys {
        let value = match key.as_deref().map(SpendCounter::parse) {
            Some(Ok(counter)) => store
                .read(&counter)
                .await
                .ok()
                .and_then(|reading| reading.value())
                .map(Number::as_f64),
            _ => None,
        };
        values.push(value);
    }
    values
}

/// `reconcile_budget_reservation(reservation, actual_cost, finalize, apply_consistent)`.
/// `window_resets` holds each entry's `window_reset_at`, which the typed record does not carry.
pub async fn reconcile(
    budgets: &Budgets,
    reservation: &mut BudgetReservation,
    window_resets: &[Option<String>],
    actual_cost: f64,
    finalize: bool,
    apply_consistent: bool,
) -> Result<Reconciled, EngineError> {
    if reservation.finalized == Some(true) {
        return Ok(Reconciled::default());
    }
    let keys = reserved_counter_keys(reservation);
    let observed = observed(budgets.store(), &keys).await;
    let settled = engine::settle(
        budgets.store(),
        &Settle {
            reservation,
            window_resets,
            observed: &observed,
            actual: actual_cost,
            apply_consistent,
        },
        &Moment::now(),
    )
    .await?;
    reseed_reserved_counters(&settled.reseed).await;
    if let Some(entries) = reservation.entries.as_mut() {
        for (entry, target) in entries.iter_mut().zip(&settled.adjustments) {
            if let (Some(entry), Some(target)) = (entry.as_mut(), target) {
                entry.applied_adjustment = Some(*target);
            }
        }
    }
    if finalize {
        reservation.finalized = Some(true);
    }
    Ok(Reconciled {
        pending: settled.pending,
    })
}

/// `release_budget_reservation`: reconcile to nothing spent and finalize.
pub async fn release(
    budgets: &Budgets,
    reservation: &mut BudgetReservation,
    window_resets: &[Option<String>],
) -> Result<(), EngineError> {
    reconcile(budgets, reservation, window_resets, 0.0, true, true)
        .await
        .map(drop)
}

/// `invalidate_budget_reservation_counters`: drop every reserved counter so the next read
/// reseeds it.
pub async fn invalidate(store: &SpendCounters, reservation: &BudgetReservation) {
    let counters: Vec<SpendCounter> = reserved_counter_keys(reservation)
        .into_iter()
        .flatten()
        .filter_map(|key| SpendCounter::parse(&key).ok())
        .collect();
    store.invalidate(&counters).await;
}

/// `release_budget_reservation`, falling back to invalidating the reserved counters and
/// finalizing the reservation when the release fails, as the cost callback does.
pub async fn release_or_invalidate(
    budgets: &Budgets,
    reservation: &mut BudgetReservation,
    window_resets: &[Option<String>],
) {
    if let Err(error) = release(budgets, reservation, window_resets).await {
        tracing::error!(%error, "Failed to release budget reservation");
        invalidate(budgets.store(), reservation).await;
        reservation.finalized = Some(true);
    }
}

/// Placeholder for the reseed of reserved counters a reconcile found inconsistent, which reads
/// the entity's spend from the DB before the spend write.
pub async fn reseed_reserved_counters(reseeds: &[Reseed]) {
    if !reseeds.is_empty() {
        tracing::debug!(
            count = reseeds.len(),
            "reserved counter reseed is not wired in Rust yet"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_callbacks::ReservationEntry;
    use rstest::rstest;

    use super::*;

    const KEY: &str = "spend:key:hashed";

    fn budgets(seed: Option<f64>) -> Budgets {
        let store = Arc::new(SpendCounters::new(None));
        if let Some(value) = seed {
            store
                .set_local(
                    &SpendCounter::parse(KEY).unwrap(),
                    Number::Float(value),
                    None,
                )
                .unwrap();
        }
        Budgets::new(store)
    }

    fn reservation(reserved: f64) -> BudgetReservation {
        BudgetReservation {
            reserved_cost: Some(reserved),
            entries: Some(vec![Some(ReservationEntry {
                counter_key: Some(KEY.to_owned()),
                entity_type: Some("key".to_owned()),
                entity_id: Some("hashed".to_owned()),
                reserved_cost: Some(reserved),
                applied_adjustment: Some(0.0),
                ..ReservationEntry::default()
            })]),
            ..BudgetReservation::default()
        }
    }

    fn value(budgets: &Budgets) -> Option<f64> {
        budgets
            .store()
            .get_local(&SpendCounter::parse(KEY).unwrap())
            .unwrap()
            .map(Number::as_f64)
    }

    #[rstest]
    #[tokio::test]
    async fn a_release_gives_the_reserved_cost_back_and_finalizes() {
        let budgets = budgets(Some(5.0));
        let mut held = reservation(2.0);

        release(&budgets, &mut held, &[None]).await.unwrap();

        assert_eq!(value(&budgets), Some(3.0));
        assert_eq!(held.finalized, Some(true));
        let entry = held.entries.unwrap()[0].clone().unwrap();
        assert_eq!(entry.applied_adjustment, Some(-2.0));
    }

    #[rstest]
    #[tokio::test]
    async fn a_reconcile_before_the_write_leaves_the_adjustment_pending() {
        let budgets = budgets(Some(5.0));
        let mut held = reservation(2.0);

        let reconciled = reconcile(&budgets, &mut held, &[None], 0.5, false, false)
            .await
            .unwrap();

        assert_eq!(value(&budgets), Some(5.0));
        assert_eq!(
            reconciled.pending,
            vec![PendingIncrement {
                counter_key: KEY.to_owned(),
                increment: -1.5
            }]
        );
        assert_eq!(held.finalized, None);
    }

    #[rstest]
    #[tokio::test]
    async fn a_finalized_reservation_is_left_alone() {
        let budgets = budgets(Some(5.0));
        let mut held = BudgetReservation {
            finalized: Some(true),
            ..reservation(2.0)
        };

        release(&budgets, &mut held, &[None]).await.unwrap();

        assert_eq!(value(&budgets), Some(5.0));
    }

    #[rstest]
    #[tokio::test]
    async fn invalidating_drops_the_reserved_counters() {
        let budgets = budgets(Some(5.0));

        invalidate(budgets.store(), &reservation(2.0)).await;

        assert_eq!(value(&budgets), None);
    }
}
