//! The flusher: statement splitting, poison-row isolation and retries for a batch of
//! spend logs, as the Python `update_spend_logs` does them.

use std::{future::Future, pin::Pin, time::Duration};

use crate::{
    Error, Retention, SpendLogRow, WriterSettings,
    statement::{jsonified, retain_metadata},
};

const STATEMENT_FRAMING_BYTES: usize = 2;
const ROW_SEPARATOR_BYTES: usize = 2;
const TRANSIENT_SQLSTATES: [&str; 5] = ["57P01", "57P02", "57P03", "57P05", "40P01"];
const REJECTED_ROW_SQLSTATE_CLASSES: [&str; 3] = ["22", "23", "54"];

/// Where a statement's rows are committed: Postgres in production, a fake in tests.
pub trait LedgerStore: Send + Sync {
    fn insert_with_deliveries(
        &self,
        rows: &[SpendLogRow],
    ) -> impl Future<Output = Result<usize, Error>> + Send;
}

/// A flush that failed, and whether its rows must go back on the queue.
#[derive(Debug, thiserror::Error)]
#[error("{source}")]
pub struct WriteFailure {
    #[source]
    pub source: WriteError,
    pub requeue: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error(transparent)]
    Store(#[from] Error),
    #[error(transparent)]
    Metadata(#[from] crate::statement::InvalidMetadata),
}

/// Failures worth retrying in the same flush: the database dropped or refused the
/// connection, is restarting, or picked this statement as a deadlock victim.
pub fn is_transient(error: &Error) -> bool {
    match error {
        Error::Database(litellm_db::Error::Query(
            sqlx::Error::Io(_) | sqlx::Error::Tls(_) | sqlx::Error::PoolClosed,
        ))
        | Error::Database(litellm_db::Error::PoolUnreachable) => true,
        error => error.sqlstate().is_some_and(|code| {
            TRANSIENT_SQLSTATES.contains(&code.as_str()) || code.starts_with("08")
        }),
    }
}

/// Failures caused by the rows themselves, which isolation narrows to the bad row.
pub fn is_rejected_row(error: &Error) -> bool {
    match error {
        Error::RejectedRow(_) | Error::MissingRequestId | Error::InvalidJsonColumn { .. } => true,
        error => error.sqlstate().is_some_and(|code| {
            REJECTED_ROW_SQLSTATE_CLASSES.contains(&code.get(..2).unwrap_or_default())
        }),
    }
}

fn is_database_failure(error: &Error) -> bool {
    !matches!(error, Error::InvalidDelivery)
}

/// Splits rows of the given sizes into statements of at most `max_rows` rows and
/// `max_bytes` bytes, counting the array framing; a row over the budget goes alone.
pub fn write_batches(sizes: &[usize], max_bytes: usize, max_rows: usize) -> Vec<(usize, usize)> {
    let mut batches = Vec::new();
    let mut start = 0;
    while start < sizes.len() {
        let mut end = start + 1;
        let mut used = STATEMENT_FRAMING_BYTES + sizes[start];
        while end < sizes.len()
            && end - start < max_rows
            && used + ROW_SEPARATOR_BYTES + sizes[end] <= max_bytes
        {
            used += ROW_SEPARATOR_BYTES + sizes[end];
            end += 1;
        }
        batches.push((start, end));
        start = end;
    }
    batches
}

/// Writes a batch of rows: chunks of `statement_chunk_rows`, statements within the
/// byte and row budgets, bisection around rows Postgres rejects, and up to
/// `write_retries` retries with `1 << attempt` second sleeps on transient failures.
pub struct SpendLogFlusher<S> {
    store: S,
    settings: WriterSettings,
}

type Isolation<'a> = Pin<Box<dyn Future<Output = Result<usize, Error>> + Send + 'a>>;

impl<S: LedgerStore> SpendLogFlusher<S> {
    pub fn new(store: S, settings: WriterSettings) -> Self {
        Self { store, settings }
    }

    pub fn settings(&self) -> &WriterSettings {
        &self.settings
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    /// Writes `rows`, retrying transient failures; a failure says whether the rows
    /// must be requeued.
    pub async fn write(
        &self,
        rows: &[SpendLogRow],
        retention: Option<&Retention>,
    ) -> Result<(), WriteFailure> {
        if !rows.is_empty() {
            tracing::info!(
                "Spend tracking - processing {} spend logs for DB write",
                rows.len()
            );
        }
        let rows_to_write = rows
            .iter()
            .map(|row| retain_metadata(row, retention, self.settings.max_depth))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| WriteFailure {
                source: error.into(),
                requeue: false,
            })?;
        let retries = self.settings.write_retries;
        for attempt in 0..=retries {
            let Err(error) = self.write_once(&rows_to_write).await else {
                return Ok(());
            };
            if !is_transient(&error) {
                let requeue = is_database_failure(&error);
                if requeue {
                    tracing::warn!(
                        "Spend tracking - DB error writing spend logs, requeued {} rows for the next flush. error={}",
                        rows.len(),
                        error
                    );
                }
                return Err(WriteFailure {
                    source: error.into(),
                    requeue,
                });
            }
            tracing::warn!(
                "Spend tracking - transient DB error writing spend logs, retry {}/{}. logs_count={}, error={}",
                attempt + 1,
                retries,
                rows.len(),
                error
            );
            if attempt >= retries {
                return Err(WriteFailure {
                    source: error.into(),
                    requeue: true,
                });
            }
            tokio::time::sleep(Duration::from_secs(1 << attempt)).await;
        }
        Ok(())
    }

    async fn write_once(&self, rows: &[SpendLogRow]) -> Result<(), Error> {
        for chunk in rows.chunks(self.settings.statement_chunk_rows.max(1)) {
            let statement_rows: Vec<SpendLogRow> = chunk.iter().map(jsonified).collect();
            let sizes: Vec<usize> = statement_rows.iter().map(SpendLogRow::bytes).collect();
            let mut budget = self.settings.isolation_failure_budget;
            for (start, end) in write_batches(
                &sizes,
                self.settings.write_batch_max_bytes,
                self.settings.write_batch_max_rows,
            ) {
                budget = self.isolate(&statement_rows[start..end], budget).await?;
            }
            tracing::debug!("Flushed {} logs to the DB.", chunk.len());
        }
        Ok(())
    }

    async fn keep_charges_of_dropped_logs(&self, rows: &[SpendLogRow]) -> Result<(), Error> {
        let charged: Vec<SpendLogRow> = rows
            .iter()
            .filter(|row| row.carries_a_charge())
            .map(SpendLogRow::accounting_only)
            .collect();
        if charged.is_empty() {
            return Ok(());
        }
        match self.store.insert_with_deliveries(&charged).await {
            Ok(_) => Ok(()),
            Err(error) if is_rejected_row(&error) => {
                tracing::error!(
                    "Spend tracking - could not keep the charges of {} dropped spend log rows. error={}",
                    charged.len(),
                    error
                );
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn isolate<'a>(&'a self, rows: &'a [SpendLogRow], failure_budget: usize) -> Isolation<'a> {
        Box::pin(async move {
            let error = match self.store.insert_with_deliveries(rows).await {
                Ok(_) => return Ok(failure_budget),
                Err(error) => error,
            };
            if !is_rejected_row(&error) {
                return Err(error);
            }
            let budget_left = failure_budget.saturating_sub(1);
            if rows.len() == 1 {
                tracing::error!(
                    "Spend tracking - dropping spend log row Postgres rejected. request_id={} error={}",
                    rows[0]
                        .get("request_id")
                        .map(litellm_python_compat::repr::to_str)
                        .unwrap_or_else(|| "None".to_owned()),
                    error
                );
                self.keep_charges_of_dropped_logs(rows).await?;
                return Ok(budget_left);
            }
            if budget_left == 0 {
                tracing::error!(
                    "Spend tracking - dropping {} spend log rows without per-row isolation; isolation failure budget exhausted for this flush",
                    rows.len()
                );
                self.keep_charges_of_dropped_logs(rows).await?;
                return Ok(0);
            }
            let mid = rows.len() / 2;
            let remaining = self.isolate(&rows[..mid], budget_left).await?;
            if remaining == 0 {
                tracing::error!(
                    "Spend tracking - dropping {} spend log rows without per-row isolation; isolation failure budget exhausted for this flush",
                    rows.len() - mid
                );
                self.keep_charges_of_dropped_logs(&rows[mid..]).await?;
                return Ok(0);
            }
            self.isolate(&rows[mid..], remaining).await
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use rstest::rstest;

    use super::{SpendLogFlusher, is_rejected_row, is_transient, write_batches};
    use crate::{
        Error, SpendLogRow, WriterSettings,
        test_support::{FakeStore, connect_error, database, request_ids, row, rows, sqlstate},
    };

    fn flusher(store: FakeStore) -> SpendLogFlusher<FakeStore> {
        SpendLogFlusher::new(store, WriterSettings::default())
    }

    fn statements_for(rows: usize) -> usize {
        (0..rows)
            .step_by(1000)
            .map(|start| (rows - start).min(1000).div_ceil(100))
            .sum()
    }

    #[rstest]
    #[case::connect_error(connect_error())]
    #[case::server_terminated(sqlstate("57P01"))]
    #[case::deadlock(sqlstate("40P01"))]
    #[tokio::test(start_paused = true)]
    async fn retries_transient_errors_until_one_attempt_succeeds(#[case] error: Error) {
        let template = Arc::new(Mutex::new(VecDeque::from([error])));
        let failures = Arc::new(Mutex::new(0usize));
        let flusher = flusher(FakeStore::new(Box::new(move |_, _| {
            let mut failures = failures.lock().expect("failures lock");
            *failures += 1;
            (*failures <= 3).then(|| {
                template.lock().expect("template lock").front().map_or_else(
                    connect_error,
                    |error| match error.sqlstate() {
                        Some(code) => sqlstate(Box::leak(code.into_boxed_str())),
                        None => connect_error(),
                    },
                )
            })
        })));
        let started = tokio::time::Instant::now();

        flusher
            .write(&rows(2), None)
            .await
            .expect("fourth attempt writes");

        assert_eq!(flusher.store.calls().statements.len(), 4);
        assert_eq!(started.elapsed(), Duration::from_secs(1 + 2 + 4));
    }

    fn pool_outage() -> Error {
        Error::Database(litellm_db::Error::PoolUnreachable)
    }

    #[rstest]
    #[case::connect_error(connect_error)]
    #[case::pool_outage(pool_outage)]
    #[tokio::test(start_paused = true)]
    async fn gives_up_after_three_retries_and_asks_for_a_requeue(#[case] error: fn() -> Error) {
        let flusher = flusher(FakeStore::new(Box::new(move |_, _| Some(error()))));
        let started = tokio::time::Instant::now();

        let failure = flusher
            .write(&rows(2), None)
            .await
            .expect_err("never writes");

        assert!(failure.requeue);
        assert_eq!(flusher.store.calls().statements.len(), 4);
        assert_eq!(started.elapsed(), Duration::from_secs(1 + 2 + 4));
    }

    #[rstest]
    #[case::systemic_database_error(sqlstate("42P01"), true)]
    #[case::capacity(sqlstate("53300"), true)]
    #[case::full_pool(database(sqlx::Error::PoolTimedOut), true)]
    #[case::invalid_payload(Error::InvalidDelivery, false)]
    #[tokio::test(start_paused = true)]
    async fn a_non_transient_error_fails_on_the_first_attempt(
        #[case] error: Error,
        #[case] requeue: bool,
    ) {
        let error = Mutex::new(Some(error));
        let flusher = flusher(FakeStore::new(Box::new(move |_, _| {
            error.lock().expect("error lock").take()
        })));

        let failure = flusher.write(&rows(2), None).await.expect_err("fails once");

        assert_eq!(failure.requeue, requeue);
        assert_eq!(flusher.store.calls().statements.len(), 1);
    }

    #[rstest]
    #[case::one_and_a_half_chunks(1500)]
    #[case::exact_chunks(2000)]
    #[tokio::test]
    async fn splits_the_flush_into_bounded_statements_in_order(#[case] count: usize) {
        let flusher = flusher(FakeStore::new(Box::new(|_, _| None)));

        flusher.write(&rows(count), None).await.expect("writes");

        let calls = flusher.store.calls();
        assert_eq!(calls.statements.len(), statements_for(count));
        assert!(
            calls
                .statements
                .iter()
                .all(|statement| statement.len() <= 100)
        );
        assert_eq!(
            calls.written,
            (0..count)
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    #[case::second_statement_fails_once()]
    #[tokio::test(start_paused = true)]
    async fn a_transient_failure_replays_the_whole_flush() {
        let flusher = flusher(FakeStore::new(Box::new(|_, call| {
            (call == 2).then(connect_error)
        })));

        flusher.write(&rows(4000), None).await.expect("writes");

        let calls = flusher.store.calls();
        assert_eq!(calls.statements.len(), 2 + statements_for(4000));
        let mut written = calls.written.clone();
        written.sort();
        written.dedup();
        assert_eq!(written.len(), 4000);
    }

    #[rstest]
    #[case::data_exception("22P02")]
    #[case::not_null("23502")]
    #[case::too_large("54000")]
    #[tokio::test]
    async fn a_rejected_row_is_dropped_alone(#[case] code: &'static str) {
        let flusher = flusher(FakeStore::new(Box::new(move |rows, _| {
            (!rows.iter().all(SpendLogRow::log_disabled)
                && request_ids(rows).iter().any(|id| id == "rejected"))
            .then(|| sqlstate(code))
        })));
        let batch = [
            row(r#"{"request_id": "good-1"}"#),
            row(r#"{"request_id": "rejected", "accounting_payload": {"spend": 1.0}}"#),
            row(r#"{"request_id": "good-2"}"#),
        ];

        flusher.write(&batch, None).await.expect("writes the rest");

        let calls = flusher.store.calls();
        assert_eq!(calls.written, ["good-1", "good-2"]);
        assert_eq!(calls.charges_kept, ["rejected"]);
    }

    #[rstest]
    #[case::undefined_table("42P01")]
    #[case::deadlock("40P01")]
    #[tokio::test]
    async fn a_systemic_failure_is_not_isolated(#[case] code: &'static str) {
        let flusher = flusher(FakeStore::new(Box::new(move |_, _| Some(sqlstate(code)))));

        let error = flusher
            .isolate(&rows(2), 10)
            .await
            .expect_err("systemic failures propagate");

        assert_eq!(error.sqlstate().as_deref(), Some(code));
        assert!(flusher.store.calls().written.is_empty());
    }

    #[rstest]
    #[case::transient_rescue_failure("40P01", Err(()))]
    #[case::rejected_rescue_failure("22P02", Ok(9))]
    #[tokio::test]
    async fn a_failed_charge_rescue_is_logged_unless_it_can_be_retried(
        #[case] rescue_code: &'static str,
        #[case] expected: Result<usize, ()>,
    ) {
        let flusher = flusher(FakeStore::new(Box::new(move |rows, _| {
            if rows.iter().all(SpendLogRow::log_disabled) {
                Some(sqlstate(rescue_code))
            } else {
                Some(sqlstate("22P02"))
            }
        })));
        let batch = [row(
            r#"{"request_id": "rejected", "accounting_payload": {"spend": 1.0}}"#,
        )];

        assert_eq!(flusher.isolate(&batch, 10).await.map_err(|_| ()), expected);
    }

    #[rstest]
    #[case::unparseable_json_column()]
    #[tokio::test]
    async fn a_row_the_statement_rejects_is_dropped_alone_and_keeps_its_charge() {
        let mut store = FakeStore::new(Box::new(|_, _| None));
        store.validate = true;
        let flusher = flusher(store);
        let batch = [
            row(
                r#"{"request_id": "good", "call_type": "acompletion", "startTime": "2026-11-02 12:00:00", "endTime": "2026-11-02 12:00:00", "messages": "[]", "accounting_payload": {}}"#,
            ),
            row(
                r#"{"request_id": "unparseable", "call_type": "acompletion", "messages": "not json", "accounting_payload": {"spend": 1.0}}"#,
            ),
        ];

        assert!(flusher.isolate(&batch, 10).await.expect("isolates") < 10);
        let calls = flusher.store.calls();
        assert_eq!(calls.written, ["good"]);
        assert_eq!(calls.charges_kept, ["unparseable"]);
    }

    #[rstest]
    #[case::budget_spent_on_the_first_half(1)]
    #[tokio::test]
    async fn an_exhausted_budget_drops_the_rest_and_keeps_their_charges(#[case] budget: usize) {
        let flusher = flusher(FakeStore::new(Box::new(|rows, _| {
            (!rows.iter().all(SpendLogRow::log_disabled)).then(|| sqlstate("22P02"))
        })));
        let batch = [
            row(r#"{"request_id": "a", "accounting_payload": {}}"#),
            row(r#"{"request_id": "b", "accounting_payload": {}}"#),
        ];

        assert_eq!(flusher.isolate(&batch, budget).await.expect("drops"), 0);
        assert_eq!(flusher.store.calls().charges_kept, ["a", "b"]);
    }

    #[rstest]
    #[case::row_budget(&[10, 10, 10], 1000, 2, vec![(0, 2), (2, 3)])]
    #[case::byte_budget_counts_framing(&[10, 10, 10], 2 + 10 + 2 + 10, 100, vec![(0, 2), (2, 3)])]
    #[case::oversized_row_alone(&[5000, 10], 100, 100, vec![(0, 1), (1, 2)])]
    fn statements_respect_both_budgets(
        #[case] sizes: &[usize],
        #[case] max_bytes: usize,
        #[case] max_rows: usize,
        #[case] expected: Vec<(usize, usize)>,
    ) {
        assert_eq!(write_batches(sizes, max_bytes, max_rows), expected);
    }

    #[rstest]
    #[case::io(connect_error(), true, false)]
    #[case::pool_full(database(sqlx::Error::PoolTimedOut), false, false)]
    #[case::pool_outage(Error::Database(litellm_db::Error::PoolUnreachable), true, false)]
    #[case::lifecycle(sqlstate("57P03"), true, false)]
    #[case::connection_exception(sqlstate("08006"), true, false)]
    #[case::unique_violation(sqlstate("23505"), false, true)]
    #[case::rejected_encoding(Error::MissingRequestId, false, true)]
    #[case::transaction_timeout(
        Error::Database(litellm_db::Error::TransactionTimedOut),
        false,
        false
    )]
    fn classifies_like_the_python_flush(
        #[case] error: Error,
        #[case] transient: bool,
        #[case] rejected: bool,
    ) {
        assert_eq!(is_transient(&error), transient);
        assert_eq!(is_rejected_row(&error), rejected);
    }
}
