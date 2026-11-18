//! The spend log writer: a byte-bounded queue of request logs and the flusher that
//! commits them to `LiteLLM_SpendLogs` and `LiteLLM_SpendLogDelivery` through the
//! spend log repository's ledger transaction (`litellm-repositories`).
//!
//! It ports the ledger-on half of the Python proxy's spend log path (`enqueue_spend_logs`,
//! `ProxyUpdateSpend.update_spend_logs`, `SpendLogsRepository.insert_with_deliveries`):
//! the same byte budget, statement sizes, poison-row isolation, retries and requeue
//! rules. The Python proxy still decides when to flush.

mod error;
mod flusher;
mod queue;
mod row;
mod statement;
mod store;
#[cfg(test)]
mod test_support;

use std::time::Duration;

use tokio::sync::{Mutex, Notify};

pub use error::Error;
pub use flusher::{
    LedgerStore, SpendLogFlusher, WriteError, WriteFailure, is_rejected_row, is_transient,
    write_batches,
};
pub use queue::{Enqueued, SpendLogQueue};
pub use row::{RowError, SpendLogRow};
pub use statement::{
    InvalidMetadata, LedgerRequest, Retention, jsonified, ledger_request, retain_metadata,
};
pub use store::PgLedgerStore;

/// The writer's limits and timings. The defaults are the Python proxy's constants;
/// the proxy passes the ones it reads from env vars.
#[derive(Clone, Debug, PartialEq)]
pub struct WriterSettings {
    pub queue_max_bytes: usize,
    pub write_batch_max_bytes: usize,
    pub write_batch_max_rows: usize,
    pub queue_size_threshold: usize,
    pub poll_interval: Duration,
    pub max_backoff: Duration,
    pub backoff_multiplier: f64,
    pub rows_per_pass: usize,
    pub statement_chunk_rows: usize,
    pub isolation_failure_budget: usize,
    pub write_retries: u32,
    pub drain_passes: usize,
    pub transaction_timeout: Duration,
    pub max_depth: usize,
}

impl Default for WriterSettings {
    fn default() -> Self {
        Self {
            queue_max_bytes: 64_000_000,
            write_batch_max_bytes: 2_000_000,
            write_batch_max_rows: 100,
            queue_size_threshold: 100,
            poll_interval: Duration::from_secs(2),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 1.5,
            rows_per_pass: 10_000,
            statement_chunk_rows: 1000,
            isolation_failure_budget: 256,
            write_retries: 3,
            drain_passes: 20,
            transaction_timeout: Duration::from_secs(60),
            max_depth: 100,
        }
    }
}

impl WriterSettings {
    fn backed_off(&self, current: Duration) -> Duration {
        current
            .mul_f64(self.backoff_multiplier)
            .min(self.max_backoff)
    }

    /// The monitor's next wait: back to the base poll once `queued` reaches the
    /// threshold, otherwise the current wait backed off up to the cap.
    pub fn next_poll_interval(&self, current: Duration, queued: usize) -> Duration {
        if queued >= self.queue_size_threshold {
            self.poll_interval
        } else {
            self.backed_off(current)
        }
    }
}

/// The queue and the flusher together, with the lock that keeps one write in flight.
/// A failed pass puts its rows back at the head of the queue when the failure says
/// they were not written.
pub struct SpendLogWriter<S> {
    queue: SpendLogQueue,
    flusher: SpendLogFlusher<S>,
    flush_requested: Notify,
    write_lock: Mutex<()>,
}

impl<S: LedgerStore> SpendLogWriter<S> {
    pub fn new(flusher: SpendLogFlusher<S>) -> Self {
        Self {
            queue: SpendLogQueue::default(),
            flusher,
            flush_requested: Notify::new(),
            write_lock: Mutex::new(()),
        }
    }

    pub fn settings(&self) -> &WriterSettings {
        self.flusher.settings()
    }

    pub fn queue(&self) -> &SpendLogQueue {
        &self.queue
    }

    pub fn flusher(&self) -> &SpendLogFlusher<S> {
        &self.flusher
    }

    /// Queues `rows` within `max_bytes`, logging how many of the oldest rows it dropped.
    pub fn enqueue(&self, rows: Vec<SpendLogRow>, at_head: bool, max_bytes: usize) -> Enqueued {
        let outcome = self.queue.enqueue(rows, at_head, max_bytes);
        if outcome.dropped > 0 {
            tracing::error!(
                "Spend tracking - spend log queue is at its {} byte budget; dropped the {} oldest spend logs",
                max_bytes,
                outcome.dropped
            );
        }
        outcome
    }

    pub fn request_flush(&self) {
        self.flush_requested.notify_one();
    }

    async fn flush_pass_locked(
        &self,
        retention: Option<&Retention>,
    ) -> Result<usize, WriteFailure> {
        let rows = self.queue.dequeue(self.settings().rows_per_pass);
        if rows.is_empty() {
            return Ok(0);
        }
        match self.flusher.write(&rows, retention).await {
            Ok(()) => Ok(rows.len()),
            Err(failure) => {
                if failure.requeue {
                    self.enqueue(rows, true, self.settings().queue_max_bytes);
                }
                Err(failure)
            }
        }
    }

    /// Writes one pass of up to `rows_per_pass` rows from the head of the queue.
    pub async fn flush_pass(&self, retention: Option<&Retention>) -> Result<usize, WriteFailure> {
        let _guard = self.write_lock.lock().await;
        self.flush_pass_locked(retention).await
    }

    /// Writes passes until the queue is empty or `drain_passes` ran out, returning how
    /// many rows are still queued.
    pub async fn drain(&self, retention: Option<&Retention>) -> Result<usize, WriteFailure> {
        let _guard = self.write_lock.lock().await;
        for _ in 0..self.settings().drain_passes {
            if self.queue.is_empty() {
                return Ok(0);
            }
            self.flush_pass_locked(retention).await?;
        }
        let remaining = self.queue.len();
        if remaining > 0 {
            tracing::error!(
                "Spend tracking - {} spend log rows still queued after {} drain passes",
                remaining,
                self.settings().drain_passes
            );
        }
        Ok(remaining)
    }

    /// The monitor loop: flushes on its poll or when a flush is requested, and backs
    /// off while the queue is small or writes keep failing.
    pub async fn run(&self, retention: impl Fn() -> Option<Retention>) {
        let settings = self.settings().clone();
        let mut interval = settings.poll_interval;
        loop {
            let queued = self.queue.len();
            if queued > 0 {
                interval = settings.next_poll_interval(interval, queued);
                if let Err(failure) = self.flush_pass(retention().as_ref()).await {
                    tracing::error!("Error in spend logs queue monitor: {}", failure);
                    interval = settings.backed_off(interval);
                    tokio::time::sleep(interval).await;
                    continue;
                }
            } else {
                interval = settings.backed_off(interval);
            }
            if tokio::time::timeout(interval, self.flush_requested.notified())
                .await
                .is_ok()
            {
                interval = settings.poll_interval;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use rstest::rstest;

    use super::{SpendLogFlusher, SpendLogWriter, WriterSettings};
    use crate::{
        Error,
        test_support::{FakeStore, connect_error, rows, sqlstate},
    };

    fn writer(store: FakeStore, settings: WriterSettings) -> Arc<SpendLogWriter<FakeStore>> {
        Arc::new(SpendLogWriter::new(SpendLogFlusher::new(store, settings)))
    }

    #[rstest]
    #[case::at_threshold(100, 2)]
    #[case::below_threshold_backs_off(99, 3)]
    #[case::capped(1, 30)]
    fn polls_like_the_python_monitor(#[case] queued: usize, #[case] expected_seconds: u64) {
        let settings = WriterSettings::default();
        let current = if expected_seconds == 30 {
            Duration::from_secs(25)
        } else {
            settings.poll_interval
        };

        assert_eq!(
            settings.next_poll_interval(current, queued),
            Duration::from_secs(expected_seconds)
        );
    }

    #[rstest]
    #[case::transient_outage(connect_error(), 3)]
    #[case::systemic(sqlstate("42P01"), 3)]
    #[tokio::test(start_paused = true)]
    async fn a_failed_pass_requeues_its_rows_at_the_head(
        #[case] error: Error,
        #[case] queued_after: usize,
    ) {
        let code = error.sqlstate();
        let writer = writer(
            FakeStore::new(Box::new(move |_, _| {
                Some(match &code {
                    Some(code) => sqlstate(Box::leak(code.clone().into_boxed_str())),
                    None => connect_error(),
                })
            })),
            WriterSettings {
                rows_per_pass: 2,
                ..WriterSettings::default()
            },
        );
        writer.enqueue(rows(3), false, usize::MAX);

        let failure = writer.flush_pass(None).await.expect_err("pass fails");

        assert!(failure.requeue);
        let order: Vec<String> = writer
            .queue()
            .dequeue(10)
            .iter()
            .map(|row| litellm_python_compat::repr::to_str(row.get("request_id").expect("id")))
            .collect();
        assert_eq!(order.len(), queued_after);
        assert_eq!(order, ["0", "1", "2"]);
    }

    #[rstest]
    #[case::drains_everything(25, 0)]
    #[tokio::test]
    async fn drain_writes_in_passes_until_the_queue_is_empty(
        #[case] count: usize,
        #[case] remaining: usize,
    ) {
        let writer = writer(
            FakeStore::new(Box::new(|_, _| None)),
            WriterSettings {
                rows_per_pass: 10,
                ..WriterSettings::default()
            },
        );
        writer.enqueue(rows(count), false, usize::MAX);

        assert_eq!(writer.drain(None).await.expect("drains"), remaining);
        assert_eq!(writer.flusher().store().calls().written.len(), count);
    }

    #[rstest]
    #[case::stops_after_the_pass_limit(3, 10)]
    #[tokio::test]
    async fn drain_gives_up_after_its_passes(#[case] passes: usize, #[case] remaining: usize) {
        let writer = writer(
            FakeStore::new(Box::new(|_, _| None)),
            WriterSettings {
                rows_per_pass: 5,
                drain_passes: passes,
                ..WriterSettings::default()
            },
        );
        writer.enqueue(rows(25), false, usize::MAX);

        assert_eq!(writer.drain(None).await.expect("drains"), remaining);
    }

    #[rstest]
    #[case::woken_before_the_poll()]
    #[tokio::test(start_paused = true)]
    async fn a_flush_request_wakes_the_monitor() {
        let writes = Arc::new(AtomicUsize::new(0));
        let counted = writes.clone();
        let writer = writer(
            FakeStore::new(Box::new(move |_, _| {
                counted.fetch_add(1, Ordering::SeqCst);
                None
            })),
            WriterSettings::default(),
        );
        let running = writer.clone();
        let monitor = tokio::spawn(async move { running.run(|| None).await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        writer.enqueue(rows(1), false, usize::MAX);
        writer.request_flush();
        tokio::time::sleep(Duration::from_millis(10)).await;

        assert_eq!(writes.load(Ordering::SeqCst), 1);
        assert!(writer.queue().is_empty());
        monitor.abort();
    }

    #[rstest]
    #[case::empty_first_poll_backs_off_to_three_seconds()]
    #[tokio::test(start_paused = true)]
    async fn the_monitor_flushes_on_its_poll() {
        let writer = writer(
            FakeStore::new(Box::new(|_, _| None)),
            WriterSettings::default(),
        );
        let running = writer.clone();
        let monitor = tokio::spawn(async move { running.run(|| None).await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        writer.enqueue(rows(1), false, usize::MAX);

        tokio::time::sleep(Duration::from_millis(2900)).await;
        assert_eq!(writer.queue().len(), 1);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(writer.queue().is_empty());
        monitor.abort();
    }
}
