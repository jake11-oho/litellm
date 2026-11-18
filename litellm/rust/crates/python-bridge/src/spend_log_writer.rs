//! Python binding for the Rust spend log writer (`litellm-spend-log-writer`).

use std::sync::Arc;

use litellm_repositories::SpendLogRepository;
use litellm_spend_log_writer::{
    PgLedgerStore, Retention, SpendLogFlusher, SpendLogRow, SpendLogWriter, WriteError,
    WriteFailure, WriterSettings,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::db::NativeDatabase;

pyo3::create_exception!(
    _native,
    RustSpendLogWriteFailure,
    pyo3::exceptions::PyException,
    "A spend log write in Rust failed. requeue says whether the rows must go back to the queue, and kind is litellm-db's ErrorKind or invalid_metadata."
);

type RetentionFields = (Option<Vec<String>>, Option<Vec<String>>, Vec<String>);

/// The Rust spend log queue and writer the proxy uses on the ledger-on path. Rows
/// cross as JSON text; `write` returns an awaitable that commits a batch.
#[pyclass(frozen, name = "RustSpendLogWriter")]
pub struct RustSpendLogWriter {
    writer: Arc<SpendLogWriter<PgLedgerStore>>,
}

fn map_failure(failure: WriteFailure) -> PyErr {
    let kind: &'static str = match &failure.source {
        WriteError::Store(error) => error.kind().into(),
        WriteError::Metadata(_) => "invalid_metadata",
    };
    let error = RustSpendLogWriteFailure::new_err(failure.to_string());
    let decided = Python::attach(|py| {
        let value = error.value(py);
        value.setattr("requeue", failure.requeue)?;
        value.setattr("kind", kind)
    });
    match decided {
        Ok(()) => error,
        Err(error) => error,
    }
}

/// Parses JSON rows, returning the rows it can hold and the indexes of those it cannot
/// (such as text with lone surrogates).
fn decoded(rows: &[String]) -> (Vec<SpendLogRow>, Vec<usize>) {
    let mut kept = Vec::with_capacity(rows.len());
    let mut refused = Vec::new();
    for (index, text) in rows.iter().enumerate() {
        match SpendLogRow::parse(text) {
            Ok(row) => kept.push(row),
            Err(_) => refused.push(index),
        }
    }
    (kept, refused)
}

#[pymethods]
impl RustSpendLogWriter {
    #[new]
    #[pyo3(signature = (database, *, queue_max_bytes, write_batch_max_bytes, write_batch_max_rows, queue_size_threshold, poll_interval_seconds, max_backoff_seconds, max_depth))]
    #[expect(
        clippy::too_many_arguments,
        reason = "Each argument is one Python constant read from its own env var"
    )]
    fn new(
        database: PyRef<'_, NativeDatabase>,
        queue_max_bytes: usize,
        write_batch_max_bytes: usize,
        write_batch_max_rows: usize,
        queue_size_threshold: usize,
        poll_interval_seconds: f64,
        max_backoff_seconds: f64,
        max_depth: usize,
    ) -> PyResult<Self> {
        let seconds = |value: f64| {
            std::time::Duration::try_from_secs_f64(value)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        };
        let settings = WriterSettings {
            queue_max_bytes,
            write_batch_max_bytes,
            write_batch_max_rows,
            queue_size_threshold,
            poll_interval: seconds(poll_interval_seconds)?,
            max_backoff: seconds(max_backoff_seconds)?,
            max_depth,
            ..WriterSettings::default()
        };
        let store = PgLedgerStore::new(
            SpendLogRepository::new(
                database.repository_database()?,
                settings.transaction_timeout,
            ),
            settings.max_depth,
        );
        Ok(Self {
            writer: Arc::new(SpendLogWriter::new(SpendLogFlusher::new(store, settings))),
        })
    }

    fn enqueue(&self, rows: Vec<String>, at_head: bool, max_bytes: usize) -> (usize, Vec<usize>) {
        let (kept, refused) = decoded(&rows);
        let outcome = self.writer.queue().enqueue(kept, at_head, max_bytes);
        (outcome.dropped, refused)
    }

    fn dequeue(&self, limit: usize) -> Vec<String> {
        self.writer
            .queue()
            .dequeue(limit)
            .iter()
            .map(SpendLogRow::to_json)
            .collect()
    }

    fn __len__(&self) -> usize {
        self.writer.queue().len()
    }

    #[getter]
    fn queued_bytes(&self) -> usize {
        self.writer.queue().bytes()
    }

    #[pyo3(signature = (rows, retention=None))]
    fn write<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<String>,
        retention: Option<RetentionFields>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (rows, refused) = decoded(&rows);
        if !refused.is_empty() {
            return Err(PyValueError::new_err(format!(
                "{} spend log rows are not JSON objects the Rust writer can hold",
                refused.len()
            )));
        }
        let retention = retention.map(|(include, exclude, always_kept)| Retention {
            include,
            exclude,
            always_kept,
        });
        let writer = Arc::clone(&self.writer);
        crate::execution::run_async(
            py,
            Box::pin(async move { writer.flusher().write(&rows, retention.as_ref()).await }),
            map_failure,
        )
    }
}
