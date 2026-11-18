//! Python binding for the Rust spend log callback (`litellm-callbacks-spend-log`). The proxy's
//! `SpendLogger` builds one [`RustSpendLog`] over the Rust spend log writer and the shared spend
//! counters, and registers its `.callback()` as a native `CustomLoggerV2`.

use std::sync::Arc;

use litellm_callbacks::ErasedCallback;
use litellm_callbacks_budgets::Budgets;
use litellm_callbacks_python::RustCallback;
use litellm_callbacks_spend_log::{SpendLogCallback, SpendLogParts};
use litellm_spend_log_writer::{PgLedgerStore, Retention};
use pyo3::prelude::*;

use crate::{
    redis::store::spend_counters::NativeSpendCounters, spend_log_writer::RustSpendLogWriter,
};

type RetentionFields = (Option<Vec<String>>, Option<Vec<String>>, Vec<String>);

/// The spend log callback over the writer the proxy flushes and the spend counters admission
/// reads.
#[pyclass(frozen, name = "RustSpendLog")]
pub(crate) struct RustSpendLog(Arc<SpendLogCallback<PgLedgerStore>>);

#[pymethods]
impl RustSpendLog {
    #[new]
    #[pyo3(signature = (writer, spend_counters, minimum_custom_key_length, word_ranges, retention=None))]
    fn new(
        py: Python<'_>,
        writer: PyRef<'_, RustSpendLogWriter>,
        spend_counters: PyRef<'_, NativeSpendCounters>,
        minimum_custom_key_length: usize,
        word_ranges: Vec<(u32, u32)>,
        retention: Option<RetentionFields>,
    ) -> Self {
        Self(Arc::new(SpendLogCallback::new(SpendLogParts {
            writer: writer.writer(),
            budgets: Arc::new(
                Budgets::new(spend_counters.store()).with_logger(crate::logger::capture(py)),
            ),
            minimum_custom_key_length,
            word_ranges,
            retention: retention.map(|(include, exclude, always_kept)| Retention {
                include,
                exclude,
                always_kept,
            }),
        })))
    }

    fn callback(&self) -> PyResult<RustCallback> {
        Ok(RustCallback::new(
            Arc::clone(&self.0) as Arc<dyn ErasedCallback>
        )?)
    }
}
