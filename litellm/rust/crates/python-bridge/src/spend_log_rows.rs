//! The spend log row builder for Python: `litellm/proxy/spend_tracking/rust_spend_log_rows.py`
//! builds a [`RustSpendLogRowBuilder`] per settings change and hands it each request's input as
//! JSON, getting the `LiteLLM_SpendLogs` row back as JSON. The GIL is released while a row builds.

use litellm_callbacks_spend_log::{RowConfig, RowError, SpendLogRowBuilder, finalize_ledger_row};
use litellm_host_python::release_gil;
use litellm_python_compat::json;
use litellm_tracing::TokenRedactor;
use pyo3::{exceptions::PyValueError, prelude::*};

fn row_error(error: RowError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// One [`SpendLogRowBuilder`] under the proxy settings it was built with.
#[pyclass(frozen, name = "RustSpendLogRowBuilder")]
pub(crate) struct RustSpendLogRowBuilder(SpendLogRowBuilder);

#[pymethods]
impl RustSpendLogRowBuilder {
    #[new]
    fn new(
        settings_json: &str,
        minimum_custom_key_length: usize,
        word_ranges: Vec<(u32, u32)>,
    ) -> PyResult<Self> {
        let settings = json::loads(settings_json).map_err(|error| row_error(error.into()))?;
        let config = RowConfig::from_settings(&settings).map_err(row_error)?;
        Ok(Self(SpendLogRowBuilder::new(
            config,
            TokenRedactor::new(minimum_custom_key_length, word_ranges),
        )))
    }

    /// The row for one request, from the glue's input JSON.
    fn build_row(&self, py: Python<'_>, input_json: &str) -> PyResult<String> {
        release_gil(py, || self.0.build_row_json(input_json)).map_err(row_error)
    }

    /// The ledger columns of a built row: billable spend, delivery id and accounting payload.
    #[staticmethod]
    fn finalize_ledger_row(py: Python<'_>, row_json: &str, input_json: &str) -> PyResult<String> {
        release_gil(py, || -> Result<String, RowError> {
            let row = json::loads(row_json)?;
            let input = json::loads(input_json)?;
            Ok(json::dumps(&finalize_ledger_row(&row, &input)?)?)
        })
        .map_err(row_error)
    }
}
