use std::{borrow::Cow, sync::Mutex};

use crate::{Error, LedgerStore, SpendLogRow, ledger_request};

#[derive(Debug)]
pub(crate) struct Sqlstate(&'static str);

impl std::fmt::Display for Sqlstate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ERROR: sqlstate {}", self.0)
    }
}

impl std::error::Error for Sqlstate {}

impl sqlx::error::DatabaseError for Sqlstate {
    fn message(&self) -> &str {
        "sqlstate"
    }

    fn code(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Borrowed(self.0))
    }

    fn as_error(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
        self
    }

    fn as_error_mut(&mut self) -> &mut (dyn std::error::Error + Send + Sync + 'static) {
        self
    }

    fn into_error(self: Box<Self>) -> Box<dyn std::error::Error + Send + Sync + 'static> {
        self
    }

    fn kind(&self) -> sqlx::error::ErrorKind {
        sqlx::error::ErrorKind::Other
    }
}

pub(crate) fn database(error: sqlx::Error) -> Error {
    Error::Database(litellm_db::Error::Query(error))
}

pub(crate) fn sqlstate(code: &'static str) -> Error {
    database(sqlx::Error::Database(Box::new(Sqlstate(code))))
}

pub(crate) fn connect_error() -> Error {
    database(sqlx::Error::Io(
        std::io::ErrorKind::ConnectionRefused.into(),
    ))
}

pub(crate) type Fail = Box<dyn Fn(&[SpendLogRow], usize) -> Option<Error> + Send + Sync>;

#[derive(Default)]
pub(crate) struct Calls {
    pub(crate) statements: Vec<Vec<String>>,
    pub(crate) written: Vec<String>,
    pub(crate) charges_kept: Vec<String>,
}

pub(crate) struct FakeStore {
    calls: Mutex<Calls>,
    fail: Fail,
    pub(crate) validate: bool,
}

pub(crate) fn request_ids(rows: &[SpendLogRow]) -> Vec<String> {
    rows.iter()
        .map(|row| {
            row.get("request_id")
                .map(litellm_python_compat::repr::to_str)
                .unwrap_or_default()
        })
        .collect()
}

impl FakeStore {
    pub(crate) fn new(fail: Fail) -> Self {
        Self {
            calls: Mutex::new(Calls::default()),
            fail,
            validate: false,
        }
    }

    pub(crate) fn calls(&self) -> std::sync::MutexGuard<'_, Calls> {
        self.calls.lock().expect("calls lock")
    }
}

impl LedgerStore for FakeStore {
    async fn insert_with_deliveries(&self, rows: &[SpendLogRow]) -> Result<usize, Error> {
        let call = {
            let mut calls = self.calls();
            calls.statements.push(request_ids(rows));
            calls.statements.len()
        };
        if let Some(error) = (self.fail)(rows, call) {
            return Err(error);
        }
        if rows.iter().all(SpendLogRow::log_disabled) {
            self.calls().charges_kept.extend(request_ids(rows));
            return Ok(rows.len());
        }
        if self.validate {
            ledger_request(rows, 100)?;
        }
        self.calls().written.extend(request_ids(rows));
        Ok(rows.len())
    }
}

pub(crate) fn rows(count: usize) -> Vec<SpendLogRow> {
    (0..count)
        .map(|index| {
            SpendLogRow::parse(&format!(r#"{{"request_id": "{index}", "spend": 10}}"#))
                .expect("row parses")
        })
        .collect()
}

pub(crate) fn row(text: &str) -> SpendLogRow {
    SpendLogRow::parse(text).expect("row parses")
}
