//! What committing a statement of spend logs can fail with.

use litellm_db::ErrorKind;

/// A statement the store could not commit: Postgres or the pool failed it, or one of its
/// rows was refused before it reached Postgres.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Database(#[from] litellm_db::Error),
    /// A row the `LiteLLM_SpendLogs` table cannot hold, such as an unknown column or a
    /// value of the wrong type.
    #[error("spend log row rejected")]
    RejectedRow(#[source] serde_json::Error),
    /// A charged row with no `request_id` to settle under.
    #[error("spend log row rejected: spend log row has no request_id")]
    MissingRequestId,
    /// A JSON column holding text that is not JSON.
    #[error("spend log row rejected: Spend log column {column} is not valid JSON")]
    InvalidJsonColumn { column: String },
    /// A charged row whose `accounting_payload` is not a JSON object.
    #[error("invalid spend log delivery: accounting_payload is not a JSON object")]
    InvalidDelivery,
}

impl Error {
    /// The database's classification of the failure; a row refused before Postgres is
    /// `DataRejected` like a row Postgres refuses, and an invalid delivery is `Other`.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::Database(error) => error.kind(),
            Self::RejectedRow(_) | Self::MissingRequestId | Self::InvalidJsonColumn { .. } => {
                ErrorKind::DataRejected
            }
            Self::InvalidDelivery => ErrorKind::Other,
        }
    }

    /// The Postgres SQLSTATE behind a database failure.
    pub fn sqlstate(&self) -> Option<String> {
        match self {
            Self::Database(error) => error.sqlstate(),
            Self::RejectedRow(_)
            | Self::MissingRequestId
            | Self::InvalidJsonColumn { .. }
            | Self::InvalidDelivery => None,
        }
    }
}
