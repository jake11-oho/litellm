//! The Postgres `LedgerStore`, over the spend log repository.

use litellm_repositories::SpendLogRepository;

use crate::{Error, LedgerStore, SpendLogRow, ledger_request};

/// Commits statements to Postgres through the spend log repository: logs with no charge go
/// in on their own, the rest through the ledger transaction. The repository reports each
/// statement, applies the deadlines, and tells an outage apart from a full pool, so the
/// flusher retries the outage.
pub struct PgLedgerStore {
    repository: SpendLogRepository,
    max_depth: usize,
}

impl PgLedgerStore {
    pub fn new(repository: SpendLogRepository, max_depth: usize) -> Self {
        Self {
            repository,
            max_depth,
        }
    }
}

impl LedgerStore for PgLedgerStore {
    async fn insert_with_deliveries(&self, rows: &[SpendLogRow]) -> Result<usize, Error> {
        let request = ledger_request(rows, self.max_depth)?;
        Ok(self
            .repository
            .insert_with_deliveries(&request.legacy, &request.ledger)
            .await?)
    }
}
