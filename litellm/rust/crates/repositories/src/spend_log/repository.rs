//! The spend log repository: one ledger commit records each request's accounting delivery in
//! `LiteLLM_SpendLogDelivery` and its request log in `LiteLLM_SpendLogs`, so a replayed flush
//! can neither charge a request twice nor lose its log.
//!
//! The policy mirrors `SpendLogsRepository.insert_with_deliveries`: logs with no charge go in
//! on their own; deliveries go in first with `ON CONFLICT DO NOTHING`, a batch or video job
//! that a pre-ledger cost row already charged loses its delivery again, and only the logs whose
//! delivery this commit accepted are written, all inside one transaction with its own deadline.

use std::{collections::BTreeMap, time::Duration};

use chrono::{NaiveDateTime, Utc};
use litellm_db::{Call, Error, Statements};
use serde_json::{Value, json};
use sqlx::PgConnection;

use super::{
    queries,
    record::{SpendLogLedgerEntry, SpendLogRecord},
};
use crate::{Database, Side};

const INSERT_SPEND_LOGS: Call = Call {
    call_type: "insert_spend_logs",
    table: Some("LiteLLM_SpendLogs"),
    operation: None,
};

/// Writes request logs and their ledger deliveries on the writer.
pub struct SpendLogRepository {
    database: Database,
    transaction_timeout: Duration,
}

impl SpendLogRepository {
    /// A repository whose ledger transaction gives up after `transaction_timeout`.
    pub fn new(database: Database, transaction_timeout: Duration) -> Self {
        Self {
            database,
            transaction_timeout,
        }
    }

    /// Writes one statement's worth of rows as one reported operation: `logs` carry no charge
    /// and go in on their own, `ledger` commits in one transaction. Returns how many logs and
    /// deliveries were new. A pool timeout with room left in the writer pool is reported as an
    /// outage, and an error that calls for fresh connections recycles the pools.
    pub async fn insert_with_deliveries(
        &self,
        logs: &[SpendLogRecord],
        ledger: &[SpendLogLedgerEntry],
    ) -> Result<usize, Error> {
        let result = self
            .database
            .run(INSERT_SPEND_LOGS, || async {
                self.write(logs, ledger)
                    .await
                    .map_err(|error| self.database.explaining_writer_timeout(error))
            })
            .await;
        if let Err(error) = &result {
            self.database.recycle_after(error);
        }
        result
    }

    async fn write(
        &self,
        logs: &[SpendLogRecord],
        ledger: &[SpendLogLedgerEntry],
    ) -> Result<usize, Error> {
        let now = Utc::now().naive_utc();
        let mut session = self.database.session(Side::Writer).await?;
        let logged = if logs.is_empty() {
            0
        } else {
            queries::insert_spend_log_rows(session.executor(), records(logs, now)?)
                .await?
                .len()
        };
        if ledger.is_empty() {
            return Ok(logged);
        }
        let named = session.names_statements();
        let accepted = commit_ledger(
            session.connection(),
            named,
            ledger,
            now,
            self.transaction_timeout,
        )
        .await?;
        Ok(logged + accepted)
    }
}

/// Commits `entries` on `connection` as one transaction, giving up with `TransactionTimedOut`
/// after `timeout` so a stuck commit cannot hold the flush forever.
async fn commit_ledger(
    connection: &mut PgConnection,
    named: bool,
    entries: &[SpendLogLedgerEntry],
    now: NaiveDateTime,
    timeout: Duration,
) -> Result<usize, Error> {
    let commit = async {
        let mut transaction = sqlx::Connection::begin(connection).await?;
        let accepted = write_ledger(&mut transaction, named, entries, now).await?;
        transaction.commit().await?;
        Ok(accepted)
    };
    tokio::time::timeout(timeout, commit)
        .await
        .unwrap_or(Err(Error::TransactionTimedOut))
}

/// The logs as the JSON array `jsonb_populate_recordset` reads, stamped with `now`.
fn records(logs: &[SpendLogRecord], now: NaiveDateTime) -> Result<Value, Error> {
    serde_json::to_value(logs.iter().map(|log| log.stamped(now)).collect::<Vec<_>>())
        .map_err(|error| Error::Query(sqlx::Error::Encode(Box::new(error))))
}

fn first_entry_per_delivery(entries: &[SpendLogLedgerEntry]) -> Vec<&SpendLogLedgerEntry> {
    let mut by_delivery: BTreeMap<&str, &SpendLogLedgerEntry> = BTreeMap::new();
    for entry in entries {
        by_delivery
            .entry(entry.delivery.delivery_id.as_str())
            .or_insert(entry);
    }
    by_delivery.into_values().collect()
}

/// Writes `entries` on `connection`, which holds an open transaction: the first entry per
/// delivery id wins, a delivery already stored is skipped, and only the logs of accepted
/// deliveries are written. Returns how many deliveries were accepted.
async fn write_ledger(
    connection: &mut PgConnection,
    named: bool,
    entries: &[SpendLogLedgerEntry],
    now: NaiveDateTime,
) -> Result<usize, Error> {
    let ledger = first_entry_per_delivery(entries);
    if ledger.is_empty() {
        return Ok(0);
    }
    let payloads = Value::Array(
        ledger
            .iter()
            .map(|entry| {
                json!({
                    "delivery_id": entry.delivery.delivery_id,
                    "request_id": entry.delivery.request_id,
                    "accounting_payload": entry.delivery.accounting_payload,
                    "is_batch": entry.delivery.is_batch,
                })
            })
            .collect(),
    );
    let accepted =
        queries::insert_spend_log_deliveries(Statements::new(&mut *connection, named), payloads)
            .await?;
    let settled_once: Vec<String> = ledger
        .iter()
        .filter(|entry| entry.delivery.is_batch)
        .map(|entry| entry.delivery.delivery_id.clone())
        .filter(|delivery_id| accepted.contains(delivery_id))
        .collect();
    let charged_by_legacy = if settled_once.is_empty() {
        Vec::new()
    } else {
        queries::spend_log_jobs_charged_by_legacy(
            Statements::new(&mut *connection, named),
            &settled_once,
        )
        .await?
    };
    if !charged_by_legacy.is_empty() {
        queries::delete_spend_log_deliveries(
            Statements::new(&mut *connection, named),
            &charged_by_legacy,
        )
        .await?;
    }
    let kept = |delivery_id: &String| {
        accepted.contains(delivery_id) && !charged_by_legacy.contains(delivery_id)
    };
    let logs: Vec<SpendLogRecord> = ledger
        .iter()
        .filter(|entry| kept(&entry.delivery.delivery_id))
        .filter_map(|entry| entry.row.clone())
        .collect();
    if !logs.is_empty() {
        queries::insert_spend_log_rows(
            Statements::new(&mut *connection, named),
            records(&logs, now)?,
        )
        .await?;
    }
    Ok(accepted.iter().filter(|id| kept(id)).count())
}

#[cfg(test)]
mod tests {
    use std::{error::Error as StdError, time::Duration};

    use chrono::NaiveDateTime;
    use litellm_db::{Error, ErrorKind, Statements};
    use rstest::{fixture, rstest};
    use serde_json::{Map, Value, json};
    use sqlx::{PgExecutor, PgPool};

    use super::{commit_ledger, queries, records, write_ledger};
    use crate::{
        spend_log::record::{
            SpendLogDelivery, SpendLogLedgerEntry, SpendLogRecord, parse_spend_log_timestamp,
        },
        test_support::database_pool,
    };

    fn now() -> NaiveDateTime {
        parse_spend_log_timestamp("2026-11-02T12:00:00+00:00").expect("timestamp parses")
    }

    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            _ => Map::new(),
        }
    }

    fn row(request_id: &str) -> SpendLogRecord {
        SpendLogRecord::from_columns(object(json!({
            "request_id": request_id,
            "call_type": "acompletion",
            "api_key": "hashed",
            "spend": 0.5,
            "startTime": "2026-11-02T12:00:00+00:00",
            "endTime": "2026-11-02 12:00:01+00:00",
            "metadata": {"accounting_savings": {"compression": 0.2}},
            "request_tags": ["a"],
            "messages": [{"role": "user", "content": "hi"}],
            "status": "success",
        })))
        .expect("record parses")
    }

    fn entry(delivery_id: &str, request_id: &str, logged: bool) -> SpendLogLedgerEntry {
        SpendLogLedgerEntry {
            delivery: SpendLogDelivery {
                delivery_id: delivery_id.into(),
                request_id: request_id.into(),
                accounting_payload: object(json!({"request_id": request_id, "spend": 0.5})),
                is_batch: false,
            },
            row: logged.then(|| row(request_id)),
        }
    }

    #[fixture]
    fn ledger() -> Vec<SpendLogLedgerEntry> {
        vec![
            entry("delivery-b", "ledger-b", true),
            entry("delivery-a", "ledger-a", true),
            entry("delivery-a", "ledger-a-replayed", true),
            entry("delivery-c", "ledger-c", false),
        ]
    }

    async fn ids(
        executor: impl PgExecutor<'_>,
        query: &'static str,
        ids: &[&str],
    ) -> Result<Vec<String>, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        sqlx::query_scalar::<_, String>(sqlx::AssertSqlSafe(query))
            .bind(ids)
            .fetch_all(executor)
            .await
    }

    const DELIVERIES: &str = r#"SELECT delivery_id FROM "LiteLLM_SpendLogDelivery" WHERE delivery_id = ANY($1) ORDER BY 1"#;
    const LOGS: &str =
        r#"SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1) ORDER BY 1"#;

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn writes_each_delivery_once_with_its_logged_row(
        ledger: Vec<SpendLogLedgerEntry>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        assert_eq!(write_ledger(&mut tx, true, &ledger, now()).await?, 3);
        assert_eq!(write_ledger(&mut tx, true, &ledger, now()).await?, 0);
        let every_id = [
            "delivery-a",
            "delivery-b",
            "delivery-c",
            "ledger-a",
            "ledger-a-replayed",
            "ledger-b",
            "ledger-c",
        ];
        assert_eq!(
            ids(&mut *tx, DELIVERIES, &every_id).await?,
            ["delivery-a", "delivery-b", "delivery-c"]
        );
        assert_eq!(
            ids(&mut *tx, LOGS, &every_id).await?,
            ["ledger-a", "ledger-b"]
        );
        tx.rollback().await?;
        Ok(())
    }

    #[rstest]
    #[case::charged_before_the_ledger(0.25, "success", vec![])]
    #[case::left_a_zero_cost_row(0.0, "success", vec!["job-1"])]
    #[case::left_a_failed_row(0.25, "failure", vec!["job-1"])]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_settle_once_job_keeps_its_delivery_unless_a_legacy_row_charged_it(
        #[case] legacy_spend: f64,
        #[case] legacy_status: &str,
        #[case] expected_deliveries: Vec<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        let legacy = SpendLogRecord {
            spend: legacy_spend,
            status: Some(legacy_status.to_owned()),
            ..row("job-1")
        };
        queries::insert_spend_log_rows(Statements::new(&mut *tx, true), records(&[legacy], now())?)
            .await?;
        let job = SpendLogLedgerEntry {
            delivery: SpendLogDelivery {
                delivery_id: "job-1".into(),
                request_id: "job-1".into(),
                accounting_payload: object(json!({"spend": 0.25})),
                is_batch: true,
            },
            row: Some(row("job-1")),
        };
        let accepted = write_ledger(&mut tx, true, &[job], now()).await?;
        assert_eq!(accepted, expected_deliveries.len());
        assert_eq!(
            ids(&mut *tx, DELIVERIES, &["job-1"]).await?,
            expected_deliveries
        );
        tx.rollback().await?;
        Ok(())
    }

    #[rstest]
    #[case::non_finite_spend(SpendLogRecord { spend: f64::NAN, ..row("bad") }, Some("23502"))]
    #[case::nul_in_json(SpendLogRecord { response: Some(json!({"text": "a\u{0}b"})), ..row("bad") }, Some("22P05"))]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_row_postgres_cannot_store_rolls_back_the_whole_ledger(
        #[case] bad: SpendLogRecord,
        #[case] sqlstate: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut outer = pool.begin().await?;
        let mut poisoned = entry("delivery-bad", "bad", true);
        poisoned.row = Some(bad);
        let error = commit_ledger(
            &mut outer,
            true,
            &[entry("delivery-good", "good", true), poisoned],
            now(),
            Duration::from_secs(60),
        )
        .await
        .expect_err("the poisoned row must fail the ledger");
        assert_eq!(error.kind(), ErrorKind::DataRejected);
        assert_eq!(error.sqlstate().as_deref(), sqlstate);
        assert!(
            ids(&mut *outer, DELIVERIES, &["delivery-good", "delivery-bad"])
                .await?
                .is_empty()
        );
        outer.rollback().await?;
        Ok(())
    }

    #[rstest]
    #[case::expired(Duration::ZERO)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_ledger_past_its_timeout_commits_nothing(
        #[case] timeout: Duration,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut outer = pool.begin().await?;
        let result = commit_ledger(
            &mut outer,
            true,
            &[entry("delivery-timeout", "timeout", true)],
            now(),
            timeout,
        )
        .await;
        assert!(matches!(result, Err(Error::TransactionTimedOut)));
        assert!(
            ids(&mut *outer, DELIVERIES, &["delivery-timeout"])
                .await?
                .is_empty()
        );
        outer.rollback().await?;
        Ok(())
    }
}
