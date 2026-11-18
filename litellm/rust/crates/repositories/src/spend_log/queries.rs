//! The spend log ledger's statements on `LiteLLM_SpendLogs` and `LiteLLM_SpendLogDelivery`,
//! mirroring the Python ledger writer (`SpendLogsRepository._insert_ledger`).

use serde_json::Value;
use sqlx::PgExecutor;

use litellm_db::Error;

/// Inserts the request logs in `records` (a JSON array of `LiteLLM_SpendLogs` rows), skipping
/// any `request_id` already stored, and returns the ids it inserted.
pub(super) async fn insert_spend_log_rows(
    executor: impl PgExecutor<'_>,
    records: Value,
) -> Result<Vec<String>, Error> {
    Ok(
        sqlx::query_file_scalar!("src/spend_log/queries/insert_spend_log_rows.sql", records)
            .fetch_all(executor)
            .await?,
    )
}

/// Inserts the deliveries in `payloads` (a JSON array of delivery rows), skipping any
/// `delivery_id` already stored, and returns the ids it accepted.
pub(super) async fn insert_spend_log_deliveries(
    executor: impl PgExecutor<'_>,
    payloads: Value,
) -> Result<Vec<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/insert_spend_log_deliveries.sql",
        payloads
    )
    .fetch_all(executor)
    .await?)
}

/// Which of the settle-once `request_ids` a pre-ledger cost row already charged.
pub(super) async fn spend_log_jobs_charged_by_legacy(
    executor: impl PgExecutor<'_>,
    request_ids: &[String],
) -> Result<Vec<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/select_spend_log_jobs_charged_by_legacy.sql",
        request_ids
    )
    .fetch_all(executor)
    .await?)
}

/// Removes the deliveries in `delivery_ids`.
pub(super) async fn delete_spend_log_deliveries(
    executor: impl PgExecutor<'_>,
    delivery_ids: &[String],
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/spend_log/queries/delete_spend_log_deliveries.sql",
        delivery_ids
    )
    .execute(executor)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use serde_json::{Map, Value, json};
    use sqlx::PgPool;

    use super::insert_spend_log_rows;
    use crate::{
        spend_log::record::{SpendLogRecord, parse_spend_log_timestamp},
        test_support::database_pool,
    };

    const SPEND_LOG_COLUMNS: [&str; 35] = [
        "call_type",
        "api_key",
        "spend",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "startTime",
        "endTime",
        "request_duration_ms",
        "completionStartTime",
        "model",
        "model_id",
        "model_group",
        "custom_llm_provider",
        "api_base",
        "user",
        "metadata",
        "cache_hit",
        "cache_key",
        "request_tags",
        "team_id",
        "organization_id",
        "end_user",
        "requester_ip_address",
        "messages",
        "response",
        "session_id",
        "status",
        "mcp_namespaced_tool_name",
        "billing_agent_id",
        "agent_id",
        "proxy_server_request",
        "litellm_call_id",
        "created_at",
        "updated_at",
    ];

    fn columns(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            _ => Map::new(),
        }
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_missing_column_takes_the_table_default(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        let now = parse_spend_log_timestamp("2026-11-02T12:00:00+00:00").ok_or("timestamp")?;
        #[expect(
            clippy::disallowed_methods,
            reason = "The reference row is inserted the way Prisma omits a column"
        )]
        sqlx::query(
            r#"INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, "startTime", "endTime", created_at, updated_at)
               VALUES ('defaults-sql', 'acompletion', '2026-11-02 12:00:00', '2026-11-02 12:00:01', $1, $1)"#,
        )
        .bind(now)
        .execute(&mut *tx)
        .await?;
        let minimal = SpendLogRecord::from_columns(columns(json!({
            "request_id": "defaults-rust",
            "call_type": "acompletion",
            "startTime": "2026-11-02 12:00:00",
            "endTime": "2026-11-02 12:00:01",
        })))?;
        insert_spend_log_rows(&mut *tx, serde_json::to_value([minimal.stamped(now)])?).await?;
        let selected = SPEND_LOG_COLUMNS
            .iter()
            .map(|column| format!("to_jsonb(\"{column}\")"))
            .collect::<Vec<_>>()
            .join(", ");
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions compare every column at runtime"
        )]
        let rows: Vec<Value> = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
            r#"SELECT jsonb_build_array({selected}) FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1) ORDER BY request_id"#
        )))
        .bind(["defaults-rust", "defaults-sql"].as_slice())
        .fetch_all(&mut *tx)
        .await?;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], rows[1]);
        tx.rollback().await?;
        Ok(())
    }
}
