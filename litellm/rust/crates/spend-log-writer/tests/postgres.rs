use std::{error::Error as StdError, str::FromStr, sync::Arc, time::SystemTime};

use litellm_db::{Endpoint, Observer, PoolLimits, Pools, QueryEvent, SslMode};
use litellm_repositories::SpendLogRepository;
use litellm_secret_value::SecretValue;
use litellm_spend_log_writer::{
    PgLedgerStore, SpendLogFlusher, SpendLogRow, SpendLogWriter, WriterSettings,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use sqlx::{PgPool, postgres::PgConnectOptions};

struct Ignored;

impl Observer for Ignored {
    fn record(&self, _event: QueryEvent) {}
}

fn endpoint() -> Result<Endpoint, Box<dyn StdError>> {
    let database_url = std::env::var("DATABASE_URL")?;
    let options = PgConnectOptions::from_str(&database_url)?;
    Ok(Endpoint {
        host: options.get_host().to_owned(),
        port: options.get_port(),
        socket: None,
        username: options.get_username().to_owned(),
        password: url::Url::parse(&database_url)?
            .password()
            .map(SecretValue::new),
        database: options.get_database().unwrap_or("postgres").to_owned(),
        ssl_mode: SslMode::Prefer,
        ssl_root_cert: None,
        ssl_client: None,
        application_name: None,
        options: Vec::new(),
        statement_cache_capacity: 100,
        pool: PoolLimits {
            max_connections: 2,
            acquire_timeout_seconds: Some(10.0),
            idle_timeout_seconds: None,
            socket_timeout_seconds: None,
        },
    })
}

struct Database {
    writer: SpendLogWriter<PgLedgerStore>,
    pool: PgPool,
    prefix: String,
}

#[fixture]
fn database() -> Result<Database, Box<dyn StdError>> {
    let endpoint = endpoint()?;
    let pools = Pools::new(&endpoint, None)?;
    let pool = pools.writer().clone();
    let settings = WriterSettings::default();
    let store = PgLedgerStore::new(
        SpendLogRepository::new(
            litellm_repositories::Database::new(pools, Arc::new(Ignored), None, true),
            settings.transaction_timeout,
        ),
        settings.max_depth,
    );
    let prefix = format!(
        "spend-log-writer-{}",
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_nanos()
    );
    Ok(Database {
        writer: SpendLogWriter::new(SpendLogFlusher::new(store, settings)),
        pool,
        prefix,
    })
}

impl Database {
    fn id(&self, name: &str) -> String {
        format!("{}-{name}", self.prefix)
    }

    fn row(&self, name: &str, messages: &str) -> Result<SpendLogRow, Box<dyn StdError>> {
        let request_id = self.id(name);
        let text = json!({
            "request_id": request_id,
            "call_type": "acompletion",
            "api_key": "hashed",
            "spend": 0.25,
            "total_tokens": 3,
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "startTime": "2026-11-02T12:00:00+00:00",
            "endTime": "2026-11-02T12:00:01+00:00",
            "completionStartTime": "2026-11-02 12:00:00.500000+00:00",
            "model": "gpt",
            "model_id": "deployment",
            "model_group": "gpt",
            "mcp_namespaced_tool_name": null,
            "agent_id": null,
            "api_base": "https://example.test",
            "user": "user",
            "metadata": "{\"status\": \"success\", \"tenant\": \"a\"}",
            "cache_hit": "False",
            "cache_key": "Cache OFF",
            "request_tags": "[\"tag\"]",
            "team_id": "team",
            "organization_id": null,
            "end_user": "",
            "requester_ip_address": null,
            "custom_llm_provider": "openai",
            "messages": messages,
            "response": {"id": "resp"},
            "proxy_server_request": "{}",
            "session_id": null,
            "request_duration_ms": 1000,
            "status": "success",
            "litellm_call_id": "call",
            "accounting_delivery_id": format!("{request_id}-delivery"),
            "accounting_payload": {"request_id": request_id, "spend": 0.25},
            "accounting_log_disabled": false,
        });
        Ok(SpendLogRow::parse(&text.to_string())?)
    }

    async fn stored(&self, query: &'static str) -> Result<Vec<Value>, Box<dyn StdError>> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let rows = sqlx::query_scalar::<_, Value>(sqlx::AssertSqlSafe(query))
            .bind(format!("{}%", self.prefix))
            .fetch_all(&self.pool)
            .await?;
        Ok(rows)
    }

    async fn clean(&self) -> Result<(), Box<dyn StdError>> {
        for query in [
            r#"DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE $1"#,
            r#"DELETE FROM "LiteLLM_SpendLogDelivery" WHERE request_id LIKE $1"#,
        ] {
            #[expect(
                clippy::disallowed_methods,
                reason = "Test cleanup is outside the compile-checked query set"
            )]
            sqlx::query(sqlx::AssertSqlSafe(query))
                .bind(format!("{}%", self.prefix))
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }
}

const LOGS: &str = r#"SELECT jsonb_build_object('request_id', request_id, 'spend', spend, 'metadata', metadata, 'request_tags', request_tags, 'messages', messages, 'response', response, 'startTime', "startTime", 'completionStartTime', "completionStartTime") FROM "LiteLLM_SpendLogs" WHERE request_id LIKE $1 ORDER BY request_id"#;
const DELIVERIES: &str = r#"SELECT jsonb_build_object('delivery_id', delivery_id, 'state', state, 'is_batch', is_batch, 'accounting_payload', accounting_payload) FROM "LiteLLM_SpendLogDelivery" WHERE request_id LIKE $1 ORDER BY delivery_id"#;

#[rstest]
#[case::replayed_once()]
#[tokio::test]
#[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
async fn a_flushed_row_lands_with_its_delivery_once(
    database: Result<Database, Box<dyn StdError>>,
) -> Result<(), Box<dyn StdError>> {
    let database = database?;
    let rows = vec![database.row("a", "[{\"role\": \"user\", \"content\": \"hi\"}]")?];
    database.writer.enqueue(rows.clone(), false, usize::MAX);
    assert_eq!(database.writer.flush_pass(None).await?, 1);
    database.writer.enqueue(rows, true, usize::MAX);
    assert_eq!(database.writer.flush_pass(None).await?, 1);

    let logs = database.stored(LOGS).await?;
    let deliveries = database.stored(DELIVERIES).await?;
    database.clean().await?;
    let id = database.id("a");
    assert_eq!(
        logs,
        [json!({
            "request_id": id,
            "spend": 0.25,
            "metadata": {"status": "success", "tenant": "a"},
            "request_tags": ["tag"],
            "messages": [{"role": "user", "content": "hi"}],
            "response": {"id": "resp"},
            "startTime": "2026-11-02T12:00:00",
            "completionStartTime": "2026-11-02T12:00:00.5",
        })]
    );
    assert_eq!(
        deliveries,
        [json!({
            "delivery_id": format!("{id}-delivery"),
            "state": "pending",
            "is_batch": false,
            "accounting_payload": {"request_id": id, "spend": 0.25},
        })]
    );
    Ok(())
}

#[rstest]
#[case::unparseable_messages()]
#[tokio::test]
#[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
async fn a_poison_row_is_dropped_but_keeps_its_charge(
    database: Result<Database, Box<dyn StdError>>,
) -> Result<(), Box<dyn StdError>> {
    let database = database?;
    let rows = vec![
        database.row("good-1", "[]")?,
        database.row("poison", "not json")?,
        database.row("good-2", "[]")?,
    ];
    database.writer.enqueue(rows, false, usize::MAX);
    assert_eq!(database.writer.flush_pass(None).await?, 3);

    let logs = database.stored(LOGS).await?;
    let deliveries = database.stored(DELIVERIES).await?;
    database.clean().await?;
    let logged: Vec<&Value> = logs.iter().map(|row| &row["request_id"]).collect();
    assert_eq!(
        logged,
        [&json!(database.id("good-1")), &json!(database.id("good-2"))]
    );
    assert_eq!(deliveries.len(), 3);
    Ok(())
}
