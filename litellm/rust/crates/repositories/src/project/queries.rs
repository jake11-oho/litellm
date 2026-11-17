//! Reads one `LiteLLM_ProjectTable` row with its budget, as project auth checks see it.

use chrono::NaiveDateTime;
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use crate::{RawJson, budget::Budget};

/// One `LiteLLM_ProjectTable` row and the `LiteLLM_BudgetTable` row its `budget_id` names, if it exists.
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct Project {
    pub(crate) project_id: String,
    pub(crate) project_alias: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) team_id: Option<String>,
    pub(crate) budget_id: Option<String>,
    pub(crate) metadata: RawJson,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) models: Option<Vec<String>>,
    pub(crate) spend: f64,
    pub(crate) model_spend: RawJson,
    pub(crate) model_rpm_limit: RawJson,
    pub(crate) model_tpm_limit: RawJson,
    pub(crate) blocked: bool,
    pub(crate) object_permission_id: Option<String>,
    #[serde(serialize_with = "crate::utc_timestamp::serialize")]
    pub(crate) created_at: NaiveDateTime,
    pub(crate) created_by: String,
    #[serde(serialize_with = "crate::utc_timestamp::serialize")]
    pub(crate) updated_at: NaiveDateTime,
    pub(crate) updated_by: String,
    pub(crate) litellm_budget_table: Option<Json<Budget>>,
}

pub(super) async fn find_project(
    executor: impl PgExecutor<'_>,
    project_id: &str,
) -> Result<Option<Project>, Error> {
    Ok(
        sqlx::query_file_as!(Project, "src/project/queries/find_project.sql", project_id)
            .fetch_optional(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::find_project;
    use crate::{RawJson, test_support::database_pool};

    async fn insert_project(
        tx: &mut Transaction<'_, Postgres>,
        project_id: &str,
        budget_id: Option<&str>,
    ) -> Result<(), sqlx::Error> {
        if let Some(budget_id) = budget_id {
            #[expect(
                clippy::disallowed_methods,
                reason = "Fixture inserts are outside the compile-checked query set"
            )]
            sqlx::query(
                r#"
                INSERT INTO "LiteLLM_BudgetTable"
                    (budget_id, max_budget, rpm_limit, model_max_budget, budget_reset_at, created_by, updated_by)
                VALUES ($1, 12.5, 7, '{"gpt": 1.5}', '2026-03-04 05:06:07.890', 'creator', 'updater')
                "#,
            )
            .bind(budget_id)
            .execute(&mut **tx)
            .await?;
        }
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_ProjectTable"
                (project_id, project_alias, team_id, budget_id, metadata, models, spend, model_spend, blocked,
                 created_at, created_by, updated_at, updated_by)
            VALUES ($1, 'alias', NULL, $2, '{"k": "v"}', '{m1,m2}', 3.25, '{"m1": 1.0}', true,
                    '2026-01-02 03:04:05.678', 'creator', '2026-02-03 04:05:06.789', 'updater')
            "#,
        )
        .bind(project_id)
        .bind(budget_id)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    #[rstest]
    #[case::with_budget(Some("project-budget"))]
    #[case::without_budget(None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_project_with_its_budget(
        #[case] budget_id: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_project(&mut tx, "project-1", budget_id).await?;

        let found = find_project(&mut *tx, "project-1")
            .await?
            .ok_or("row missing")?;

        assert_eq!(found.project_id, "project-1");
        assert_eq!(found.project_alias.as_deref(), Some("alias"));
        assert_eq!(found.budget_id.as_deref(), budget_id);
        assert_eq!(found.metadata.get(), r#"{"k": "v"}"#);
        assert_eq!(found.models, Some(vec!["m1".to_owned(), "m2".to_owned()]));
        assert_eq!(found.spend, 3.25);
        assert_eq!(found.model_spend.get(), r#"{"m1": 1.0}"#);
        assert!(found.blocked);
        assert_eq!(found.created_at.to_string(), "2026-01-02 03:04:05.678");
        assert_eq!(
            found.litellm_budget_table.as_ref().map(|budget| (
                budget.budget_id.as_str(),
                budget.max_budget,
                budget.rpm_limit,
                budget.model_max_budget.as_ref().map(RawJson::get),
                budget.budget_reset_at.map(|at| at.to_string()),
            )),
            budget_id.map(|budget_id| (
                budget_id,
                Some(12.5),
                Some(7),
                Some(r#"{"gpt": 1.5}"#),
                Some("2026-03-04 05:06:07.890".to_owned()),
            ))
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_row_is_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        assert!(find_project(&pool, "project-missing").await?.is_none());
        Ok(())
    }
}
