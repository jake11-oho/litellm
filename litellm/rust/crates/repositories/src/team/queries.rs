//! `LiteLLM_TeamTable` reads for auth, by id or by alias, with each team's `LiteLLM_ModelTable` aliases nested in by the SQL.

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use crate::RawJson;

/// One `LiteLLM_TeamTable` row as auth reads it, with its JSON columns kept opaque for the host's codec.
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Serialize)]
pub struct Team {
    pub(crate) team_id: String,
    pub(crate) team_alias: Option<String>,
    pub(crate) organization_id: Option<String>,
    pub(crate) admins: Vec<String>,
    pub(crate) members: Vec<String>,
    pub(crate) members_with_roles: RawJson,
    pub(crate) metadata: RawJson,
    pub(crate) max_budget: Option<f64>,
    pub(crate) soft_budget: Option<f64>,
    pub(crate) spend: f64,
    pub(crate) models: Vec<String>,
    pub(crate) max_parallel_requests: Option<i32>,
    pub(crate) tpm_limit: Option<i64>,
    pub(crate) rpm_limit: Option<i64>,
    pub(crate) tpd_limit: Option<i64>,
    pub(crate) budget_duration: Option<String>,
    #[serde(serialize_with = "crate::utc_timestamp::option::serialize")]
    pub(crate) budget_reset_at: Option<NaiveDateTime>,
    pub(crate) blocked: bool,
    #[serde(serialize_with = "crate::utc_timestamp::serialize")]
    pub(crate) created_at: NaiveDateTime,
    #[serde(serialize_with = "crate::utc_timestamp::serialize")]
    pub(crate) updated_at: NaiveDateTime,
    pub(crate) model_spend: RawJson,
    pub(crate) model_max_budget: RawJson,
    pub(crate) router_settings: Option<RawJson>,
    pub(crate) team_member_permissions: Vec<String>,
    pub(crate) access_group_ids: Vec<String>,
    pub(crate) policies: Vec<String>,
    pub(crate) default_team_member_models: Vec<String>,
    pub(crate) budget_limits: Option<RawJson>,
    pub(crate) model_id: Option<i32>,
    pub(crate) allow_team_guardrail_config: bool,
    pub(crate) object_permission_id: Option<String>,
    pub(crate) litellm_model_table: Option<Json<TeamModelAliases>>,
}

/// The team's `LiteLLM_ModelTable` row, which holds its model aliases, nested in by the SQL as `to_jsonb(m)`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub(crate) struct TeamModelAliases {
    pub(crate) id: i32,
    #[serde(rename(deserialize = "aliases"))]
    pub(crate) model_aliases: Option<RawJson>,
    #[serde(serialize_with = "crate::utc_timestamp::serialize")]
    pub(crate) created_at: NaiveDateTime,
    pub(crate) created_by: String,
    #[serde(serialize_with = "crate::utc_timestamp::serialize")]
    pub(crate) updated_at: NaiveDateTime,
    pub(crate) updated_by: String,
}

/// The team `team_id`, or `None` when no row has that id.
pub(super) async fn find_team(
    executor: impl PgExecutor<'_>,
    team_id: &str,
) -> Result<Option<Team>, Error> {
    Ok(
        sqlx::query_file_as!(Team, "src/team/queries/find_team.sql", team_id)
            .fetch_optional(executor)
            .await?,
    )
}

/// Up to two teams whose alias is `team_alias`, so a caller can tell a unique alias from a shared one.
pub(super) async fn find_teams_by_alias(
    executor: impl PgExecutor<'_>,
    team_alias: &str,
) -> Result<Vec<Team>, Error> {
    Ok(
        sqlx::query_file_as!(Team, "src/team/queries/find_teams_by_alias.sql", team_alias)
            .fetch_all(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{find_team, find_teams_by_alias};
    use crate::test_support::database_pool;
    use rstest::rstest;
    use sqlx::PgPool;

    #[rstest]
    #[case::with_model_aliases(true)]
    #[case::without_model_aliases(false)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_team_with_its_model_aliases(
        #[case] with_aliases: bool,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let model_id: Option<i32> = if with_aliases {
            Some(
                sqlx::query_scalar(
                    r#"INSERT INTO "LiteLLM_ModelTable" (aliases, created_by, updated_by)
                       VALUES ('{"gpt": "gpt-4o"}', 'a', 'b') RETURNING id"#,
                )
                .fetch_one(&mut *tx)
                .await?,
            )
        } else {
            None
        };
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let _inserted = sqlx::query(
            r#"INSERT INTO "LiteLLM_TeamTable"
                   (team_id, team_alias, models, members_with_roles, metadata, tpm_limit,
                    budget_reset_at, created_at, updated_at, model_id, admins)
               VALUES ('find-team', 'alias', '{gpt-4o}', '[{"role": "admin", "user_id": "u"}]',
                       '{"k": 1}', 9007199254740993, '2026-01-02 03:04:05.678',
                       '2026-01-01', '2026-01-01', $1, NULL)"#,
        )
        .bind(model_id)
        .execute(&mut *tx)
        .await?;

        let team = find_team(&mut *tx, "find-team")
            .await?
            .ok_or("team missing")?;

        assert_eq!(team.team_alias.as_deref(), Some("alias"));
        assert_eq!(team.models, ["gpt-4o"]);
        assert!(team.admins.is_empty());
        assert_eq!(team.tpm_limit, Some(9_007_199_254_740_993));
        assert_eq!(
            team.members_with_roles.get(),
            r#"[{"role": "admin", "user_id": "u"}]"#
        );
        assert_eq!(team.model_id, model_id);
        assert_eq!(
            team.litellm_model_table.as_ref().map(|aliases| (
                aliases.id,
                aliases
                    .model_aliases
                    .as_ref()
                    .map(|json| json.get().to_owned())
            )),
            model_id.map(|id| (id, Some(r#"{"gpt": "gpt-4o"}"#.to_owned())))
        );
        let encoded = serde_json::to_value(&team)?;
        assert_eq!(encoded["budget_reset_at"], "2026-01-02T03:04:05.678+00:00");
        assert_eq!(encoded["created_at"], "2026-01-01T00:00:00+00:00");
        if with_aliases {
            assert!(
                encoded["litellm_model_table"]["created_at"]
                    .as_str()
                    .is_some_and(|at| at.ends_with("+00:00"))
            );
            assert_eq!(
                encoded["litellm_model_table"]["model_aliases"],
                serde_json::json!({"gpt": "gpt-4o"})
            );
        }
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_team_is_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        assert!(find_team(&pool, "no-such-team").await?.is_none());
        Ok(())
    }

    #[rstest]
    #[case::unique("unique-alias", &["by-alias-a"])]
    #[case::shared("shared-alias", &["by-alias-b", "by-alias-c"])]
    #[case::missing("no-such-alias", &[])]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_teams_by_alias(
        #[case] alias: &str,
        #[case] expected: &[&str],
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        for (team_id, team_alias) in [
            ("by-alias-a", "unique-alias"),
            ("by-alias-b", "shared-alias"),
            ("by-alias-c", "shared-alias"),
            ("by-alias-d", "shared-alias"),
        ] {
            #[expect(
                clippy::disallowed_methods,
                reason = "Fixture writes are outside the compile-checked query set"
            )]
            let _inserted = sqlx::query(
                r#"INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, created_at, updated_at)
                   VALUES ($1, $2, '2026-01-01', '2026-01-01')"#,
            )
            .bind(team_id)
            .bind(team_alias)
            .execute(&mut *tx)
            .await?;
        }

        let teams = find_teams_by_alias(&mut *tx, alias).await?;

        assert_eq!(
            teams
                .iter()
                .map(|team| team.team_id.as_str())
                .collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }
}
