//! The post-call spend counter fan-out's decisions: which `spend:*` counters one charged request
//! bumps.
//!
//! The functions here are pure. They pick every counter key for a charge, skip the ones a budget
//! reservation already holds, name the auth cache entries a cold counter falls back to, and say
//! which scopes may warm their counters concurrently. [`crate::charge`] runs the plan against the
//! spend counter store.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const COMMITTED_TWIN_PREFIXES: [&str; 5] = [
    "spend:key:",
    "spend:team:",
    "spend:user:",
    "spend:org:",
    "spend:end_user:",
];

/// One charged request as the proxy's cost callback sees it. A tag or access group that is not a
/// string arrives as `None` and names no counter.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Charge {
    pub token: Option<String>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub org_id: Option<String>,
    pub end_user_id: Option<String>,
    pub project_id: Option<String>,
    pub tags: Option<Vec<Option<String>>>,
    pub model_access_groups: Option<Vec<Option<String>>>,
    pub response_cost: Option<f64>,
    pub reserved_counter_keys: Vec<String>,
    pub committed_counter_keys: Vec<String>,
    pub customer_windows: Vec<CustomerWindow>,
}

/// A customer hierarchy budget counter the request was admitted against. Only the ones with a
/// `window_duration` are window counters the fan-out charges.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CustomerWindow {
    pub counter_key: String,
    pub entity_type: String,
    pub entity_id: String,
    pub window_duration: Option<String>,
    /// The start of the window the request falls in, ISO 8601; `None` when it is not valid.
    pub window_start: Option<String>,
}

/// An entity whose budget windows live on its cached auth object. The strum string is the entity
/// label the window reseed reads spend logs by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, strum::IntoStaticStr)]
pub enum WindowedEntity {
    #[serde(rename = "key")]
    #[strum(serialize = "Key")]
    Key,
    #[serde(rename = "team")]
    #[strum(serialize = "Team")]
    Team,
    #[serde(rename = "user")]
    #[strum(serialize = "User")]
    User,
}

/// One step of a scope, run by the proxy.
///
/// `Counter` warms a lifetime counter and charges it. `CachedWindows` reads the entity's budget
/// windows from the auth cache and charges each through [`window_counters`], or defers to the
/// stored row when the entity is not cached. `CustomerWindow` charges the customer counter at
/// `index` of the charge's `customer_windows`. `Committed` seeds and charges the committed twins.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "task", rename_all = "snake_case")]
pub enum Task {
    Counter {
        counter_key: String,
        source_cache_keys: Vec<String>,
    },
    CachedWindows {
        entity_type: WindowedEntity,
        entity_id: String,
        cache_key: String,
    },
    CustomerWindow {
        index: usize,
        counter_key: String,
        entity_label: String,
        reserved: bool,
    },
    Committed {
        counter_keys: Vec<String>,
    },
}

/// A group of tasks for one entity. A sequential scope stops at its first failure and drops what it
/// already prepared. A concurrent scope runs every task and keeps each failure as its own item.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Scope {
    pub concurrent: bool,
    pub tasks: Vec<Task>,
}

impl Scope {
    fn sequential(tasks: impl IntoIterator<Item = Task>) -> Self {
        Self {
            concurrent: false,
            tasks: tasks.into_iter().collect(),
        }
    }

    fn concurrent(tasks: impl IntoIterator<Item = Task>) -> Self {
        Self {
            concurrent: true,
            tasks: tasks.into_iter().collect(),
        }
    }
}

/// Every scope one charge touches, in the order their increments join the pipeline.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChargePlan {
    pub scopes: Vec<Scope>,
}

/// One budget window's counter, and whether a reservation already holds it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WindowCounter {
    pub counter_key: String,
    pub entity_label: &'static str,
    pub reserved: bool,
}

/// Whether a counter keeps a charges-only committed twin: lifetime key, team, user, org and end
/// user counters, never a window.
pub fn has_committed_twin(counter_key: &str) -> bool {
    COMMITTED_TWIN_PREFIXES
        .iter()
        .any(|prefix| counter_key.starts_with(prefix))
        && !counter_key.contains(":window:")
}

/// The tracked counters whose committed twin takes this charge, none unless the cost is positive.
pub fn committed_counter_keys(tracked: &[String], response_cost: Option<f64>) -> Vec<String> {
    match response_cost {
        Some(cost) if cost > 0.0 => tracked
            .iter()
            .filter(|key| has_committed_twin(key))
            .cloned()
            .collect(),
        _ => Vec::new(),
    }
}

fn named(values: Option<&[Option<String>]>) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .unwrap_or_default()
        .iter()
        .flatten()
        .filter(|name| !name.is_empty() && seen.insert(name.as_str()))
        .cloned()
        .collect()
}

/// The cost the fan-out charges, `None` when there is no cost or it is zero.
pub fn charged(response_cost: Option<f64>) -> Option<f64> {
    response_cost.filter(|cost| *cost != 0.0)
}

fn hashed_token(token: &str) -> String {
    if token.starts_with("sk-") {
        format!("{:x}", Sha256::digest(token.as_bytes()))
    } else {
        token.to_owned()
    }
}

fn entity_label(entity_type: &str) -> String {
    match entity_type {
        "key" => "Key",
        "team" => "Team",
        "user" => "User",
        other => other,
    }
    .to_owned()
}

struct Planner<'a> {
    reserved: HashSet<&'a str>,
}

impl Planner<'_> {
    fn counter(&self, counter_key: String, source_cache_keys: Vec<String>) -> Option<Task> {
        (!self.reserved.contains(counter_key.as_str())).then_some(Task::Counter {
            counter_key,
            source_cache_keys,
        })
    }
}

/// The scopes and tasks for one charge, with reserved counters already left out.
pub fn plan(charge: &Charge) -> ChargePlan {
    let planner = Planner {
        reserved: charge
            .reserved_counter_keys
            .iter()
            .map(String::as_str)
            .collect(),
    };
    let mut scopes = Vec::new();
    if !charge.customer_windows.is_empty() {
        scopes.push(Scope::concurrent(
            charge
                .customer_windows
                .iter()
                .enumerate()
                .filter(|(_, window)| window.window_duration.is_some())
                .map(|(index, window)| Task::CustomerWindow {
                    index,
                    counter_key: window.counter_key.clone(),
                    entity_label: entity_label(&window.entity_type),
                    reserved: planner.reserved.contains(window.counter_key.as_str()),
                }),
        ));
    }
    if let Some(token) = &charge.token {
        let hashed = hashed_token(token);
        scopes.push(Scope::sequential(
            planner
                .counter(format!("spend:key:{hashed}"), vec![hashed.clone()])
                .into_iter()
                .chain([Task::CachedWindows {
                    entity_type: WindowedEntity::Key,
                    entity_id: hashed.clone(),
                    cache_key: hashed,
                }]),
        ));
    }
    if let Some(team_id) = &charge.team_id {
        let cache_key = format!("team_id:{team_id}");
        scopes.push(Scope::sequential(
            planner
                .counter(format!("spend:team:{team_id}"), vec![cache_key.clone()])
                .into_iter()
                .chain([Task::CachedWindows {
                    entity_type: WindowedEntity::Team,
                    entity_id: team_id.clone(),
                    cache_key,
                }]),
        ));
    }
    if let (Some(user_id), Some(team_id)) = (&charge.user_id, &charge.team_id) {
        scopes.push(Scope::sequential(planner.counter(
            format!("spend:team_member:{user_id}:{team_id}"),
            vec![format!("team_membership:{user_id}:{team_id}")],
        )));
    }
    if let Some(user_id) = &charge.user_id {
        scopes.push(Scope::sequential(
            planner
                .counter(format!("spend:user:{user_id}"), vec![user_id.clone()])
                .into_iter()
                .chain([Task::CachedWindows {
                    entity_type: WindowedEntity::User,
                    entity_id: user_id.clone(),
                    cache_key: user_id.clone(),
                }]),
        ));
    }
    if charge.end_user_id.is_some() || charge.tags.is_some() {
        let end_user = charge.end_user_id.iter().map(|end_user| {
            (
                format!("spend:end_user:{end_user}"),
                format!("end_user_id:{end_user}"),
            )
        });
        let tags = named(charge.tags.as_deref())
            .into_iter()
            .map(|tag| (format!("spend:tag:{tag}"), format!("tag:{tag}")));
        scopes.push(Scope::concurrent(end_user.chain(tags).filter_map(
            |(counter_key, cache_key)| planner.counter(counter_key, vec![cache_key]),
        )));
    }
    if charge
        .model_access_groups
        .as_deref()
        .is_some_and(|groups| !groups.is_empty())
    {
        scopes.push(Scope::concurrent(
            named(charge.model_access_groups.as_deref())
                .into_iter()
                .filter_map(|group| {
                    planner.counter(
                        format!("spend:model_access_group:{group}"),
                        vec![format!("model_access_group:{group}")],
                    )
                }),
        ));
    }
    if let Some(org_id) = &charge.org_id {
        scopes.push(Scope::sequential(planner.counter(
            format!("spend:org:{org_id}"),
            vec![
                format!("org_id:{org_id}:with_budget"),
                format!("org_id:{org_id}"),
            ],
        )));
    }
    if let Some(project_id) = &charge.project_id {
        scopes.push(Scope::sequential(planner.counter(
            format!("spend:project:{project_id}"),
            vec![format!("project_id:{project_id}")],
        )));
    }
    if !charge.committed_counter_keys.is_empty() {
        scopes.push(Scope::sequential([Task::Committed {
            counter_keys: charge.committed_counter_keys.clone(),
        }]));
    }
    ChargePlan { scopes }
}

/// The counter for each of an entity's budget windows, named by its duration.
pub fn window_counters(
    entity_type: WindowedEntity,
    entity_id: &str,
    durations: &[String],
    reserved_counter_keys: &[String],
) -> Vec<WindowCounter> {
    durations
        .iter()
        .map(|duration| {
            let counter_key = match entity_type {
                WindowedEntity::Key => format!("spend:key:{entity_id}:window:{duration}"),
                WindowedEntity::User => format!("spend:user_window:{entity_id}:{duration}"),
                WindowedEntity::Team => format!("spend:team:{entity_id}:window:{duration}"),
            };
            WindowCounter {
                reserved: reserved_counter_keys.contains(&counter_key),
                counter_key,
                entity_label: entity_type.into(),
            }
        })
        .collect()
}

/// A customer scope's lifetime or period counter and the spend its auth snapshot recorded.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CustomerScopeSpend {
    pub counter_key: String,
    pub spend: f64,
}

/// The writes that settle one charge on the customer counters: cold counters are seeded with their
/// snapshot spend first, then every scope takes the cost.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CustomerSettlement {
    pub seeds: Vec<(String, f64)>,
    pub increments: Vec<(String, f64)>,
}

/// Seeds every counter whose reading came back empty, then charges every scope.
pub fn customer_settlement(
    scopes: &[CustomerScopeSpend],
    readings: &[Option<f64>],
    cost: f64,
) -> CustomerSettlement {
    CustomerSettlement {
        seeds: scopes
            .iter()
            .zip(readings)
            .filter(|(_, reading)| reading.is_none())
            .map(|(scope, _)| (scope.counter_key.clone(), scope.spend))
            .collect(),
        increments: scopes
            .iter()
            .map(|scope| (scope.counter_key.clone(), cost))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn counter_keys(plan: &ChargePlan) -> Vec<Vec<String>> {
        plan.scopes
            .iter()
            .map(|scope| {
                scope
                    .tasks
                    .iter()
                    .map(|task| match task {
                        Task::Counter { counter_key, .. }
                        | Task::CustomerWindow { counter_key, .. } => counter_key.clone(),
                        Task::CachedWindows { cache_key, .. } => format!("windows:{cache_key}"),
                        Task::Committed { counter_keys } => counter_keys.join(","),
                    })
                    .collect()
            })
            .collect()
    }

    fn full_charge() -> Charge {
        Charge {
            token: Some("hashed".into()),
            team_id: Some("t1".into()),
            user_id: Some("u1".into()),
            org_id: Some("o1".into()),
            end_user_id: Some("eu1".into()),
            project_id: Some("p1".into()),
            tags: Some(vec![
                Some("a".into()),
                Some(String::new()),
                None,
                Some("b".into()),
                Some("a".into()),
            ]),
            model_access_groups: Some(vec![Some("g".into()), None, Some("g".into())]),
            response_cost: Some(1.5),
            committed_counter_keys: vec!["spend:key:hashed".into()],
            customer_windows: vec![
                CustomerWindow {
                    counter_key: "spend:customer_org:c:window:1mo@x".into(),
                    entity_type: "customer_org".into(),
                    window_duration: Some("1mo".into()),
                    ..CustomerWindow::default()
                },
                CustomerWindow {
                    counter_key: "spend:customer_org:c:lifetime".into(),
                    entity_type: "customer_org".into(),
                    window_duration: None,
                    ..CustomerWindow::default()
                },
            ],
            ..Charge::default()
        }
    }

    #[test]
    fn plan_orders_scopes_and_counters_like_the_python_fan_out() {
        assert_eq!(
            counter_keys(&plan(&full_charge())),
            vec![
                vec!["spend:customer_org:c:window:1mo@x".to_owned()],
                vec!["spend:key:hashed".into(), "windows:hashed".into()],
                vec!["spend:team:t1".into(), "windows:team_id:t1".into()],
                vec!["spend:team_member:u1:t1".into()],
                vec!["spend:user:u1".into(), "windows:u1".into()],
                vec![
                    "spend:end_user:eu1".into(),
                    "spend:tag:a".into(),
                    "spend:tag:b".into()
                ],
                vec!["spend:model_access_group:g".into()],
                vec!["spend:org:o1".into()],
                vec!["spend:project:p1".into()],
                vec!["spend:key:hashed".into()],
            ]
        );
    }

    #[test]
    fn reserved_counters_are_skipped_but_their_windows_still_resolve() {
        let charge = Charge {
            reserved_counter_keys: vec![
                "spend:key:hashed".into(),
                "spend:org:o1".into(),
                "spend:tag:a".into(),
                "spend:customer_org:c:window:1mo@x".into(),
            ],
            ..full_charge()
        };
        let keys = counter_keys(&plan(&charge));
        assert_eq!(keys[1], vec!["windows:hashed".to_owned()]);
        assert_eq!(
            keys[5],
            vec!["spend:end_user:eu1".to_owned(), "spend:tag:b".into()]
        );
        assert!(keys[7].is_empty());
        assert!(matches!(
            plan(&charge).scopes[0].tasks[0],
            Task::CustomerWindow {
                reserved: true,
                index: 0,
                ..
            }
        ));
    }

    #[test]
    fn source_cache_keys_name_the_cached_auth_objects() {
        let plan = plan(&full_charge());
        let sources: Vec<Vec<String>> = plan
            .scopes
            .iter()
            .flat_map(|scope| &scope.tasks)
            .filter_map(|task| match task {
                Task::Counter {
                    source_cache_keys, ..
                } => Some(source_cache_keys.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            sources,
            vec![
                vec!["hashed".to_owned()],
                vec!["team_id:t1".into()],
                vec!["team_membership:u1:t1".into()],
                vec!["u1".into()],
                vec!["end_user_id:eu1".into()],
                vec!["tag:a".into()],
                vec!["tag:b".into()],
                vec!["model_access_group:g".into()],
                vec!["org_id:o1:with_budget".into(), "org_id:o1".into()],
                vec!["project_id:p1".into()],
            ]
        );
    }

    #[test]
    fn a_raw_key_is_hashed_before_it_names_a_counter() {
        let plan = plan(&Charge {
            token: Some("sk-1234".into()),
            ..Charge::default()
        });
        let hashed = format!("{:x}", Sha256::digest(b"sk-1234"));
        assert_eq!(
            plan.scopes[0].tasks[0],
            Task::Counter {
                counter_key: format!("spend:key:{hashed}"),
                source_cache_keys: vec![hashed],
            }
        );
    }

    #[rstest]
    #[case::empty_tags_keep_an_empty_scope(Some(vec![]), None, 1)]
    #[case::empty_groups_have_no_scope(None, Some(vec![]), 0)]
    #[case::no_tags_no_scope(None, None, 0)]
    fn empty_lists_follow_the_python_truthiness(
        #[case] tags: Option<Vec<Option<String>>>,
        #[case] model_access_groups: Option<Vec<Option<String>>>,
        #[case] scopes: usize,
    ) {
        let plan = plan(&Charge {
            tags,
            model_access_groups,
            ..Charge::default()
        });
        assert_eq!(plan.scopes.len(), scopes);
        assert!(plan.scopes.iter().all(|scope| scope.tasks.is_empty()));
    }

    #[rstest]
    #[case::key("spend:key:k", true)]
    #[case::team("spend:team:t", true)]
    #[case::user("spend:user:u", true)]
    #[case::org("spend:org:o", true)]
    #[case::end_user("spend:end_user:e", true)]
    #[case::key_window("spend:key:k:window:1d", false)]
    #[case::team_named_like_a_window("spend:team:x:window:", false)]
    #[case::user_window("spend:user_window:u:1d", false)]
    #[case::tag("spend:tag:t", false)]
    #[case::team_member("spend:team_member:u:t", false)]
    fn committed_twins_exist_only_for_lifetime_entity_counters(
        #[case] key: &str,
        #[case] twin: bool,
    ) {
        assert_eq!(has_committed_twin(key), twin);
    }

    #[rstest]
    #[case::charged(Some(0.5), vec!["spend:key:k", "spend:org:o"])]
    #[case::zero(Some(0.0), vec![])]
    #[case::negative(Some(-1.0), vec![])]
    #[case::unknown(None, vec![])]
    fn committed_keys_only_take_a_positive_charge(
        #[case] cost: Option<f64>,
        #[case] expected: Vec<&str>,
    ) {
        let tracked = vec![
            "spend:key:k".to_owned(),
            "spend:tag:t".into(),
            "spend:org:o".into(),
        ];
        assert_eq!(committed_counter_keys(&tracked, cost), expected);
    }

    #[rstest]
    #[case::none(None, None)]
    #[case::zero(Some(0.0), None)]
    #[case::negative_zero(Some(-0.0), None)]
    #[case::refund(Some(-2.0), Some(-2.0))]
    #[case::cost(Some(1e-9), Some(1e-9))]
    fn only_a_nonzero_cost_is_charged(#[case] cost: Option<f64>, #[case] expected: Option<f64>) {
        assert_eq!(charged(cost), expected);
    }

    #[rstest]
    #[case::key(WindowedEntity::Key, "spend:key:id:window:1d", "Key")]
    #[case::team(WindowedEntity::Team, "spend:team:id:window:1d", "Team")]
    #[case::user(WindowedEntity::User, "spend:user_window:id:1d", "User")]
    fn window_counters_name_each_budget_window(
        #[case] entity: WindowedEntity,
        #[case] key: &str,
        #[case] label: &str,
    ) {
        let counters = window_counters(entity, "id", &["1d".into(), "7d".into()], &[key.into()]);
        assert_eq!(
            counters[0],
            WindowCounter {
                counter_key: key.into(),
                entity_label: match entity {
                    WindowedEntity::Key => "Key",
                    WindowedEntity::Team => "Team",
                    WindowedEntity::User => "User",
                },
                reserved: true,
            }
        );
        assert_eq!(counters[0].entity_label, label);
        assert!(!counters[1].reserved);
        assert!(counters[1].counter_key.ends_with("7d"));
    }

    #[test]
    fn customer_settlement_seeds_only_cold_counters_and_charges_every_scope() {
        let scopes = [
            CustomerScopeSpend {
                counter_key: "spend:customer_org:a:lifetime".into(),
                spend: 2.0,
            },
            CustomerScopeSpend {
                counter_key: "spend:customer_team:b:lifetime".into(),
                spend: 3.0,
            },
        ];
        assert_eq!(
            customer_settlement(&scopes, &[None, Some(9.0)], 0.25),
            CustomerSettlement {
                seeds: vec![("spend:customer_org:a:lifetime".into(), 2.0)],
                increments: vec![
                    ("spend:customer_org:a:lifetime".into(), 0.25),
                    ("spend:customer_team:b:lifetime".into(), 0.25),
                ],
            }
        );
    }
}
