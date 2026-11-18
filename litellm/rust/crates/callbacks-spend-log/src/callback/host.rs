//! The calls the spend log callback makes back into the Python proxy, for state only Python
//! holds. The alerts, the cached spend and proxy spend bumps and the batch accounting outcome are
//! fire and forget: Python schedules or records them and answers at once. Enrichment, the
//! accounting context and the search free tier counter are awaited because the row needs them.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use litellm_callbacks::HostCall;
use litellm_callbacks_budgets::CustomerWindow;

/// `_enrich_failure_metadata_with_key_info`: the key's and team's identity from the auth cache
/// or DB, for a request whose key carried no user.
#[derive(Debug, Serialize)]
pub struct EnrichKeyIdentity {
    pub api_key: String,
    pub resolve_missing_key_identity: bool,
    pub key_alias: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub project_id: Option<String>,
    pub team_alias: Option<String>,
}

/// The enriched `user_api_key_*` metadata, each `None` when still unknown.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct EnrichedKeyIdentity {
    pub key_alias: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub project_id: Option<String>,
    pub team_alias: Option<String>,
}

impl HostCall for EnrichKeyIdentity {
    const NAME: &'static str = "enrich_key_identity";
    type Reply = EnrichedKeyIdentity;
}

/// The accounting inputs that need the auth cache, the DB, the router and the pricing map: the
/// budget windows of the key, team, user and customer (`accounting_window_keys`), the savings
/// priced from the built row (`_savings_or_zero`), the model access groups the served model debits,
/// whether a Responses turn is replayed from its spend log (`_turn_chains_through_the_db`), and
/// for the counter step the customer window counters and the warmed committed twins.
#[derive(Debug, Serialize)]
pub struct AccountingContext {
    /// The built row, as JSON text.
    pub row: String,
    pub token: Option<String>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub windows: bool,
    pub customer: Option<Map<String, Value>>,
    pub customer_scopes: Option<Vec<Value>>,
    pub model_access_groups: Vec<String>,
    pub org_id: Option<String>,
    pub end_user_id: Option<String>,
    /// The request carried no committed twins and charges a positive cost, so the host warms
    /// them (`warm_committed_counter_keys`) when committed spend is tracked.
    pub warm_committed: bool,
    /// The row is a Responses turn, so the host asks the router whether its next turn reads it
    /// back from the database.
    pub responses_turn: bool,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Accounting {
    pub windows: Vec<Value>,
    pub savings: Value,
    pub model_access_groups: Vec<String>,
    pub customer_windows: Vec<CustomerWindow>,
    pub committed_tracked: Vec<String>,
    pub chains_through_db: bool,
}

impl HostCall for AccountingContext {
    const NAME: &'static str = "accounting_context";
    type Reply = Accounting;
}

/// `_add_spend_to_global_proxy_spend`, which the proxy `max_budget` check reads.
#[derive(Debug, Serialize)]
pub struct IncrementProxySpend {
    pub response_cost: f64,
}

impl HostCall for IncrementProxySpend {
    const NAME: &'static str = "increment_proxy_spend";
    type Reply = ();
}

/// The cached key's projected soft budget alert from `update_cache`.
#[derive(Debug, Serialize)]
pub struct SoftBudgetAlert {
    pub token: String,
    pub response_cost: f64,
}

impl HostCall for SoftBudgetAlert {
    const NAME: &'static str = "soft_budget_alert";
    type Reply = ();
}

/// `slack_alerting_instance.customer_spend_alert`.
#[derive(Debug, Serialize)]
pub struct CustomerSpendAlert {
    pub token: Option<String>,
    pub key_alias: Option<String>,
    pub end_user_id: Option<String>,
    pub response_cost: f64,
    pub max_budget: Option<f64>,
}

impl HostCall for CustomerSpendAlert {
    const NAME: &'static str = "customer_spend_alert";
    type Reply = ();
}

/// `proxy_logging_obj.failed_tracking_alert`, raised when the cost callback cannot track a
/// request's cost.
#[derive(Debug, Serialize)]
pub struct FailedTrackingAlert {
    pub error_message: String,
    pub failing_model: String,
}

impl HostCall for FailedTrackingAlert {
    const NAME: &'static str = "failed_tracking_alert";
    type Reply = ();
}

/// The `BatchAccountingOutcome` CheckBatchCost reads to decide whether a batch's costing may be
/// retried. It lives in a context variable of the task that logs the batch, so only Python can
/// set it; the host call runs inline in that task.
#[derive(Debug, Serialize)]
pub struct BatchAccounting {
    pub call_id: String,
    pub write_started: bool,
    pub failed: bool,
}

impl HostCall for BatchAccounting {
    const NAME: &'static str = "batch_accounting";
    type Reply = ();
}

/// `add_spend_to_cached_spend_objects`: the cached user, end user, team and tag objects' spend.
#[derive(Debug, Serialize)]
pub struct CachedSpendBump {
    pub user_id: Option<String>,
    pub end_user_id: Option<String>,
    pub team_id: Option<String>,
    pub tags: Option<Vec<String>>,
    pub response_cost: f64,
}

impl HostCall for CachedSpendBump {
    const NAME: &'static str = "cached_spend_bump";
    type Reply = ();
}

/// The search free tier's shared allowance counter in the internal usage cache's Redis, which
/// the Python path also counts in and which may not be the spend counter store's Redis. The reply
/// is the total after the increment, or `None` when the proxy has no Redis.
#[derive(Debug, Serialize)]
pub struct IncrementSearchFreeTier {
    pub key: String,
    pub units: i64,
    pub ttl: i64,
}

impl HostCall for IncrementSearchFreeTier {
    const NAME: &'static str = "increment_search_free_tier";
    type Reply = Option<f64>;
}
