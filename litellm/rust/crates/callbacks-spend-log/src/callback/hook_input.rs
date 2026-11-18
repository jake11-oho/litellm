//! The `spend_log_input` field, typed. The Python derivation
//! (`litellm/proxy/spend_tracking/rust_spend_log_inputs.py`) reads what only Python holds (the
//! response object, the auth object, the router, the settings) and hands over one document per
//! hook: the proxy flags the writer reads, the identity the request is charged to, the raw facts
//! about the response and the logging payload, and each row's builder input and cost. Every
//! decision over those facts is made in Rust ([`super::policy`]).

use litellm_callbacks_budgets::CustomerScopeSpend;
use serde::Deserialize;
use serde_json::Value;

/// The proxy settings the cost callback and `update_database` read on each request.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub disable_spend_updates: bool,
    pub disable_spend_logs: bool,
    pub disable_error_logs: bool,
    pub disable_responses_id_security: bool,
    /// `spend_log_row_config()`, the row builder's settings.
    pub row_config: Value,
    /// `litellm_proxy_budget_name` when a proxy `max_budget` is set.
    pub proxy_budget_name: Option<String>,
    /// `litellm.apply_search_free_tier`.
    pub apply_search_free_tier: bool,
    /// `BACKGROUND_INTERACTION_COST_POLLING_ENABLED`.
    pub background_interaction_cost_polling: bool,
}

/// Who a request is charged to, from the proxy-stamped `user_api_key_*` metadata (success) or
/// the auth object (failure).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Identity {
    /// The key as logged: a hash, or a raw `sk-` key the writer hashes.
    pub token: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub org_id: Option<String>,
    pub project_id: Option<String>,
    pub end_user_id: Option<String>,
    pub key_alias: Option<String>,
    pub agent_id: Option<String>,
}

/// One spend log row to write: the row builder's input and what the row charges.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct RowRequest {
    pub row: Value,
    pub response_cost: Option<f64>,
    /// `get_request_model_access_groups`, before the router narrows them to the served model.
    pub model_access_groups: Vec<String>,
    /// `customer_hierarchy_ids_from_request(...).entities()`.
    pub customer_entities: Option<serde_json::Map<String, Value>>,
    /// `customer_budget_scopes_from_request`, the customer scopes admission granted, each as
    /// its JSON record, or `None` when the request carries no snapshot.
    pub customer_scopes: Option<Vec<Value>>,
    /// The entity counters whose committed twin the request carried, `None` when it carried
    /// none and the accounting context decides.
    pub committed_tracked: Option<Vec<String>>,
    /// `user_api_key_customer_hierarchy`'s scopes, settled first by the counter step.
    pub customer_hierarchy: Vec<CustomerScopeSpend>,
}

/// What only the response object can tell: a batch whose cost is not final yet
/// (`defers_batch_cost`), and an interaction create with no usage
/// (`_is_unbilled_interaction_response`) that may still be running in the background.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ResponseFacts {
    pub defers_batch_cost: bool,
    pub unbilled_interaction: bool,
    pub interaction_in_progress: bool,
}

/// The logging payload facts the row builder input leaves out.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct PayloadFacts {
    /// `cost_finalized`: the cost was settled before the callback, allowance included.
    pub cost_finalized: bool,
    pub cache_hit: bool,
    /// `str(response_cost_failure_debug_info)` when it is set.
    pub cost_failure_debug_info: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Success {
    pub settings: Settings,
    pub identity: Identity,
    pub call_type: Option<String>,
    pub response: ResponseFacts,
    pub payload: PayloadFacts,
    /// `kwargs["stream"] is True`.
    pub stream: bool,
    /// The kwargs carry `complete_streaming_response`.
    pub stream_complete: bool,
    pub end_user_max_budget: Option<f64>,
    /// A CheckBatchCost costing waits on this request's `BatchAccountingOutcome`.
    pub batch_accounting: bool,
    /// The `window_reset_at` of each reservation entry, which the typed reservation lacks.
    pub window_resets: Vec<Option<String>>,
    pub row: RowRequest,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Failure {
    pub settings: Settings,
    pub identity: Identity,
    /// The failure is no `DBLookupDeadlineExceeded`, so the key identity may be looked up.
    pub enrich: bool,
    /// The standard logging payload says `cost_finalized`.
    pub completed_cost: bool,
    /// The failure is one the proxy writes rows for (error logs on, a trackable route).
    pub track: bool,
    pub window_resets: Vec<Option<String>>,
    pub rows: Vec<RowRequest>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpendLogInput {
    Success(Box<Success>),
    Failure(Box<Failure>),
}
