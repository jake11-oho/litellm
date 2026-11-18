//! The byte-bounded spend log queue, which keeps a long database outage from growing
//! the proxy's memory without bound.

use std::{collections::VecDeque, sync::Mutex};

use crate::SpendLogRow;

#[derive(Debug, Default)]
struct State {
    rows: VecDeque<SpendLogRow>,
    bytes: usize,
}

/// The in-memory spend log queue, bounded by the bytes its rows dump to.
#[derive(Debug, Default)]
pub struct SpendLogQueue {
    state: Mutex<State>,
}

/// What an enqueue did to the queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Enqueued {
    pub dropped: usize,
}

fn within_budget(rows: &[SpendLogRow], queued_bytes: usize, max_bytes: usize) -> (usize, usize) {
    if queued_bytes <= max_bytes || rows.len() <= 1 {
        return (0, queued_bytes);
    }
    let mut remaining = queued_bytes;
    for (drops, row) in rows[..rows.len() - 1].iter().enumerate() {
        remaining = remaining.saturating_sub(row.bytes());
        if remaining <= max_bytes {
            return (drops + 1, remaining);
        }
    }
    (rows.len() - 1, rows[rows.len() - 1].bytes())
}

impl SpendLogQueue {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Adds `rows` at the head or the tail, then drops the oldest rows until the queue
    /// fits `max_bytes`, always keeping the newest row. A dropped row that carries a
    /// charge comes back as its accounting-only row so the request is still charged.
    pub fn enqueue(&self, rows: Vec<SpendLogRow>, at_head: bool, max_bytes: usize) -> Enqueued {
        let added: usize = rows.iter().map(SpendLogRow::bytes).sum();
        let mut state = self.state();
        let previous = std::mem::take(&mut state.rows);
        let queued: Vec<SpendLogRow> = if at_head {
            rows.into_iter().chain(previous).collect()
        } else {
            previous.into_iter().chain(rows).collect()
        };
        let (dropped, kept_bytes) = within_budget(&queued, state.bytes + added, max_bytes);
        let rescued: Vec<SpendLogRow> = queued[..dropped]
            .iter()
            .filter(|row| row.carries_a_charge())
            .map(SpendLogRow::accounting_only)
            .collect();
        state.bytes = kept_bytes + rescued.iter().map(SpendLogRow::bytes).sum::<usize>();
        state.rows = rescued
            .into_iter()
            .chain(queued.into_iter().skip(dropped))
            .collect();
        Enqueued { dropped }
    }

    /// Takes up to `limit` rows off the head.
    pub fn dequeue(&self, limit: usize) -> Vec<SpendLogRow> {
        let mut state = self.state();
        let take = limit.min(state.rows.len());
        let popped: Vec<SpendLogRow> = state.rows.drain(..take).collect();
        let freed: usize = popped.iter().map(SpendLogRow::bytes).sum();
        state.bytes = state.bytes.saturating_sub(freed);
        popped
    }

    pub fn len(&self) -> usize {
        self.state().rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> usize {
        self.state().bytes
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{Enqueued, SpendLogQueue};
    use crate::SpendLogRow;

    fn row(request_id: &str, charged: bool) -> SpendLogRow {
        let payload = if charged {
            r#", "accounting_payload": {"spend": 1.0}"#
        } else {
            ""
        };
        SpendLogRow::parse(&format!(
            r#"{{"request_id": "{request_id}", "call_type": "c", "messages": "{}"{payload}}}"#,
            "x".repeat(50)
        ))
        .expect("row parses")
    }

    fn ids(rows: &[SpendLogRow]) -> Vec<String> {
        rows.iter().map(SpendLogRow::to_json).collect()
    }

    #[rstest]
    #[case::appends(false, vec!["old", "new"])]
    #[case::replays_at_head(true, vec!["new", "old"])]
    fn keeps_order_within_budget(#[case] at_head: bool, #[case] expected: Vec<&str>) {
        let queue = SpendLogQueue::default();
        queue.enqueue(vec![row("old", false)], false, usize::MAX);
        let outcome = queue.enqueue(vec![row("new", false)], at_head, usize::MAX);

        assert_eq!(outcome, Enqueued { dropped: 0 });
        assert_eq!(
            ids(&queue.dequeue(10)),
            ids(&expected.iter().map(|id| row(id, false)).collect::<Vec<_>>())
        );
        assert_eq!(queue.bytes(), 0);
    }

    #[rstest]
    #[case::drops_oldest_log_only(false, vec![row("c", false)])]
    #[case::keeps_the_charge_of_a_dropped_row(true, vec![row("a", true).accounting_only(), row("c", false)])]
    fn past_the_budget_drops_the_oldest(#[case] charged: bool, #[case] expected: Vec<SpendLogRow>) {
        let queue = SpendLogQueue::default();
        let first = row("a", charged);
        let budget = row("c", false).bytes() + row("b", false).bytes() / 2;
        queue.enqueue(vec![first, row("b", false)], false, usize::MAX);
        let outcome = queue.enqueue(vec![row("c", false)], false, budget);

        assert_eq!(outcome, Enqueued { dropped: 2 });
        let expected_bytes: usize = expected.iter().map(SpendLogRow::bytes).sum();
        assert_eq!(queue.bytes(), expected_bytes);
        assert_eq!(ids(&queue.dequeue(10)), ids(&expected));
    }

    #[rstest]
    #[case::newest_row_over_budget()]
    fn keeps_the_newest_row_whatever_it_costs() {
        let queue = SpendLogQueue::default();
        let outcome = queue.enqueue(vec![row("a", false)], false, 1);

        assert_eq!(outcome, Enqueued { dropped: 0 });
        assert_eq!(queue.len(), 1);
    }

    #[rstest]
    #[case::partial(1, 1, 1)]
    #[case::everything(5, 2, 0)]
    fn dequeues_the_oldest_rows(#[case] limit: usize, #[case] popped: usize, #[case] left: usize) {
        let queue = SpendLogQueue::default();
        queue.enqueue(vec![row("a", false), row("b", false)], false, usize::MAX);

        assert_eq!(queue.dequeue(limit).len(), popped);
        assert_eq!(queue.len(), left);
        assert_eq!(queue.bytes(), left * row("b", false).bytes());
    }
}
