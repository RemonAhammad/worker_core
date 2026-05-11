//! Context manager.
//!
//! Loads a session's full message history, recomputes any missing token
//! counts, and trims to fit the model's context window. Trim policy:
//!
//! 1. Always preserve the system prompt.
//! 2. Always preserve the most recent user message.
//! 3. When over budget, drop the oldest user/assistant *pair* — never split
//!    a turn so the assistant sees a dangling reply with no question.

use sqlx::SqlitePool;

use crate::db::{memories as mem_db, messages as msg_db};
use crate::error::AppError;
use crate::model::engine::{ChatTurn, SharedEngine};
use crate::types::{Message, Role, Session};

/// How many tokens to leave free for the assistant's reply by default.
const DEFAULT_RESPONSE_BUDGET: u32 = 1024;

pub struct ContextManager<'a> {
    pub engine: &'a SharedEngine,
    pub pool: &'a SqlitePool,
}

impl<'a> ContextManager<'a> {
    pub fn new(engine: &'a SharedEngine, pool: &'a SqlitePool) -> Self {
        Self { engine, pool }
    }

    /// Build the chat turn list for `session`, fitting it into
    /// `context_length - response_budget` tokens.
    ///
    /// Pairs are dropped from the oldest end first. The system prompt and
    /// the trailing user message are never dropped.
    pub async fn build(
        &self,
        session: &Session,
        context_length: u32,
        response_budget: Option<u32>,
    ) -> Result<Vec<ChatTurn>, AppError> {
        let response_budget = response_budget.unwrap_or(DEFAULT_RESPONSE_BUDGET);
        let budget = context_length.saturating_sub(response_budget) as i64;

        let history = msg_db::list_for_session(self.pool, session.id).await?;

        // Build the effective system prompt: long-term memories (if any) +
        // the session's configured system prompt. Memories come first so the
        // session prompt can refine or override them.
        let memories = mem_db::list(self.pool).await?;
        let mut system_text = String::new();
        if !memories.is_empty() {
            system_text.push_str(
                "The following facts about the user persist across all conversations. Honor them in every reply:\n",
            );
            for m in &memories {
                system_text.push_str("- ");
                system_text.push_str(&m.content);
                system_text.push('\n');
            }
        }
        if let Some(prompt) = session.system_prompt.as_deref() {
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(prompt);
        }

        let system_turn = if !system_text.is_empty() {
            let tokens = self.engine.count_tokens(&system_text).await? as i64;
            Some((
                ChatTurn {
                    role: Role::System,
                    content: system_text,
                },
                tokens,
            ))
        } else {
            None
        };

        // Recompute token counts for any messages with token_count == 0
        // (older rows or hand-inserted data); use the stored count otherwise.
        let mut counted: Vec<(ChatTurn, i64)> = Vec::with_capacity(history.len());
        for m in &history {
            let tokens = if m.token_count > 0 {
                m.token_count
            } else {
                self.engine.count_tokens(&m.content).await? as i64
            };
            counted.push((
                ChatTurn {
                    role: m.role,
                    content: m.content.clone(),
                },
                tokens,
            ));
        }

        let pairs = group_into_pairs(&counted);
        let trailing_user_idx = trailing_user_index(&counted);

        // Always-include cost: system prompt + trailing user message.
        let mut fixed_cost = system_turn.as_ref().map(|(_, t)| *t).unwrap_or(0);
        if let Some(idx) = trailing_user_idx {
            fixed_cost += counted[idx].1;
        }

        // Decide which historical pairs to keep, newest first.
        let mut pairs_to_keep: Vec<usize> = Vec::new();
        let mut running = fixed_cost;
        for (pi, p) in pairs.iter().enumerate().rev() {
            // Skip the pair that contains the trailing user message; it's
            // already counted in fixed_cost.
            if let Some(t_idx) = trailing_user_idx
                && p.indices.contains(&t_idx)
            {
                continue;
            }
            if running + p.tokens <= budget {
                running += p.tokens;
                pairs_to_keep.push(pi);
            }
        }
        pairs_to_keep.sort();

        // Reassemble: system, kept pairs (in original order), trailing user.
        let mut out: Vec<ChatTurn> = Vec::new();
        if let Some((turn, _)) = system_turn {
            out.push(turn);
        }
        for pi in pairs_to_keep {
            for &idx in &pairs[pi].indices {
                out.push(counted[idx].0.clone());
            }
        }
        if let Some(idx) = trailing_user_idx {
            out.push(counted[idx].0.clone());
        }

        Ok(out)
    }
}

#[derive(Debug)]
struct Pair {
    indices: Vec<usize>,
    tokens: i64,
}

/// Group consecutive non-system messages into user/assistant pairs. A trailing
/// user message with no assistant reply forms a one-element pair.
fn group_into_pairs(history: &[(ChatTurn, i64)]) -> Vec<Pair> {
    let mut pairs = Vec::new();
    let mut current: Option<Pair> = None;
    for (i, (turn, tokens)) in history.iter().enumerate() {
        if matches!(turn.role, Role::System) {
            continue;
        }
        match turn.role {
            Role::User => {
                if let Some(p) = current.take() {
                    pairs.push(p);
                }
                current = Some(Pair {
                    indices: vec![i],
                    tokens: *tokens,
                });
            }
            Role::Assistant | Role::Tool => {
                if let Some(p) = current.as_mut() {
                    p.indices.push(i);
                    p.tokens += *tokens;
                } else {
                    pairs.push(Pair {
                        indices: vec![i],
                        tokens: *tokens,
                    });
                }
            }
            Role::System => {}
        }
    }
    if let Some(p) = current {
        pairs.push(p);
    }
    pairs
}

/// Find the index of the trailing user message (if any) — the last user turn
/// with no assistant reply after it. This is the message we just inserted
/// before generating, and it must be preserved.
fn trailing_user_index(history: &[(ChatTurn, i64)]) -> Option<usize> {
    for (i, (turn, _)) in history.iter().enumerate().rev() {
        match turn.role {
            Role::User => return Some(i),
            Role::Assistant | Role::Tool => return None,
            Role::System => {}
        }
    }
    None
}

#[allow(dead_code)]
pub fn debug_summary(history: &[Message]) -> String {
    history
        .iter()
        .map(|m| format!("{}({}t)", m.role.as_str(), m.token_count))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(role: Role, content: &str, tokens: i64) -> (ChatTurn, i64) {
        (
            ChatTurn {
                role,
                content: content.into(),
            },
            tokens,
        )
    }

    #[test]
    fn pairs_group_user_then_assistant() {
        let h = vec![
            t(Role::User, "q1", 10),
            t(Role::Assistant, "a1", 20),
            t(Role::User, "q2", 5),
            t(Role::Assistant, "a2", 15),
            t(Role::User, "q3", 7),
        ];
        let pairs = group_into_pairs(&h);
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].tokens, 30);
        assert_eq!(pairs[1].tokens, 20);
        assert_eq!(pairs[2].tokens, 7); // trailing user
    }

    #[test]
    fn trailing_user_detected() {
        let h = vec![
            t(Role::User, "q1", 10),
            t(Role::Assistant, "a1", 20),
            t(Role::User, "q2", 5),
        ];
        assert_eq!(trailing_user_index(&h), Some(2));

        let h2 = vec![
            t(Role::User, "q1", 10),
            t(Role::Assistant, "a1", 20),
        ];
        assert_eq!(trailing_user_index(&h2), None);
    }
}
