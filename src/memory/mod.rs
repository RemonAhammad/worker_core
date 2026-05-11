//! Auto-extraction of long-term facts from user messages.
//!
//! Runs after each user turn (best-effort). Catches a handful of common
//! self-introduction patterns and persists them via `db::memories::upsert`.
//! Extractor failures are *never* fatal — they're logged and skipped so
//! generation always proceeds.

use sqlx::SqlitePool;

use crate::db::memories as mem_db;
use crate::error::AppError;

/// Inspect `content` and persist any facts we recognize. Returns the number
/// of memories actually inserted (existing duplicates count as 0).
pub async fn extract_and_store(
    pool: &SqlitePool,
    content: &str,
) -> Result<usize, AppError> {
    let facts = extract_facts(content);
    let mut inserted = 0usize;
    for fact in facts {
        // Best-effort: log and continue on individual failures.
        match mem_db::upsert(pool, &fact, "auto").await {
            Ok(_) => inserted += 1,
            Err(e) => tracing::warn!(error = %e, fact = %fact, "failed to persist auto memory"),
        }
    }
    Ok(inserted)
}

/// Pure function — extract candidate facts from a single user message.
/// Lowercase scan; captures the original-cased value from the source string.
fn extract_facts(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = content.to_lowercase();

    // "my name is X", "i'm X", "i am X", "call me X"
    for pattern in &["my name is ", "i'm ", "i am ", "call me "] {
        if let Some(start) = lower.find(pattern) {
            let value_start = start + pattern.len();
            if let Some(value) = take_proper_noun(&content[value_start..]) {
                out.push(format!("The user's name is {value}."));
                break; // one name fact per message
            }
        }
    }

    // "i live in X", "i'm from X", "i am from X"
    for pattern in &["i live in ", "i'm from ", "i am from "] {
        if let Some(start) = lower.find(pattern) {
            let value_start = start + pattern.len();
            if let Some(value) = take_proper_noun(&content[value_start..]) {
                out.push(format!("The user is from {value}."));
                break;
            }
        }
    }

    // "i work at X", "i work for X"
    for pattern in &["i work at ", "i work for "] {
        if let Some(start) = lower.find(pattern) {
            let value_start = start + pattern.len();
            if let Some(value) = take_proper_noun(&content[value_start..]) {
                out.push(format!("The user works at {value}."));
                break;
            }
        }
    }

    out
}

/// Take the first capitalized word (or first word if none capitalized) from
/// `s`, stripping trailing punctuation. Returns `None` if nothing usable.
///
/// The first-word fallback handles lowercase input ("my name is rimon") which
/// is far more common in chat than properly capitalized prose.
fn take_proper_noun(s: &str) -> Option<String> {
    let s = s.trim_start();
    if s.is_empty() {
        return None;
    }
    // Pull the first whitespace-delimited token.
    let token: String = s
        .chars()
        .take_while(|c| !c.is_whitespace())
        .collect();
    // Strip trailing punctuation: . , ! ? ; :
    let trimmed = token.trim_end_matches(|c: char| ".,!?;:".contains(c));
    if trimmed.is_empty() || trimmed.len() > 64 {
        return None;
    }
    // Reject obvious filler ("a", "the", "an").
    let lower = trimmed.to_lowercase();
    if matches!(lower.as_str(), "a" | "the" | "an" | "not" | "no") {
        return None;
    }
    Some(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_my_name_is() {
        let facts = extract_facts("Hi, my name is Rimon and I'm new here.");
        assert!(facts.iter().any(|f| f.contains("Rimon")));
    }

    #[test]
    fn extracts_lowercase_name() {
        let facts = extract_facts("my name is rimon");
        assert!(
            facts.iter().any(|f| f.contains("rimon")),
            "expected lowercase name capture, got: {facts:?}"
        );
    }

    #[test]
    fn extracts_im_pattern() {
        let facts = extract_facts("hey i'm Alice");
        assert!(facts.iter().any(|f| f.contains("Alice")));
    }

    #[test]
    fn strips_trailing_punctuation() {
        // Input "Bob." should produce "...Bob." with exactly one period, not
        // "...Bob.." — i.e. the trailing punctuation was stripped from the
        // captured value before the format string added its own period.
        let facts = extract_facts("my name is Bob.");
        assert!(facts.iter().any(|f| f.contains("Bob") && !f.contains("Bob..")));
    }

    #[test]
    fn empty_on_no_match() {
        let facts = extract_facts("what is the weather like");
        assert!(facts.is_empty());
    }

    #[test]
    fn no_double_name_fact() {
        // Both "my name is" and "i am" would match, but we stop after one.
        let facts = extract_facts("my name is Rimon and i am Rimon");
        let name_facts: Vec<_> = facts.iter().filter(|f| f.contains("name is")).collect();
        assert_eq!(name_facts.len(), 1);
    }
}
