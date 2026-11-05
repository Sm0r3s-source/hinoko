//! Hinoko's built-in prompt for plain chats.

use crate::computer;

const CHAT: &str = include_str!("../../../assets/default-system-prompt.md");

/// Knowledge cut-off told to the model. It is the person's model that knows its own;
/// this line only keeps answers about dates honest.
const CUTOFF: &str = "unknown (it depends on the model; search the web for anything recent)";

/// Sections kept for small local models, whose context window is short.
const SHORT: &[&str] = &["Identity & Personality", "System Security - CRITICAL", "Communication Style", "Support"];

/// The system prompt for a plain chat. `local` keeps only the core sections so a
/// small model has room left for the conversation. `web` says whether this message
/// comes with web results.
pub fn chat(local: bool, web: bool) -> String {
    let body = if local { sections(CHAT, SHORT) } else { CHAT.to_string() };
    fill(&body, &computer::today(), web)
}

fn fill(text: &str, date: &str, web: bool) -> String {
    let (state, tools) = if web {
        ("on. The newest web results are included with the user's message", "none you can call yourself; web results, when present, are already in the message")
    } else {
        ("off", "none")
    };
    text.replace("{{CURRENT_DATE}}", date).replace("{{KNOWLEDGE_CUTOFF}}", CUTOFF).replace("{{WEB_SEARCH_STATE}}", state).replace("{{TOOL_LIST}}", tools)
}

/// Keep the text before the first `## ` heading plus the named sections.
fn sections(text: &str, keep: &[&str]) -> String {
    let mut out = String::new();
    let mut on = true;
    for line in text.lines() {
        if let Some(h) = line.strip_prefix("## ") {
            on = keep.iter().any(|k| h.trim() == *k);
        }
        if on {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_are_filled() {
        for local in [false, true] {
            let p = chat(local, false);
            assert!(!p.contains("{{"), "left over placeholder: {local}");
            assert!(p.contains("Hinoko"));
        }
        assert!(chat(false, true).contains("web results, when present"));
    }

    #[test]
    fn local_prompt_is_much_shorter() {
        let (full, short) = (chat(false, false), chat(true, false));
        assert!(short.len() * 3 < full.len(), "{} vs {}", short.len(), full.len());
        assert!(short.contains("Communication Style") && !short.contains("Copyright"));
    }

    #[test]
    fn dates_read_well() {
        assert_eq!(computer::date_line(1_761_600_000), "Monday, 2025-10-27");
        assert_eq!(computer::date_line(0), "Thursday, 1970-01-01");
    }
}
