use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub const VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Store {
    pub version: u32,
    pub bots: Vec<Bot>,
    pub messages: BTreeMap<String, Vec<Message>>,
    pub computers: BTreeMap<String, Computer>,
    pub memories: BTreeMap<String, Vec<Memory>>,
    pub routines: Vec<Routine>,
    pub approvals: Vec<Approval>,
    /// Plain conversations with a model: no computer, no tools.
    #[serde(default)]
    pub chats: Vec<Chat>,
    /// Coding-agent sessions, each tied to a project folder.
    #[serde(default)]
    pub code: Vec<CodeSession>,
    /// Lifetime count of model calls (informational).
    #[serde(default)]
    pub api_calls: u32,
    /// Timestamps of recent model calls; the budget is a rolling hour.
    #[serde(default)]
    pub call_times: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bot {
    pub id: String,
    pub name: String,
    pub title: String,
    pub mandate: String,
    pub shape: String,
    pub presence: String,
    /// Provider id to use instead of the desk default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Read-only bots can look but not change anything ("plan" mode).
    #[serde(default)]
    pub read_only: bool,
    /// Ids of machines (Settings → Machines) this bot may reach over SSH.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub machines: Vec<String>,
}

/// A normal chatbot conversation. Messages reuse [`Message`] with kinds
/// `user`, `bot` and `error`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chat {
    pub id: String,
    pub title: String,
    /// Provider id; `None` follows the desk default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Optional system prompt for this chat.
    #[serde(default)]
    pub system: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    pub updated_ms: u64,
    /// Short summary of the older messages (see compact.rs), and how many
    /// leading messages it already covers.
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub covered: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub kind: String,
    pub text: String,
    pub at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_id: Option<String>,
    /// The model's reasoning, when it sent any (shown folded away).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub think: String,
    /// Names of files attached to this message.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<String>,
    /// Text of those files, kept so follow-up questions can still use them.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub file_text: String,
    /// Code sessions: a diff, command output or file excerpt that goes with `text`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
    /// Code sessions: "pending", "allowed", "always" or "denied" for approval cards.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub state: String,
}

/// A coding-agent session working inside one folder on the user's computer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeSession {
    pub id: String,
    pub title: String,
    /// The project folder. The agent cannot touch anything outside it.
    pub dir: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// "build" (can change things, asks first) or "plan" (read-only).
    pub mode: String,
    /// "ask" (approve every edit and command), "edits" (edits go through, commands ask), "all".
    pub autonomy: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    pub updated_ms: u64,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub covered: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub text: String,
    pub at_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Routine {
    pub id: String,
    pub bot_id: String,
    pub name: String,
    pub schedule: String,
    pub prompt: String,
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Approval {
    pub id: String,
    pub bot_id: String,
    pub title: String,
    pub detail: String,
    pub status: String,
    pub action: String,
    pub path: String,
    /// For `ssh`: the exact command, kept so Allow runs what was shown.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub command: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Computer {
    pub cwd: String,
    pub files: BTreeMap<String, String>,
    pub terminal: Vec<TermLine>,
    pub browser_url: String,
    pub browser_title: String,
    pub browser_excerpt: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TermLine {
    pub cmd: String,
    pub out: String,
}

impl Computer {
    pub fn empty(home: &str) -> Self {
        Self {
            cwd: home.to_string(),
            files: BTreeMap::new(),
            terminal: Vec::new(),
            browser_url: String::new(),
            browser_title: "Nothing open".into(),
            browser_excerpt: "The browser is idle.".into(),
        }
    }

    pub fn show(&mut self, url: &str, title: &str, excerpt: &str) {
        self.browser_url = url.to_string();
        self.browser_title = title.to_string();
        self.browser_excerpt = clip(excerpt, 700);
    }

    pub fn push_term(&mut self, cmd: &str, out: &str) {
        self.terminal.push(TermLine {
            cmd: clip(cmd, 200),
            out: clip(out, 1500),
        });
        if self.terminal.len() > 30 {
            let drop_n = self.terminal.len() - 30;
            self.terminal.drain(0..drop_n);
        }
    }
}

const WINDOW_MS: u64 = 3_600_000;

impl Store {
    /// Reserve one model call against the rolling hourly budget.
    pub fn take_call(&mut self, per_hour: u32) -> bool {
        let now = now_ms();
        let cutoff = now.saturating_sub(WINDOW_MS);
        self.call_times.retain(|t| *t > cutoff);
        if self.call_times.len() as u64 >= per_hour as u64 {
            return false;
        }
        self.call_times.push(now);
        self.api_calls = self.api_calls.saturating_add(1);
        true
    }

    pub fn calls_left(&self, per_hour: u32) -> u32 {
        let cutoff = now_ms().saturating_sub(WINDOW_MS);
        let used = self.call_times.iter().filter(|t| **t > cutoff).count() as u32;
        per_hour.saturating_sub(used)
    }

    pub fn bot(&self, id: &str) -> Option<&Bot> {
        self.bots.iter().find(|b| b.id == id)
    }

    pub fn bot_mut(&mut self, id: &str) -> Option<&mut Bot> {
        self.bots.iter_mut().find(|b| b.id == id)
    }

    pub fn push_msg(&mut self, bot_id: &str, kind: &str, text: impl Into<String>, approval_id: Option<String>) {
        let msg = Message {
            id: nid("m"),
            kind: kind.to_string(),
            text: clip(&text.into(), 8000),
            at_ms: now_ms(),
            approval_id,
            think: String::new(),
            files: Vec::new(),
            file_text: String::new(),
            detail: String::new(),
            state: String::new(),
        };
        let list = self.messages.entry(bot_id.to_string()).or_default();
        list.push(msg);
        if list.len() > 80 {
            let drop_n = list.len() - 80;
            list.drain(0..drop_n);
        }
    }
}

pub fn home_of(id: &str) -> String {
    format!("/home/{id}")
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn nid(prefix: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(1);
    let n = N.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{}-{n}", now_ms())
}

pub fn clip(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let t: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{t}…")
}

pub fn clean(s: &str, max: usize) -> String {
    let t: String = s
        .chars()
        .filter(|c| *c == '\n' || *c == '\t' || !c.is_control())
        .take(max)
        .collect();
    t.trim().to_string()
}

impl Store {
    pub fn empty() -> Self {
        Self {
            version: VERSION,
            bots: Vec::new(),
            messages: BTreeMap::new(),
            computers: BTreeMap::new(),
            memories: BTreeMap::new(),
            routines: Vec::new(),
            approvals: Vec::new(),
            chats: Vec::new(),
            code: Vec::new(),
            api_calls: 0,
            call_times: Vec::new(),
        }
    }

    /// Load the state file, or start empty. A file that exists but cannot be
    /// read is moved aside rather than overwritten.
    pub fn load(path: &Path) -> Self {
        let Ok(text) = fs::read_to_string(path) else {
            return Self::empty();
        };
        match serde_json::from_str::<Store>(&text) {
            Ok(store) if store.version == VERSION => store,
            _ => {
                let _ = fs::rename(path, path.with_extension("json.bad"));
                Self::empty()
            }
        }
    }

    pub fn save(&self, path: &Path) {
        if let Some(dir) = path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let Ok(body) = serde_json::to_vec_pretty(self) else {
            return;
        };
        let tmp = path.with_extension("json.tmp");
        if fs::write(&tmp, body).is_ok() {
            let _ = fs::rename(&tmp, path);
        }
    }

    /// Create a bot with its own empty computer. Returns its id.
    pub fn add_bot(&mut self, name: &str, title: &str, mandate: &str) -> Result<String, &'static str> {
        let (name, title, mandate) = (clean(name, 32), clean(title, 40), clean(mandate, 280));
        if name.is_empty() || title.is_empty() || mandate.is_empty() {
            return Err("name, title, and mandate are required");
        }
        if self.bots.len() >= 16 {
            return Err("this desk holds sixteen bots");
        }
        let mut id = slug(&name);
        if self.bot(&id).is_some() {
            id = format!("{id}-{}", self.bots.len() + 1);
        }
        let shape = ["ring", "slab", "hex", "sun"][self.bots.len() % 4];
        let home = home_of(&id);
        let mut computer = Computer::empty(&home);
        let welcome = format!("# {name}\n\n{mandate}\n");
        computer.files.insert(format!("{home}/welcome.md"), welcome.clone());
        computer.show(&format!("{home}/welcome.md"), "welcome.md", &welcome);
        self.bots.push(Bot {
            id: id.clone(),
            name,
            title,
            mandate,
            shape: shape.into(),
            presence: "idle".into(),
            provider: None,
            read_only: false,
            machines: Vec::new(),
        });
        self.messages.insert(id.clone(), Vec::new());
        self.memories.insert(id.clone(), Vec::new());
        self.computers.insert(id.clone(), computer);
        Ok(id)
    }

    pub fn add_chat(&mut self) -> String {
        let id = nid("chat");
        self.chats.insert(
            0,
            Chat { id: id.clone(), title: "New chat".into(), provider: None, system: String::new(), messages: Vec::new(), updated_ms: now_ms(), summary: String::new(), covered: 0 },
        );
        id
    }

    pub fn chat(&self, id: &str) -> Option<&Chat> {
        self.chats.iter().find(|c| c.id == id)
    }

    pub fn chat_mut(&mut self, id: &str) -> Option<&mut Chat> {
        self.chats.iter_mut().find(|c| c.id == id)
    }

    /// Append a message to a chat; the first user message names an untitled chat.
    pub fn push_chat_msg(&mut self, chat_id: &str, kind: &str, text: impl Into<String>) {
        let text = text.into();
        let Some(chat) = self.chat_mut(chat_id) else { return };
        if kind == "user" && chat.title == "New chat" {
            let first = clean(&text, 60);
            if !first.is_empty() {
                chat.title = clip(first.lines().next().unwrap_or(""), 40);
            }
        }
        chat.messages.push(Message { id: nid("m"), kind: kind.into(), text, at_ms: now_ms(), approval_id: None, think: String::new(), files: Vec::new(), file_text: String::new(), detail: String::new(), state: String::new() });
        chat.updated_ms = now_ms();
    }

    pub fn remove_chat(&mut self, id: &str) {
        self.chats.retain(|c| c.id != id);
    }

    pub fn add_code(&mut self, dir: &str) -> String {
        let id = nid("code");
        let title = std::path::Path::new(dir).file_name().map(|n| n.to_string_lossy().to_string()).filter(|n| !n.is_empty()).unwrap_or_else(|| "project".into());
        self.code.insert(
            0,
            CodeSession { id: id.clone(), title, dir: dir.to_string(), provider: None, mode: "build".into(), autonomy: "ask".into(), messages: Vec::new(), updated_ms: now_ms(), summary: String::new(), covered: 0 },
        );
        id
    }

    pub fn code_session(&self, id: &str) -> Option<&CodeSession> {
        self.code.iter().find(|c| c.id == id)
    }

    pub fn code_mut(&mut self, id: &str) -> Option<&mut CodeSession> {
        self.code.iter_mut().find(|c| c.id == id)
    }

    /// Add a line to a coding session. Returns the message id.
    pub fn push_code(&mut self, id: &str, kind: &str, text: impl Into<String>, detail: impl Into<String>, state: &str) -> String {
        let Some(c) = self.code_mut(id) else { return String::new() };
        let mid = nid("m");
        c.messages.push(Message {
            id: mid.clone(),
            kind: kind.into(),
            text: text.into(),
            at_ms: now_ms(),
            approval_id: None,
            think: String::new(),
            files: Vec::new(),
            file_text: String::new(),
            detail: detail.into(),
            state: state.into(),
        });
        c.updated_ms = now_ms();
        if c.messages.len() > 400 {
            let cut = c.messages.len() - 400;
            c.messages.drain(..cut);
        }
        mid
    }

    pub fn remove_bot(&mut self, id: &str) {
        self.bots.retain(|b| b.id != id);
        self.messages.remove(id);
        self.memories.remove(id);
        self.computers.remove(id);
        self.routines.retain(|r| r.bot_id != id);
        self.approvals.retain(|a| a.bot_id != id);
    }
}

pub fn slug(name: &str) -> String {
    let mut s = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_lowercase());
        } else if (c.is_whitespace() || c == '-' || c == '_') && !s.ends_with('-') && !s.is_empty() {
            s.push('-');
        }
        if s.len() >= 18 {
            break;
        }
    }
    let s = s.trim_matches('-').to_string();
    if s.is_empty() { "bot".into() } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_desk_has_no_bots() {
        let s = Store::empty();
        assert!(s.bots.is_empty() && s.approvals.is_empty() && s.routines.is_empty());
    }

    #[test]
    fn add_and_remove_bot() {
        let mut s = Store::empty();
        let id = s.add_bot("Ada Voss", "Research", "Keep notes.").unwrap();
        assert_eq!(id, "ada-voss");
        let id2 = s.add_bot("Ada Voss", "Research", "Again.").unwrap();
        assert_ne!(id, id2);
        assert!(s.computers[&id].files.contains_key("/home/ada-voss/welcome.md"));
        assert!(s.add_bot("", "x", "y").is_err());
        s.remove_bot(&id);
        assert!(s.bot(&id).is_none() && !s.computers.contains_key(&id));
    }

    #[test]
    fn chats_get_a_title_from_the_first_message() {
        let mut s = Store::empty();
        let id = s.add_chat();
        s.push_chat_msg(&id, "user", "How do I sort a list in Rust?\nthanks");
        s.push_chat_msg(&id, "bot", "Use sort().");
        assert_eq!(s.chat(&id).unwrap().title, "How do I sort a list in Rust?");
        s.push_chat_msg(&id, "user", "another");
        assert_eq!(s.chat(&id).unwrap().title, "How do I sort a list in Rust?");
        assert_eq!(s.chat(&id).unwrap().messages.len(), 3);
        let newer = s.add_chat();
        assert_eq!(s.chats[0].id, newer, "newest first");
    }

    #[test]
    fn budget_is_a_rolling_window() {
        let mut s = Store::empty();
        assert!(s.take_call(2));
        assert!(s.take_call(2));
        assert!(!s.take_call(2));
        assert_eq!(s.calls_left(2), 0);
        for t in s.call_times.iter_mut() {
            *t = now_ms() - WINDOW_MS - 1;
        }
        assert_eq!(s.calls_left(2), 2);
        assert!(s.take_call(2));
        assert_eq!(s.api_calls, 3);
    }

    #[test]
    fn save_and_load_roundtrip_and_bad_file_is_kept() {
        let dir = std::env::temp_dir().join(format!("hinoko-test-{}", nid("t")));
        let path = dir.join("state.json");
        let mut s = Store::empty();
        s.add_bot("Sol", "Design", "Marks.").unwrap();
        s.save(&path);
        assert_eq!(Store::load(&path).bots.len(), 1);
        fs::write(&path, "{not json").unwrap();
        assert!(Store::load(&path).bots.is_empty());
        assert!(path.with_extension("json.bad").exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn slug_is_stable() {
        assert_eq!(slug("Ada Voss"), "ada-voss");
        assert_eq!(slug("..."), "bot");
    }
}
