//! Desk settings: model providers (API, local server, CLI agent), budget, web.
//! Stored as JSON in the Hinoko home folder. API keys are kept in that file
//! (owner-only permissions on Unix), never in the state file.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Any OpenAI-compatible `/chat/completions` endpoint: hosted APIs,
    /// OpenRouter, Ollama, LM Studio, llama.cpp, vLLM.
    OpenAi,
    /// A command-line agent run once per model call (claude, codex, gemini, ...).
    Cli,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::OpenAi => "API / local server",
            Kind::Cli => "CLI agent",
        }
    }
}

/// A standing answer to a bot's approval requests, so routine things stop asking. Made by the
/// person (from an approval card or in Settings), never by a bot. A deny rule beats an allow rule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    pub id: String,
    /// Only this bot ("" = any bot).
    #[serde(default)]
    pub bot_id: String,
    /// "ssh", "delete" or "send" ("" = any kind of request).
    #[serde(default)]
    pub action: String,
    /// Matches when the command, path or title contains this text (case-insensitive). Never empty.
    pub contains: String,
    pub allow: bool,
}

impl Rule {
    pub fn matches(&self, bot_id: &str, action: &str, haystack: &str) -> bool {
        !self.contains.trim().is_empty()
            && (self.bot_id.is_empty() || self.bot_id == bot_id)
            && (self.action.is_empty() || self.action == action)
            && haystack.to_lowercase().contains(&self.contains.trim().to_lowercase())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub kind: Kind,
    #[serde(default)]
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    /// Name of an environment variable that holds the key. Wins over `api_key`
    /// when it is set, so a key never has to be written to disk.
    #[serde(default)]
    pub api_key_env: String,
    #[serde(default)]
    pub model: String,
    /// CLI: executable name or path.
    #[serde(default)]
    pub command: String,
    /// CLI: arguments. The prompt is written to stdin unless an argument is `{prompt}`.
    #[serde(default)]
    pub args: Vec<String>,
    /// API: send tool schemas natively. Off = tools are described in the prompt,
    /// which works with models that have no tool-calling support.
    #[serde(default = "yes")]
    pub native_tools: bool,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Sampling for this model only (blank = the global temperature / the server's default).
    /// Some models want their own values, e.g. Spark recommends temperature 1.0 and top-p 0.95.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Id of another provider to switch to when this one is down, rate-limited or out of quota.
    #[serde(default)]
    pub backup: String,
}

fn yes() -> bool {
    true
}
fn default_timeout() -> u64 {
    120
}

impl Provider {
    pub fn blank(kind: Kind) -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            kind,
            base_url: String::new(),
            api_key: String::new(),
            api_key_env: String::new(),
            model: String::new(),
            command: String::new(),
            args: Vec::new(),
            native_tools: true,
            timeout_secs: default_timeout(),
            temperature: None,
            top_p: None,
            backup: String::new(),
        }
    }

    /// The key to send, if any: the environment variable first, then the stored key.
    pub fn key(&self) -> String {
        let env_name = self.api_key_env.trim();
        if !env_name.is_empty() {
            if let Ok(v) = std::env::var(env_name) {
                if !v.trim().is_empty() {
                    return v.trim().to_string();
                }
            }
        }
        self.api_key.trim().to_string()
    }

    pub fn ready(&self) -> bool {
        match self.kind {
            Kind::OpenAi => !self.base_url.trim().is_empty() && !self.model.trim().is_empty(),
            Kind::Cli => !self.command.trim().is_empty(),
        }
    }

    pub fn summary(&self) -> String {
        match self.kind {
            Kind::OpenAi => format!("{} @ {}", self.model, self.base_url),
            Kind::Cli => format!("{} {}", self.command, self.args.join(" ")).trim().to_string(),
        }
    }
}

/// A computer a bot may reach over SSH.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Machine {
    pub id: String,
    pub name: String,
    pub host: String,
    #[serde(default = "ssh_port")]
    pub port: u16,
    pub user: String,
    #[serde(default)]
    pub notes: String,
    /// Run plain inspection commands (ls, df, uptime...) without asking.
    #[serde(default)]
    pub auto_readonly: bool,
}

fn ssh_port() -> u16 {
    22
}

impl Machine {
    pub fn blank() -> Self {
        Self { id: String::new(), name: String::new(), host: String::new(), port: 22, user: String::new(), notes: String::new(), auto_readonly: false }
    }
}

/// Look and feel. Colours are sRGB triples. A theme name other than
/// `custom` is just a starting palette; editing any colour switches to `custom`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Appearance {
    pub theme: String,
    pub accent: [u8; 3],
    pub background: [u8; 3],
    pub panel: [u8; 3],
    pub text: [u8; 3],
    /// Whole-window zoom, 0.8 to 1.6.
    pub ui_scale: f32,
    pub spacing: f32,
    pub rounding: f32,
    /// Maximum width of the reading column, in points. 0 = fill the window.
    pub chat_width: f32,
    pub monospace_chat: bool,
    pub show_events: bool,
    pub show_computer_panel: bool,
    /// Enter sends (Shift+Enter = new line). Off: Ctrl+Enter sends.
    pub enter_sends: bool,
    /// Pixel-art interface: notched pixel corners, hard borders, hard drop shadows, pixel sprites.
    #[serde(default)]
    pub pixel: bool,
    /// Use the pixel font (Pixelify Sans) instead of the default soft sans (Inter).
    #[serde(default)]
    pub pixel_font: bool,
}

/// Models a command-line agent can be told to use (for the model menu). Claude Code takes aliases.
pub fn cli_models(command: &str) -> &'static [(&'static str, &'static str)] {
    let name = command.trim().rsplit(['/', '\\']).next().unwrap_or("").to_lowercase();
    match name.trim_end_matches(".exe").trim_end_matches(".cmd") {
        "claude" => &[("", "Default"), ("haiku", "Haiku"), ("sonnet", "Sonnet"), ("opus", "Opus")],
        _ => &[],
    }
}

/// Arguments that select `model` for an agent that has a model menu; none when it is "Default".
pub fn cli_model_args(command: &str, model: &str) -> Vec<String> {
    if model.trim().is_empty() || cli_models(command).is_empty() { vec![] } else { vec!["--model".into(), model.trim().into()] }
}

pub const THEMES: &[(&str, &str)] = &[
    ("astral", "Astral (dark)"),
    ("pixel-astral", "Pixel Astral"),
    ("crimson", "Crimson (dark)"),
    ("sakura", "Sakura (dark)"),
    ("sakura-light", "Sakura (light)"),
    ("pixel", "Pixel (crimson)"),
    ("pixel-sakura", "Pixel Sakura"),
    ("kojo", "Kojo (indigo)"),
    ("ember", "Ember (dark)"),
    ("paper", "Paper (light)"),
    ("midnight", "Midnight"),
    ("forest", "Forest"),
    ("mono", "Mono"),
    ("custom", "Custom"),
];

impl Default for Appearance {
    fn default() -> Self {
        Self::preset("astral")
    }
}

impl Appearance {
    /// Colours for a named theme. Layout options keep their defaults.
    pub fn preset(name: &str) -> Self {
        let (accent, background, panel, text) = match name {
            "paper" => ([0x9a, 0x34, 0x12], [0xf4, 0xf0, 0xe7], [0xe9, 0xe3, 0xd6], [0x1c, 0x19, 0x16]),
            "ember" => ([0xd0, 0x5a, 0x24], [0x1c, 0x19, 0x16], [0x24, 0x20, 0x1c], [0xf4, 0xf0, 0xe7]),
            "midnight" => ([0x5b, 0x8d, 0xef], [0x0f, 0x14, 0x1f], [0x16, 0x1d, 0x2c], [0xe6, 0xea, 0xf2]),
            "forest" => ([0x6f, 0xb3, 0x7b], [0x12, 0x1a, 0x15], [0x1a, 0x25, 0x1d], [0xe4, 0xee, 0xe6]),
            "mono" => ([0xd8, 0xd8, 0xd8], [0x10, 0x10, 0x10], [0x1a, 0x1a, 0x1a], [0xee, 0xee, 0xee]),
            // the universe: near-black navy, starlight text, a soft astral violet
            "astral" => ([0x9a, 0x8c, 0xff], [0x07, 0x09, 0x1c], [0x0e, 0x12, 0x30], [0xe8, 0xec, 0xff]),
            "pixel-astral" => ([0x9a, 0x8c, 0xff], [0x0a, 0x0b, 0x22], [0x14, 0x16, 0x38], [0xee, 0xf0, 0xff]),
            "crimson" => ([0xf0, 0x38, 0x4f], [0x0c, 0x08, 0x09], [0x17, 0x0d, 0x10], [0xf5, 0xec, 0xee]),
            // pixel art: a limited, slightly purple palette so the hard edges read well
            "pixel" => ([0xff, 0x4d, 0x6d], [0x14, 0x0c, 0x1a], [0x22, 0x14, 0x2b], [0xf8, 0xe8, 0xf0]),
            "pixel-sakura" => ([0xff, 0x9e, 0xc4], [0x1a, 0x10, 0x24], [0x2a, 0x1a, 0x38], [0xff, 0xf0, 0xf6]),
            "sakura" => ([0xf4, 0xa6, 0xc0], [0x12, 0x0a, 0x10], [0x1d, 0x11, 0x19], [0xfb, 0xee, 0xf3]),
            "sakura-light" => ([0xd6, 0x36, 0x6f], [0xff, 0xf5, 0xf8], [0xfd, 0xe4, 0xec], [0x2a, 0x16, 0x20]),
            // deep indigo panels on near-black, in the spirit of Kojo and Grok
            "kojo" | "custom" | _ => ([0x8b, 0x7c, 0xff], [0x0d, 0x0b, 0x1a], [0x15, 0x12, 0x2e], [0xec, 0xea, 0xf6]),
        };
        let pixel = name.starts_with("pixel");
        let rounding = if pixel { 0.0 } else if matches!(name, "ember" | "paper" | "midnight" | "forest" | "mono") { 6.0 } else { 14.0 };
        Self {
            theme: if name == "custom" { "custom".into() } else { name.into() },
            accent,
            background,
            panel,
            text,
            ui_scale: 1.0,
            spacing: 8.0,
            rounding,
            chat_width: 760.0,
            monospace_chat: false,
            show_events: true,
            show_computer_panel: true,
            enter_sends: true,
            pixel,
            pixel_font: false,
        }
    }

    /// Switch palette but keep the person's layout choices.
    pub fn with_theme(&self, name: &str) -> Self {
        let p = Self::preset(name);
        Self {
            theme: p.theme,
            accent: p.accent,
            background: p.background,
            panel: p.panel,
            text: p.text,
            // choosing a pixel theme turns the style on; other themes leave the person's choice
            pixel: p.pixel || self.pixel,
            rounding: if p.pixel { 0.0 } else { self.rounding },
            ..self.clone()
        }
    }

    /// Clamp everything a hand-edited or imported file could push out of range.
    pub fn sanitized(mut self) -> Self {
        self.ui_scale = self.ui_scale.clamp(0.8, 1.6);
        self.spacing = self.spacing.clamp(2.0, 16.0);
        self.rounding = self.rounding.clamp(0.0, 20.0);
        self.chat_width = if self.chat_width <= 0.0 { 0.0 } else { self.chat_width.clamp(420.0, 2000.0) };
        self
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str::<Appearance>(text)
            .map(Appearance::sanitized)
            .map_err(|_| "that is not a Hinoko theme".to_string())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub appearance: Appearance,
    pub providers: Vec<Provider>,
    /// Id of the default provider.
    #[serde(default)]
    pub active: String,
    /// Custom instructions added to every chat and bot, like a global system prompt.
    #[serde(default)]
    pub instructions: String,
    /// SearXNG base URL for web search (may be on your own network). Blank:
    /// fall back to scraping DuckDuckGo.
    #[serde(default)]
    pub search_url: String,
    #[serde(default)]
    pub machines: Vec<Machine>,
    #[serde(default = "default_calls")]
    pub calls_per_hour: u32,
    /// Look for a newer Hinoko when it starts (it only tells you; installing is always your click).
    #[serde(default = "yes")]
    pub auto_update_check: bool,
    /// Standing allow/deny answers to bot approval requests.
    #[serde(default)]
    pub rules: Vec<Rule>,
    /// Routines stay quiet when they find nothing: no message, just a muted "nothing to report" line.
    #[serde(default = "yes")]
    pub quiet_routines: bool,
    #[serde(default = "default_temp")]
    pub temperature: f32,
    #[serde(default = "default_steps")]
    pub max_steps: u32,
    /// Let page reads use system proxies. Off by default: a proxy resolves
    /// names itself and so bypasses the public-address check.
    #[serde(default)]
    pub web_allow_proxy: bool,
    /// Model file the built-in local runner started last, so it can start again by itself.
    #[serde(default)]
    pub local_last: String,
    /// Which engine runs local models: "" (automatic), "builtin" or "llama".
    #[serde(default)]
    pub local_engine: String,
    /// Start every plain chat with Hinoko's built-in identity and behavior prompt.
    #[serde(default = "yes")]
    pub builtin_prompt: bool,
    /// How many tokens of history a model reads before older messages are
    /// summarized. 0 picks by model: 5,000 for local ones, 20,000 otherwise.
    #[serde(default)]
    pub compact_tokens: u32,
    /// Write a JSON-lines record of every model call and tool run to the data folder.
    #[serde(default)]
    pub traces: bool,
    /// Longest chat reply in tokens, thinking included. 0 picks 8,192, or 16,384 while Think is on.
    #[serde(default)]
    pub max_reply_tokens: u32,
    /// Longest model turn in the Code tab, in tokens. 0 picks 4,096.
    #[serde(default)]
    pub max_code_tokens: u32,
    /// Code tab: after the agent finishes a request that changed files, a reviewer pass checks
    /// the changes and sends it back with problems, up to this many times. 0 = off.
    #[serde(default)]
    pub code_review: u32,
    /// Provider id for the reviewer ("" = the same model that wrote the code).
    #[serde(default)]
    pub review_provider: String,
}

fn default_calls() -> u32 {
    60
}
fn default_temp() -> f32 {
    0.4
}
fn default_steps() -> u32 {
    6
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            appearance: Appearance::default(),
            providers: Vec::new(),
            active: String::new(),
            instructions: String::new(),
            search_url: String::new(),
            machines: Vec::new(),
            calls_per_hour: default_calls(),
            auto_update_check: true,
            rules: Vec::new(),
            quiet_routines: true,
            temperature: default_temp(),
            max_steps: default_steps(),
            web_allow_proxy: false,
            local_last: String::new(),
            local_engine: String::new(),
            builtin_prompt: true,
            compact_tokens: 0,
            traces: false,
            max_reply_tokens: 0,
            max_code_tokens: 0,
            code_review: 0,
            review_provider: String::new(),
        }
    }
}

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_default().trim().to_string()
}

impl Settings {
    pub fn provider(&self, id: &str) -> Option<&Provider> {
        self.providers.iter().find(|p| p.id == id)
    }

    /// The provider a bot should use: its own choice, else the desk default.
    pub fn provider_for(&self, bot_choice: Option<&str>) -> Option<&Provider> {
        bot_choice
            .and_then(|id| self.provider(id))
            .or_else(|| self.provider(&self.active))
            .or_else(|| self.providers.first())
    }

    /// The provider to switch to when `p` is unavailable: its configured backup, if that is
    /// set up and is a different provider.
    pub fn backup_for(&self, p: &Provider) -> Option<&Provider> {
        if p.backup.trim().is_empty() || p.backup == p.id {
            return None;
        }
        self.provider(p.backup.trim()).filter(|b| b.ready())
    }

    pub fn machine(&self, id_or_name: &str) -> Option<&Machine> {
        let q = id_or_name.trim();
        self.machines.iter().find(|m| m.id == q).or_else(|| self.machines.iter().find(|m| m.name.eq_ignore_ascii_case(q)))
    }

    pub fn upsert_machine(&mut self, mut m: Machine) -> String {
        if m.id.is_empty() {
            let base = crate::model::slug(if m.name.is_empty() { &m.host } else { &m.name });
            let mut id = base.clone();
            let mut n = 2;
            while self.machines.iter().any(|x| x.id == id) {
                id = format!("{base}-{n}");
                n += 1;
            }
            m.id = id;
        }
        let id = m.id.clone();
        match self.machines.iter_mut().find(|x| x.id == id) {
            Some(slot) => *slot = m,
            None => self.machines.push(m),
        }
        id
    }

    pub fn upsert(&mut self, mut p: Provider) -> String {
        if p.id.is_empty() {
            let base = crate::model::slug(if p.name.is_empty() { "provider" } else { &p.name });
            let mut id = base.clone();
            let mut n = 2;
            while self.provider(&id).is_some() {
                id = format!("{base}-{n}");
                n += 1;
            }
            p.id = id;
        }
        let id = p.id.clone();
        match self.providers.iter_mut().find(|x| x.id == id) {
            Some(slot) => *slot = p,
            None => self.providers.push(p),
        }
        if self.active.is_empty() {
            self.active = id.clone();
        }
        id
    }

    pub fn remove(&mut self, id: &str) {
        self.providers.retain(|p| p.id != id);
        if self.active == id {
            self.active = self.providers.first().map(|p| p.id.clone()).unwrap_or_default();
        }
    }

    /// First run: pick up the older environment variables so an existing
    /// setup keeps working.
    pub fn from_env() -> Self {
        let mut s = Settings::default();
        s.search_url = env("SEARXNG_URL");
        let base = env("LLM_BASE_URL");
        let mut key = env("LLM_API_KEY");
        let mut model = env("LLM_MODEL");
        let mut base_url = base.clone();
        if base_url.is_empty() && key.is_empty() {
            let xai = env("XAI_API_KEY");
            if !xai.is_empty() {
                key = xai;
                base_url = "https://api.x.ai/v1".into();
                if model.is_empty() {
                    model = "grok-4.5".into();
                }
            }
        }
        if base_url.is_empty() && !key.is_empty() {
            base_url = "https://api.openai.com/v1".into();
        }
        if !base_url.is_empty() {
            let mut p = Provider::blank(Kind::OpenAi);
            p.name = "From environment".into();
            p.base_url = base_url.trim_end_matches('/').to_string();
            p.api_key = key;
            p.model = model;
            s.upsert(p);
        }
        s
    }

    pub fn load(path: &Path) -> Self {
        match fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|_| {
                let _ = fs::rename(path, path.with_extension("json.bad"));
                Settings::default()
            }),
            Err(_) => {
                let s = Settings::from_env();
                if !s.providers.is_empty() {
                    s.save(path);
                }
                s
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
        if fs::write(&tmp, body).is_err() {
            return;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600));
        }
        let _ = fs::rename(&tmp, path);
    }
}

/// Where Hinoko keeps its files. `HINOKO_HOME` overrides the platform default.
pub struct Paths {
    pub home: PathBuf,
}

impl Paths {
    pub fn discover() -> Self {
        let home = match env("HINOKO_HOME") {
            h if !h.is_empty() => PathBuf::from(h),
            _ => {
                let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
                let (old, new) = (base.join("Hearth"), base.join("Hinoko"));
                // one-time move from the app's old name (Hearth), including its SSH key
                if old.is_dir() && !new.exists() && fs::rename(&old, &new).is_ok() {
                    for ext in ["", ".pub"] {
                        let _ = fs::rename(new.join(format!("ssh/hearth_ed25519{ext}")), new.join(format!("ssh/hinoko_ed25519{ext}")));
                    }
                }
                new
            }
        };
        Self { home }
    }
    pub fn at(home: impl Into<PathBuf>) -> Self {
        Self { home: home.into() }
    }
    pub fn state(&self) -> PathBuf {
        self.home.join("state.json")
    }
    pub fn settings(&self) -> PathBuf {
        self.home.join("settings.json")
    }
    /// Hinoko's own SSH key (private half; the `.pub` sits beside it).
    pub fn ssh_key(&self) -> PathBuf {
        self.home.join("ssh").join("hinoko_ed25519")
    }
    pub fn known_hosts(&self) -> PathBuf {
        self.home.join("ssh").join("known_hosts")
    }
}

/// One-click starting points shown in Settings. Fields stay editable.
pub struct Preset {
    pub label: &'static str,
    pub kind: Kind,
    pub base_url: &'static str,
    pub model: &'static str,
    pub command: &'static str,
    pub args: &'static [&'static str],
    pub needs_key: bool,
}

pub const PRESETS: &[Preset] = &[
    Preset { label: "OpenAI", kind: Kind::OpenAi, base_url: "https://api.openai.com/v1", model: "gpt-4o-mini", command: "", args: &[], needs_key: true },
    Preset { label: "Anthropic", kind: Kind::OpenAi, base_url: "https://api.anthropic.com/v1", model: "claude-sonnet-4-5", command: "", args: &[], needs_key: true },
    Preset { label: "OpenRouter", kind: Kind::OpenAi, base_url: "https://openrouter.ai/api/v1", model: "openai/gpt-4o-mini", command: "", args: &[], needs_key: true },
    Preset { label: "xAI", kind: Kind::OpenAi, base_url: "https://api.x.ai/v1", model: "grok-4.5", command: "", args: &[], needs_key: true },
    Preset { label: "Ollama (local)", kind: Kind::OpenAi, base_url: "http://127.0.0.1:11434/v1", model: "llama3.1", command: "", args: &[], needs_key: false },
    Preset { label: "LM Studio (local)", kind: Kind::OpenAi, base_url: "http://127.0.0.1:1234/v1", model: "local-model", command: "", args: &[], needs_key: false },
    Preset { label: "llama.cpp server", kind: Kind::OpenAi, base_url: "http://127.0.0.1:8080/v1", model: "local-model", command: "", args: &[], needs_key: false },
    Preset { label: "Claude Code CLI", kind: Kind::Cli, base_url: "", model: "", command: "claude", args: &["-p"], needs_key: false },
    Preset { label: "Codex CLI", kind: Kind::Cli, base_url: "", model: "", command: "codex", args: &["exec", "-"], needs_key: false },
    Preset { label: "Gemini CLI", kind: Kind::Cli, base_url: "", model: "", command: "gemini", args: &["-p", "Follow the instructions on stdin."], needs_key: false },
];

impl Preset {
    pub fn to_provider(&self) -> Provider {
        let mut p = Provider::blank(self.kind);
        p.name = self.label.to_string();
        p.base_url = self.base_url.to_string();
        p.model = self.model.to_string();
        p.command = self.command.to_string();
        p.args = self.args.iter().map(|a| a.to_string()).collect();
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_picks_active_and_unique_ids() {
        let mut s = Settings::default();
        let mut a = Provider::blank(Kind::OpenAi);
        a.name = "Local".into();
        let id_a = s.upsert(a.clone());
        let id_b = s.upsert(a);
        assert_ne!(id_a, id_b);
        assert_eq!(s.active, id_a);
        s.remove(&id_a);
        assert_eq!(s.active, id_b);
    }

    #[test]
    fn bot_choice_beats_default() {
        let mut s = Settings::default();
        let mut a = Provider::blank(Kind::OpenAi);
        a.name = "A".into();
        let ia = s.upsert(a);
        let mut b = Provider::blank(Kind::Cli);
        b.name = "B".into();
        let ib = s.upsert(b);
        assert_eq!(s.provider_for(None).unwrap().id, ia);
        assert_eq!(s.provider_for(Some(&ib)).unwrap().id, ib);
        assert_eq!(s.provider_for(Some("gone")).unwrap().id, ia);
    }

    #[test]
    fn claude_cli_gets_a_model_menu_and_other_agents_are_left_alone() {
        let names: Vec<&str> = cli_models(r"C:\Users\me\.local\bin\claude.exe").iter().map(|m| m.1).collect();
        assert_eq!(names, ["Default", "Haiku", "Sonnet", "Opus"]);
        assert_eq!(cli_model_args("claude", "sonnet"), ["--model", "sonnet"]);
        assert!(cli_model_args("claude", "").is_empty(), "Default adds nothing");
        assert!(cli_models("codex").is_empty() && cli_model_args("codex", "gpt-x").is_empty());
    }

    #[test]
    fn local_server_needs_no_key_and_cli_needs_command() {
        let mut p = Provider::blank(Kind::OpenAi);
        p.base_url = "http://127.0.0.1:11434/v1".into();
        p.model = "llama3.1".into();
        assert!(p.ready());
        assert!(!Provider::blank(Kind::Cli).ready());
        assert!(PRESETS.iter().all(|x| x.to_provider().ready()));
    }

    #[test]
    fn env_key_wins_over_stored_key() {
        let mut p = Provider::blank(Kind::OpenAi);
        p.api_key = "stored".into();
        assert_eq!(p.key(), "stored");
        p.api_key_env = "HINOKO_TEST_KEY_XYZ".into();
        assert_eq!(p.key(), "stored"); // unset variable falls back
        std::env::set_var("HINOKO_TEST_KEY_XYZ", " from-env ");
        assert_eq!(p.key(), "from-env");
        std::env::remove_var("HINOKO_TEST_KEY_XYZ");
    }

    #[test]
    fn themes_import_export_and_clamp() {
        let a = Appearance::preset("paper");
        let back = Appearance::from_json(&a.to_json()).unwrap();
        assert_eq!(a, back);
        let wild = r#"{"ui_scale": 9.0, "rounding": -5, "chat_width": 3.0, "accent": [1,2,3]}"#;
        let c = Appearance::from_json(wild).unwrap();
        assert_eq!(c.ui_scale, 1.6);
        assert_eq!(c.rounding, 0.0);
        assert_eq!(c.chat_width, 420.0);
        assert_eq!(c.accent, [1, 2, 3]);
        assert_eq!(c.background, Appearance::default().background, "missing fields take defaults");
        assert!(Appearance::from_json("nope").is_err());
        let kept = Appearance { enter_sends: false, ..Appearance::default() }.with_theme("forest");
        assert!(!kept.enter_sends && kept.theme == "forest");
    }

    #[test]
    fn old_settings_files_get_default_appearance() {
        let s: Settings = serde_json::from_str(r#"{"providers": []}"#).unwrap();
        assert_eq!(s.appearance.theme, "astral");
    }

    #[test]
    fn roundtrip_through_disk() {
        let dir = std::env::temp_dir().join(format!("hinoko-set-{}", crate::model::nid("t")));
        let path = dir.join("settings.json");
        let mut s = Settings::default();
        s.upsert(PRESETS[0].to_provider());
        s.save(&path);
        let back = Settings::load(&path);
        assert_eq!(back.providers.len(), 1);
        assert_eq!(back.providers[0].base_url, "https://api.openai.com/v1");
        let _ = fs::remove_dir_all(dir);
    }
}
