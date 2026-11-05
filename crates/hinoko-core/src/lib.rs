pub mod code;
pub mod compare;
pub mod compact;
pub mod computer;
#[cfg(feature = "builtin")]
pub mod embedded;
pub mod engine;
pub mod git;
pub mod llm;
pub mod local;
pub mod model;
pub mod prompts;
pub mod settings;
pub mod ssh;
pub mod trace;
pub mod update;
pub mod web;

pub use engine::{Attach, ChatOpts, Engine};
pub use model::Store;
pub use settings::{Appearance, Kind, Machine, Paths, Provider, Rule, Settings, PRESETS, THEMES};
