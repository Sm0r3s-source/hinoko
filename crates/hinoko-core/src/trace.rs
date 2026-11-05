//! Optional run traces: one JSON object per line, one file per day, in the
//! Hinoko data folder under `traces/`. Off by default (Settings → Traces).
//! Useful for debugging an agent run or building a dataset from your own use.

use std::io::Write;

use serde_json::{json, Value};

use crate::engine::Engine;
use crate::model::{clip, now_ms};

impl Engine {
    /// Append one record. Never fails the caller: a trace is best-effort.
    pub fn trace(&self, kind: &str, data: Value) {
        if !self.settings().traces {
            return;
        }
        let dir = self.paths.home.join("traces");
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let day = crate::computer::date_line((now_ms() / 1000) as i64);
        let day = day.split(", ").nth(1).unwrap_or("log").to_string();
        let line = record(kind, data);
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("{day}.jsonl"))) {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// One trace line. Long strings are clipped so a trace never fills the disk.
pub fn record(kind: &str, data: Value) -> String {
    json!({"t": now_ms(), "kind": kind, "data": clip_values(data)}).to_string()
}

fn clip_values(v: Value) -> Value {
    match v {
        Value::String(s) => Value::String(clip(&s, 4000)),
        Value::Array(a) => Value::Array(a.into_iter().map(clip_values).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, clip_values(v))).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_one_line_and_clipped() {
        let line = record("tool", json!({"name": "read_file", "result": "x".repeat(50_000)}));
        assert!(!line.contains('\n'));
        assert!(line.len() < 6000);
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["kind"], "tool");
        assert_eq!(v["data"]["name"], "read_file");
    }
}
