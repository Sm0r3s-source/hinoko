//! The usage meter: calls and tokens per day, provider and model, saved beside the other data.
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::{collections::BTreeMap, path::Path, sync::Mutex};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub calls: u64,
    pub prompt: u64,
    pub completion: u64,
    pub cached: u64,
}

impl Row {
    pub fn add(&mut self, o: &Row) {
        (self.calls, self.prompt, self.completion, self.cached) = (self.calls + o.calls, self.prompt + o.prompt, self.completion + o.completion, self.cached + o.cached);
    }
}

static ROWS: Mutex<BTreeMap<String, Row>> = Mutex::new(BTreeMap::new()); // "day|provider id|model"
static DIRTY: AtomicBool = AtomicBool::new(false);

fn rows() -> std::sync::MutexGuard<'static, BTreeMap<String, Row>> {
    ROWS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Today as YYYY-MM-DD (UTC).
pub fn day() -> String {
    let d = crate::computer::today();
    d[d.len().saturating_sub(10)..].to_string()
}

/// The first day (YYYY-MM-DD) inside the last `n` days, today included.
pub fn since(n: i64) -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let d = crate::computer::date_line(secs - (n - 1).max(0) * 86_400);
    d[d.len().saturating_sub(10)..].to_string()
}

/// Counts one model call. `prompt` and `completion` are the server's numbers when it gave them.
pub fn record(provider: &str, model: &str, prompt: u64, completion: u64, cached: u64) {
    rows().entry(format!("{}|{provider}|{model}", day())).or_default().add(&Row { calls: 1, prompt, completion, cached });
    DIRTY.store(true, Ordering::Relaxed);
}

/// Every (day, provider id, model, totals), oldest first.
pub fn all() -> Vec<(String, String, String, Row)> {
    rows()
        .iter()
        .filter_map(|(k, r)| {
            let mut p = k.splitn(3, '|');
            Some((p.next()?.to_string(), p.next()?.to_string(), p.next()?.to_string(), r.clone()))
        })
        .collect()
}

pub fn load(path: &Path) {
    if let Some(m) = std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()) {
        *rows() = m;
    }
}

/// Writes the meter if anything was counted since the last write.
pub fn save(path: &Path) {
    if DIRTY.swap(false, Ordering::Relaxed) {
        let tmp = path.with_extension("tmp");
        if serde_json::to_string(&*rows()).ok().and_then(|s| std::fs::write(&tmp, s).ok()).is_some() {
            let _ = std::fs::rename(&tmp, path);
        }
    }
}

/// Estimated cost in dollars from prices per million tokens.
pub fn cost(r: &Row, price_in: f64, price_out: f64) -> f64 {
    (r.prompt as f64 * price_in + r.completion as f64 * price_out) / 1e6
}

/// What the server said it used: (prompt tokens, completion tokens, cached prompt tokens).
pub fn read(usage: &serde_json::Value) -> Option<(u64, u64, u64)> {
    let p = usage["prompt_tokens"].as_u64()?;
    Some((p, usage["completion_tokens"].as_u64().unwrap_or(0), usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_add_up_survive_a_restart_and_price_out() {
        let dir = std::env::temp_dir().join(format!("hinoko-usage-{}", crate::model::nid("u")));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("usage.json");
        record("unit-test", "m1", 1000, 200, 800);
        record("unit-test", "m1", 500, 100, 0);
        record("unit-test", "m2", 10, 1, 0);
        save(&file);
        *rows() = BTreeMap::new();
        load(&file);
        let mine: Vec<_> = all().into_iter().filter(|r| r.1 == "unit-test").collect();
        let m1 = &mine.iter().find(|r| r.2 == "m1").unwrap().3;
        assert_eq!((m1.calls, m1.prompt, m1.completion, m1.cached), (2, 1500, 300, 800));
        assert_eq!(mine.len(), 2, "each model has its own row");
        assert!((cost(m1, 3.0, 15.0) - 0.009).abs() < 1e-9, "1500 in at $3 and 300 out at $15 per million");
        assert_eq!(read(&serde_json::json!({"prompt_tokens": 9, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 2}})), Some((9, 4, 2)));
        assert_eq!(read(&serde_json::json!({})), None);
        let _ = std::fs::remove_dir_all(dir);
    }
}
