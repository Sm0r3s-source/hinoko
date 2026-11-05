//! Token savers for the agent loops: shrink noisy tool output, drop stale results,
//! hold rare tools back until asked for, and stop a model that is going in circles.
use serde_json::{json, Value};
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

const BUDGET: usize = 5000; // characters kept from one tool result (about 1.2k tokens)
const PROTECT: usize = 160_000; // recent tool output left alone (about 40k tokens)
const MIN_FREE: usize = 80_000; // pruning breaks the provider's prompt cache, so only when it frees this much
const NOISY: &[&str] = &["run_command", "ssh_run", "search", "web_read", "web_search"];
/// Code tools that are only described until the model asks for them.
pub const DEFERRED: &[&str] = &["find_files", "web_search", "web_read"];

/// Shrinks a long tool result, keeping what the model needs: a JSON array becomes a sample and a
/// count; logs keep their start, end and every error line (with context) and collapse repeats.
/// Only output-like tools are touched, never file reads (those have offset/limit).
pub fn squeeze(tool: &str, s: &str) -> String {
    if s.len() <= BUDGET || !NOISY.contains(&tool) {
        return s.to_string();
    }
    if let Ok(Value::Array(a)) = serde_json::from_str(s) {
        if a.len() > 8 {
            return format!("{}\n... {} more items ({} in all)", serde_json::to_string_pretty(&a[..5]).unwrap_or_default(), a.len() - 5, a.len());
        }
    }
    let mut rows: Vec<(String, usize)> = Vec::new();
    for l in s.lines() {
        let l: String = l.chars().take(400).collect();
        match rows.last_mut() {
            Some((p, n)) if *p == l => *n += 1,
            _ => rows.push((l, 1)),
        }
    }
    let text = |i: usize| if rows[i].1 > 1 { format!("{}  (x{})", rows[i].0, rows[i].1) } else { rows[i].0.clone() };
    let (mut keep, mut used) = (vec![false; rows.len()], 0);
    let n = rows.len();
    let bad = |l: &str| ["error", "fail", "panic", "warn", "exception", "traceback", "assert"].iter().any(|w| l.to_lowercase().contains(w));
    let wanted = (0..n.min(15)).chain(n.saturating_sub(15)..n).chain((0..n).filter(|&i| bad(&rows[i].0)).flat_map(|i| i.saturating_sub(2)..(i + 3).min(n)));
    for i in wanted {
        if !keep[i] && used < BUDGET {
            keep[i] = true;
            used += text(i).len() + 1;
        }
    }
    let (mut out, mut gap) = (Vec::new(), 0);
    for i in 0..n {
        if keep[i] {
            if gap > 0 {
                out.push(format!("... {gap} lines omitted ..."));
                gap = 0;
            }
            out.push(text(i));
        } else {
            gap += 1;
        }
    }
    if gap > 0 {
        out.push(format!("... {gap} lines omitted ..."));
    }
    out.join("\n")
}

/// Replaces stale tool results in the history with a short stub: a file read that was read again
/// later, and output older than the most recent `PROTECT` characters. The last two tool rounds are
/// never touched. Does nothing unless it frees `MIN_FREE`, and is safe to run every step.
pub fn prune(msgs: &mut [Value]) {
    let mut calls: HashMap<String, (String, String)> = HashMap::new();
    for m in msgs.iter() {
        for c in m["tool_calls"].as_array().into_iter().flatten() {
            let a: Value = serde_json::from_str(c["function"]["arguments"].as_str().unwrap_or("")).unwrap_or_default();
            calls.insert(c["id"].as_str().unwrap_or("").into(), (c["function"]["name"].as_str().unwrap_or("").into(), a["path"].as_str().unwrap_or("").into()));
        }
    }
    let rounds: Vec<usize> = (0..msgs.len()).filter(|&i| msgs[i]["tool_calls"].is_array()).collect();
    let guard = if rounds.len() >= 2 { rounds[rounds.len() - 2] } else { 0 };
    let size = |m: &Value| m["content"].as_str().map_or(0, str::len);
    let live = |i: usize| msgs[i]["role"] == "tool" && !msgs[i]["content"].as_str().unwrap_or("").starts_with("[old ");
    let read_of = |i: usize| calls.get(msgs[i]["tool_call_id"].as_str().unwrap_or("")).filter(|(n, p)| n == "read_file" && !p.is_empty()).map(|(_, p)| p.clone());
    let mut stale = vec![false; msgs.len()];
    let mut seen = std::collections::HashSet::new();
    for i in (0..msgs.len()).rev().filter(|&i| msgs[i]["role"] == "tool") {
        if let Some(p) = read_of(i) {
            stale[i] = !seen.insert(p) && i < guard;
        }
    }
    let mut kept = 0;
    for i in (0..guard).rev().filter(|&i| live(i)) {
        kept += size(&msgs[i]);
        stale[i] |= kept > PROTECT;
    }
    let idx: Vec<usize> = (0..guard).filter(|&i| stale[i] && live(i)).collect();
    if idx.iter().map(|&i| size(&msgs[i])).sum::<usize>() < MIN_FREE {
        return;
    }
    for i in idx {
        let name = calls.get(msgs[i]["tool_call_id"].as_str().unwrap_or("")).map_or("tool", |c| c.0.as_str()).to_string();
        msgs[i]["content"] = json!(format!("[old {name} output removed to save tokens; run it again if you need it]"));
    }
}

/// The `tool_search` tool: asks for a held-back tool by name or topic.
pub fn search_tool() -> Value {
    json!({"type": "function", "function": {"name": "tool_search", "description": "Load a tool that is listed under 'Other tools' in the instructions. Say its name or what you need.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": false}}})
}

/// Splits tools into the ones sent every time (plus `tool_search`) and the held-back ones, and
/// returns the instruction line that tells the model the held-back ones exist.
pub fn hold_back(all: Vec<Value>, deferred: &[&str]) -> (Vec<Value>, Vec<Value>, String) {
    let (held, mut core): (Vec<Value>, Vec<Value>) = all.into_iter().partition(|t| deferred.contains(&t["function"]["name"].as_str().unwrap_or("")));
    if held.is_empty() {
        return (core, held, String::new());
    }
    core.push(search_tool());
    let list: Vec<String> = held.iter().map(|t| format!("{} ({})", t["function"]["name"].as_str().unwrap_or(""), t["function"]["description"].as_str().unwrap_or("").split(['.', ',']).next().unwrap_or(""))).collect();
    (core, held, format!("\nOther tools, not loaded yet (call tool_search with the name to use one): {}.\n", list.join("; ")))
}

/// Moves the held-back tools that match `query` into the offered list; tells the model what happened.
pub fn load(query: &str, held: &mut Vec<Value>, offered: &mut Vec<Value>) -> String {
    let q = query.to_lowercase();
    let words: Vec<&str> = q.split(|c: char| !c.is_alphanumeric() && c != '_').filter(|w| w.len() >= 3).collect();
    let hit = |t: &Value| {
        let (n, d) = (t["function"]["name"].as_str().unwrap_or(""), t["function"]["description"].as_str().unwrap_or("").to_lowercase());
        q.contains(n) || words.iter().any(|w| n.contains(w) || d.contains(w))
    };
    let (take, rest): (Vec<Value>, Vec<Value>) = std::mem::take(held).into_iter().partition(hit);
    *held = rest;
    if take.is_empty() {
        let names: Vec<&str> = held.iter().filter_map(|t| t["function"]["name"].as_str()).collect();
        return format!("No tool matches that. Not loaded yet: {}.", names.join(", "));
    }
    let names: Vec<&str> = take.iter().filter_map(|t| t["function"]["name"].as_str()).collect();
    let msg = format!("Loaded: {}. You can call it now.", names.join(", "));
    offered.extend(take);
    msg
}

/// Notices a model that keeps repeating itself: the same call with the same result, or a streak
/// of failures. The first times it returns a nudge to add to the result; after two nudges `stuck`.
#[derive(Default)]
pub struct LoopGuard {
    last: u64,
    same: u32,
    fails: u32,
    nudges: u32,
}

impl LoopGuard {
    pub fn note(&mut self, name: &str, args: &str, result: &str) -> Option<String> {
        let mut h = DefaultHasher::new();
        (name, args, result).hash(&mut h);
        let key = h.finish();
        self.same = if key == self.last { self.same + 1 } else { 0 };
        self.last = key;
        self.fails = if result.trim_start().to_lowercase().starts_with("error") { self.fails + 1 } else { 0 };
        if self.same >= 2 || self.fails >= 3 {
            (self.same, self.fails, self.nudges) = (0, 0, self.nudges + 1);
            return Some("\n[You are repeating yourself without getting anywhere. Stop, try a different approach, or tell the user what is blocking you.]".into());
        }
        None
    }

    pub fn stuck(&self) -> bool {
        self.nudges >= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(id: &str, name: &str, path: &str) -> Value {
        json!({"role": "assistant", "content": "", "tool_calls": [{"id": id, "type": "function", "function": {"name": name, "arguments": json!({"path": path}).to_string()}}]})
    }
    fn result(id: &str, body: &str) -> Value {
        json!({"role": "tool", "tool_call_id": id, "content": body})
    }

    #[test]
    fn long_logs_keep_errors_and_shrink_but_files_and_short_output_stay() {
        let mut log: String = (0..3000).map(|i| format!("compiling crate number {i} of many\n")).collect();
        log.push_str("error[E0308]: mismatched types in src/main.rs\n");
        log.push_str(&"still going\n".repeat(500));
        let out = squeeze("run_command", &log);
        assert!(out.len() < 6500 && out.contains("error[E0308]") && out.contains("lines omitted") && out.contains("(x500)"), "{out}");
        assert_eq!(squeeze("read_file", &log), log, "file reads are never cut");
        assert_eq!(squeeze("run_command", "ok"), "ok");
        let arr = format!("[{}]", (0..400).map(|i| format!("{{\"id\": {i}, \"name\": \"item number {i} with a long name\"}}")).collect::<Vec<_>>().join(","));
        let out = squeeze("web_read", &arr);
        assert!(out.contains("395 more items (400 in all)") && out.len() < 1500, "{out}");
    }

    #[test]
    fn stale_results_are_stubbed_only_when_it_pays_and_recent_rounds_survive() {
        let (big, tiny) = ("x".repeat(50_000), "tiny".to_string());
        let history = |reads: &[(&str, &String)]| {
            let mut m = vec![json!({"role": "system", "content": "s"})];
            for (i, (p, body)) in reads.iter().enumerate() {
                m.push(call(&format!("c{i}"), "read_file", p));
                m.push(result(&format!("c{i}"), body));
            }
            m
        };
        let body = |m: &[Value], i: usize| m[2 + i * 2]["content"].as_str().unwrap().to_string();
        // a file read again later: the older copy goes, the newer stays
        let mut m = history(&[("a.rs", &big), ("b.rs", &big), ("a.rs", &tiny), ("b.rs", &tiny), ("c.rs", &tiny), ("d.rs", &tiny)]);
        prune(&mut m);
        assert!(body(&m, 0).starts_with("[old read_file") && body(&m, 1).starts_with("[old read_file"));
        assert!((2..6).all(|i| body(&m, i) == "tiny"), "newer reads and the last two rounds stay");
        // old output beyond the protected amount goes, the recent part stays
        let mut m = history(&[("a", &big), ("b", &big), ("c", &big), ("d", &big), ("e", &big), ("f", &big), ("g", &big), ("h", &big)]);
        prune(&mut m);
        assert!((0..3).all(|i| body(&m, i).starts_with("[old")) && (3..8).all(|i| body(&m, i).len() == 50_000));
        let again = m.clone();
        prune(&mut m);
        assert_eq!(m, again, "running it twice changes nothing more");
        // small savings are not worth breaking the provider's cache
        let mut small = history(&[("a", &tiny), ("a", &tiny), ("b", &tiny)]);
        prune(&mut small);
        assert_eq!(body(&small, 0), "tiny");
    }

    #[test]
    fn held_back_tools_load_on_request_and_loops_are_noticed() {
        let t = |n: &str, d: &str| json!({"type": "function", "function": {"name": n, "description": d}});
        let all = vec![t("read_file", "Read a file."), t("web_search", "Search the web (docs)."), t("find_files", "Find files by name glob.")];
        let (mut offered, mut held, note) = hold_back(all, DEFERRED);
        let names = |v: &[Value]| v.iter().map(|t| t["function"]["name"].as_str().unwrap().to_string()).collect::<Vec<_>>();
        assert_eq!(names(&offered), ["read_file", "tool_search"]);
        assert!(note.contains("web_search") && note.contains("find_files"));
        assert!(load("zzz", &mut held, &mut offered).starts_with("No tool matches"));
        assert!(load("I need to search the web", &mut held, &mut offered).contains("web_search"));
        assert_eq!(names(&offered), ["read_file", "tool_search", "web_search"]);
        assert_eq!(names(&held), ["find_files"]);

        let mut g = LoopGuard::default();
        assert!(g.note("run_command", "{}", "same").is_none() && g.note("run_command", "{}", "same").is_none());
        assert!(g.note("run_command", "{}", "same").is_some(), "third identical result");
        assert!(g.note("read_file", "{}", "ERROR: no").is_none() && g.note("edit_file", "{}", "ERROR: no").is_none());
        assert!(!g.stuck(), "one nudge so far");
        assert!(g.note("x", "{}", "error: still").is_some() && g.stuck(), "two nudges and the run is stopped");
    }
}
