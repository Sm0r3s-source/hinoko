#![cfg(unix)]
//! Engine end-to-end with a scripted CLI agent standing in for the model.

use std::sync::Arc;

use hinoko_core::{Engine, Kind, Paths, Provider};

fn engine(tag: &str, script: &str) -> (Engine, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("hinoko-eng-{tag}-{}", hinoko_core::model::nid("x")));
    let e = Engine::new(Paths::at(&dir), tokio::runtime::Handle::current(), Arc::new(|| {})).unwrap();
    if !script.is_empty() {
        let mut s = e.settings();
        let mut p = Provider::blank(Kind::Cli);
        p.name = "Fake".into();
        p.command = "sh".into();
        p.args = vec!["-c".into(), script.into()];
        p.timeout_secs = 20;
        s.upsert(p);
        e.set_settings(s);
    }
    (e, dir)
}

const WRITER: &str = r#"in=$(cat); n=$(printf '%s' "$in" | grep -c '<tool_result id=')
if [ "$n" = 0 ]; then echo 'Writing it.'; echo '<tool_call>{"name":"write_file","arguments":{"path":"note.md","content":"hi"}}</tool_call>'
else echo 'Saved.'; fi"#;

fn texts(e: &Engine, bot: &str, kind: &str) -> Vec<String> {
    e.store().messages[bot].iter().filter(|m| m.kind == kind).map(|m| m.text.clone()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_loop_writes_a_file_and_answers() {
    let (e, dir) = engine("w", WRITER);
    let id = e.store().add_bot("Ada", "Research", "Keep notes.").unwrap();
    e.run_turn(&id, "save a note", "save a note").await;
    let st = e.store();
    assert_eq!(st.computers[&id].files["/home/ada/note.md"], "hi");
    drop(st);
    assert_eq!(texts(&e, &id, "bot").last().unwrap(), "Saved.");
    assert!(texts(&e, &id, "event").iter().any(|t| t == "Wrote note.md"));
    assert_eq!(e.store().api_calls, 2);
    assert_eq!(e.store().bot(&id).unwrap().presence, "done");
    // state reached disk and reloads
    let again = Engine::new(Paths::at(&dir), tokio::runtime::Handle::current(), Arc::new(|| {})).unwrap();
    assert_eq!(again.store().bots.len(), 1);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn read_only_bot_cannot_write() {
    let (e, dir) = engine("ro", WRITER);
    let id = e.store().add_bot("Ada", "Research", "Keep notes.").unwrap();
    e.store().bot_mut(&id).unwrap().read_only = true;
    e.run_turn(&id, "save a note", "save a note").await;
    assert!(!e.store().computers[&id].files.contains_key("/home/ada/note.md"));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn no_provider_points_to_settings() {
    let (e, dir) = engine("none", "");
    let id = e.store().add_bot("Ada", "Research", "Keep notes.").unwrap();
    e.run_turn(&id, "hello", "hello").await;
    assert!(texts(&e, &id, "bot").last().unwrap().contains("Settings"));
    assert_eq!(e.store().api_calls, 0);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn broken_cli_reports_a_readable_error_and_budget_applies() {
    let (e, dir) = engine("bad", "echo boom >&2; exit 3");
    let id = e.store().add_bot("Ada", "Research", "Keep notes.").unwrap();
    e.run_turn(&id, "hello", "hello").await;
    let last = texts(&e, &id, "bot").last().unwrap().clone();
    assert!(last.contains("couldn't reach the model") && last.contains("boom"), "{last}");
    // budget of 1 call per hour
    let mut s = e.settings();
    s.calls_per_hour = 1;
    e.set_settings(s);
    e.store().call_times.clear();
    e.run_turn(&id, "again", "again").await; // takes the only call
    e.run_turn(&id, "and again", "and again").await;
    assert!(texts(&e, &id, "bot").last().unwrap().contains("model calls for the hour"));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn one_turn_per_bot() {
    let (e, dir) = engine("busy", WRITER);
    let id = e.store().add_bot("Ada", "Research", "Keep notes.").unwrap();
    let g = e.try_busy(&id).unwrap();
    assert!(e.try_busy(&id).is_none());
    assert_eq!(e.chat(&id, "hi"), Err("that bot is mid-turn"));
    drop(g);
    assert!(e.try_busy(&id).is_some());
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn approvals_run_the_stored_action_on_allow() {
    let script = r#"in=$(cat); n=$(printf '%s' "$in" | grep -c '<tool_result id=')
if [ "$n" = 0 ]; then echo '<tool_call>{"name":"write_file","arguments":{"path":"old.md","content":"x"}}</tool_call>'
elif [ "$n" = 1 ]; then echo '<tool_call>{"name":"propose_delete","arguments":{"path":"old.md","reason":"Archive it"}}</tool_call>'
else echo 'Waiting on you.'; fi"#;
    let (e, dir) = engine("ap", script);
    let id = e.store().add_bot("Ada", "Research", "Keep notes.").unwrap();
    e.run_turn(&id, "archive", "archive").await;
    let ap = e.store().approvals.iter().find(|a| a.status == "pending").cloned().unwrap();
    assert!(e.store().computers[&id].files.contains_key("/home/ada/old.md"), "nothing deleted before approval");
    assert_eq!(e.store().bot(&id).unwrap().presence, "waiting");
    e.decide(&ap.id, true).unwrap();
    assert!(!e.store().computers[&id].files.contains_key("/home/ada/old.md"));
    assert!(e.decide(&ap.id, true).is_err(), "decisions are final");
    let _ = std::fs::remove_dir_all(dir);
}


// ---- plain chats, streaming and deep research ----

use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Minimal HTTP server: `handler(request_body)` -> (content_type, body).
async fn serve(handler: fn(&str) -> (&'static str, String)) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                let (mut head_end, mut want) = (None, 0usize);
                loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if head_end.is_none() {
                        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            head_end = Some(p + 4);
                            let head = String::from_utf8_lossy(&buf[..p]).to_lowercase();
                            want = head
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:").and_then(|v| v.trim().parse().ok()))
                                .unwrap_or(0);
                        }
                    }
                    if let Some(h) = head_end {
                        if buf.len() >= h + want {
                            break;
                        }
                    }
                }
                let body = String::from_utf8_lossy(&buf[head_end.unwrap_or(0)..]).to_string();
                let (ctype, out) = handler(&body);
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nConnection: close\r\n\r\n{out}");
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}")
}

fn sse(_: &str) -> (&'static str, String) {
    let chunk = |t: &str| format!("data: {}\n\n", serde_json::json!({"choices":[{"delta":{"content":t}}]}));
    ("text/event-stream", format!("{}{}{}data: [DONE]\n\n", chunk("Hel"), chunk("lo "), chunk("world")))
}

fn api_engine(tag: &str, base: &str) -> (Engine, std::path::PathBuf) {
    std::env::set_var("NO_PROXY", "127.0.0.1");
    let (e, dir) = engine(tag, "");
    let mut s = e.settings();
    let mut p = Provider::blank(Kind::OpenAi);
    p.name = "Fake API".into();
    p.base_url = format!("{base}/v1");
    p.model = "fake".into();
    s.upsert(p);
    e.set_settings(s);
    (e, dir)
}

#[tokio::test(flavor = "multi_thread")]
async fn plain_chat_streams_an_answer_and_names_itself() {
    let base = serve(sse).await;
    let (e, dir) = api_engine("stream", &base);
    let id = e.new_chat();
    e.store().chat_mut(&id).unwrap().system = "Be brief.".into();
    e.run_chat(&id, "Say hello to the world please", false).await;
    let st = e.store();
    let chat = st.chat(&id).unwrap();
    assert_eq!(chat.title, "Say hello to the world please");
    assert_eq!(chat.messages.last().unwrap().kind, "bot");
    assert_eq!(chat.messages.last().unwrap().text, "Hello world");
    assert_eq!(st.api_calls, 0, "plain chats do not use the bot budget");
    drop(st);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_errors_are_readable_and_keep_the_question() {
    let (e, dir) = engine("chaterr", "echo nope >&2; exit 2");
    let id = e.new_chat();
    e.run_chat(&id, "hello", false).await;
    let st = e.store();
    let kinds: Vec<_> = st.chat(&id).unwrap().messages.iter().map(|m| m.kind.as_str()).collect();
    assert_eq!(kinds, ["user", "error"]);
    assert!(st.chat(&id).unwrap().messages[1].text.contains("nope"));
    drop(st);
    let _ = std::fs::remove_dir_all(dir);
}

fn fake_searx(_: &str) -> (&'static str, String) {
    ("application/json", r#"{"results":[
        {"title":"Tides explained","url":"https://tides.invalid/a","content":"Tides come from the Moon's gravity pulling on the oceans."},
        {"title":"Ocean basics","url":"https://ocean.invalid/b","content":"Spring tides happen at new and full moon."}]}"#.into())
}

#[tokio::test(flavor = "multi_thread")]
async fn deep_research_plans_searches_and_writes_with_citations() {
    let searx = serve(fake_searx).await;
    // scripted model: answers the planner with queries and the writer with a cited report
    let script = r#"in=$(cat)
if printf '%s' "$in" | grep -q 'JSON array of 3 to 5'; then echo '["why tides happen", "spring tides"]'
elif printf '%s' "$in" | grep -q '\[1\]'; then echo 'Tides come from lunar gravity [1]. Spring tides follow new and full moons [2].'
else echo 'no sources seen'; fi"#;
    let (e, dir) = engine("research", script);
    let mut s = e.settings();
    s.search_url = searx;
    e.set_settings(s);
    std::env::set_var("NO_PROXY", "127.0.0.1");
    let id = e.new_chat();
    e.run_chat(&id, "why do tides happen?", true).await;
    let st = e.store();
    let msgs = &st.chat(&id).unwrap().messages;
    let events: Vec<&str> = msgs.iter().filter(|m| m.kind == "event").map(|m| m.text.as_str()).collect();
    assert!(events.iter().any(|t| t.starts_with("Searching: why tides happen")), "{events:?}");
    assert!(events.iter().any(|t| t.starts_with("Writing from 2 sources")), "{events:?}");
    let answer = msgs.last().unwrap();
    assert_eq!(answer.kind, "bot");
    assert!(answer.text.contains("[1]") && answer.text.contains("[2]"), "{}", answer.text);
    drop(st);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn research_without_working_search_says_how_to_fix_it() {
    let (e, dir) = engine("nosearch", "echo '[\"q\"]'");
    let mut s = e.settings();
    s.search_url = "http://127.0.0.1:9".into(); // nothing listens here
    e.set_settings(s);
    std::env::set_var("NO_PROXY", "127.0.0.1");
    let id = e.new_chat();
    e.run_chat(&id, "anything", true).await;
    let st = e.store();
    let last = st.chat(&id).unwrap().messages.last().unwrap();
    assert_eq!(last.kind, "error");
    assert!(last.text.contains("Settings"), "{}", last.text);
    drop(st);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn query_plans_parse_from_messy_replies() {
    use hinoko_core::engine::parse_queries;
    assert_eq!(parse_queries("Sure!\n```json\n[\"a\", \"b\"]\n```"), ["a", "b"]);
    assert!(parse_queries("no array").is_empty());
    assert!(parse_queries("] [").is_empty());
    assert_eq!(parse_queries(r#"["1","2","3","4","5","6","7"]"#).len(), 5);
}

// ---- SSH ----

/// Put a fake `ssh` first on PATH that logs its last argument (the command).
fn fake_ssh() -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("hinoko-fakessh-{}", hinoko_core::model::nid("s")));
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("log");
    let script = format!("#!/bin/sh\nfor a; do last=\"$a\"; done\necho \"$last\" >> {}\necho \"output of: $last\"\n", log.display());
    let bin = dir.join("ssh");
    std::fs::write(&bin, script).unwrap();
    std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("PATH", format!("{}:{}", dir.display(), std::env::var("PATH").unwrap_or_default()));
    log
}

const SSH_AGENT: &str = r#"in=$(cat); n=$(printf '%s' "$in" | grep -c '<tool_result id=')
if printf '%s' "$in" | grep -q 'user approved your command'; then echo 'Cleaned up on the PC.'
elif [ "$n" = 0 ]; then
  printf '%s\n' '<tool_call>{"name":"ssh_run","arguments":{"machine":"Workshop PC","command":"df -h"}}</tool_call>'
  printf '%s\n' '<tool_call>{"name":"ssh_run","arguments":{"machine":"Workshop PC","command":"rm -rf /tmp/old"}}</tool_call>'
elif printf '%s' "$in" | grep -q 'waiting for the user'; then echo 'The disk report is above; the cleanup waits for your OK.'
else echo 'ok'; fi"#;

fn with_machine(e: &Engine, auto: bool) -> String {
    let mut s = e.settings();
    let mut m = hinoko_core::Machine::blank();
    m.name = "Workshop PC".into();
    m.host = "192.168.1.20".into();
    m.user = "sakor".into();
    m.auto_readonly = auto;
    let mid = s.upsert_machine(m);
    e.set_settings(s);
    let id = e.store().add_bot("Ada", "Ops", "Tend the PC.").unwrap();
    e.store().bot_mut(&id).unwrap().machines = vec![mid];
    id
}

#[tokio::test(flavor = "multi_thread")]
async fn ssh_readonly_runs_now_and_everything_else_waits_for_allow() {
    let log = fake_ssh();
    let (e, dir) = engine("ssh", SSH_AGENT);
    let id = with_machine(&e, true);
    e.run_turn(&id, "check the disk and clear temp", "check the disk and clear temp").await;
    let logged = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(logged.trim(), "df -h", "only the read-only command ran: {logged:?}");
    let ap = e.store().approvals.iter().find(|a| a.status == "pending" && a.action == "ssh").cloned().expect("rm waits");
    assert_eq!(ap.command, "rm -rf /tmp/old");
    assert_eq!(e.store().bot(&id).unwrap().presence, "waiting");
    // deny: nothing runs
    let denied_before = std::fs::read_to_string(&log).unwrap_or_default();
    // allow: the stored command runs and the bot answers with the result
    e.decide_and_wait(&ap.id, true).await.unwrap();
    let after = std::fs::read_to_string(&log).unwrap();
    assert!(after.contains("rm -rf /tmp/old") && after.len() > denied_before.len());
    assert_eq!(texts(&e, &id, "bot").last().unwrap(), "Cleaned up on the PC.");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn ssh_denied_command_never_runs_and_without_auto_even_ls_asks() {
    let log = fake_ssh();
    let (e, dir) = engine("ssh2", SSH_AGENT);
    let id = with_machine(&e, false);
    e.run_turn(&id, "go", "go").await;
    assert!(std::fs::read_to_string(&log).unwrap_or_default().trim().is_empty(), "nothing ran before approval");
    let pending: Vec<_> = e.store().approvals.iter().filter(|a| a.status == "pending").cloned().collect();
    assert_eq!(pending.len(), 2);
    e.decide_and_wait(&pending[1].id, false).await.unwrap();
    assert!(std::fs::read_to_string(&log).unwrap_or_default().trim().is_empty(), "denied means never");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn bots_only_see_machines_they_were_given_and_read_only_bots_see_none() {
    let (e, dir) = engine("ssh3", SSH_AGENT);
    let id = with_machine(&e, true);
    e.store().bot_mut(&id).unwrap().machines.clear();
    e.run_turn(&id, "go", "go").await;
    assert!(e.store().approvals.is_empty(), "no machines assigned: no ssh tool, nothing queued");
    e.store().bot_mut(&id).unwrap().machines = e.settings().machines.iter().map(|m| m.id.clone()).collect();
    e.store().bot_mut(&id).unwrap().read_only = true;
    e.run_turn(&id, "go again", "go again").await;
    assert!(e.store().approvals.is_empty(), "read-only bots cannot use ssh");
    let _ = std::fs::remove_dir_all(dir);
}

fn echo_body(body: &str) -> (&'static str, String) {
    // answers with the request it got, so tests can see what was sent
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let said = v.to_string();
    let think = format!("data: {}\n\n", serde_json::json!({"choices":[{"delta":{"reasoning_content":"hmm"}}]}));
    let chunk = format!("data: {}\n\n", serde_json::json!({"choices":[{"delta":{"content":said}}]}));
    ("text/event-stream", format!("{think}{chunk}data: [DONE]\n\n"))
}

#[tokio::test(flavor = "multi_thread")]
async fn attachments_effort_and_thinking_reach_the_model_and_the_chat() {
    let base = serve(echo_body).await;
    let (e, dir) = api_engine("attach", &base);
    let id = e.new_chat();
    let opts = hinoko_core::ChatOpts {
        effort: "high".into(),
        files: vec![
            hinoko_core::Attach { name: "notes.txt".into(), text: "secret-42".into(), image: None },
            hinoko_core::Attach { name: "pic.png".into(), text: String::new(), image: Some("data:image/png;base64,AAAA".into()) },
        ],
        ..Default::default()
    };
    e.run_chat_with(&id, "what is in my file?", opts).await;
    let st = e.store();
    let msgs = &st.chat(&id).unwrap().messages;
    assert_eq!(msgs[0].files, ["notes.txt", "pic.png"]);
    assert!(msgs[0].file_text.contains("secret-42"));
    assert!(!msgs[0].file_text.contains("AAAA"), "image data is not stored");
    let reply = &msgs.last().unwrap();
    assert_eq!(reply.think, "hmm");
    assert!(reply.text.contains("secret-42"), "file text was sent: {}", reply.text);
    assert!(reply.text.contains("image_url") && reply.text.contains("AAAA"), "image was sent");
    assert!(reply.text.contains("\"reasoning_effort\":\"high\""), "effort was sent");
    drop(st);
    e.delete_chat(&id);
    assert!(e.store().chat(&id).is_none());
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn local_runner_starts_registers_and_answers_chats_on_demand() {
    use std::os::unix::fs::PermissionsExt;
    std::env::set_var("NO_PROXY", "127.0.0.1");
    let (e, dir) = engine("localrun", "");
    // a stand-in llama-server: answers /health and one chat completion on --port
    let runner = e.local_dir().join("runner");
    std::fs::create_dir_all(&runner).unwrap();
    let exe = runner.join("llama-server");
    std::fs::write(
        &exe,
        r#"#!/usr/bin/env python3
import sys, json
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        n = int(self.headers.get("content-length", 0)); self.rfile.read(n)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"local says hi"}}]}\n\ndata: [DONE]\n\n')
HTTPServer(("127.0.0.1", port), H).serve_forever()
"#,
    )
    .unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::create_dir_all(e.models_dir()).unwrap();
    let model = e.models_dir().join("tiny-Q4_K_M.gguf");
    std::fs::write(&model, b"not a real model").unwrap();

    let hw = hinoko_core::local::Hardware { ram_gb: 8.0, threads: 4, ..Default::default() };
    let mut st = e.settings();
    st.local_engine = "llama".into(); // this test is about the llama.cpp runner path
    e.set_settings(st);
    e.local_start(&model, &hw).await.expect("starts");
    let info = e.local_info();
    assert!(info.running && info.port > 0);
    let s = e.settings();
    let p = s.provider(hinoko_core::local::PROVIDER_ID).expect("provider registered");
    assert_eq!(s.active, "local");
    assert_eq!(p.model, "tiny-Q4_K_M");

    let id = e.new_chat();
    e.run_chat(&id, "hi", false).await;
    assert_eq!(e.store().chat(&id).unwrap().messages.last().unwrap().text, "local says hi");

    // stop, then a chat starts it again by itself
    e.local_stop().await;
    assert!(!e.local_info().running);
    e.run_chat(&id, "again", false).await;
    let st = e.store();
    assert_eq!(st.chat(&id).unwrap().messages.last().unwrap().text, "local says hi", "restarted on demand: {:?}", st.chat(&id).unwrap().messages.iter().map(|m| &m.text).collect::<Vec<_>>());
    drop(st);
    e.local_stop().await;
    let _ = std::fs::remove_dir_all(dir);
}

fn scripted_coder(body: &str) -> (&'static str, String) {
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let done = v["messages"].as_array().map(|m| m.iter().filter(|x| x["role"] == "tool").count()).unwrap_or(0);
    let call = |name: &str, args: serde_json::Value| {
        serde_json::json!({"choices":[{"message":{"role":"assistant","content":"","tool_calls":[
            {"id":format!("c{done}"),"type":"function","function":{"name":name,"arguments":args.to_string()}}]}}]})
    };
    let reply = match done {
        0 => call("edit_file", serde_json::json!({"path":"hello.txt","old":"world","new":"Hinoko"})),
        1 => call("run_command", serde_json::json!({"command":"echo ran-by-agent > made.txt"})),
        2 => call("read_file", serde_json::json!({"path":"../../../etc/passwd"})),
        _ => serde_json::json!({"choices":[{"message":{"role":"assistant","content":"All done."}}]}),
    };
    ("application/json", reply.to_string())
}

async fn wait_for_ask(e: &Engine, id: &str, not: &str) -> (String, String) {
    for _ in 0..200 {
        {
            let st = e.store();
            if let Some(m) = st.code_session(id).unwrap().messages.iter().find(|m| m.kind == "ask" && m.state == "pending" && m.approval_id.as_deref() != Some(not)) {
                return (m.approval_id.clone().unwrap(), m.detail.clone());
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("no approval card appeared");
}

#[tokio::test(flavor = "multi_thread")]
async fn coding_agent_edits_with_approval_denies_runs_and_stays_in_the_folder() {
    let base = serve(scripted_coder).await;
    let (e, dir) = api_engine("coder", &base);
    let proj = dir.join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    std::fs::write(proj.join("hello.txt"), "hello world\n").unwrap();
    let id = e.code_new(proj.to_str().unwrap()).unwrap();
    e.code_send(&id, "greet Hinoko instead of the world").unwrap();

    // 1. the edit waits for approval and shows a diff; nothing changed yet
    let (ask, diff) = wait_for_ask(&e, &id, "").await;
    assert!(diff.contains("-hello world") && diff.contains("+hello Hinoko"), "{diff}");
    assert_eq!(std::fs::read_to_string(proj.join("hello.txt")).unwrap(), "hello world\n");
    e.code_decide(&ask, 1);

    // 2. the command asks too; deny it
    let (ask2, cmd) = wait_for_ask(&e, &id, &ask).await;
    let ask = ask2;
    assert!(cmd.contains("echo ran-by-agent"));
    e.code_decide(&ask, 0);

    for _ in 0..200 {
        if !e.is_code_busy(&id) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(std::fs::read_to_string(proj.join("hello.txt")).unwrap(), "hello Hinoko\n");
    assert!(!proj.join("made.txt").exists(), "a denied command must not run");
    {
        let st = e.store();
        let msgs = &st.code_session(&id).unwrap().messages;
        assert!(msgs.iter().any(|m| m.kind == "edit" && m.text == "Edited hello.txt"));
        assert!(msgs.iter().any(|m| m.kind == "ask" && m.state == "denied"));
        assert!(msgs.iter().any(|m| m.kind == "tool" && m.state == "failed" && m.text.contains("outside the project")), "path escape refused");
        assert_eq!(msgs.last().unwrap().text, "All done.");
    }
    // undo puts the file back
    assert!(e.code_can_undo(&id));
    e.code_undo(&id).unwrap();
    assert_eq!(std::fs::read_to_string(proj.join("hello.txt")).unwrap(), "hello world\n");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn auto_accept_edits_still_asks_before_commands_but_all_runs_them() {
    let base = serve(scripted_coder).await;
    let (e, dir) = api_engine("coder2", &base);
    let proj = dir.join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    std::fs::write(proj.join("hello.txt"), "hello world\n").unwrap();
    let id = e.code_new(proj.to_str().unwrap()).unwrap();
    e.store().code_mut(&id).unwrap().autonomy = "all".into();
    e.code_send(&id, "go").unwrap();
    for _ in 0..300 {
        if !e.is_code_busy(&id) && e.store().code_session(&id).unwrap().messages.iter().any(|m| m.text == "All done.") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(std::fs::read_to_string(proj.join("hello.txt")).unwrap(), "hello Hinoko\n");
    assert!(proj.join("made.txt").exists(), "autonomy=all runs commands without asking");
    assert!(e.store().code_session(&id).unwrap().messages.iter().all(|m| m.kind != "ask"));
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(feature = "builtin")]
#[tokio::test]
async fn builtin_engine_reports_a_readable_error_for_a_broken_model() {
    let (e, _dir) = engine("builtinbad", "");
    std::fs::create_dir_all(e.models_dir()).unwrap();
    let model = e.models_dir().join("broken-Q4_K_M.gguf");
    std::fs::write(&model, b"this is not a gguf file").unwrap();
    let mut st = e.settings();
    st.local_engine = "builtin".into();
    e.set_settings(st);
    let hw = hinoko_core::local::Hardware { ram_gb: 8.0, threads: 4, ..Default::default() };
    assert!(e.uses_builtin(&hw));
    let err = e.local_start(&model, &hw).await.expect_err("a junk file can't load");
    assert!(err.contains("couldn't load the model"), "{err}");
    assert!(!e.local_info().running);
}

fn fake_tool_caller(body: &str) -> (&'static str, String) {
    // a small model that "calls a tool" until it is told there are none
    let said = if body.contains("do not write a tool call") { "Here is a plain answer." } else { "<tool_call>\n<function=Web Search>\n</function>\n</tool_call>" };
    let chunk = format!("data: {}\n\n", serde_json::json!({"choices":[{"delta":{"content":said}}]}));
    ("text/event-stream", format!("{chunk}data: [DONE]\n\n"))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_made_up_tool_call_in_a_plain_chat_gets_one_nudge_and_then_an_answer() {
    let base = serve(fake_tool_caller).await;
    let (e, dir) = api_engine("nudge", &base);
    let id = e.new_chat();
    e.run_chat(&id, "what is new?", false).await;
    let st = e.store();
    let last = st.chat(&id).unwrap().messages.last().unwrap();
    assert_eq!(last.text, "Here is a plain answer.");
    drop(st);
    let _ = std::fs::remove_dir_all(dir);
}

fn summarizing_model(body: &str) -> (&'static str, String) {
    let said = if body.contains("You compress a conversation") {
        "SUMMARY-TEXT: the user is planning a trip"
    } else if body.contains("SUMMARY-TEXT") {
        "saw-summary"
    } else {
        "plain"
    };
    if !body.contains("\"stream\":true") {
        // the summarizer asks for a whole answer, not a stream
        return ("application/json", serde_json::json!({"choices":[{"message":{"role":"assistant","content":said}}]}).to_string());
    }
    let chunk = format!("data: {}\n\n", serde_json::json!({"choices":[{"delta":{"content":said}}]}));
    ("text/event-stream", format!("{chunk}data: [DONE]\n\n"))
}

#[tokio::test(flavor = "multi_thread")]
async fn long_chats_are_compacted_and_the_summary_reaches_the_model() {
    let base = serve(summarizing_model).await;
    let (e, dir) = api_engine("compact", &base);
    let id = e.new_chat();
    {
        // a long history: 14 messages of 8000 characters
        let mut st = e.store();
        for i in 0..14 {
            st.push_chat_msg(&id, if i % 2 == 0 { "user" } else { "bot" }, "x".repeat(8000));
        }
    }
    e.run_chat(&id, "and what next?", false).await;
    {
        let st = e.store();
        let c = st.chat(&id).unwrap();
        assert!(c.summary.starts_with("SUMMARY-TEXT"), "summary: {:?}", c.summary);
        assert!(c.covered > 4 && c.covered < c.messages.len(), "covered {}", c.covered);
        assert_eq!(c.messages[c.covered].kind, "user", "the kept part starts on a user turn");
        assert!(c.messages.iter().any(|m| m.kind == "event" && m.text.contains("summarized")));
        assert!(c.messages.len() > 14, "nothing was deleted");
    }
    // the next turn is sent with the summary and without the folded messages
    e.run_chat(&id, "thanks", false).await;
    let st = e.store();
    let last = st.chat(&id).unwrap().messages.iter().rev().find(|m| m.kind == "bot").unwrap();
    assert_eq!(last.text, "saw-summary");
    drop(st);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn compacting_a_short_chat_by_hand_says_so_and_a_failure_is_readable() {
    let base = serve(summarizing_model).await;
    let (e, dir) = api_engine("compact2", &base);
    let id = e.new_chat();
    e.run_chat(&id, "hi", false).await;
    assert_eq!(e.compact_chat(&id, true).await, Ok(false), "one exchange is too short to fold");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn citations_become_links_and_a_sources_list_is_added() {
    let src = vec![("Kauno liga".to_string(), "https://ktml.lt/a".to_string()), ("Other [x]".to_string(), "https://example.org/b".to_string())];
    let out = hinoko_core::engine::link_citations("He plays in KTML [1]. Also [2] and [9] and [1](kept).", &src);
    assert!(out.contains("KTML [[1]](https://ktml.lt/a)."), "{out}");
    assert!(out.contains("[[2]](https://example.org/b)"));
    assert!(out.contains("[9]") && !out.contains("[[9]]"), "unknown numbers stay as they are");
    assert!(out.contains("**Sources**\n1. [Kauno liga](https://ktml.lt/a)\n2. [Other x](https://example.org/b)"), "{out}");
    // nothing cited: still show what was looked at
    let none = hinoko_core::engine::link_citations("No numbers here.", &src);
    assert!(none.contains("**Sources looked at**"));
    // no sources: untouched
    assert_eq!(hinoko_core::engine::link_citations("Hi [1]", &[]), "Hi [1]");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_builtin_prompt_reaches_the_model_and_can_be_switched_off() {
    let base = serve(echo_body).await;
    let (e, dir) = api_engine("prompt", &base);
    let id = e.new_chat();
    e.run_chat(&id, "hi there", false).await;
    let reply = e.store().chat(&id).unwrap().messages.last().unwrap().text.clone();
    assert!(reply.contains("You are Hinoko"), "identity sent: {reply}");
    assert!(reply.contains("Today's date: "), "date filled in");
    assert!(!reply.contains("{{"), "no placeholder is left");

    let mut s = e.settings();
    s.builtin_prompt = false;
    e.set_settings(s);
    let id2 = e.new_chat();
    e.run_chat(&id2, "hi again", false).await;
    let reply = e.store().chat(&id2).unwrap().messages.last().unwrap().text.clone();
    assert!(!reply.contains("Today's date"), "prompt is off: {reply}");
    let _ = std::fs::remove_dir_all(dir);
}

/// First request: 429 with Retry-After: 1. Later ones: a streamed answer.
async fn serve_busy_once() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            let n = hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];
                let mut got = 0;
                // read until the whole request has arrived (headers + short body)
                for _ in 0..50 {
                    let k = sock.read(&mut buf[got..]).await.unwrap_or(0);
                    got += k;
                    if k == 0 || String::from_utf8_lossy(&buf[..got]).contains("\"messages\"") && String::from_utf8_lossy(&buf[..got]).trim_end().ends_with('}') {
                        break;
                    }
                }
                let resp = if n == 0 {
                    "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string()
                } else {
                    let chunk = format!("data: {}\n\n", serde_json::json!({"choices":[{"delta":{"content":"Back again"}}]}));
                    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{chunk}data: [DONE]\n\n")
                };
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}")
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rate_limited_provider_is_retried_after_its_retry_after() {
    let base = serve_busy_once().await;
    let (e, dir) = api_engine("busy", &base);
    let id = e.new_chat();
    let t = std::time::Instant::now();
    e.run_chat(&id, "are you there?", false).await;
    assert!(t.elapsed().as_millis() >= 900, "it waited the Retry-After second");
    assert_eq!(e.store().chat(&id).unwrap().messages.last().unwrap().text, "Back again");
    let _ = std::fs::remove_dir_all(dir);
}

fn spark_text_caller(body: &str) -> (&'static str, String) {
    // a model whose template had no tool support: it writes Spark-style call text as plain content
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let used = v["messages"].as_array().map(|m| m.iter().any(|x| x["role"] == "tool")).unwrap_or(false);
    let content = if used {
        "Done.".to_string()
    } else {
        "<tool_call>write_file<arg_key>path</arg_key><arg_value>spark.txt</arg_value><arg_key>content</arg_key><arg_value>made by spark</arg_value></tool_call>".to_string()
    };
    ("application/json", serde_json::json!({"choices":[{"message":{"role":"assistant","content":content}}]}).to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_calls_written_as_plain_spark_text_still_run_in_the_code_agent() {
    let base = serve(spark_text_caller).await;
    let (e, dir) = api_engine("spark", &base);
    let proj = dir.join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    let id = e.code_new(proj.to_str().unwrap()).unwrap();
    e.store().code_mut(&id).unwrap().autonomy = "all".into();
    e.code_send(&id, "make spark.txt").unwrap();
    for _ in 0..200 {
        if !e.is_code_busy(&id) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(std::fs::read_to_string(proj.join("spark.txt")).unwrap(), "made by spark");
    assert_eq!(e.store().code_session(&id).unwrap().messages.last().unwrap().text, "Done.");
    let _ = std::fs::remove_dir_all(dir);
}

fn double_editor(body: &str) -> (&'static str, String) {
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let done = v["messages"].as_array().map(|m| m.iter().filter(|x| x["role"] == "tool").count()).unwrap_or(0);
    let call = |args: serde_json::Value| {
        serde_json::json!({"choices":[{"message":{"role":"assistant","content":"","tool_calls":[
            {"id":format!("c{done}"),"type":"function","function":{"name":"edit_file","arguments":args.to_string()}}]}}]})
    };
    let reply = match done {
        0 => call(serde_json::json!({"path":"hello.txt","old":"world","new":"Hinoko"})),
        1 => call(serde_json::json!({"path":"hello.txt","old":"Hinoko","new":"Hinoko!!"})),
        _ => serde_json::json!({"choices":[{"message":{"role":"assistant","content":"Edited twice."}}]}),
    };
    ("application/json", reply.to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn undo_request_reverts_everything_the_last_request_changed() {
    let base = serve(double_editor).await;
    let (e, dir) = api_engine("undorun", &base);
    let proj = dir.join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    std::fs::write(proj.join("hello.txt"), "hello world\n").unwrap();
    let id = e.code_new(proj.to_str().unwrap()).unwrap();
    e.store().code_mut(&id).unwrap().autonomy = "edits".into();
    e.code_send(&id, "shout the greeting").unwrap();
    for _ in 0..200 {
        if !e.is_code_busy(&id) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(std::fs::read_to_string(proj.join("hello.txt")).unwrap(), "hello Hinoko!!\n");
    assert_eq!(e.code_undo_run(&id).unwrap(), 2);
    assert_eq!(std::fs::read_to_string(proj.join("hello.txt")).unwrap(), "hello world\n");
    assert!(!e.code_can_undo(&id));
    assert!(e.code_undo_run(&id).is_err(), "nothing left to undo");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn traces_are_written_only_when_switched_on() {
    let base = serve(sse).await;
    let (e, dir) = api_engine("trace", &base);
    let id = e.new_chat();
    e.run_chat(&id, "hello", false).await;
    assert!(!dir.join("traces").exists(), "off by default");
    let mut s = e.settings();
    s.traces = true;
    e.set_settings(s);
    e.run_chat(&id, "hello again", false).await;
    let file = std::fs::read_dir(dir.join("traces")).unwrap().next().unwrap().unwrap().path();
    let text = std::fs::read_to_string(file).unwrap();
    let line: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(line["kind"], "chat_reply");
    assert_eq!(line["data"]["text"], "Hello world");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn comparing_two_models_fills_both_sides_and_times_them() {
    let base = serve(sse).await;
    let (e, dir) = api_engine("compare", &base);
    let mut s = e.settings();
    let mut p = Provider::blank(Kind::OpenAi);
    p.name = "Second".into();
    p.base_url = format!("{base}/v1");
    p.model = "other".into();
    s.upsert(p);
    e.set_settings(s);
    let ids: Vec<Option<String>> = e.settings().providers.iter().map(|p| Some(p.id.clone())).collect();
    assert!(ids.len() >= 2);
    let cmp = e.compare_start("say hello", ids[..2].to_vec());
    for _ in 0..200 {
        if cmp.finished() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let run = cmp.run.lock().unwrap();
    assert_eq!(run.lanes.len(), 2);
    for lane in &run.lanes {
        assert_eq!(lane.text, "Hello world");
        assert!(lane.done && lane.error.is_empty());
        assert!(!lane.model.is_empty());
    }
    drop(run);
    let _ = std::fs::remove_dir_all(dir);
}

fn sse_with_usage(_: &str) -> (&'static str, String) {
    let chunk = |t: &str| format!("data: {}\n\n", serde_json::json!({"choices":[{"delta":{"content":t}}]}));
    let usage = format!("data: {}\n\n", serde_json::json!({"choices":[],"usage":{"completion_tokens":37}}));
    ("text/event-stream", format!("{}{}{usage}data: [DONE]\n\n", chunk("Some "), chunk("words")))
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_replies_carry_a_speed_line_with_the_servers_token_count() {
    let base = serve(sse_with_usage).await;
    let (e, dir) = api_engine("speed", &base);
    let id = e.new_chat();
    e.run_chat(&id, "count for me", false).await;
    let line = e.store().chat(&id).unwrap().messages.last().unwrap().detail.clone();
    assert!(line.contains("tok/s") && line.contains("37 tokens") && !line.contains('~'), "{line}");
    // a server that reports nothing gets an estimate, marked with ~
    let base2 = serve(sse).await;
    let (e2, dir2) = api_engine("speed2", &base2);
    let id2 = e2.new_chat();
    e2.run_chat(&id2, "count for me", false).await;
    let line = e2.store().chat(&id2).unwrap().messages.last().unwrap().detail.clone();
    assert!(line.starts_with('~') && line.contains("tokens"), "{line}");
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(dir2);
}

fn only_thinks(_: &str) -> (&'static str, String) {
    let think = format!("data: {}\n\n", serde_json::json!({"choices":[{"delta":{"reasoning_content":"let me think about it for a very long time"}}]}));
    ("text/event-stream", format!("{think}data: [DONE]\n\n"))
}

#[tokio::test(flavor = "multi_thread")]
async fn the_reply_limit_is_a_setting_and_a_model_that_only_thinks_keeps_its_thoughts() {
    let base = serve(echo_body).await;
    let (e, dir) = api_engine("limit", &base);
    let id = e.new_chat();
    e.run_chat(&id, "hi", false).await;
    let reply = e.store().chat(&id).unwrap().messages.last().unwrap().text.clone();
    assert!(reply.contains("\"max_tokens\":16384"), "auto with thinking left on auto: {reply}");
    let mut s = e.settings();
    s.max_reply_tokens = 3000;
    e.set_settings(s);
    let id2 = e.new_chat();
    e.run_chat(&id2, "hi", false).await;
    let reply = e.store().chat(&id2).unwrap().messages.last().unwrap().text.clone();
    assert!(reply.contains("\"max_tokens\":3000"), "{reply}");

    let base2 = serve(only_thinks).await;
    let (e2, dir2) = api_engine("thinkonly", &base2);
    let id3 = e2.new_chat();
    e2.run_chat(&id3, "hard question", false).await;
    let st = e2.store();
    let msgs = &st.chat(&id3).unwrap().messages;
    let last = msgs.last().unwrap();
    assert_eq!(last.kind, "bot");
    assert!(last.think.contains("very long time"), "thoughts are kept");
    assert!(last.text.contains("reply budget") && last.text.contains("Max reply length"), "{}", last.text);
    assert!(msgs.iter().all(|m| m.kind != "error"), "no error line");
    drop(st);
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(dir2);
}

#[tokio::test(flavor = "multi_thread")]
async fn off_on_a_local_server_falls_back_to_the_template_flag_before_giving_up() {
    // the fallback order: both settings, then only the template flag, then a plain request
    let mut p = Provider::blank(Kind::OpenAi);
    p.base_url = "http://127.0.0.1:1/v1".into();
    let v = hinoko_core::llm::request_variants(&p, Some("off"));
    assert!(v.len() == 3 && v[1].len() == 1 && v[1][0].0 == "chat_template_kwargs");
}

fn stream_of(deltas: Vec<serde_json::Value>) -> (&'static str, String) {
    let mut out = String::new();
    for d in deltas {
        out.push_str(&format!("data: {}\n\n", serde_json::json!({"choices":[{"delta":d}]})));
    }
    out.push_str("data: [DONE]\n\n");
    ("text/event-stream", out)
}

fn thinks_in_circles(_: &str) -> (&'static str, String) {
    let line = "Wait, let me reconsider the whole problem from the beginning and check the first step again before I decide anything at all. ";
    stream_of((0..14).map(|_| serde_json::json!({"reasoning_content": line})).collect())
}

fn writes_in_circles(_: &str) -> (&'static str, String) {
    let line = "I will now check the file again and then check the file once more to be sure about the result of the check, which I will do again. ";
    let mut v = vec![serde_json::json!({"content": "Here is the start of a good answer. "})];
    v.extend((0..14).map(|_| serde_json::json!({"content": line})));
    stream_of(v)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_stuck_in_a_thinking_loop_is_stopped_and_keeps_its_thoughts() {
    let base = serve(thinks_in_circles).await;
    let (e, dir) = api_engine("thinkloop", &base);
    let id = e.new_chat();
    e.run_chat(&id, "hard question", false).await;
    let st = e.store();
    let msgs = &st.chat(&id).unwrap().messages;
    let last = msgs.last().unwrap();
    assert_eq!(last.kind, "bot");
    assert!(last.think.contains("reconsider"), "thoughts kept");
    assert!(last.text.contains("thinking loop") && last.text.contains("Temperature"), "{}", last.text);
    assert!(msgs.iter().all(|m| m.kind != "error"));
    drop(st);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn repeating_text_is_cut_back_to_the_first_time_it_was_said() {
    let base = serve(writes_in_circles).await;
    let (e, dir) = api_engine("textloop", &base);
    let id = e.new_chat();
    e.run_chat(&id, "write something", false).await;
    let text = e.store().chat(&id).unwrap().messages.last().unwrap().text.clone();
    assert!(text.starts_with("Here is the start of a good answer."), "{text}");
    assert!(text.contains("repeating itself"), "{text}");
    assert_eq!(text.matches("check the file once more").count(), 1, "only the first copy stays: {text}");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dead_provider_hands_over_to_its_backup() {
    let base = serve(sse).await;
    let (e, dir) = api_engine("failover", &base);
    let mut s = e.settings();
    let backup_id = s.providers[0].id.clone();
    let mut dead = Provider::blank(Kind::OpenAi);
    dead.name = "Down".into();
    dead.base_url = "http://127.0.0.1:1/v1".into();
    dead.model = "gone".into();
    dead.backup = backup_id;
    let dead_id = s.upsert(dead);
    s.active = dead_id;
    e.set_settings(s);
    let id = e.new_chat();
    e.run_chat(&id, "hello there", false).await;
    let st = e.store();
    let msgs = &st.chat(&id).unwrap().messages;
    assert_eq!(msgs.last().unwrap().text, "Hello world", "the backup answered");
    assert!(msgs.iter().any(|m| m.kind == "event" && m.text.contains("Switched to Fake API")), "{:?}", msgs.iter().map(|m| (&m.kind, &m.text)).collect::<Vec<_>>());
    drop(st);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_can_have_its_own_temperature_and_top_p() {
    let base = serve(echo_body).await;
    let (e, dir) = api_engine("sampling", &base);
    let mut s = e.settings();
    s.providers[0].temperature = Some(1.0);
    s.providers[0].top_p = Some(0.95);
    e.set_settings(s);
    let id = e.new_chat();
    e.run_chat(&id, "hi", false).await;
    let reply = e.store().chat(&id).unwrap().messages.last().unwrap().text.clone();
    assert!(reply.contains("\"temperature\":1.0") && reply.contains("\"top_p\":0.95"), "{reply}");
    let _ = std::fs::remove_dir_all(dir);
}

static REVIEWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn coder_with_a_reviewer(body: &str) -> (&'static str, String) {
    use std::sync::atomic::Ordering;
    let say = |t: &str| serde_json::json!({"choices":[{"message":{"role":"assistant","content":t}}]}).to_string();
    if body.contains("strict, practical code reviewer") {
        let n = REVIEWS.fetch_add(1, Ordering::SeqCst);
        return ("application/json", say(if n == 0 { "1. hello.txt: the greeting should end with a period." } else { "LGTM" }));
    }
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let tools = v["messages"].as_array().map(|m| m.iter().filter(|x| x["role"] == "tool").count()).unwrap_or(0);
    if tools == 0 {
        let args = serde_json::json!({"path":"hello.txt","old":"world","new":"Hinoko"});
        return ("application/json", serde_json::json!({"choices":[{"message":{"role":"assistant","content":"","tool_calls":[
            {"id":"c0","type":"function","function":{"name":"edit_file","arguments":args.to_string()}}]}}]}).to_string());
    }
    let fixed = body.contains("A reviewer checked your changes");
    ("application/json", say(if fixed { "Looked at the review; no change needed." } else { "Edited." }))
}

#[tokio::test(flavor = "multi_thread")]
async fn the_reviewer_sends_work_back_with_problems_and_signs_it_off() {
    REVIEWS.store(0, std::sync::atomic::Ordering::SeqCst);
    let base = serve(coder_with_a_reviewer).await;
    let (e, dir) = api_engine("review", &base);
    let mut s = e.settings();
    s.code_review = 2;
    e.set_settings(s);
    let proj = dir.join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    std::fs::write(proj.join("hello.txt"), "hello world\n").unwrap();
    let id = e.code_new(proj.to_str().unwrap()).unwrap();
    e.store().code_mut(&id).unwrap().autonomy = "edits".into();
    e.code_send(&id, "greet Hinoko").unwrap();
    for _ in 0..300 {
        if !e.is_code_busy(&id) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let st = e.store();
    let texts: Vec<String> = st.code_session(&id).unwrap().messages.iter().map(|m| m.text.clone()).collect();
    drop(st);
    assert!(texts.iter().any(|t| t.starts_with("Reviewer found problems") && t.contains("should end with a period")), "{texts:?}");
    assert!(texts.iter().any(|t| t == "Looked at the review; no change needed."), "the agent answered the review: {texts:?}");
    assert!(texts.iter().any(|t| t == "Reviewer: no problems found"), "{texts:?}");
    assert_eq!(REVIEWS.load(std::sync::atomic::Ordering::SeqCst), 2);
    let _ = std::fs::remove_dir_all(dir);
}

const DELETER: &str = r#"in=$(cat); n=$(printf '%s' "$in" | grep -c '<tool_result id=')
if [ "$n" = 0 ]; then echo '<tool_call>{"name":"write_file","arguments":{"path":"old.md","content":"x"}}</tool_call>'
elif [ "$n" = 1 ]; then echo '<tool_call>{"name":"propose_delete","arguments":{"path":"old.md","reason":"Archive it"}}</tool_call>'
else echo 'Done for now.'; fi"#;

#[tokio::test(flavor = "multi_thread")]
async fn a_standing_rule_answers_the_approval_so_the_bot_does_not_wait() {
    let (e, dir) = engine("rule", DELETER);
    let id = e.store().add_bot("Ada", "Research", "Keep notes.").unwrap();
    let mut s = e.settings();
    s.rules.push(hinoko_core::Rule { id: "r1".into(), bot_id: id.clone(), action: "delete".into(), contains: "OLD.MD".into(), allow: true });
    e.set_settings(s);
    e.run_turn(&id, "archive", "archive").await;
    let st = e.store();
    assert!(st.approvals.iter().all(|a| a.status != "pending"), "nothing left waiting");
    assert!(!st.computers[&id].files.contains_key("/home/ada/old.md"), "the rule allowed the delete");
    assert!(st.messages[&id].iter().any(|m| m.kind == "event" && m.text.starts_with("Standing rule allows")));
    assert_eq!(st.bot(&id).unwrap().presence, "done");
    drop(st);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn deny_rules_beat_allow_rules_and_other_bots_are_not_covered() {
    let (e, dir) = engine("rule2", DELETER);
    let id = e.store().add_bot("Ada", "Research", "Keep notes.").unwrap();
    let other = e.store().add_bot("Bo", "Research", "Keep notes.").unwrap();
    let mut s = e.settings();
    s.rules.push(hinoko_core::Rule { id: "a".into(), bot_id: String::new(), action: String::new(), contains: "old.md".into(), allow: true });
    s.rules.push(hinoko_core::Rule { id: "d".into(), bot_id: id.clone(), action: "delete".into(), contains: "old".into(), allow: false });
    s.rules.push(hinoko_core::Rule { id: "blank".into(), bot_id: String::new(), action: String::new(), contains: "  ".into(), allow: true });
    e.set_settings(s);
    e.run_turn(&id, "archive", "archive").await;
    let st = e.store();
    assert!(st.computers[&id].files.contains_key("/home/ada/old.md"), "denied, so the file stays");
    assert!(st.approvals.iter().any(|a| a.status == "denied"));
    drop(st);
    // a rule for another bot, and a blank rule, never match
    let mut s = e.settings();
    s.rules.retain(|r| r.id == "blank");
    s.rules.push(hinoko_core::Rule { id: "x".into(), bot_id: other, action: "delete".into(), contains: "old.md".into(), allow: true });
    e.set_settings(s);
    e.run_turn(&id, "archive again", "archive again").await;
    assert!(e.store().approvals.iter().any(|a| a.status == "pending"), "no rule covers this bot, so it waits");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_quiet_routine_with_nothing_to_report_leaves_one_muted_line() {
    let (e, dir) = engine("quiet", "echo NOTHING_TO_REPORT");
    let id = e.store().add_bot("Ada", "Research", "Watch things.").unwrap();
    e.store().routines.push(hinoko_core::model::Routine { id: "r1".into(), bot_id: id.clone(), name: "Check site".into(), schedule: "daily".into(), prompt: "Look at the site.".into(), enabled: true });
    e.run_routine("r1").unwrap();
    for _ in 0..100 {
        if !e.is_busy(&id) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let st = e.store();
    let last = st.messages[&id].last().unwrap();
    assert_eq!(last.kind, "event");
    assert!(last.text.contains("nothing to report") && last.text.contains("Check site"), "{}", last.text);
    assert!(st.messages[&id].iter().all(|m| m.kind != "bot"), "no bot message");
    drop(st);
    // with quiet off the bot's words stay a message
    let mut s = e.settings();
    s.quiet_routines = false;
    e.set_settings(s);
    e.run_routine("r1").unwrap();
    for _ in 0..100 {
        if !e.is_busy(&id) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(e.store().messages[&id].last().unwrap().kind, "bot");
    let _ = std::fs::remove_dir_all(dir);
}

fn slow_answer(_: &str) -> (&'static str, String) {
    std::thread::sleep(std::time::Duration::from_millis(400));
    ("application/json", serde_json::json!({"choices":[{"message":{"role":"assistant","content":"All done here."}}]}).to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn code_sessions_run_side_by_side() {
    let base = serve(slow_answer).await;
    let (e, dir) = api_engine("parallel", &base);
    let (pa, pb) = (dir.join("a"), dir.join("b"));
    std::fs::create_dir_all(&pa).unwrap();
    std::fs::create_dir_all(&pb).unwrap();
    let a = e.code_new(pa.to_str().unwrap()).unwrap();
    let b = e.code_new(pb.to_str().unwrap()).unwrap();
    let t0 = std::time::Instant::now();
    e.code_send(&a, "look around").unwrap();
    e.code_send(&b, "look around too").unwrap();
    assert!(e.is_code_busy(&a) && e.is_code_busy(&b), "both are working at the same moment");
    assert_eq!(e.code_send(&a, "again"), Err("still working"), "one request at a time per session");
    for _ in 0..200 {
        if !e.is_code_busy(&a) && !e.is_code_busy(&b) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    for id in [&a, &b] {
        assert_eq!(e.store().code_session(id).unwrap().messages.last().unwrap().text, "All done here.");
    }
    assert!(t0.elapsed() < std::time::Duration::from_millis(780), "ran together, not one after the other: {:?}", t0.elapsed());
    let _ = std::fs::remove_dir_all(dir);
}

fn tool_reply(name: &str, args: serde_json::Value) -> (&'static str, String) {
    let r = serde_json::json!({"choices":[{"message":{"role":"assistant","content":"","tool_calls":[
        {"id":"c1","type":"function","function":{"name":name,"arguments":args.to_string()}}]}}]});
    ("application/json", r.to_string())
}

fn say(text: &str) -> (&'static str, String) {
    ("application/json", serde_json::json!({"choices":[{"message":{"role":"assistant","content":text}}]}).to_string())
}

async fn code_finish(e: &Engine, id: &str) {
    for _ in 0..300 {
        if !e.is_code_busy(id) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("the coding session never finished");
}

fn defer_probe(body: &str) -> (&'static str, String) {
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let has = v["tools"].as_array().is_some_and(|t| t.iter().any(|x| x["function"]["name"] == "web_search"));
    let asked = v["messages"].as_array().is_some_and(|m| m.iter().any(|x| x["role"] == "tool"));
    match (asked, has) {
        (false, true) => say("FAIL: web_search was sent before anyone asked for it"),
        (false, false) => tool_reply("tool_search", serde_json::json!({"query": "search the web"})),
        (true, true) => say("OK"),
        (true, false) => say("FAIL: web_search was not loaded"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn rare_tools_are_held_back_until_the_model_asks_for_them() {
    let base = serve(defer_probe).await;
    let (e, dir) = api_engine("defer", &base);
    let proj = dir.join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    let id = e.code_new(proj.to_str().unwrap()).unwrap();
    e.code_send(&id, "look something up").unwrap();
    code_finish(&e, &id).await;
    assert_eq!(e.store().code_session(&id).unwrap().messages.last().unwrap().text, "OK");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_stuck_repeating_the_same_call_is_stopped() {
    let base = serve(|_| tool_reply("list_dir", serde_json::json!({"path": "."}))).await;
    let (e, dir) = api_engine("stuck", &base);
    let proj = dir.join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    let id = e.code_new(proj.to_str().unwrap()).unwrap();
    e.code_send(&id, "keep looking").unwrap();
    code_finish(&e, &id).await;
    let last = e.store().code_session(&id).unwrap().messages.last().unwrap().text.clone();
    assert!(last.contains("kept repeating itself"), "{last}");
    let _ = std::fs::remove_dir_all(dir);
}
