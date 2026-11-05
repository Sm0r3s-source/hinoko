//! Command-line mode: the same engine, no window.

use std::io::{BufRead, Write};
use std::sync::Arc;

use hinoko_core::{Engine, Kind, Paths, Provider};
use serde_json::json;

const HELP: &str = "\
hinoko                         open the desktop app
hinoko bots                    list bots
hinoko new <name> <role> <mandate...>
hinoko run <bot> <prompt...>   one turn; add --json for one JSON line per message
hinoko chat <bot>              interactive chat (/quit to leave)
hinoko approvals               list pending approvals
hinoko approve <id> | deny <id>
hinoko provider list
hinoko provider add --name N --kind api|cli [--url U --model M --key K --key-env VAR --no-tools]
                    [--cmd C --arg A --arg B ...]
hinoko provider use <id> | remove <id> | test [id]
hinoko machine list | add --name N --host H --user U [--port P] [--auto-readonly] | remove <id> | test <id> | key
hinoko path                    print the data folder
Run only one Hinoko (window or CLI) at a time: they share one state file.
";

pub fn wants_cli(args: &[String]) -> bool {
    args.get(1).is_some()
}

pub fn run(args: &[String]) -> i32 {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("runtime");
    let engine = match Engine::new(Paths::discover(), rt.handle().clone(), Arc::new(|| {})) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("hinoko: {e}");
            return 1;
        }
    };
    let rest: Vec<String> = args[2..].to_vec();
    let code = match args[1].as_str() {
        "help" | "--help" | "-h" => {
            print!("{HELP}");
            0
        }
        "path" => {
            println!("{}", engine.home().display());
            0
        }
        "bots" => {
            let st = engine.store();
            if st.bots.is_empty() {
                println!("no bots yet. Create one: hinoko new <name> <role> <mandate>");
            }
            for b in &st.bots {
                println!("{}\t{}\t{}{}", b.id, b.name, b.title, if b.read_only { "\t(read-only)" } else { "" });
            }
            0
        }
        "new" => {
            if rest.len() < 3 {
                eprintln!("usage: hinoko new <name> <role> <mandate...>");
                return 2;
            }
            let res = engine.store().add_bot(&rest[0], &rest[1], &rest[2..].join(" "));
            match res {
                Ok(id) => {
                    engine.save();
                    println!("{id}");
                    0
                }
                Err(e) => {
                    eprintln!("hinoko: {e}");
                    1
                }
            }
        }
        "run" => {
            let json_out = rest.iter().any(|a| a == "--json");
            let words: Vec<&String> = rest.iter().filter(|a| *a != "--json").collect();
            if words.len() < 2 {
                eprintln!("usage: hinoko run <bot> <prompt...> [--json]");
                return 2;
            }
            let bot = words[0].clone();
            let prompt = words[1..].iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ");
            rt.block_on(one_turn(&engine, &bot, &prompt, json_out))
        }
        "chat" => {
            let Some(bot) = rest.first().cloned() else {
                eprintln!("usage: hinoko chat <bot>");
                return 2;
            };
            if engine.store().bot(&bot).is_none() {
                eprintln!("hinoko: no bot called {bot}");
                return 1;
            }
            let stdin = std::io::stdin();
            loop {
                print!("you> ");
                let _ = std::io::stdout().flush();
                let mut line = String::new();
                if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                let line = line.trim();
                if line == "/quit" || line == "/exit" {
                    break;
                }
                if line.is_empty() {
                    continue;
                }
                rt.block_on(one_turn(&engine, &bot, line, false));
            }
            0
        }
        "approvals" => {
            for a in engine.store().approvals.iter().filter(|a| a.status == "pending") {
                println!("{}\t{}\t{}\t{}", a.id, a.bot_id, a.title, a.detail);
            }
            0
        }
        "approve" | "deny" => {
            let Some(id) = rest.first() else {
                eprintln!("usage: hinoko approve|deny <id>");
                return 2;
            };
            match rt.block_on(engine.decide_and_wait(id, args[1] == "approve")) {
                Ok(()) => {
                    println!("{}d", args[1]);
                    0
                }
                Err(e) => {
                    eprintln!("hinoko: {e}");
                    1
                }
            }
        }
        "provider" => provider(&engine, &rt, &rest),
        "machine" => machine(&engine, &rt, &rest),
        _ => {
            eprint!("unknown command: {}\n\n{HELP}", args[1]);
            2
        }
    };
    engine.save();
    code
}

async fn one_turn(engine: &Engine, bot: &str, prompt: &str, json_out: bool) -> i32 {
    let before = match engine.store().messages.get(bot) {
        Some(v) => v.len(),
        None => {
            eprintln!("hinoko: no bot called {bot}");
            return 1;
        }
    };
    let Some(_guard) = engine.try_busy(bot) else {
        eprintln!("hinoko: that bot is mid-turn");
        return 1;
    };
    engine.run_turn(bot, prompt, prompt).await;
    let st = engine.store();
    let new: Vec<_> = st.messages[bot].iter().skip(before + 1).cloned().collect(); // +1 skips the user line
    for m in &new {
        if json_out {
            println!("{}", json!({"kind": m.kind, "text": m.text, "approval_id": m.approval_id}));
        } else {
            match m.kind.as_str() {
                "bot" => println!("{}", m.text),
                "event" => println!("  · {}", m.text),
                "approval" => println!("  ! needs approval: {} (hinoko approve {})", m.text, m.approval_id.clone().unwrap_or_default()),
                _ => {}
            }
        }
    }
    0
}

fn provider(engine: &Engine, rt: &tokio::runtime::Runtime, rest: &[String]) -> i32 {
    let mut s = engine.settings();
    match rest.first().map(String::as_str) {
        Some("list") | None => {
            if s.providers.is_empty() {
                println!("no providers yet. Add one: hinoko provider add --name Local --kind api --url http://127.0.0.1:11434/v1 --model llama3.1");
            }
            for p in &s.providers {
                println!("{}{}\t{}\t{}\t{}", if p.id == s.active { "* " } else { "  " }, p.id, p.name, p.kind.label(), p.summary());
            }
            0
        }
        Some("add") => {
            let mut p = Provider::blank(Kind::OpenAi);
            let mut it = rest[1..].iter();
            while let Some(flag) = it.next() {
                let mut val = || it.next().cloned().unwrap_or_default();
                match flag.as_str() {
                    "--name" => p.name = val(),
                    "--kind" => p.kind = if val() == "cli" { Kind::Cli } else { Kind::OpenAi },
                    "--url" => p.base_url = val().trim_end_matches('/').to_string(),
                    "--model" => p.model = val(),
                    "--key" => p.api_key = val(),
                    "--key-env" => p.api_key_env = val(),
                    "--cmd" => p.command = val(),
                    "--arg" => p.args.push(val()),
                    "--no-tools" => p.native_tools = false,
                    other => {
                        eprintln!("unknown flag {other}");
                        return 2;
                    }
                }
            }
            if p.name.is_empty() || !p.ready() {
                eprintln!("a provider needs --name and either (--url and --model) or --kind cli --cmd");
                return 2;
            }
            let id = s.upsert(p);
            engine.set_settings(s);
            println!("{id}");
            0
        }
        Some("use") => match rest.get(1) {
            Some(id) if s.provider(id).is_some() => {
                s.active = id.clone();
                engine.set_settings(s);
                0
            }
            _ => {
                eprintln!("hinoko: no such provider");
                1
            }
        },
        Some("remove") => match rest.get(1) {
            Some(id) => {
                s.remove(id);
                engine.set_settings(s);
                0
            }
            None => 2,
        },
        Some("test") => {
            let Some(p) = rest.get(1).and_then(|id| s.provider(id)).or_else(|| s.provider_for(None)).cloned() else {
                eprintln!("hinoko: no provider to test");
                return 1;
            };
            let (tx, rx) = std::sync::mpsc::channel();
            engine.test_provider(p, move |r| {
                let _ = tx.send(r);
            });
            let _guard = rt.enter();
            match rx.recv() {
                Ok(Ok(reply)) => {
                    println!("ok: {reply}");
                    0
                }
                Ok(Err(e)) => {
                    eprintln!("failed: {e}");
                    1
                }
                Err(_) => 1,
            }
        }
        _ => {
            eprint!("{HELP}");
            2
        }
    }
}

fn machine(engine: &Engine, rt: &tokio::runtime::Runtime, rest: &[String]) -> i32 {
    let mut s = engine.settings();
    match rest.first().map(String::as_str) {
        Some("list") | None => {
            for m in &s.machines {
                println!("{}\t{}\t{}@{}:{}{}", m.id, m.name, m.user, m.host, m.port, if m.auto_readonly { "\t(read-only commands run without asking)" } else { "" });
            }
            0
        }
        Some("add") => {
            let mut m = hinoko_core::Machine::blank();
            let mut it = rest[1..].iter();
            while let Some(flag) = it.next() {
                let mut val = || it.next().cloned().unwrap_or_default();
                match flag.as_str() {
                    "--name" => m.name = val(),
                    "--host" => m.host = val(),
                    "--user" => m.user = val(),
                    "--port" => m.port = val().parse().unwrap_or(22),
                    "--notes" => m.notes = val(),
                    "--auto-readonly" => m.auto_readonly = true,
                    other => {
                        eprintln!("unknown flag {other}");
                        return 2;
                    }
                }
            }
            if m.name.is_empty() {
                m.name = m.host.clone();
            }
            if let Err(e) = hinoko_core::ssh::validate(&m) {
                eprintln!("hinoko: {e}");
                return 2;
            }
            let id = s.upsert_machine(m);
            engine.set_settings(s);
            println!("{id}");
            0
        }
        Some("remove") => {
            if let Some(id) = rest.get(1) {
                s.machines.retain(|m| &m.id != id);
                engine.set_settings(s);
            }
            0
        }
        Some("test") => {
            let Some(m) = rest.get(1).and_then(|id| s.machine(id)).cloned() else {
                eprintln!("hinoko: no such machine");
                return 1;
            };
            let (tx, rx) = std::sync::mpsc::channel();
            engine.test_machine(m, move |r| {
                let _ = tx.send(r);
            });
            let _guard = rt.enter();
            match rx.recv() {
                Ok(Ok(_)) => {
                    println!("connected");
                    0
                }
                Ok(Err(e)) => {
                    eprintln!("failed: {e}");
                    1
                }
                Err(_) => 1,
            }
        }
        Some("key") => match rt.block_on(engine.public_key()) {
            Ok(k) => {
                println!("{k}\n\nRun this once on the other computer:\n{}", hinoko_core::ssh::authorize_command(&k));
                0
            }
            Err(e) => {
                eprintln!("hinoko: {e}");
                1
            }
        },
        _ => {
            eprint!("{HELP}");
            2
        }
    }
}
