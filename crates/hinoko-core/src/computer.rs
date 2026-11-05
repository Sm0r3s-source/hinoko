use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::model::{clip, Computer};

pub fn exec(comp: &mut Computer, home: &str, raw: &str) -> String {
    let shown = raw.trim();
    if shown.is_empty() {
        return err("type help for the short shell");
    }
    if shown.chars().count() > 200 {
        return record(comp, shown, err("that command is too long"));
    }
    if shown.chars().any(|c| matches!(c, '|' | ';' | '&' | '>' | '<' | '`' | '$' | '\\' | '\n')) {
        return record(comp, shown, err("pipes and redirects are not on this computer"));
    }
    let tokens = match tokenize(shown) {
        Ok(t) if !t.is_empty() => t,
        Ok(_) => return record(comp, shown, err("empty command")),
        Err(()) => return record(comp, shown, err("unbalanced quotes")),
    };
    let cmd = tokens[0].clone();
    let arg = tokens.get(1).cloned();
    let out = match cmd.as_str() {
        "help" => help_text().to_string(),
        "pwd" => comp.cwd.clone(),
        "date" => utc_now(),
        "echo" => tokens[1..].join(" "),
        "cd" => change_dir(comp, home, arg.as_deref().unwrap_or(home)),
        "ls" => list_path(comp, home, arg.as_deref()),
        "find" => find(comp, home, arg.as_deref()),
        "cat" | "head" | "wc" => file_cmd(comp, home, &cmd, arg.as_deref()),
        other => err(&format!("{other} is not on this computer")),
    };
    record(comp, shown, out)
}

pub fn write_file(comp: &mut Computer, home: &str, path: &str, content: &str) -> Result<String, String> {
    let path = resolve(home, &comp.cwd, path)?;
    if content.chars().count() > 12_000 {
        return Err("that file is too long for this computer".into());
    }
    if !comp.files.contains_key(&path) && comp.files.len() >= 40 {
        return Err("this computer is full".into());
    }
    let content = clip(content, 12_000);
    comp.files.insert(path.clone(), content.clone());
    let name = file_name(&path);
    comp.show(&path, &name, &content);
    comp.push_term(&format!("write {path}"), "wrote the file");
    Ok(path)
}

pub fn read_file(comp: &mut Computer, home: &str, path: &str) -> Result<String, String> {
    let path = resolve(home, &comp.cwd, path)?;
    let body = comp
        .files
        .get(&path)
        .cloned()
        .ok_or_else(|| format!("no such file: {path}"))?;
    let name = file_name(&path);
    comp.show(&path, &name, &body);
    Ok(clip(&body, 4000))
}

pub fn remove_file(comp: &mut Computer, home: &str, path: &str) -> Result<String, String> {
    let path = resolve(home, &comp.cwd, path)?;
    if comp.files.remove(&path).is_none() {
        return Err(format!("no such file: {path}"));
    }
    comp.push_term(&format!("rm {path}"), "removed");
    if comp.browser_url == path {
        comp.show("", "Nothing open", "That file was removed.");
    }
    Ok(path)
}

pub fn resolve(home: &str, cwd: &str, input: &str) -> Result<String, String> {
    if input.contains('\0') {
        return Err("bad path".into());
    }
    let home = home.trim_end_matches('/');
    let joined = if input.starts_with('/') {
        input.to_string()
    } else {
        format!("{}/{}", cwd.trim_end_matches('/'), input)
    };
    let mut stack: Vec<String> = Vec::new();
    for part in joined.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if stack.pop().is_none() {
                    return Err("that path leaves this computer".into());
                }
            }
            other => {
                if other.contains('\\') {
                    return Err("bad path".into());
                }
                stack.push(other.to_string());
            }
        }
    }
    let path = format!("/{}", stack.join("/"));
    if path != home && !path.starts_with(&format!("{home}/")) {
        return Err("that path is outside this bot's home".into());
    }
    Ok(path)
}

fn record(comp: &mut Computer, cmd: &str, out: String) -> String {
    comp.push_term(cmd, &out);
    out
}

fn err(msg: &str) -> String {
    format!("error: {msg}")
}

fn help_text() -> &'static str {
    "pwd, ls, cat, head, wc, date, echo, find, cd, help\nNo pipes, no network, no path outside this home."
}

fn change_dir(comp: &mut Computer, home: &str, path: &str) -> String {
    let path = match resolve(home, &comp.cwd, path) {
        Ok(p) => p,
        Err(e) => return err(&e),
    };
    if !is_dir(&comp.files, home, &path) {
        return err("not a directory");
    }
    comp.cwd = path.clone();
    path
}

fn list_path(comp: &Computer, home: &str, path: Option<&str>) -> String {
    let cwd = comp.cwd.clone();
    let target = match path {
        Some(p) => match resolve(home, &cwd, p) {
            Ok(v) => v,
            Err(e) => return err(&e),
        },
        None => cwd,
    };
    if comp.files.contains_key(&target) && !is_dir(&comp.files, home, &target) {
        return err("not a directory");
    }
    if !is_dir(&comp.files, home, &target) {
        return err("no such path");
    }
    let mut names = BTreeSet::new();
    let prefix = format!("{target}/");
    for key in comp.files.keys() {
        if let Some(rest) = key.strip_prefix(&prefix) {
            if let Some(name) = rest.split('/').next() {
                if !name.is_empty() {
                    let child = format!("{target}/{name}");
                    let mark = if is_dir(&comp.files, home, &child) && !comp.files.contains_key(&child) {
                        format!("{name}/")
                    } else {
                        name.to_string()
                    };
                    names.insert(mark);
                }
            }
        }
    }
    if names.is_empty() {
        "(empty)".into()
    } else {
        names.into_iter().collect::<Vec<_>>().join("\n")
    }
}

fn find(comp: &Computer, home: &str, path: Option<&str>) -> String {
    let cwd = comp.cwd.clone();
    let target = match path {
        Some(p) => match resolve(home, &cwd, p) {
            Ok(v) => v,
            Err(e) => return err(&e),
        },
        None => cwd,
    };
    if !is_dir(&comp.files, home, &target) && !comp.files.contains_key(&target) {
        return err("no such path");
    }
    let mut hits: Vec<String> = comp
        .files
        .keys()
        .filter(|k| *k == &target || k.starts_with(&format!("{target}/")))
        .cloned()
        .collect();
    hits.sort();
    if hits.is_empty() {
        "(empty)".into()
    } else {
        hits.join("\n")
    }
}

fn file_cmd(comp: &Computer, home: &str, cmd: &str, path: Option<&str>) -> String {
    let Some(path) = path else {
        return err(&format!("{cmd} needs a path"));
    };
    let path = match resolve(home, &comp.cwd, path) {
        Ok(p) => p,
        Err(e) => return err(&e),
    };
    let Some(body) = comp.files.get(&path) else {
        return err(&format!("no such file: {path}"));
    };
    match cmd {
        "cat" => clip(body, 4000),
        "head" => body.lines().take(15).collect::<Vec<_>>().join("\n"),
        "wc" => {
            let lines = body.lines().count();
            let words = body.split_whitespace().count();
            let bytes = body.len();
            format!("{lines} {words} {bytes} {path}")
        }
        _ => err("unknown"),
    }
}

pub fn is_dir(files: &BTreeMap<String, String>, home: &str, dir: &str) -> bool {
    if dir == home.trim_end_matches('/') {
        return true;
    }
    let prefix = format!("{dir}/");
    files.keys().any(|k| k.starts_with(&prefix))
}

pub fn file_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

fn tokenize(s: &str) -> Result<Vec<String>, ()> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    for c in s.chars() {
        match c {
            '"' => in_q = !in_q,
            ' ' | '\t' if !in_q => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if in_q {
        return Err(());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    Ok(out)
}

/// Today's date for prompts, like "Wednesday, 2026-10-28" (UTC).
pub fn today() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    date_line(secs)
}

pub fn date_line(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let (y, mo, d) = civil_from_days(days);
    const NAMES: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    format!("{}, {y:04}-{mo:02}-{d:02}", NAMES[(days + 4).rem_euclid(7) as usize])
}

fn utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    civil_utc(secs)
}

fn civil_utc(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let h = tod / 3600;
    let m = (tod % 3600) / 60;
    let s = tod % 60;
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{m:02}:{s:02} UTC")
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo as i64, d as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Computer;

    fn desk() -> (Computer, String) {
        let mut c = Computer::empty("/home/ada");
        c.files
            .insert("/home/ada/briefing.md".into(), "hello\nworld\n".into());
        c.files
            .insert("/home/ada/notes/a.txt".into(), "nested".into());
        (c, "/home/ada".into())
    }

    #[test]
    fn ls_and_cat() {
        let (mut c, home) = desk();
        let out = exec(&mut c, &home, "ls");
        assert!(out.contains("briefing.md"), "{out}");
        assert!(out.contains("notes/"), "{out}");
        let out = exec(&mut c, &home, "cat briefing.md");
        assert!(out.contains("hello"), "{out}");
    }

    #[test]
    fn rejects_escape_and_pipes() {
        let (mut c, home) = desk();
        let out = exec(&mut c, &home, "cat ../../etc/passwd");
        assert!(out.contains("outside") || out.contains("leaves"), "{out}");
        let out = exec(&mut c, &home, "ls | cat");
        assert!(out.contains("pipes"), "{out}");
    }

    #[test]
    fn write_stays_home() {
        let (mut c, home) = desk();
        let path = write_file(&mut c, &home, "notes/b.txt", "desk").unwrap();
        assert_eq!(path, "/home/ada/notes/b.txt");
        assert!(write_file(&mut c, &home, "/etc/passwd", "no").is_err());
    }

    #[test]
    fn epoch_is_1970() {
        assert_eq!(civil_utc(0), "1970-01-01 00:00:00 UTC");
    }
}


/// Command-line tools found on the PATH that are useful to a coding agent, e.g. `git, gh, cargo`.
/// Looked up once; names only, nothing is run.
pub fn dev_tools() -> &'static str {
    static FOUND: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    FOUND.get_or_init(|| {
        const WANTED: &[&str] = &[
            "git", "gh", "cargo", "rustc", "node", "npm", "pnpm", "yarn", "bun", "python3", "python", "pip", "go", "make", "cmake", "docker",
            "kubectl", "vercel", "netlify", "railway", "flyctl", "aws", "gcloud", "az", "rg", "jq", "curl",
        ];
        let path = std::env::var_os("PATH").unwrap_or_default();
        let dirs: Vec<std::path::PathBuf> = std::env::split_paths(&path).collect();
        let exts: &[&str] = if cfg!(windows) { &["", ".exe", ".cmd", ".bat"] } else { &[""] };
        WANTED
            .iter()
            .filter(|t| dirs.iter().any(|d| exts.iter().any(|e| d.join(format!("{t}{e}")).is_file())))
            .copied()
            .collect::<Vec<_>>()
            .join(", ")
    })
}
