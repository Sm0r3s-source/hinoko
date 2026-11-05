//! A small look at a project's git state, for the Code tab. Read-only: it runs
//! `git status` and `git diff` and never changes the repository.

use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GitStatus {
    /// Branch name, or a short commit id when HEAD is detached.
    pub branch: String,
    /// Changed paths with git's two-letter state, e.g. (" M", "src/main.rs").
    pub changed: Vec<(String, String)>,
}

fn git(dir: &str, args: &[&str]) -> Option<String> {
    let mut c = Command::new("git");
    c.args(args).current_dir(dir).stdin(Stdio::null()).stderr(Stdio::null()).env("GIT_OPTIONAL_LOCKS", "0");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    let out = c.output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).to_string())
}

/// `None` when the folder is not inside a git repository (or git is not installed).
pub fn status(dir: &str) -> Option<GitStatus> {
    if !Path::new(dir).is_dir() {
        return None;
    }
    parse_status(&git(dir, &["status", "--porcelain=v1", "--branch", "--untracked-files=normal"])?)
}

pub fn parse_status(text: &str) -> Option<GitStatus> {
    let mut st = GitStatus::default();
    for line in text.lines() {
        if let Some(b) = line.strip_prefix("## ") {
            let b = b.split("...").next().unwrap_or(b);
            st.branch = match b.strip_prefix("No commits yet on ") {
                Some(name) => name.to_string(),
                None if b.starts_with("HEAD (no branch)") => "detached HEAD".into(),
                None => b.to_string(),
            };
        } else if line.len() > 3 {
            st.changed.push((line[..2].to_string(), line[3..].trim_matches('"').to_string()));
        }
    }
    (!st.branch.is_empty()).then_some(st)
}

/// The working-tree diff against HEAD (tracked files), trimmed to `max` bytes.
pub fn diff(dir: &str, max: usize) -> String {
    let d = git(dir, &["diff", "HEAD", "--no-color", "--stat", "-p"]).or_else(|| git(dir, &["diff", "--no-color", "--stat", "-p"])).unwrap_or_default();
    if d.len() <= max {
        return d;
    }
    let mut cut = max;
    while !d.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}\n… (diff cut at {} bytes)", &d[..cut], max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_branch_and_changes() {
        let s = parse_status("## main...origin/main [ahead 1]\n M src/a.rs\n?? new.txt\nA  b.rs\n").unwrap();
        assert_eq!(s.branch, "main");
        assert_eq!(s.changed.len(), 3);
        assert_eq!(s.changed[0], (" M".to_string(), "src/a.rs".to_string()));
        assert_eq!(s.changed[1].0, "??");
    }

    #[test]
    fn odd_branch_lines() {
        assert_eq!(parse_status("## No commits yet on trunk\n").unwrap().branch, "trunk");
        assert_eq!(parse_status("## HEAD (no branch)\n").unwrap().branch, "detached HEAD");
        assert!(parse_status("").is_none());
    }

    #[test]
    fn works_on_a_real_repo() {
        if Command::new("git").arg("--version").output().is_err() {
            return;
        }
        let d = std::env::temp_dir().join(format!("hinoko-git-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let ds = d.to_string_lossy().to_string();
        assert!(status(&ds).is_none() || status(&ds).is_some()); // may sit inside another repo
        let run = |a: &[&str]| Command::new("git").args(a).current_dir(&d).output().unwrap();
        run(&["init", "-q"]);
        std::fs::write(d.join("a.txt"), "one\n").unwrap();
        let s = status(&ds).unwrap();
        assert_eq!(s.changed, vec![("??".to_string(), "a.txt".to_string())]);
        let _ = std::fs::remove_dir_all(&d);
    }
}
