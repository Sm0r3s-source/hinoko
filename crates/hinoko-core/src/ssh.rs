//! Reaching other computers over SSH, using the system `ssh` client.
//! Hinoko never stores passwords: access is by key (Hinoko's own, or one you point to).

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

use crate::model::clip;
use crate::settings::Machine;

/// Only plain, side-effect-free inspection commands may run without asking.
const READ_ONLY: &[&str] = &[
    "ls", "df", "du", "free", "uptime", "uname", "whoami", "hostname", "pwd", "date", "ps", "id", "lsblk", "lscpu", "lspci", "lsusb",
    "nvidia-smi", "sensors",
];

/// True for a single read-only command with no shell tricks.
pub fn allowed_readonly(cmd: &str) -> bool {
    let cmd = cmd.trim();
    if cmd.is_empty() || cmd.len() > 200 {
        return false;
    }
    // anything that chains, redirects, substitutes or expands is out
    if cmd.chars().any(|c| matches!(c, ';' | '&' | '|' | '>' | '<' | '`' | '$' | '(' | ')' | '{' | '}' | '\n' | '\r' | '\\' | '*' | '?' | '~' | '!' | '"' | '\'')) {
        return false;
    }
    let mut words = cmd.split_whitespace();
    let first = words.next().unwrap_or("");
    if !READ_ONLY.contains(&first) {
        return false;
    }
    // du/ls/ps are inspection only; ip is excluded entirely
    true
}

fn safe_name(s: &str, extra: &[char]) -> bool {
    !s.is_empty() && !s.starts_with('-') && s.len() <= 253 && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' || extra.contains(&c))
}

/// Host and user go into an argument list; refuse anything that could be read as an option.
pub fn validate(m: &Machine) -> Result<(), String> {
    if !safe_name(&m.host, &[':']) {
        return Err("host may only contain letters, digits, . - _ and :".into());
    }
    if !safe_name(&m.user, &[]) {
        return Err("user may only contain letters, digits, . - _".into());
    }
    if m.port == 0 {
        return Err("port must be 1 to 65535".into());
    }
    Ok(())
}

pub fn args(m: &Machine, key: &Path, known_hosts: &Path, command: &str) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "-o".into(),
        "BatchMode=yes".into(),
        "-o".into(),
        "ConnectTimeout=10".into(),
        "-o".into(),
        "StrictHostKeyChecking=accept-new".into(),
        "-o".into(),
        format!("UserKnownHostsFile={}", known_hosts.display()),
    ];
    if key.exists() {
        a.extend(["-o".into(), "IdentitiesOnly=yes".into(), "-i".into(), key.display().to_string()]);
    }
    a.extend(["-p".into(), m.port.to_string(), format!("{}@{}", m.user, m.host), "--".into(), command.to_string()]);
    a
}

/// Run one command. Output is capped; the result says how it ended.
pub async fn run(m: &Machine, key: &Path, known_hosts: &Path, command: &str, timeout_secs: u64) -> Result<String, String> {
    validate(m)?;
    if command.trim().is_empty() || command.len() > 500 || command.contains('\0') {
        return Err("that command is empty or too long".into());
    }
    if let Some(dir) = known_hosts.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let child = Command::new("ssh")
        .args(args(m, key, known_hosts, command))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("could not start ssh ({e}). Install the OpenSSH client."))?;
    let out = tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait_with_output())
        .await
        .map_err(|_| format!("the command did not finish within {timeout_secs}s"))?
        .map_err(|e| format!("ssh failed: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let code = out.status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into());
    let mut text = format!("exit {code}");
    if !stdout.trim().is_empty() {
        text.push_str(&format!("\n{}", clip(stdout.trim(), 4000)));
    }
    if !stderr.trim().is_empty() {
        text.push_str(&format!("\nstderr: {}", clip(stderr.trim(), 1500)));
    }
    Ok(text)
}

/// Make Hinoko's own key if it is missing; return the public half.
pub async fn ensure_key(key: &Path) -> Result<String, String> {
    let pubpath = key.with_extension("pub");
    if !key.exists() {
        if let Some(dir) = key.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let out = Command::new("ssh-keygen")
            .args(["-t", "ed25519", "-N", "", "-C", "hinoko", "-f"])
            .arg(key)
            .stdin(Stdio::null())
            .output()
            .await
            .map_err(|e| format!("could not run ssh-keygen ({e}). Install the OpenSSH client."))?;
        if !out.status.success() {
            return Err(format!("ssh-keygen failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
    }
    std::fs::read_to_string(&pubpath).map(|s| s.trim().to_string()).map_err(|e| format!("cannot read the public key: {e}"))
}

/// One line to paste on the other computer, once, to let Hinoko in.
pub fn authorize_command(public_key: &str) -> String {
    format!("mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '{public_key}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(host: &str, user: &str) -> Machine {
        Machine { id: "pc".into(), name: "PC".into(), host: host.into(), port: 22, user: user.into(), notes: String::new(), auto_readonly: false }
    }

    #[test]
    fn readonly_allowlist_is_strict() {
        for ok in ["ls -la /home", "df -h", "uptime", "nvidia-smi", "ps aux", "free -m"] {
            assert!(allowed_readonly(ok), "{ok}");
        }
        for bad in [
            "rm -rf /", "ls; rm x", "ls && reboot", "ls | sh", "cat /etc/passwd", "ls $(whoami)", "ls `id`", "ls > f", "df\nreboot",
            "echo hi", "ls *", "sudo ls", "ls ~", "ls \"a b\"", "", "uptime & ", "ssh other", "ip a", "find / -delete",
        ] {
            assert!(!allowed_readonly(bad), "{bad:?}");
        }
    }

    #[test]
    fn host_and_user_cannot_smuggle_options() {
        assert!(validate(&m("192.168.1.20", "sakor")).is_ok());
        assert!(validate(&m("my-pc.local", "a.b_c")).is_ok());
        assert!(validate(&m("fe80::1", "u")).is_ok());
        assert!(validate(&m("-oProxyCommand=evil", "u")).is_err());
        assert!(validate(&m("host", "-oProxyCommand=evil")).is_err());
        assert!(validate(&m("host name", "u")).is_err());
        assert!(validate(&m("host;rm", "u")).is_err());
        assert!(validate(&m("", "u")).is_err());
    }

    #[test]
    fn argument_list_ends_options_before_the_command() {
        let a = args(&m("h", "u"), Path::new("/nonexistent/key"), Path::new("/tmp/kh"), "ls -la");
        let dd = a.iter().position(|x| x == "--").unwrap();
        assert_eq!(a[dd - 1], "u@h");
        assert_eq!(a[dd + 1], "ls -la");
        assert!(a.contains(&"BatchMode=yes".to_string()));
        assert!(!a.contains(&"-i".to_string()), "no key file, so no -i");
    }

    #[test]
    fn authorize_line_is_a_single_command() {
        let l = authorize_command("ssh-ed25519 AAAA hinoko");
        assert!(l.contains("authorized_keys") && !l.contains('\n'));
    }
}
