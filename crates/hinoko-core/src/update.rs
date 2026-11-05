//! Self-update from GitHub releases: check for a newer version, download its installer, run it.
use std::sync::{Arc, Mutex};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::Engine;

/// Where releases are published. One place to change if the app is renamed.
pub const REPO: &str = "Sm0r3s-source/hinoko";

#[derive(Clone, Debug)]
pub struct Release {
    pub tag: String,
    pub page: String,
    /// The Windows installer asset ("" when the release has none).
    pub url: String,
    pub sha256: Option<String>,
    pub notes: String,
}

#[derive(Clone, Debug, Default)]
pub enum Update {
    #[default]
    Idle,
    Checking,
    UpToDate,
    Available(Release),
    Downloading,
    Ready(std::path::PathBuf),
    Failed(String),
}

/// Whether release `tag` (like "v0.14.0") is newer than `current` ("0.13.0").
pub fn newer(tag: &str, current: &str) -> bool {
    let parts = |s: &str| {
        let mut v: Vec<u64> = s.trim_start_matches('v').split('.').map(|p| p.split(|c: char| !c.is_ascii_digit()).next().unwrap_or("").parse().unwrap_or(0)).collect();
        v.resize(3, 0);
        v
    };
    parts(tag) > parts(current)
}

async fn latest(http: &reqwest::Client) -> Result<Option<Release>, String> {
    let r = http.get(format!("https://api.github.com/repos/{REPO}/releases/latest")).header("User-Agent", "Hinoko-updater").header("Accept", "application/vnd.github+json").send().await.map_err(|_| "couldn't reach GitHub")?;
    let status = r.status();
    if !status.is_success() {
        return Err(if status.as_u16() == 404 { "no release has been published yet".into() } else { format!("GitHub answered {status}") });
    }
    let v: Value = r.json().await.map_err(|_| "GitHub's answer was unreadable")?;
    let tag = v["tag_name"].as_str().unwrap_or_default();
    if !newer(tag, env!("CARGO_PKG_VERSION")) {
        return Ok(None);
    }
    let asset = v["assets"].as_array().and_then(|a| a.iter().find(|x| x["name"].as_str().is_some_and(|n| n.ends_with("Setup.exe"))));
    let text = |x: Option<&Value>, k: &str| x.and_then(|x| x[k].as_str()).unwrap_or_default().to_string();
    Ok(Some(Release {
        tag: tag.to_string(),
        page: v["html_url"].as_str().unwrap_or_default().to_string(),
        url: text(asset, "browser_download_url"),
        sha256: asset.and_then(|a| a["digest"].as_str()).and_then(|d| d.strip_prefix("sha256:")).map(str::to_lowercase),
        notes: v["body"].as_str().unwrap_or_default().chars().take(600).collect(),
    }))
}

async fn download(http: &reqwest::Client, rel: &Release) -> Result<std::path::PathBuf, String> {
    let mut r = http.get(&rel.url).header("User-Agent", "Hinoko-updater").send().await.map_err(|_| "the download failed")?.error_for_status().map_err(|_| "the download failed")?;
    let path = std::env::temp_dir().join(format!("Hinoko-Setup-{}.exe", rel.tag.trim_start_matches('v')));
    let (mut file, mut hash) = (std::fs::File::create(&path).map_err(|e| e.to_string())?, Sha256::new());
    while let Some(chunk) = r.chunk().await.map_err(|_| "the download broke off")? {
        hash.update(&chunk);
        std::io::Write::write_all(&mut file, &chunk).map_err(|e| e.to_string())?;
    }
    let got: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if rel.sha256.as_ref().is_some_and(|want| *want != got) {
        let _ = std::fs::remove_file(&path);
        return Err("the download didn't match its checksum, so it was thrown away".into());
    }
    Ok(path)
}

/// Run the installer quietly (it closes Hinoko, replaces it and starts it again). The caller then exits.
pub fn run_installer(path: &std::path::Path) -> Result<(), String> {
    if !cfg!(windows) {
        return Err("automatic install is for the Windows installer".into());
    }
    std::process::Command::new(path).args(["/VERYSILENT", "/SUPPRESSMSGBOXES", "/CLOSEAPPLICATIONS"]).spawn().map(|_| ()).map_err(|e| e.to_string())
}

impl Engine {
    pub fn update_check(&self, st: Arc<Mutex<Update>>) {
        *st.lock().unwrap_or_else(|e| e.into_inner()) = Update::Checking;
        let (http, repaint) = (self.http.clone(), self.repaint.clone());
        self.rt.spawn(async move {
            let next = match latest(&http).await {
                Ok(Some(rel)) => Update::Available(rel),
                Ok(None) => Update::UpToDate,
                Err(e) => Update::Failed(e),
            };
            *st.lock().unwrap_or_else(|e| e.into_inner()) = next;
            repaint();
        });
    }

    pub fn update_download(&self, st: Arc<Mutex<Update>>, rel: Release) {
        *st.lock().unwrap_or_else(|e| e.into_inner()) = Update::Downloading;
        let (http, repaint) = (self.http.clone(), self.repaint.clone());
        self.rt.spawn(async move {
            let next = download(&http, &rel).await.map_or_else(Update::Failed, Update::Ready);
            *st.lock().unwrap_or_else(|e| e.into_inner()) = next;
            repaint();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::newer;

    #[test]
    fn versions_compare_by_number_not_by_text() {
        assert!(newer("v0.14.0", "0.13.0") && newer("v0.13.1", "0.13.0") && newer("v1.0.0", "0.13.9"));
        assert!(newer("v0.100.0", "0.13.0"), "100 is more than 13");
        assert!(!newer("v0.13.0", "0.13.0") && !newer("v0.12.9", "0.13.0"));
        assert!(!newer("v0.13", "0.13.0"), "a missing part counts as zero");
        assert!(!newer("", "0.13.0"));
    }
}
