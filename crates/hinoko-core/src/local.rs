//! Local models without installing anything yourself.
//!
//! * [`Hardware::detect`] looks at RAM, CPU and GPU.
//! * [`discover`] is the "cookbook": it asks Hugging Face for GGUF models (trending
//!   or a search), reads their file sizes and says which quantisation of each one
//!   fits this computer, and how well.
//! * The engine downloads a `llama-server` build and the model file (with
//!   progress), starts the server on a free local port, and adds it as a provider.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use serde_json::Value;
use tokio::io::AsyncWriteExt;

use crate::engine::Engine;
use crate::settings::{Kind, Provider};

// ---------------------------------------------------------------- hardware

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gpu {
    pub name: String,
    /// Dedicated video memory in GB; 0 when unknown or shared with the system.
    pub vram_gb: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Hardware {
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub threads: usize,
    pub ram_gb: f64,
    pub gpus: Vec<Gpu>,
    /// Apple Silicon: the GPU shares system memory.
    pub unified: bool,
}

fn run_quiet(cmd: &str, args: &[&str]) -> Option<String> {
    let mut c = std::process::Command::new(cmd);
    c.args(args).stdin(Stdio::null()).stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // no console window
    }
    let out = c.output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).to_string())
}

/// `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`
pub fn parse_nvidia(out: &str) -> Vec<Gpu> {
    out.lines()
        .filter_map(|l| {
            let (name, mb) = l.rsplit_once(',')?;
            let mb: f64 = mb.trim().parse().ok()?;
            Some(Gpu { name: name.trim().to_string(), vram_gb: mb / 1024.0 })
        })
        .collect()
}

impl Hardware {
    /// Blocking: runs a few small commands. Call from a worker thread.
    pub fn detect() -> Hardware {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.refresh_cpu_all();
        let cpu = sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
        let mut hw = Hardware {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu,
            threads: sys.cpus().len().max(1),
            ram_gb: sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
            gpus: Vec::new(),
            unified: cfg!(all(target_os = "macos", target_arch = "aarch64")),
        };
        if let Some(out) = run_quiet("nvidia-smi", &["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]) {
            hw.gpus = parse_nvidia(&out);
        }
        if hw.gpus.is_empty() && cfg!(windows) {
            if let Some(out) = run_quiet(
                "powershell",
                &["-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"],
            ) {
                // AdapterRAM is capped at 4 GB by Windows, so names only (memory unknown)
                hw.gpus = out.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).map(|n| Gpu { name: n.to_string(), vram_gb: 0.0 }).collect();
            }
        }
        if hw.gpus.is_empty() && cfg!(target_os = "linux") {
            if let Some(out) = run_quiet("lspci", &[]) {
                hw.gpus = out
                    .lines()
                    .filter(|l| l.contains("VGA compatible") || l.contains("3D controller"))
                    .filter_map(|l| l.split(": ").nth(1))
                    .map(|n| Gpu { name: n.trim().to_string(), vram_gb: 0.0 })
                    .collect();
            }
        }
        hw
    }

    /// GB of memory a model can use on the GPU (0 when there is no usable GPU memory figure).
    pub fn gpu_budget_gb(&self) -> f64 {
        if self.unified {
            return self.ram_gb * 0.66;
        }
        let best = self.gpus.iter().map(|g| g.vram_gb).fold(0.0, f64::max);
        if best <= 0.0 {
            0.0
        } else {
            (best * 0.92 - 0.6).max(0.0)
        }
    }

    pub fn ram_budget_gb(&self) -> f64 {
        self.ram_gb * 0.7
    }

    /// Any GPU worth using Vulkan/Metal on.
    pub fn has_gpu(&self) -> bool {
        self.unified || self.gpus.iter().any(|g| {
            let n = g.name.to_lowercase();
            g.vram_gb > 1.0 || n.contains("nvidia") || n.contains("geforce") || n.contains("radeon rx") || n.contains("arc ")
        })
    }

    pub fn summary(&self) -> String {
        let gpu = if self.unified {
            "Apple GPU (shares memory)".to_string()
        } else if self.gpus.is_empty() {
            "no GPU found".to_string()
        } else {
            self.gpus
                .iter()
                .map(|g| if g.vram_gb > 0.0 { format!("{} ({:.0} GB)", g.name, g.vram_gb) } else { g.name.clone() })
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!("{:.0} GB RAM · {} ({} threads) · {}", self.ram_gb, if self.cpu.is_empty() { "CPU" } else { &self.cpu }, self.threads, gpu)
    }
}

// ---------------------------------------------------------------- fit

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Fit {
    Fast,
    Good,
    Mixed,
    Slow,
    TooBig,
}

impl Fit {
    pub fn label(self) -> &'static str {
        match self {
            Fit::Fast => "Runs fast",
            Fit::Good => "Runs well",
            Fit::Mixed => "Runs, shared GPU/CPU",
            Fit::Slow => "Runs slowly",
            Fit::TooBig => "Too big",
        }
    }
}

/// How well a model file of `size_gb` runs here. `moe`: mixture-of-experts models
/// only use a fraction of their weights per token, so they stay quick on CPU.
pub fn fit(hw: &Hardware, size_gb: f64, moe: bool) -> Fit {
    let need = size_gb * 1.08 + 0.8; // weights + context/cache
    let (gpu, ram) = (hw.gpu_budget_gb(), hw.ram_budget_gb());
    if gpu > 0.0 && need <= gpu {
        Fit::Fast
    } else if gpu > 0.0 && need <= gpu + ram * 0.8 && !hw.unified {
        Fit::Mixed
    } else if need <= ram {
        if size_gb <= 5.5 || moe {
            Fit::Good
        } else {
            Fit::Slow
        }
    } else {
        Fit::TooBig
    }
}

// ---------------------------------------------------------------- catalogue

#[derive(Clone, Debug)]
pub struct GgufFile {
    pub name: String,
    pub size: u64,
    pub quant: String,
}

impl GgufFile {
    pub fn gb(&self) -> f64 {
        self.size as f64 / 1e9
    }
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub repo: String,
    pub params_b: Option<f64>,
    pub downloads: u64,
    pub likes: u64,
    pub moe: bool,
    pub vision: bool,
    /// The quantisation recommended for this computer.
    pub best: GgufFile,
    pub fit: Fit,
    pub others: usize,
}

/// `Q4_K_M`, `UD-Q4_K_XL`, `BF16`… from a GGUF file name.
pub fn quant_of(file: &str) -> String {
    let stem = file.trim_end_matches(".gguf").trim_end_matches(".GGUF");
    let toks: Vec<&str> = stem.split(['-', '.']).collect();
    for (i, t) in toks.iter().enumerate() {
        let u = t.to_uppercase();
        let after = u.strip_prefix("IQ").or_else(|| u.strip_prefix('Q')).unwrap_or("");
        let is_q = after.chars().next().is_some_and(|c| c.is_ascii_digit()) && u.len() <= 9;
        if is_q || matches!(u.as_str(), "BF16" | "F16" | "F32" | "MXFP4") {
            return if i > 0 && toks[i - 1].eq_ignore_ascii_case("UD") { format!("UD-{u}") } else { u };
        }
    }
    "GGUF".into()
}

fn quality_rank(quant: &str) -> u8 {
    let q = quant.trim_start_matches("UD-");
    if q.starts_with("IQ1") || q.starts_with("Q1") {
        0
    } else if q.starts_with("IQ2") || q.starts_with("Q2") {
        1
    } else if q.starts_with("IQ3") || q.starts_with("Q3") {
        2
    } else {
        3
    }
}

/// Pick the best file of one repo for this computer: the largest that fits
/// comfortably, skipping very low-quality quantisations unless nothing else fits.
pub fn pick_file(hw: &Hardware, files: &[GgufFile], moe: bool) -> Option<(GgufFile, Fit)> {
    let mut usable: Vec<(&GgufFile, Fit)> = files
        .iter()
        .filter(|f| !f.quant.starts_with("F32"))
        .map(|f| (f, fit(hw, f.gb(), moe)))
        .filter(|(_, ft)| *ft != Fit::TooBig)
        .collect();
    if usable.is_empty() {
        let smallest = files.iter().min_by_key(|f| f.size)?;
        return Some((smallest.clone(), Fit::TooBig));
    }
    let good: Vec<_> = usable.iter().filter(|(f, _)| quality_rank(&f.quant) >= 2).cloned().collect();
    if !good.is_empty() {
        usable = good;
    }
    // prefer the best fit class, then the largest (= highest quality) file
    let best_fit = usable.iter().map(|(_, ft)| *ft).min()?;
    // a bigger file is only worth it while it keeps the same fit class
    usable.retain(|(_, ft)| *ft == best_fit);
    if best_fit != Fit::Fast {
        // off the GPU, speed follows file size, so a 4-bit file is the sweet spot
        let q4: Vec<_> = usable.iter().filter(|(f, _)| f.quant.trim_start_matches("UD-").contains("Q4")).cloned().collect();
        if !q4.is_empty() {
            usable = q4;
        }
    }
    let (f, ft) = usable.into_iter().max_by_key(|(f, _)| f.size)?;
    Some((f.clone(), ft))
}

fn files_from_tree(tree: &Value) -> Vec<GgufFile> {
    let mut out = Vec::new();
    for e in tree.as_array().cloned().unwrap_or_default() {
        if e["type"] != "file" {
            continue;
        }
        let path = e["path"].as_str().unwrap_or("");
        let lower = path.to_lowercase();
        if !lower.ends_with(".gguf") || lower.contains("mmproj") || lower.contains("imatrix") || lower.contains("-of-0") {
            continue;
        }
        let size = e["lfs"]["size"].as_u64().or_else(|| e["size"].as_u64()).unwrap_or(0);
        if size == 0 {
            continue;
        }
        out.push(GgufFile { name: path.to_string(), size, quant: quant_of(path) });
    }
    out
}

/// Drop files too small to be the whole model: a real quantisation needs at least
/// ~0.17 GB per billion parameters, so a 2 GB file in a 177B repo is a side file
/// (draft/MTP head, one shard), not something that would run.
pub fn plausible_files(files: Vec<GgufFile>, params_b: Option<f64>) -> Vec<GgufFile> {
    files
        .into_iter()
        .filter(|f| {
            let n = f.name.to_lowercase();
            !(n.contains("mtp") || n.contains("draft") || n.contains("eagle"))
                && params_b.map_or(true, |p| p < 0.5 || f.gb() >= p * 0.17)
        })
        .collect()
}

/// Ask Hugging Face for GGUF models (`query` empty = trending) and rank them for this PC.
pub async fn discover(http: &reqwest::Client, hw: &Hardware, query: &str) -> Result<Vec<Candidate>, String> {
    let mut req = http.get("https://huggingface.co/api/models").query(&[
        ("filter", "gguf"),
        ("sort", if query.trim().is_empty() { "trendingScore" } else { "downloads" }),
        ("limit", "40"),
    ]);
    for field in ["downloads", "likes", "pipeline_tag", "gated", "gguf", "tags"] {
        req = req.query(&[("expand[]", field)]);
    }
    if !query.trim().is_empty() {
        req = req.query(&[("search", query.trim())]);
    }
    let list: Value = req
        .timeout(Duration::from_secs(25))
        .send()
        .await
        .map_err(|_| "couldn't reach Hugging Face (check your connection)".to_string())?
        .json()
        .await
        .map_err(|_| "Hugging Face sent something unreadable".to_string())?;
    let mut metas = Vec::new();
    for m in list.as_array().cloned().unwrap_or_default() {
        let id = m["id"].as_str().unwrap_or("").to_string();
        let gated = m["gated"].as_bool().unwrap_or(false) || m["gated"].is_string();
        let task = m["pipeline_tag"].as_str().unwrap_or("");
        let textual = matches!(task, "" | "text-generation" | "image-text-to-text" | "any-to-any" | "conversational");
        if id.is_empty() || gated || !textual {
            continue;
        }
        metas.push(m);
        if metas.len() >= 24 {
            break;
        }
    }
    let mut set = tokio::task::JoinSet::new();
    for (i, m) in metas.iter().enumerate() {
        let id = m["id"].as_str().unwrap_or("").to_string();
        let http = http.clone();
        set.spawn(async move {
            let tree = match http.get(format!("https://huggingface.co/api/models/{id}/tree/main")).timeout(Duration::from_secs(25)).send().await {
                Ok(r) => r.json::<Value>().await.unwrap_or(Value::Null),
                Err(_) => Value::Null,
            };
            (i, tree)
        });
    }
    let mut trees: HashMap<usize, Value> = HashMap::new();
    while let Some(r) = set.join_next().await {
        if let Ok((i, t)) = r {
            trees.insert(i, t);
        }
    }
    let mut out = Vec::new();
    for (i, m) in metas.iter().enumerate() {
        let id = m["id"].as_str().unwrap_or("").to_string();
        let params_b = m["gguf"]["total"].as_f64().map(|t| t / 1e9);
        let files = plausible_files(files_from_tree(trees.get(&i).unwrap_or(&Value::Null)), params_b);
        if files.is_empty() {
            continue;
        }
        let lname = id.to_lowercase();
        let tags: Vec<String> = m["tags"].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(|s| s.to_lowercase())).collect()).unwrap_or_default();
        let moe = tags.iter().any(|t| t == "moe" || t.contains("mixture")) || has_active_params(&lname);
        let vision = m["pipeline_tag"].as_str() == Some("image-text-to-text");
        let Some((best, fit)) = pick_file(hw, &files, moe) else { continue };
        out.push(Candidate {
            repo: id,
            params_b,
            downloads: m["downloads"].as_u64().unwrap_or(0),
            likes: m["likes"].as_u64().unwrap_or(0),
            moe,
            vision,
            best,
            fit,
            others: files.len().saturating_sub(1),
        });
    }
    // fits first (stable: keeps Hugging Face's own order within a class)
    out.sort_by_key(|c| c.fit);
    Ok(out)
}

/// "-A3B" / "-A4B": mixture-of-experts naming ("30B-A3B").
fn has_active_params(lower_name: &str) -> bool {
    let b = lower_name.as_bytes();
    (0..b.len().saturating_sub(3)).any(|i| b[i] == b'a' && (i == 0 || matches!(b[i - 1], b'-' | b'_' | b'.')) && b[i + 1].is_ascii_digit() && lower_name[i + 1..].trim_start_matches(|c: char| c.is_ascii_digit() || c == '.').starts_with('b'))
}

// ---------------------------------------------------------------- runner choice

/// Which llama.cpp release file suits this computer. Matches on the words in the
/// file name (os, cpu type, "vulkan") instead of exact names, so a renamed build
/// still works. `assets` are the names in the release.
pub fn pick_asset(hw: &Hardware, assets: &[String]) -> Option<String> {
    let arm = hw.arch == "aarch64";
    let has = |n: &str, words: &[&str]| words.iter().any(|w| n.contains(w));
    let mut best: Option<(u8, &String)> = None;
    for a in assets {
        let n = a.to_lowercase();
        if !(n.ends_with(".zip") || n.ends_with(".tar.gz")) || n.starts_with("cudart") {
            continue;
        }
        if has(&n, &["cuda", "sycl", "hip", "rocm", "openvino", "opencl", "s390x", "ppc64", "xcframework", "android", "musa", "cann", "kleidi"]) {
            continue;
        }
        let os_ok = match hw.os.as_str() {
            "windows" => has(&n, &["win"]),
            "linux" => has(&n, &["ubuntu", "linux"]),
            "macos" => has(&n, &["macos", "darwin", "osx"]),
            _ => false,
        };
        let arch_ok = if arm { has(&n, &["arm64", "aarch64"]) } else { has(&n, &["x64", "x86_64", "x86-64", "amd64"]) };
        if !os_ok || !arch_ok {
            continue;
        }
        let vulkan = n.contains("vulkan");
        let score = if hw.os == "macos" {
            1
        } else if hw.has_gpu() == vulkan {
            2 // the GPU build for a PC with a GPU, the plain build for one without
        } else {
            1
        };
        if best.map_or(true, |(s, _)| score > s) {
            best = Some((score, a));
        }
    }
    best.map(|(_, a)| a.clone())
}

/// Likely file names of the llama.cpp build for `tag`, best first.
pub fn runner_names(hw: &Hardware, tag: &str) -> Vec<String> {
    let arch = if hw.arch == "aarch64" { "arm64" } else { "x64" };
    let gpu = hw.has_gpu();
    match hw.os.as_str() {
        "windows" => {
            let mut v = vec![];
            if gpu && arch == "x64" {
                v.push(format!("llama-{tag}-bin-win-vulkan-x64.zip"));
            }
            v.push(format!("llama-{tag}-bin-win-cpu-{arch}.zip"));
            v.push(format!("llama-{tag}-bin-win-avx2-{arch}.zip"));
            v
        }
        "linux" => {
            let mut v = vec![];
            if gpu && arch == "x64" {
                v.push(format!("llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz"));
            }
            v.push(format!("llama-{tag}-bin-ubuntu-{arch}.tar.gz"));
            v
        }
        _ => vec![format!("llama-{tag}-bin-macos-{arch}.tar.gz"), format!("llama-{tag}-bin-macos-{arch}.zip")],
    }
}

/// Try the release download addresses for `tag` directly (no API needed).
async fn direct_runner_url(http: &reqwest::Client, hw: &Hardware, tag: &str) -> Option<String> {
    for name in runner_names(hw, tag) {
        let url = format!("https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}");
        if let Ok(r) = http.head(&url).timeout(Duration::from_secs(20)).send().await {
            if r.status().is_success() {
                return Some(url);
            }
        }
    }
    None
}

// ---------------------------------------------------------------- state

#[derive(Clone, Debug, PartialEq)]
pub enum Phase {
    Running,
    Done,
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct Progress {
    pub label: String,
    pub done: u64,
    pub total: u64,
    pub phase: Phase,
}

#[derive(Clone, Debug, Default)]
pub struct ServerInfo {
    pub running: bool,
    pub starting: bool,
    pub model: String,
    pub port: u16,
    pub error: String,
}

#[derive(Default)]
pub struct LocalState {
    pub progress: Mutex<HashMap<String, Progress>>,
    pub info: Mutex<ServerInfo>,
    child: tokio::sync::Mutex<Option<tokio::process::Child>>,
    /// The built-in engine's server task, when that is what is running.
    embedded: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

pub const PROVIDER_ID: &str = "local";

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").and_then(|l| l.local_addr()).map(|a| a.port()).unwrap_or(8187)
}

impl Engine {
    pub fn local_dir(&self) -> PathBuf {
        self.paths.home.join("local")
    }
    pub fn models_dir(&self) -> PathBuf {
        self.local_dir().join("models")
    }
    fn runner_dir(&self) -> PathBuf {
        self.local_dir().join("runner")
    }

    pub fn local_progress(&self, key: &str) -> Option<Progress> {
        lock(&self.local.progress).get(key).cloned()
    }
    pub fn local_info(&self) -> ServerInfo {
        lock(&self.local.info).clone()
    }

    fn set_progress(&self, key: &str, label: &str, done: u64, total: u64, phase: Phase) {
        lock(&self.local.progress).insert(key.to_string(), Progress { label: label.to_string(), done, total, phase });
        (self.repaint)();
    }

    /// The `llama-server` we installed (or that was dropped into the runner folder).
    pub fn runner_path(&self) -> Option<PathBuf> {
        let exe = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
        find_file(&self.runner_dir(), exe)
    }

    pub fn runner_version(&self) -> String {
        std::fs::read_to_string(self.runner_dir().join("VERSION")).unwrap_or_default().trim().to_string()
    }

    /// Models already on disk: (file name, path, bytes).
    pub fn downloaded_models(&self) -> Vec<(String, PathBuf, u64)> {
        let mut v: Vec<(String, PathBuf, u64)> = std::fs::read_dir(self.models_dir())
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                let name = p.file_name()?.to_string_lossy().to_string();
                if !name.ends_with(".gguf") {
                    return None;
                }
                Some((name, p, e.metadata().ok()?.len()))
            })
            .collect();
        v.sort();
        v
    }

    pub fn delete_model(&self, path: &Path) {
        if path.starts_with(self.models_dir()) {
            let _ = std::fs::remove_file(path);
        }
    }

    pub fn detect_hardware_async(&self, done: impl FnOnce(Hardware) + Send + 'static) {
        let repaint = self.repaint.clone();
        self.rt.spawn(async move {
            let hw = tokio::task::spawn_blocking(Hardware::detect).await.unwrap_or_default();
            done(hw);
            repaint();
        });
    }

    pub fn discover_async(&self, hw: Hardware, query: String, done: impl FnOnce(Result<Vec<Candidate>, String>) + Send + 'static) {
        let (http, repaint) = (self.big_client(), self.repaint.clone());
        self.rt.spawn(async move {
            done(discover(&http, &hw, &query).await);
            repaint();
        });
    }

    fn big_client(&self) -> reqwest::Client {
        reqwest::Client::builder().user_agent("Hinoko").connect_timeout(Duration::from_secs(15)).build().unwrap_or_else(|_| self.http.clone())
    }

    // ---- runner download ----

    /// Download and unpack the right `llama-server` build for this computer.
    pub async fn install_runner(&self, hw: &Hardware) -> Result<PathBuf, String> {
        if let Some(p) = self.runner_path() {
            return Ok(p);
        }
        let key = "runner";
        let fail = |e: String| {
            self.set_progress(key, "Runner", 0, 0, Phase::Failed(e.clone()));
            e
        };
        self.set_progress(key, "Looking up the latest llama.cpp", 0, 0, Phase::Running);
        let http = self.big_client();
        let releases: Value = http
            .get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30")
            .header("Accept", "application/vnd.github+json")
            .timeout(Duration::from_secs(25))
            .send()
            .await
            .map_err(|_| fail("couldn't reach GitHub to find the model runner".into()))?
            .json()
            .await
            .map_err(|_| fail("GitHub sent something unreadable (rate limit?). Try again in a minute".into()))?;
        // newest release that has a build for this computer
        let mut found: Option<(String, String)> = None;
        let mut seen: Vec<String> = Vec::new();
        for rel in releases.as_array().cloned().unwrap_or_default() {
            // nightly builds may be flagged pre-release, so those count too
            if rel["draft"].as_bool().unwrap_or(false) {
                continue;
            }
            let assets: Vec<(String, String)> = rel["assets"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| Some((x["name"].as_str()?.to_string(), x["browser_download_url"].as_str()?.to_string()))).collect())
                .unwrap_or_default();
            let names: Vec<String> = assets.iter().map(|a| a.0.clone()).collect();
            if seen.is_empty() {
                seen = names.iter().take(6).cloned().collect();
            }
            if let Some(name) = pick_asset(hw, &names) {
                let url = assets.iter().find(|a| a.0 == name).map(|a| a.1.clone()).unwrap_or_default();
                found = Some((rel["tag_name"].as_str().unwrap_or("").to_string(), url));
                break;
            }
            // Some releases only carry a `nightly-tag.txt` pointing at the real build.
            if let Some(tag_url) = assets.iter().find(|a| a.0 == "nightly-tag.txt").map(|a| a.1.clone()) {
                let tag = match http.get(&tag_url).timeout(Duration::from_secs(20)).send().await {
                    Ok(r) => r.text().await.unwrap_or_default().trim().to_string(),
                    Err(_) => String::new(),
                };
                if tag.is_empty() || tag.len() > 64 || tag.contains(char::is_whitespace) {
                    continue;
                }
                if let Some(url) = direct_runner_url(&http, hw, &tag).await {
                    found = Some((tag, url));
                    break;
                }
            }
        }
        // last resort: the newest tag we saw, built by name
        if found.is_none() {
            if let Some(tag) = releases.as_array().and_then(|a| a.iter().find_map(|r| r["tag_name"].as_str())).filter(|t| t.starts_with('b')) {
                if let Some(url) = direct_runner_url(&http, hw, tag).await {
                    found = Some((tag.to_string(), url));
                }
            }
        }
        let Some((tag, url)) = found else {
            return Err(fail(format!(
                "no llama.cpp build for {} {} found. Newest release offered: {}. You can put llama-server{} (with its files) into {} yourself",
                hw.os,
                hw.arch,
                if seen.is_empty() { "no files".to_string() } else { seen.join(", ") },
                if cfg!(windows) { ".exe" } else { "" },
                self.runner_dir().display()
            )));
        };
        let name = url.rsplit('/').next().unwrap_or("runner.zip").to_string();
        let dir = self.runner_dir();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| fail(e.to_string()))?;
        let archive = dir.join(&name);
        self.download(&http, &url, &archive, key, &format!("Downloading runner {tag}")).await.map_err(&fail)?;
        self.set_progress(key, "Unpacking runner", 0, 0, Phase::Running);
        let (a2, d2) = (archive.clone(), dir.clone());
        tokio::task::spawn_blocking(move || unpack(&a2, &d2)).await.map_err(|e| fail(e.to_string()))?.map_err(&fail)?;
        let _ = std::fs::remove_file(&archive);
        let _ = std::fs::write(dir.join("VERSION"), &tag);
        match self.runner_path() {
            Some(p) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755));
                }
                self.set_progress(key, "Runner ready", 1, 1, Phase::Done);
                Ok(p)
            }
            None => Err(fail("the runner download didn't contain llama-server".into())),
        }
    }

    // ---- model download ----

    /// Download one GGUF into the models folder (resumes a partial download).
    pub async fn download_model(&self, repo: &str, file: &str) -> Result<PathBuf, String> {
        let safe = file.replace(['/', '\\'], "_");
        let dest = self.models_dir().join(&safe);
        let key = format!("{repo}/{file}");
        if dest.exists() {
            return Ok(dest);
        }
        std::fs::create_dir_all(self.models_dir()).map_err(|e| e.to_string())?;
        let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
        let http = self.big_client();
        match self.download(&http, &url, &dest, &key, &format!("Downloading {safe}")).await {
            Ok(()) => {
                self.set_progress(&key, &safe, 1, 1, Phase::Done);
                Ok(dest)
            }
            Err(e) => {
                self.set_progress(&key, &safe, 0, 0, Phase::Failed(e.clone()));
                Err(e)
            }
        }
    }

    /// Stream `url` to `dest` through a `.part` file, reporting progress under `key`.
    async fn download(&self, http: &reqwest::Client, url: &str, dest: &Path, key: &str, label: &str) -> Result<(), String> {
        let part = PathBuf::from(format!("{}.part", dest.display()));
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let mut req = http.get(url);
        if have > 0 {
            req = req.header("Range", format!("bytes={have}-"));
        }
        let mut res = req.send().await.map_err(|e| if e.is_connect() { "couldn't connect (check your internet)".to_string() } else { format!("download failed: {e}") })?;
        let status = res.status();
        if status.as_u16() == 416 {
            // already complete
            std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
            return Ok(());
        }
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err("this model needs a Hugging Face login (gated) and can't be downloaded here".into());
        }
        if !status.is_success() {
            return Err(format!("download refused: HTTP {status}"));
        }
        let resumed = status.as_u16() == 206;
        let total = res.content_length().map(|l| l + if resumed { have } else { 0 }).unwrap_or(0);
        let mut done = if resumed { have } else { 0 };
        let mut f = tokio::fs::OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part).await.map_err(|e| e.to_string())?;
        let mut last = std::time::Instant::now();
        loop {
            let chunk = tokio::time::timeout(Duration::from_secs(60), res.chunk())
                .await
                .map_err(|_| "the download stalled (it will resume if you try again)".to_string())?
                .map_err(|_| "the connection dropped (it will resume if you try again)".to_string())?;
            let Some(bytes) = chunk else { break };
            f.write_all(&bytes).await.map_err(|e| format!("couldn't write the file: {e}"))?;
            done += bytes.len() as u64;
            if last.elapsed() > Duration::from_millis(150) {
                self.set_progress(key, label, done, total, Phase::Running);
                last = std::time::Instant::now();
            }
        }
        f.flush().await.map_err(|e| e.to_string())?;
        drop(f);
        if total > 0 && done < total {
            return Err("the download ended early (try again to resume)".into());
        }
        std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
        Ok(())
    }

    // ---- server ----

    /// Start `llama-server` for a model file and register it as the "Local model" provider.
    pub async fn local_start(&self, model: &Path, hw: &Hardware) -> Result<(), String> {
        self.local_stop().await;
        let name = model.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "local".into());
        {
            let mut i = lock(&self.local.info);
            *i = ServerInfo { running: false, starting: true, model: name.clone(), port: 0, error: String::new() };
        }
        (self.repaint)();
        let result = self.local_start_inner(model, hw, &name).await;
        {
            let mut i = lock(&self.local.info);
            i.starting = false;
            match &result {
                Ok(port) => {
                    i.running = true;
                    i.port = *port;
                }
                Err(e) => i.error = e.clone(),
            }
        }
        if result.is_ok() {
            let mut st = self.settings();
            st.local_last = model.to_string_lossy().to_string();
            self.set_settings(st);
        }
        (self.repaint)();
        result.map(|_| ())
    }

    /// Whether local models run on the built-in engine (true) or the llama.cpp runner.
    /// llama.cpp is the default everywhere because it is much faster, on a CPU too; the
    /// built-in engine (CPU only, small models) is used only when chosen in Settings, or as
    /// a fallback when the runner can't be downloaded (see `local_start_inner`).
    pub fn uses_builtin(&self, _hw: &Hardware) -> bool {
        cfg!(feature = "builtin") && self.settings().local_engine == "builtin"
    }

    /// The machine as the chosen engine will use it: the built-in engine runs on the CPU.
    pub fn engine_hw(&self, hw: &Hardware) -> Hardware {
        let mut h = hw.clone();
        if self.uses_builtin(hw) {
            h.gpus.clear();
            h.unified = false;
        }
        h
    }

    async fn local_start_inner(&self, model: &Path, hw: &Hardware, name: &str) -> Result<u16, String> {
        #[cfg(feature = "builtin")]
        if self.uses_builtin(hw) {
            let t0 = std::time::Instant::now();
            let run = crate::embedded::start(model, name).await?;
            let _ = std::fs::create_dir_all(self.local_dir());
            let line = format!("built-in engine loaded {name} in {:.1}s\n", t0.elapsed().as_secs_f32());
            let _ = std::fs::OpenOptions::new().create(true).append(true).open(self.local_dir().join("server.log")).and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
            *self.local.embedded.lock().await = Some(run.task);
            self.register_local(name, run.port);
            return Ok(run.port);
        }
        let exe = match self.runner_path() {
            Some(p) => p,
            None => match self.install_runner(hw).await {
                Ok(p) => p,
                // offline or blocked: on Automatic, a small model can still run on the built-in engine
                #[cfg(feature = "builtin")]
                Err(e) if self.settings().local_engine.is_empty() => {
                    let run = crate::embedded::start(model, name).await.map_err(|b| format!("{e}. The built-in engine couldn't start it either: {b}"))?;
                    *self.local.embedded.lock().await = Some(run.task);
                    self.register_local(name, run.port);
                    return Ok(run.port);
                }
                Err(e) => return Err(e),
            },
        };
        let port = free_port();
        let size_gb = std::fs::metadata(model).map(|m| m.len() as f64 / 1e9).unwrap_or(0.0);
        let ctx = if hw.ram_gb >= 24.0 { "16384" } else { "8192" };
        let mut cmd = tokio::process::Command::new(&exe);
        cmd.arg("-m").arg(model).args(["--host", "127.0.0.1", "--port", &port.to_string(), "-c", ctx, "--jinja"]);
        match fit(hw, size_gb, false) {
            Fit::Fast => {
                cmd.args(["-ngl", "99"]);
            }
            Fit::Mixed => {} // let llama.cpp split layers to what fits
            _ => {
                cmd.args(["-ngl", "0"]);
            }
        }
        let _ = std::fs::create_dir_all(self.local_dir());
        let log_path = self.local_dir().join("server.log");
        let log = std::fs::File::create(&log_path).map_err(|e| e.to_string())?;
        cmd.stdin(Stdio::null()).stdout(log.try_clone().map_err(|e| e.to_string())?).stderr(log).kill_on_drop(true);
        if let Some(dir) = exe.parent() {
            cmd.current_dir(dir);
        }
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000);
        let child = cmd.spawn().map_err(|e| format!("couldn't start the model runner: {e}"))?;
        *self.local.child.lock().await = Some(child);
        // wait until the model is loaded
        let http = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{port}/health");
        for _ in 0..360 {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if let Some(c) = self.local.child.lock().await.as_mut() {
                if let Ok(Some(status)) = c.try_wait() {
                    let tail = tail_of(&log_path);
                    return Err(format!("the model runner stopped ({status}). {tail}"));
                }
            }
            if let Ok(r) = http.get(&url).timeout(Duration::from_secs(2)).send().await {
                if r.status().is_success() {
                    self.register_local(name, port);
                    return Ok(port);
                }
            }
        }
        self.local_stop().await;
        Err("the model took too long to load".into())
    }

    fn register_local(&self, name: &str, port: u16) {
        let mut s = self.settings();
        let mut p = Provider::blank(Kind::OpenAi);
        p.id = PROVIDER_ID.into();
        p.name = format!("Local: {name}");
        p.base_url = format!("http://127.0.0.1:{port}/v1");
        p.model = name.to_string();
        p.timeout_secs = 300;
        s.upsert(p);
        s.active = PROVIDER_ID.into();
        self.set_settings(s);
    }

    /// Chats and bots that use the built-in local model start it on demand.
    pub async fn ensure_local(&self, provider: &Provider) -> Result<(), String> {
        if provider.id != PROVIDER_ID {
            return Ok(());
        }
        let info = self.local_info();
        if info.running && !info.starting {
            return Ok(());
        }
        let last = PathBuf::from(self.settings().local_last);
        if info.starting {
            for _ in 0..720 {
                tokio::time::sleep(Duration::from_millis(500)).await;
                if self.local_info().running {
                    return Ok(());
                }
            }
            return Err("the local model is still loading".into());
        }
        if !last.exists() {
            return Err("the local model isn't set up. Open Settings → Local models".into());
        }
        let hw = tokio::task::spawn_blocking(Hardware::detect).await.unwrap_or_default();
        self.local_start(&last, &hw).await
    }

    pub async fn local_stop(&self) {
        if let Some(mut c) = self.local.child.lock().await.take() {
            let _ = c.kill().await;
        }
        if let Some(t) = self.local.embedded.lock().await.take() {
            t.abort();
        }
        let mut i = lock(&self.local.info);
        i.running = false;
        i.starting = false;
        drop(i);
        (self.repaint)();
    }

    pub fn local_stop_async(&self) {
        let me = self.clone();
        self.rt.spawn(async move { me.local_stop().await });
    }

    /// Download (if needed) and start a model. The one button behind "Download & run".
    pub fn local_run_async(&self, repo: String, file: String, hw: Hardware) {
        let me = self.clone();
        self.rt.spawn(async move {
            let path = match me.download_model(&repo, &file).await {
                Ok(p) => p,
                Err(_) => return,
            };
            if let Err(e) = me.local_start(&path, &hw).await {
                me.set_progress("server", "Model runner", 0, 0, Phase::Failed(e));
            }
        });
    }

    /// Start a model that is already on disk.
    pub fn local_start_async(&self, path: PathBuf, hw: Hardware) {
        let me = self.clone();
        self.rt.spawn(async move {
            if let Err(e) = me.local_start(&path, &hw).await {
                me.set_progress("server", "Model runner", 0, 0, Phase::Failed(e));
            }
        });
    }

    /// Install the runner on its own, ahead of time.
    pub fn install_runner_async(&self, hw: Hardware) {
        let me = self.clone();
        self.rt.spawn(async move {
            let _ = me.install_runner(&hw).await;
        });
    }
}

fn tail_of(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let n = lines.len().saturating_sub(3);
    lines[n..].join(" | ").chars().take(300).collect()
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(f) = find_file(&p, name) {
                return Some(f);
            }
        } else if p.file_name().map(|n| n == name).unwrap_or(false) {
            return Some(p);
        }
    }
    None
}

/// Unpack a `.zip` or `.tar.gz` into `dir`, dropping a single top-level folder.
fn unpack(archive: &Path, dir: &Path) -> Result<(), String> {
    let name = archive.to_string_lossy().to_string();
    if name.ends_with(".zip") {
        let f = std::fs::File::open(archive).map_err(|e| e.to_string())?;
        let mut z = zip::ZipArchive::new(f).map_err(|e| format!("bad runner archive: {e}"))?;
        for i in 0..z.len() {
            let mut entry = z.by_index(i).map_err(|e| e.to_string())?;
            let Some(rel) = entry.enclosed_name() else { continue };
            let out = dir.join(rel);
            if entry.is_dir() {
                std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            } else {
                if let Some(parent) = out.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
                let mut w = std::fs::File::create(&out).map_err(|e| e.to_string())?;
                std::io::copy(&mut entry, &mut w).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    } else {
        let f = std::fs::File::open(archive).map_err(|e| e.to_string())?;
        let mut t = tar::Archive::new(flate2::read::GzDecoder::new(f));
        t.set_preserve_permissions(true);
        // `unpack` refuses entries that escape the folder
        t.unpack(dir).map_err(|e| format!("bad runner archive: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pc(ram: f64, vram: f64) -> Hardware {
        Hardware {
            os: "windows".into(),
            arch: "x86_64".into(),
            cpu: "Test CPU".into(),
            threads: 8,
            ram_gb: ram,
            gpus: if vram > 0.0 { vec![Gpu { name: "NVIDIA GeForce Test".into(), vram_gb: vram }] } else { vec![] },
            unified: false,
        }
    }

    #[test]
    fn quant_names() {
        assert_eq!(quant_of("flint-UD-Q4_K_XL.gguf"), "UD-Q4_K_XL");
        assert_eq!(quant_of("google_gemma-4-E4B-it-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(quant_of("model.Q8_0.gguf"), "Q8_0");
        assert_eq!(quant_of("flint-UD-IQ1_M.gguf"), "UD-IQ1_M");
        assert_eq!(quant_of("x-BF16.gguf"), "BF16");
        assert_eq!(quant_of("weights.gguf"), "GGUF");
        assert_eq!(quant_of("Qwen3.8-27B-UD-Q4_K_XL.gguf"), "UD-Q4_K_XL", "model names like Qwen3 are not quants");
    }

    #[test]
    fn fit_depends_on_the_machine() {
        let laptop = pc(8.0, 0.0);
        assert_eq!(fit(&laptop, 2.5, false), Fit::Good);
        assert_eq!(fit(&laptop, 9.0, false), Fit::TooBig);
        let gaming = pc(32.0, 12.0);
        assert_eq!(fit(&gaming, 7.0, false), Fit::Fast);
        assert_eq!(fit(&gaming, 16.0, false), Fit::Mixed);
        assert_eq!(fit(&gaming, 40.0, false), Fit::TooBig);
        let cpu_big = pc(64.0, 0.0);
        assert_eq!(fit(&cpu_big, 20.0, false), Fit::Slow);
        assert_eq!(fit(&cpu_big, 20.0, true), Fit::Good, "mixture-of-experts stays quick on CPU");
        let mac = Hardware { os: "macos".into(), arch: "aarch64".into(), ram_gb: 32.0, unified: true, ..Default::default() };
        assert_eq!(fit(&mac, 14.0, false), Fit::Fast);
    }

    #[test]
    fn picks_the_largest_quant_that_keeps_the_best_fit() {
        // the real file list of a 33B-class model
        let f = |n: &str, gb: f64| GgufFile { name: n.into(), size: (gb * 1e9) as u64, quant: quant_of(n) };
        let files = vec![f("m-UD-IQ1_M.gguf", 11.3), f("m-UD-Q2_K_XL.gguf", 13.5), f("m-UD-Q3_K_XL.gguf", 18.4), f("m-UD-Q4_K_XL.gguf", 25.0), f("m-UD-Q6_K_XL.gguf", 36.3), f("m-Q8_0.gguf", 42.3)];
        let (best, ft) = pick_file(&pc(32.0, 24.0), &files, false).unwrap();
        assert_eq!(best.name, "m-UD-Q3_K_XL.gguf");
        assert_eq!(ft, Fit::Fast, "Q4 would spill out of VRAM, so the Q3 that stays on the GPU wins");
        let (best, _) = pick_file(&pc(64.0, 0.0), &files, false).unwrap();
        assert_eq!(best.name, "m-UD-Q4_K_XL.gguf", "CPU-only: a 4-bit file, not the biggest that fits");
        let (best, ft) = pick_file(&pc(8.0, 0.0), &files, false).unwrap();
        assert_eq!(ft, Fit::TooBig);
        assert_eq!(best.name, "m-UD-IQ1_M.gguf", "nothing fits: show the smallest, flagged");
    }

    #[test]
    fn tree_listing_skips_split_vision_and_imatrix_files() {
        let tree = serde_json::json!([
            {"type": "dir", "path": "BF16"},
            {"type": "file", "path": "a-Q4_K_M.gguf", "size": 1000, "lfs": {"size": 4_900_000_000u64}},
            {"type": "file", "path": "a-00001-of-00003.gguf", "size": 1000, "lfs": {"size": 9_000_000_000u64}},
            {"type": "file", "path": "mmproj-a-F16.gguf", "size": 1000, "lfs": {"size": 900_000_000u64}},
            {"type": "file", "path": "imatrix_unsloth.gguf_file", "size": 5},
            {"type": "file", "path": "README.md", "size": 5}
        ]);
        let files = files_from_tree(&tree);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].size, 4_900_000_000);
    }

    #[test]
    fn side_files_are_not_mistaken_for_the_model() {
        let f = |n: &str, gb: f64| GgufFile { name: n.into(), size: (gb * 1e9) as u64, quant: quant_of(n) };
        let kept = plausible_files(vec![f("x-Q4_0.gguf", 2.2), f("x-MTP-Q4_0.gguf", 60.0), f("x-IQ2_XXS.gguf", 52.0)], Some(177.0));
        assert_eq!(kept.len(), 1, "2.2 GB cannot hold 177B parameters; MTP heads are not the model");
        assert_eq!(kept[0].name, "x-IQ2_XXS.gguf");
        assert_eq!(plausible_files(vec![f("t-Q4_0.gguf", 0.2)], Some(0.3)).len(), 1, "tiny models are fine");
        assert_eq!(plausible_files(vec![f("t-Q4_0.gguf", 0.2)], None).len(), 1, "unknown size: keep");
    }

    #[test]
    fn moe_names() {
        assert!(has_active_params("unsloth/qwen3-30b-a3b-gguf"));
        assert!(has_active_params("bartowski/google_gemma-4-26b-a4b-it-gguf"));
        assert!(!has_active_params("bartowski/gemma-4-12b-it-gguf"));
        assert!(!has_active_params("unsloth/flint-gguf"));
    }

    #[test]
    fn runner_asset_matches_the_machine() {
        let assets: Vec<String> = [
            "llama-b8694-bin-win-cpu-x64.zip",
            "llama-b8694-bin-win-cpu-arm64.zip",
            "llama-b8694-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
            "llama-b8694-bin-win-vulkan-x64.zip",
            "llama-b8694-bin-ubuntu-x64.tar.gz",
            "llama-b8694-bin-ubuntu-vulkan-x64.tar.gz",
            "llama-b8694-bin-macos-arm64.tar.gz",
            "llama-b8694-bin-macos-x64.tar.gz",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(pick_asset(&pc(16.0, 8.0), &assets).unwrap(), "llama-b8694-bin-win-vulkan-x64.zip");
        assert_eq!(pick_asset(&pc(16.0, 0.0), &assets).unwrap(), "llama-b8694-bin-win-cpu-x64.zip");
        let lin = Hardware { os: "linux".into(), arch: "x86_64".into(), ..pc(16.0, 0.0) };
        assert_eq!(pick_asset(&lin, &assets).unwrap(), "llama-b8694-bin-ubuntu-x64.tar.gz");
        let mac = Hardware { os: "macos".into(), arch: "aarch64".into(), ram_gb: 16.0, unified: true, ..Default::default() };
        assert_eq!(pick_asset(&mac, &assets).unwrap(), "llama-b8694-bin-macos-arm64.tar.gz");
    }

    #[test]
    fn built_names_are_ones_the_picker_accepts() {
        for hw in [pc(16.0, 8.0), pc(16.0, 0.0), Hardware { os: "linux".into(), arch: "x86_64".into(), ..pc(16.0, 0.0) }] {
            let names = runner_names(&hw, "b9999");
            assert_eq!(pick_asset(&hw, &names).as_deref(), Some(names[0].as_str()));
        }
        assert_eq!(runner_names(&pc(16.0, 8.0), "b1")[0], "llama-b1-bin-win-vulkan-x64.zip");
    }

    #[test]
    fn unpacks_zip_and_tar_gz_with_a_top_folder() {
        let dir = std::env::temp_dir().join(format!("hinoko-unpack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // zip, flat like the Windows builds
        let zpath = dir.join("a.zip");
        {
            let mut z = zip::ZipWriter::new(std::fs::File::create(&zpath).unwrap());
            z.start_file("llama-server.exe", zip::write::SimpleFileOptions::default()).unwrap();
            std::io::Write::write_all(&mut z, b"MZ").unwrap();
            z.finish().unwrap();
        }
        let out1 = dir.join("z");
        std::fs::create_dir_all(&out1).unwrap();
        unpack(&zpath, &out1).unwrap();
        assert!(find_file(&out1, "llama-server.exe").is_some());
        // tar.gz, one folder on top like the Linux/macOS builds
        let tpath = dir.join("a.tar.gz");
        {
            let gz = flate2::write::GzEncoder::new(std::fs::File::create(&tpath).unwrap(), flate2::Compression::fast());
            let mut t = tar::Builder::new(gz);
            let data = b"#!/bin/sh\n";
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o755);
            h.set_cksum();
            t.append_data(&mut h, "llama-b1/llama-server", &data[..]).unwrap();
            t.into_inner().unwrap().finish().unwrap();
        }
        let out2 = dir.join("t");
        std::fs::create_dir_all(&out2).unwrap();
        unpack(&tpath, &out2).unwrap();
        assert!(find_file(&out2, "llama-server").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn runner_asset_survives_renamed_builds() {
        let assets: Vec<String> = ["llama-v0.8.0-windows-x86_64-vulkan.zip", "llama-v0.8.0-windows-x86_64-cpu.zip", "llama-v0.8.0-windows-x86_64-cuda-12.zip", "llama-v0.8.0-linux-x86_64.tar.gz"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(pick_asset(&pc(16.0, 8.0), &assets).unwrap(), "llama-v0.8.0-windows-x86_64-vulkan.zip");
        assert_eq!(pick_asset(&pc(16.0, 0.0), &assets).unwrap(), "llama-v0.8.0-windows-x86_64-cpu.zip");
        assert!(pick_asset(&pc(16.0, 0.0), &["notes.txt".to_string()]).is_none());
    }

    #[test]
    fn nvidia_smi_output() {
        let gpus = parse_nvidia("NVIDIA GeForce RTX 3060, 12288\nNVIDIA GeForce GTX 1050, 4096\n");
        assert_eq!(gpus.len(), 2);
        assert!((gpus[0].vram_gb - 12.0).abs() < 0.01);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 3060");
    }
}
