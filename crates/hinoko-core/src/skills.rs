//! Skills: folders holding a SKILL.md (a name and description, then instructions). The model sees
//! only the list; it loads a skill's full text when a task matches, so unused skills cost almost nothing.
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file: PathBuf,
}

/// Skills in the project's `.hinoko/skills` and `.claude/skills` and in the data folder's `skills`.
pub fn find(project: Option<&str>, home: &Path) -> Vec<Skill> {
    let mut dirs: Vec<PathBuf> = project.map(|p| vec![Path::new(p).join(".hinoko/skills"), Path::new(p).join(".claude/skills")]).unwrap_or_default();
    dirs.push(home.join("skills"));
    let mut out: Vec<Skill> = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(d) else { continue };
        let mut found: Vec<_> = rd.flatten().map(|e| e.path().join("SKILL.md")).filter(|f| f.is_file()).collect();
        found.sort();
        for f in found {
            let text = std::fs::read_to_string(&f).unwrap_or_default();
            let folder = f.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let (name, description, _) = parse(&text, &folder);
            if !out.iter().any(|s| s.name == name) {
                out.push(Skill { name, description, file: f });
            }
        }
    }
    out
}

/// (name, description, instructions) from a SKILL.md with optional `---` front matter.
fn parse(text: &str, folder: &str) -> (String, String, String) {
    let (mut name, mut desc, mut body) = (folder.to_string(), String::new(), text);
    if let Some(rest) = text.trim_start().strip_prefix("---") {
        if let Some((head, tail)) = rest.split_once("\n---") {
            body = tail.trim_start_matches(['-', '\r']).trim_start_matches('\n');
            for l in head.lines() {
                match l.split_once(':') {
                    Some(("name", v)) if !v.trim().is_empty() => name = v.trim().trim_matches(['"', '\'']).to_string(),
                    Some(("description", v)) => desc = v.trim().trim_matches(['"', '\'']).to_string(),
                    _ => {}
                }
            }
        }
    }
    if desc.is_empty() {
        desc = body.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim_start_matches('#').trim().to_string();
    }
    (name, desc.chars().take(200).collect(), body.trim().to_string())
}

/// The instructions of a skill (front matter removed), cut to a sensible size.
pub fn body(s: &Skill) -> String {
    let (_, _, b) = parse(&std::fs::read_to_string(&s.file).unwrap_or_default(), "");
    crate::model::clip(&b, 20_000)
}

/// The line of instructions that tells the model which skills exist.
pub fn index(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let list: Vec<String> = skills.iter().map(|s| format!("- {}: {}", s.name, s.description)).collect();
    format!("\nSkills (saved instructions for particular tasks). When the request matches one, call the skill tool with its name first, then follow it:\n{}\n", list.join("\n"))
}

/// The `skill` tool.
pub fn tool() -> serde_json::Value {
    serde_json::json!({"type": "function", "function": {"name": "skill", "description": "Load the full instructions of one of the listed skills.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": false}}})
}

/// What the `skill` tool returns.
pub fn run(skills: &[Skill], name: &str) -> String {
    match skills.iter().find(|s| s.name.eq_ignore_ascii_case(name.trim())) {
        Some(s) => format!("Skill {}:\n{}", s.name, body(s)),
        None => format!("ERROR: no skill called {name}. Skills: {}", skills.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ")),
    }
}

/// What to fetch: (owner, repo, ref, subfolder) from `owner/repo`, `owner/repo/sub/folder`
/// or a GitHub URL (`.../tree/<branch>/<folder>`).
pub fn parse_spec(spec: &str) -> Option<(String, String, String, String)> {
    let t = spec.trim().trim_end_matches('/').trim_end_matches(".git");
    let t = t.strip_prefix("https://github.com/").or_else(|| t.strip_prefix("github.com/")).unwrap_or(t);
    let p: Vec<&str> = t.split('/').filter(|x| !x.is_empty()).collect();
    let ok = |x: &str| !x.is_empty() && x.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
    if p.len() < 2 || !ok(p[0]) || !ok(p[1]) || p.iter().any(|x| *x == "..") {
        return None;
    }
    let (r#ref, sub) = if p.get(2) == Some(&"tree") && p.len() > 3 { (p[3].to_string(), p[4..].join("/")) } else { ("HEAD".into(), p[2..].join("/")) };
    Some((p[0].into(), p[1].into(), r#ref, sub))
}

/// Unpacks every skill folder (one with a SKILL.md) found under `sub` in a GitHub tar.gz into `dest`.
/// Returns the names installed. Paths are sanitised; files over 1 MB and very large skills are skipped.
pub fn install_archive(gz: &[u8], sub: &str, repo: &str, dest: &Path) -> Result<Vec<String>, String> {
    use std::io::Read;
    let sub = sub.trim_matches('/');
    let mut files: Vec<(Vec<String>, Vec<u8>)> = Vec::new();
    let mut ar = tar::Archive::new(flate2::read::GzDecoder::new(gz));
    for e in ar.entries().map_err(|e| format!("bad archive: {e}"))?.flatten() {
        if !e.header().entry_type().is_file() || e.size() > 1_000_000 {
            continue;
        }
        let path = e.path().map_err(|e| e.to_string())?.to_string_lossy().replace('\\', "/");
        let parts: Vec<String> = path.split('/').skip(1).map(String::from).collect(); // drop "<repo>-<ref>/"
        if parts.is_empty() || parts.iter().any(|p| p.is_empty() || p == ".." || p.contains(':')) || (!sub.is_empty() && !parts.join("/").starts_with(&format!("{sub}/"))) {
            continue;
        }
        let mut buf = Vec::new();
        e.take(1_000_000).read_to_end(&mut buf).map_err(|e| e.to_string())?;
        files.push((parts, buf));
    }
    let root = sub.split('/').filter(|x| !x.is_empty()).count();
    let mut names = Vec::new();
    for (parts, _) in files.iter().filter(|f| f.0.last().map(|n| n == "SKILL.md").unwrap_or(false)) {
        let dir = &parts[..parts.len() - 1];
        let name = dir.last().cloned().unwrap_or_else(|| repo.to_string());
        let name: String = name.chars().filter(|c| c.is_ascii_alphanumeric() || "-_.".contains(*c)).collect();
        if name.is_empty() || name.starts_with('.') || dir.len() < root {
            continue;
        }
        let (mut total, out) = (0usize, dest.join(&name));
        let _ = std::fs::remove_dir_all(&out);
        for (p, data) in files.iter().filter(|f| f.0.len() > dir.len() && f.0[..dir.len()] == *dir) {
            total += data.len();
            if total > 5_000_000 {
                return Err(format!("{name} is larger than 5 MB"));
            }
            let target = p[dir.len()..].iter().fold(out.clone(), |a, x| a.join(x));
            if let Some(d) = target.parent() {
                std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
            }
            std::fs::write(&target, data).map_err(|e| e.to_string())?;
        }
        names.push(name);
    }
    if names.is_empty() {
        return Err("no SKILL.md found there".into());
    }
    names.sort();
    names.dedup();
    Ok(names)
}

impl crate::Engine {
    /// Downloads a skill (or a repo of skills) from GitHub into the skills folder.
    pub async fn skill_install(&self, spec: &str) -> Result<Vec<String>, String> {
        let (owner, repo, r, sub) = parse_spec(spec).ok_or("use owner/repo, owner/repo/folder or a GitHub link")?;
        let url = format!("https://codeload.github.com/{owner}/{repo}/tar.gz/{r}");
        let res = self.http.get(&url).timeout(std::time::Duration::from_secs(60)).send().await.map_err(|_| "could not reach GitHub")?;
        if !res.status().is_success() {
            return Err(format!("GitHub answered {} (is the repo public?)", res.status()));
        }
        let bytes = res.bytes().await.map_err(|_| "the download broke")?;
        if bytes.len() > 40_000_000 {
            return Err("that repo is too large".into());
        }
        install_archive(&bytes, &sub, &repo, &self.home().join("skills"))
    }
}

/// Removes an installed skill (by its name or its folder's name) from the data folder.
pub fn remove(home: &Path, name: &str) -> bool {
    let root = home.join("skills");
    let by_name = find(None, home).into_iter().find(|s| s.name.eq_ignore_ascii_case(name.trim())).and_then(|s| s.file.parent().map(Path::to_path_buf));
    let folder = by_name.or_else(|| (!name.trim().is_empty() && !name.contains(['/', '\\']) && !name.trim().starts_with('.')).then(|| root.join(name.trim())));
    folder.is_some_and(|f| f.parent() == Some(root.as_path()) && std::fs::remove_dir_all(f).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skills_are_found_in_project_and_home_and_load_on_demand() {
        let root = std::env::temp_dir().join(format!("hinoko-skills-{}", crate::model::nid("s")));
        let (proj, home) = (root.join("proj"), root.join("home"));
        let put = |dir: &Path, folder: &str, text: &str| {
            std::fs::create_dir_all(dir.join(folder)).unwrap();
            std::fs::write(dir.join(folder).join("SKILL.md"), text).unwrap();
        };
        put(&proj.join(".claude/skills"), "release", "---\nname: release\ndescription: \"Cut a release\"\n---\n# Release\nBump the version, then tag it.\n");
        put(&home.join("skills"), "plain", "# Plain notes\nNo front matter here.\n");
        put(&home.join("skills"), "release", "---\nname: release\ndescription: shadowed by the project\n---\nignored");
        let found = find(proj.to_str(), &home);
        let names: Vec<_> = found.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["release", "plain"], "a project skill wins over a same-named global one");
        assert_eq!(found[0].description, "Cut a release");
        assert_eq!(found[1].description, "Plain notes", "no front matter: the first line describes it");
        assert!(index(&found).contains("- release: Cut a release") && index(&[]).is_empty());
        assert!(run(&found, "Release").contains("Bump the version") && !run(&found, "release").contains("description:"));
        assert!(run(&found, "nope").starts_with("ERROR: no skill called nope"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn github_specs_parse_and_archives_install_safely() {
        assert_eq!(parse_spec("anthropics/skills").unwrap(), ("anthropics".into(), "skills".into(), "HEAD".into(), "".into()));
        assert_eq!(parse_spec("a/b/skills/pdf").unwrap().3, "skills/pdf");
        assert_eq!(parse_spec("https://github.com/a/b/tree/main/skills/pdf").unwrap(), ("a".into(), "b".into(), "main".into(), "skills/pdf".into()));
        assert!(parse_spec("nope").is_none() && parse_spec("a/../b").is_none() && parse_spec("a b/c").is_none());

        let mut tb = tar::Builder::new(Vec::new());
        let mut add = |path: &str, body: &str| {
            let mut h = tar::Header::new_gnu();
            h.set_size(body.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            tb.append_data(&mut h, path, body.as_bytes()).unwrap();
        };
        add("repo-abc/skills/pdf/SKILL.md", "---\nname: pdf\ndescription: Work with PDFs\n---\nSteps");
        add("repo-abc/skills/pdf/scripts/run.py", "print(1)");
        add("repo-abc/skills/docx/SKILL.md", "# Docx\nSteps");
        add("repo-abc/other/SKILL.md", "# Elsewhere");
        let gz = {
            use std::io::Write;
            let mut g = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            g.write_all(&tb.into_inner().unwrap()).unwrap();
            g.finish().unwrap()
        };
        let root = std::env::temp_dir().join(format!("hinoko-inst-{}", crate::model::nid("i")));
        assert_eq!(install_archive(&gz, "skills", "repo", &root).unwrap(), ["docx", "pdf"], "only the chosen folder is installed");
        assert!(root.join("pdf/scripts/run.py").is_file() && !root.join("other").exists());
        assert_eq!(install_archive(&gz, "skills/pdf", "repo", &root.join("two")).unwrap(), ["pdf"]);
        assert!(install_archive(&gz, "missing", "repo", &root).is_err());
        // removing goes by the skill's name even when its folder is called something else
        let home = root.join("home");
        std::fs::create_dir_all(home.join("skills/pdf-tools.v2")).unwrap();
        std::fs::write(home.join("skills/pdf-tools.v2/SKILL.md"), "---\nname: pdf\n---\nx").unwrap();
        assert!(!remove(&home, "..") && !remove(&home, "nope"));
        assert!(remove(&home, "pdf") && !home.join("skills/pdf-tools.v2").exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
