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
}
