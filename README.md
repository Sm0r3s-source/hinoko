# Hinoko

A native desktop AI workspace in pure Rust (egui). Plain **chats**, **deep research**, and persistent **bots**
(own computer, memory, routines, approvals) in one app, with any model: OpenAI-compatible APIs, local servers, or CLI agents.
Crimson by default; sakura, ember, paper, midnight, forest, mono, kojo and fully custom themes in Settings. The default font is Inter (soft, professional). Two optional pixel-art looks: the **Pixel / Pixel Sakura** themes (or *Pixel art interface* in Settings → Appearance) give the whole UI notched pixel corners, hard borders, hard drop shadows and a pixel flame logo, and *Pixel font* switches the text to Pixelify Sans. They are independent, so you can have either or both. Fonts: Inter and Pixelify Sans, SIL Open Font License (`assets/fonts`).

## Install as an app

| OS | Command | Result |
|---|---|---|
| Windows | `.\install-windows.ps1` in PowerShell | Desktop + Start Menu shortcut with icon |
| Linux | `sh install-linux.sh` | Entry in your app menu |
| macOS | `cargo build --release`, drag `target/release/hinoko` anywhere | Double-click to open |

Needs Rust (https://rustup.rs). Windows also needs the Visual Studio C++ Build Tools (rustup offers them).
Push to GitHub and tag `v0.4.0`: `.github/workflows/release.yml` builds a real **Hinoko-Setup.exe** (Inno Setup, per-user, no admin) plus Linux/macOS binaries and attaches them to the release. Run it by hand from the Actions tab too. You never need to compile locally.
The Windows build is untested by me (built and tested on Linux only).

## First launch

**⚙ Settings → Models:** pick a starting point (OpenAI, Anthropic, OpenRouter, xAI, Ollama, LM Studio, llama.cpp,
Claude/Codex/Gemini CLI, or custom) → key + model → **Test** → **Save**. Then **＋ New chat** or **Bots → ＋ New bot**.
No premade bots. Keys can be read from an environment variable instead of stored.

## Features

- **Chats:** streaming, model picker per chat, markdown code blocks with Copy. Delete or rename from the sidebar (✕ or right-click).
- **Composer:** the **+** button attaches files (text is read in, images are shown to vision models; you can also drag files onto the window) and turns on Web search or Deep research. **Think:** Auto/Off/Low/Medium/High sets reasoning effort for models that support it, and their reasoning shows folded under the answer.
- **Local models, nothing else to install:** Settings → *Local models* (or the button on the welcome screen). Hinoko reads your RAM/CPU/GPU, asks Hugging Face for trending or searched GGUF models, and shows which quantisation fits your PC and how well (runs fast / well / slowly / too big). *Download & run* fetches the model with progress and resume and runs it on **llama.cpp** (MIT; Hinoko downloads a ~40 MB `llama-server` build once, and uses your GPU when it can), and adds it as your default model. There is also an experimental **built-in engine** ([mistral.rs](https://github.com/EricLBuehler/mistral.rs), MIT, inside Hinoko, CPU only, much slower, tiny models only): choose it under *Engine* in the same screen. If the runner can't be downloaded (offline), Automatic falls back to it. Always run a release build (`cargo run --release`); a debug build of the built-in engine is 20-50x slower. It starts again by itself when a chat or bot needs it, and stops when you close Hinoko. Models live in the Hinoko data folder under `local/models`.
- **Code:** a coding agent for one project folder (Code tab → Open a project). Uses any model you set up (API, local or CLI). It lists, searches and reads your code, edits files with exact-match edits, and runs build/test commands. Every edit shows a diff and every command waits for your Allow, unless you pick *Auto-accept edits* or *Auto-run everything*. *Plan* mode is read-only. It cannot leave the project folder or touch `.git`, *Undo request* reverts every file the last request changed (*Undo step* only the newest change), the header shows the git branch and changed files with a *Diff* button, and `AGENTS.md`/`CLAUDE.md` in the project are read as instructions. Models that write their tool calls as text (Spark, Qwen-Coder, Hermes style) are understood too.
- **Deep research:** tick the box in a chat: plans queries, searches, reads pages, writes a cited report. Set a SearXNG URL in Settings for reliable search (blank falls back to scraping DuckDuckGo).
- **Bots:** web search/read, files in their own sandbox, memory, routines (proposed paused), approvals for delete/send/ssh, read-only mode.
- **SSH machines:** Settings → Machines. Show Hinoko's key, run the one-line setup on the other PC, Test. Tick which machines each bot may reach (Bot Options). Plain read-only commands (ls, df, uptime…) can auto-run; everything else waits for your Allow.
- **Compare models:** *⇄ Compare models* in the Chats sidebar asks two models the same question at once and shows both answers with time to first text and words per second. Nothing is saved to a chat.
- **Built-in chat prompt:** plain chats start with Hinoko's identity, today's date and guidance on searching, sources and tone (a shortened version for small local models). Switch it off in Settings → Instructions; your own instructions come after it.
- **Speed counter:** every chat answer shows its writing speed (`38 tok/s · 412 tokens · first token 1.2s`), live while it types and with the server's own token count at the end (a `~` marks an estimate). Speed is measured from the first piece of output, so waiting for the prompt isn't counted. The compare window and the Code tab show it too.
- **Reply length:** Settings → Limits → *Max reply length* (chat) and *Max Code step length*. Auto is 8,192 tokens, doubled while Think is on, since a reasoning model spends part of the budget thinking. If a model still uses it all thinking, its thoughts are kept and the reply tells you what to change. For local servers, Think Off/Low is also sent as a chat-template setting (`enable_thinking`, `reasoning_effort`), with fallbacks if the server rejects it.
- **Busy providers:** a rate-limited or overloaded provider (429, 502, 503) is retried up to three times, honoring `Retry-After`, then reported in plain words.
- **Code reviewer:** Settings → Limits → *Code reviewer*. When a Code request changed files, a second pass (the same model, or a stronger one you pick) reads the diffs against your request and sends the agent back with concrete problems, up to 1-3 rounds, until it says LGTM. Off by default.
- **Activity:** the *Activity* button in the Code tab lists every file touched and command run in the session, with failed commands in red. The agent is also told which dev tools are installed (`git`, `gh`, `cargo`, `npm`, `docker`, cloud CLIs...) so it can use them.
- **Loop stop:** a model that starts repeating itself, or gets stuck in a thinking loop, is stopped. The repetition is cut and the reply says what happened and what to change.
- **Per-model sampling and backup:** in a model's settings you can give it its own Temperature and Top-p (Spark recommends 1.0 and 0.95) and pick a **Backup model**. If the model is down, rate-limited, out of quota or its key is rejected, Hinoko switches to the backup for that request and says so.
- **Standing rules (bots):** on a bot's approval card, *Always allow this* / *Always deny this* saves a rule for that exact command or file and bot. From then on that request is answered for you, with a line in the chat saying so. Rules are listed (and removed) in Settings → Bots: attention, a deny rule beats an allow rule, a bot can never make rules itself.
- **Quiet routines:** a routine that finds nothing new leaves one muted line ("Checked · name · nothing to report") instead of a message, so bots only speak when there is something for you. Switch it off in Settings → Bots: attention.
- **Parallel Code sessions:** open several projects and let them all work at once, each with its own model. The sidebar marks the ones working (●) and the ones waiting for you (?), with a *Stop all* button.
- **Updates:** Settings → Updates checks GitHub for a newer release (also when Hinoko starts, if you leave that on). *Download and install* fetches `Hinoko-Setup.exe`, checks its SHA-256, closes Hinoko, updates it and opens it again; your chats, bots and settings stay. To publish a version: bump `version` in `Cargo.toml`, run `.\release.ps1` (commit, tag `vX.Y.Z`, push); the CI builds the installer and creates the release. The repo is set in `crates/hinoko-core/src/update.rs`.
- **Run traces:** Settings → Limits → *Write run traces* saves every model call and tool run as JSON lines under `traces/` in the data folder. Off by default.
- **Customize:** themes, accent/background, UI scale, spacing, rounding, chat width, Enter-to-send, global instructions.

## Command line

```
hinoko bots | new <name> <role> <mandate...>   hinoko run <bot> <prompt...> [--json]   hinoko chat <bot>
hinoko approvals | approve <id> | deny <id>
hinoko provider list | add ... | use <id> | remove <id> | test [id]
hinoko machine list | add --name N --host H --user U [--port P] [--auto-readonly] | remove <id> | test <id> | key
hinoko path
```
Run only one Hinoko (window or CLI) at a time; they share one state file.

## Safety

- Bot shell is a simulation in memory; it never touches your PC. SSH is the only way out, and it asks.
- Page reads: https, public addresses only (checked at connect time), tool output fenced as untrusted.
- Model calls capped per rolling hour (default 60; plain chats exempt). CLI agents run outside the sandbox with your account's permissions.
- Data: `%APPDATA%\Hinoko` / `~/.local/share/Hinoko` / `~/Library/Application Support/Hinoko` (override `HINOKO_HOME`). `settings.json` holds keys, owner-only on Linux/macOS.

## Layout

`crates/hinoko-core` engine + tests. `crates/hinoko` window + CLI. `assets/` icon (`packaging/make_icon.py` regenerates it).


- **Long conversations stay usable:** when a chat or coding session outgrows what the model can read (about 20k tokens, 5k for the built-in local model; change it in Settings → Limits), Hinoko has the model summarize the older part and sends only the summary plus the recent messages. Nothing is deleted. *Chat options → Compact now* and the *Compact* button in Code do it on demand.

## Credits

The built-in model engine is [mistral.rs](https://github.com/EricLBuehler/mistral.rs) by Eric Buehler and contributors (MIT). The optional GPU runner is [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
