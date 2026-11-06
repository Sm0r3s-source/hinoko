//! The desktop window (egui). All state lives in `hinoko-core`; this file only draws it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use base64::Engine as _;
use eframe::egui::{self, Color32, Key, Modifiers, RichText};
use hinoko_core::model::{clean, home_of, Message, Routine};
use hinoko_core::settings::{Machine, Appearance, Kind, Preset, Provider, PRESETS, THEMES};
use hinoko_core::{computer, Engine};

/// Colours in use right now. Rebuilt from the Appearance settings.
#[derive(Clone, Copy)]
struct Pal {
    ink: Color32,
    panel: Color32,
    raised: Color32,
    text: Color32,
    muted: Color32,
    accent: Color32,
    good: Color32,
}

static PAL: RwLock<Pal> = RwLock::new(Pal {
    ink: Color32::from_rgb(0x1c, 0x19, 0x16),
    panel: Color32::from_rgb(0x24, 0x20, 0x1c),
    raised: Color32::from_rgb(0x2e, 0x29, 0x24),
    text: Color32::from_rgb(0xf4, 0xf0, 0xe7),
    muted: Color32::from_rgb(0xa0, 0x98, 0x8c),
    accent: Color32::from_rgb(0xd0, 0x5a, 0x24),
    good: Color32::from_rgb(0x7f, 0xb0, 0x7a),
});

fn pal() -> Pal {
    *PAL.read().unwrap_or_else(|e| e.into_inner())
}
fn ink() -> Color32 {
    pal().ink
}
fn panel() -> Color32 {
    pal().panel
}
fn raised() -> Color32 {
    pal().raised
}
fn paper() -> Color32 {
    pal().text
}
fn muted() -> Color32 {
    pal().muted
}
fn ember() -> Color32 {
    pal().accent
}
fn good() -> Color32 {
    pal().good
}

fn rgb(c: [u8; 3]) -> Color32 {
    Color32::from_rgb(c[0], c[1], c[2])
}

fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
}

fn luminance(c: Color32) -> f32 {
    (0.2126 * c.r() as f32 + 0.7152 * c.g() as f32 + 0.0722 * c.b() as f32) / 255.0
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Files,
    Terminal,
    Browser,
    Memory,
}

#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Chats,
    Bots,
    Code,
}

enum Composed {
    Nothing,
    Send(String),
    Stop,
}

#[derive(PartialEq, Clone, Copy)]
enum Dialog {
    None,
    ChatOptions,
    NewBot,
    Settings,
    Routines,
    BotOptions,
    About,
    Cookbook,
    Compare,
}

#[derive(Default)]
struct NewBot {
    name: String,
    title: String,
    mandate: String,
    error: String,
}

#[derive(Default)]
struct RoutineForm {
    name: String,
    schedule: String,
    prompt: String,
}

/// A provider being edited in Settings. `args` is one argument per line.
struct Editing {
    provider: Provider,
    args: String,
    is_new: bool,
}

/// State of the Local models window.
#[derive(Default)]
struct CookList {
    loading: bool,
    error: String,
    items: Vec<hinoko_core::local::Candidate>,
}

#[derive(Default)]
struct Cook {
    hw: Arc<Mutex<Option<hinoko_core::local::Hardware>>>,
    list: Arc<Mutex<CookList>>,
    query: String,
    show_big: bool,
    started: bool,
}

/// What the composer has queued for the next message in one chat.
#[derive(Default)]
struct ChatDraft {
    files: Vec<hinoko_core::Attach>,
    web: bool,
    research: bool,
    effort: String,
}

const MAX_TEXT_FILE: usize = 60_000;
const MAX_IMAGE: u64 = 4_000_000;

/// Read a file the person attached: text is inlined, images are sent as pictures.
fn read_attachment(path: &std::path::Path) -> Result<hinoko_core::Attach, String> {
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "file".into());
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let size = std::fs::metadata(path).map_err(|e| format!("{name}: {e}"))?.len();
    let mime = match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        _ => None,
    };
    if let Some(mime) = mime {
        if size > MAX_IMAGE {
            return Err(format!("{name} is over 4 MB"));
        }
        let bytes = std::fs::read(path).map_err(|e| format!("{name}: {e}"))?;
        let url = format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes));
        return Ok(hinoko_core::Attach { name, text: String::new(), image: Some(url) });
    }
    if size > 5_000_000 {
        return Err(format!("{name} is too large (text files up to 5 MB)"));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{name}: {e}"))?;
    if bytes.iter().take(4096).any(|b| *b == 0) {
        return Err(format!("{name}: only text files and images can be attached for now"));
    }
    let text = String::from_utf8_lossy(&bytes).to_string();
    let text = if text.chars().count() > MAX_TEXT_FILE {
        let cut: String = text.chars().take(MAX_TEXT_FILE).collect();
        format!("{cut}\n[…file cut at {MAX_TEXT_FILE} characters]")
    } else {
        text
    };
    Ok(hinoko_core::Attach { name, text, image: None })
}

impl Drop for HinokoApp {
    fn drop(&mut self) {
        // don't leave a model server running after the window closes
        self._rt.block_on(self.engine.local_stop());
    }
}

pub struct HinokoApp {
    engine: Engine,
    _rt: tokio::runtime::Runtime,
    cook: Cook,
    chat_drafts: HashMap<String, ChatDraft>,
    pending_delete: Option<String>,
    editing_machine: Option<Machine>,
    machine_msg: Arc<Mutex<String>>,
    pubkey: Arc<Mutex<Option<String>>>,
    mode: Mode,
    active_chat: Option<String>,
    active_code: Option<String>,
    confirm_delete_code: bool,
    look: Appearance,
    applied: String,
    theme_text: String,
    theme_msg: String,
    skill_spec: String,
    sidebar_open: bool,
    skill_msg: Arc<Mutex<String>>,
    active: Option<String>,
    drafts: HashMap<String, String>,
    tab: Tab,
    dialog: Dialog,
    new_bot: NewBot,
    routine_form: RoutineForm,
    editing: Option<Editing>,
    test_result: Arc<Mutex<Option<Result<String, String>>>>,
    testing: bool,
    term_input: String,
    mem_input: String,
    open_file: Option<String>,
    toast: String,
    stick_bottom: bool,
    confirm_delete_bot: bool,
    md: std::cell::RefCell<egui_commonmark::CommonMarkCache>,
    compare_prompt: String,
    compare_pick: [Option<String>; 2],
    ctx: egui::Context,
    git_diff: Option<(String, String)>,
    activity_open: bool,
    update: Arc<Mutex<hinoko_core::update::Update>>,
    compare: Option<hinoko_core::compare::Compare>,
    git: Arc<Mutex<HashMap<String, GitEntry>>>,
}

/// A cached look at a project's git state, refreshed in the background.
struct GitEntry {
    at: std::time::Instant,
    status: Option<hinoko_core::git::GitStatus>,
    busy: bool,
}

static PIXEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static SAKURA_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A corner radius, squared off in the pixel style.
fn rad(r: f32) -> f32 {
    if PIXEL.load(std::sync::atomic::Ordering::Relaxed) { 0.0 } else { r }
}


fn pixel_on() -> bool {
    PIXEL.load(std::sync::atomic::Ordering::Relaxed)
}

/// A panel in the pixel style: corners notched by one "pixel", a hard 2px border and, if
/// asked, a hard drop shadow. The two overlapping rectangles make the notch.
fn notched(rect: egui::Rect, fill: Color32, edge: Color32, shadow: bool) -> egui::Shape {
    let u = 4.0;
    let parts = |r: egui::Rect| [egui::Rect::from_min_max(egui::pos2(r.left() + u, r.top()), egui::pos2(r.right() - u, r.bottom())), egui::Rect::from_min_max(egui::pos2(r.left(), r.top() + u), egui::pos2(r.right(), r.bottom() - u))];
    let mut shapes = Vec::new();
    if shadow {
        for p in parts(rect.translate(egui::vec2(4.0, 4.0))) {
            shapes.push(egui::Shape::rect_filled(p, 0.0, mix(ember(), Color32::BLACK, 0.72)));
        }
    }
    for p in parts(rect) {
        shapes.push(egui::Shape::rect_filled(p, 0.0, fill));
    }
    let (l, t, r, b) = (rect.left(), rect.top(), rect.right(), rect.bottom());
    let pts = [(l, t + u), (l + u, t + u), (l + u, t), (r - u, t), (r - u, t + u), (r, t + u), (r, b - u), (r - u, b - u), (r - u, b), (l + u, b), (l + u, b - u), (l, b - u)];
    shapes.push(egui::Shape::closed_line(pts.iter().map(|p| egui::pos2(p.0, p.1)).collect(), egui::Stroke::new(2.0, edge)));
    egui::Shape::Vec(shapes)
}

/// `Frame` with a fill, an optional border and rounded corners in the normal look; the
/// notched pixel panel in the pixel look.
fn card<R>(ui: &mut egui::Ui, fill: Color32, border: Option<(f32, Color32)>, margin: f32, radius: f32, shadow: bool, add: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R> {
    if !pixel_on() {
        let mut f = egui::Frame::new().fill(fill).corner_radius(radius).inner_margin(margin);
        if let Some((w, c)) = border {
            f = f.stroke(egui::Stroke::new(w, c));
        }
        return f.show(ui, add);
    }
    let idx = ui.painter().add(egui::Shape::Noop);
    let inner = egui::Frame::new().inner_margin(margin).show(ui, add);
    let edge = border.map(|b| b.1).unwrap_or_else(|| mix(fill, ui.visuals().text_color(), 0.30));
    ui.painter().set(idx, notched(inner.response.rect, fill, edge, shadow));
    inner
}

/// Paints a pixel sprite from rows of characters ('.' is clear, others look up `pal`) with each pixel
/// a whole number of screen pixels, so it stays crisp at any display scale.
fn paint_sprite(ui: &egui::Ui, at: egui::Pos2, rows: &[&str], pal: &[(char, Color32)], px: f32) {
    let k = ui.ctx().pixels_per_point();
    let (px, at) = ((px * k).round().max(1.0) / k, ((at.to_vec2() * k).round() / k).to_pos2());
    for (y, row) in rows.iter().enumerate() {
        for (x, ch) in row.chars().enumerate() {
            if let Some((_, c)) = pal.iter().find(|(k, _)| *k == ch) {
                ui.painter().rect_filled(egui::Rect::from_min_size(at + egui::vec2(x as f32 * px, y as f32 * px), egui::vec2(px, px)), 0.0, *c);
            }
        }
    }
}

fn sprite(ui: &mut egui::Ui, rows: &[&str], pal: &[(char, Color32)], px: f32) -> egui::Response {
    let w = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0) as f32 * px;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, rows.len() as f32 * px), egui::Sense::hover());
    paint_sprite(ui, rect.min, rows, pal, px);
    resp
}

/// A busy indicator: the round spinner normally, three blocks that light up in turn in pixel mode.
fn spin(ui: &mut egui::Ui) {
    if !pixel_on() {
        ui.spinner();
        return;
    }
    let t = ui.input(|i| i.time);
    let lit = ((t * 5.0) as usize) % 4; // 0..2 lights one block, 3 is a beat of rest
    let (rect, _) = ui.allocate_exact_size(egui::vec2(34.0, 12.0), egui::Sense::hover());
    for k in 0..3 {
        let c = if k == lit { ember() } else { mix(ui.visuals().extreme_bg_color, ember(), 0.25) };
        let r = egui::Rect::from_min_size(rect.min + egui::vec2(k as f32 * 12.0, 0.0), egui::vec2(10.0, 10.0));
        ui.painter().rect_filled(r, 0.0, c);
    }
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
}

const SAKURA: [&str; 9] = ["..pp.pp..", ".pwwpwwp.", "pwwwwwwwp", ".pwwyywp.", "pwwyyyywp", ".pwwyywp.", "pwwwwwwwp", ".pwwpwwp.", "..pp.pp.."];

fn sakura_sprite(ui: &mut egui::Ui) {
    sprite(ui, &SAKURA, &[('p', Color32::from_rgb(0xd9, 0x5f, 0x8f)), ('w', Color32::from_rgb(0xff, 0xc4, 0xd9)), ('y', Color32::from_rgb(0xff, 0xd1, 0x66))], 4.0);
}

/// The Hinoko star (the app icon), 16x16: 'w' starlight, 'b' and 'c' its shaded cells.
pub const STAR: [&str; 16] = [
    "................", "..............w.", "....w..w......b.", "....b..wc.......", ".......wc...c...", ".w....wwww..w...", ".b...cwwwww.....", "...wwwwcwwwcww..",
    "..bbbwwcwcwww...", ".....bwcwcw.....", "......bwww......", "....w..bw....w..", "..w.b..bw....b..", "..b.....b.w.....", "..........c.....", "................",
];

const STAR_PAL: [(char, Color32); 3] = [('w', Color32::WHITE), ('b', Color32::from_gray(236)), ('c', Color32::from_gray(250))];

fn logo(ui: &mut egui::Ui, px: f32) {
    if SAKURA_ON.load(std::sync::atomic::Ordering::Relaxed) {
        return sakura_sprite(ui);
    }
    sprite(ui, &STAR, &STAR_PAL, px);
}

pub fn apply_appearance(ctx: &egui::Context, look: &Appearance) {
    PIXEL.store(look.pixel, std::sync::atomic::Ordering::Relaxed);
    SAKURA_ON.store(look.theme.contains("sakura"), std::sync::atomic::Ordering::Relaxed);
    let (bg, panel_c, text, accent) = (rgb(look.background), rgb(look.panel), rgb(look.text), rgb(look.accent));
    let dark = luminance(bg) < 0.5;
    let p = Pal {
        ink: bg,
        panel: panel_c,
        raised: mix(panel_c, text, 0.07),
        text,
        muted: mix(text, bg, 0.45),
        accent,
        good: if dark { Color32::from_rgb(0x7f, 0xb0, 0x7a) } else { Color32::from_rgb(0x2f, 0x7a, 0x3a) },
    };
    *PAL.write().unwrap_or_else(|e| e.into_inner()) = p;
    ctx.set_theme(if dark { egui::Theme::Dark } else { egui::Theme::Light });
    let mut v = if dark { egui::Visuals::dark() } else { egui::Visuals::light() };
    v.panel_fill = bg;
    v.window_fill = panel_c;
    v.extreme_bg_color = mix(bg, if dark { Color32::BLACK } else { Color32::WHITE }, 0.25);
    v.faint_bg_color = panel_c;
    v.override_text_color = Some(text);
    v.hyperlink_color = accent;
    v.selection.bg_fill = accent.linear_multiply(0.55);
    v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, mix(bg, text, 0.18));
    v.widgets.inactive.bg_fill = p.raised;
    v.widgets.inactive.weak_bg_fill = p.raised;
    v.widgets.hovered.weak_bg_fill = mix(p.raised, text, 0.1);
    v.widgets.active.weak_bg_fill = accent.linear_multiply(0.7);
    let r = egui::CornerRadius::same(if look.pixel { 0 } else { look.rounding.round() as u8 });
    v.widgets.noninteractive.corner_radius = r;
    v.widgets.inactive.corner_radius = r;
    v.widgets.hovered.corner_radius = r;
    v.widgets.active.corner_radius = r;
    v.widgets.open.corner_radius = r;
    v.window_corner_radius = egui::CornerRadius::same(if look.pixel { 0 } else { (look.rounding + 2.0).round() as u8 });
    if look.pixel {
        // hard 2px borders, no soft shadows, a hard offset drop shadow like a sprite's
        let edge = mix(bg, text, 0.32);
        let hard = egui::Shadow { offset: [4, 4], blur: 0, spread: 0, color: Color32::from_black_alpha(190) };
        v.window_shadow = hard;
        v.popup_shadow = hard;
        v.window_stroke = egui::Stroke::new(2.0, accent);
        v.widgets.noninteractive.bg_stroke = egui::Stroke::new(2.0, mix(bg, text, 0.22));
        v.widgets.inactive.bg_stroke = egui::Stroke::new(2.0, edge);
        v.widgets.hovered.bg_stroke = egui::Stroke::new(2.0, accent);
        v.widgets.active.bg_stroke = egui::Stroke::new(2.0, text);
        v.widgets.hovered.expansion = 0.0;
        v.widgets.active.expansion = 0.0;
        v.menu_corner_radius = egui::CornerRadius::ZERO;
    }
    ctx.set_visuals(v);
    set_pixel_font(ctx, look.pixel_font);
    ctx.global_style_mut(|s| {
        s.spacing.item_spacing = egui::vec2(look.spacing, look.spacing);
        s.spacing.button_padding = if look.pixel { egui::vec2(12.0, 6.0) } else { egui::vec2(10.0, 5.0) };
        // the pixel font reads small, so its sizes are a little larger
        s.spacing.scroll = if look.pixel { let mut sc = egui::style::ScrollStyle::solid(); sc.bar_width = 10.0; sc } else { egui::style::ScrollStyle::default() };
        let (body, small, head, mono) = if look.pixel_font { (16.0, 13.0, 26.0, 14.0) } else { (14.5, 12.0, 22.0, 13.0) };
        use egui::{FontFamily::{Monospace, Proportional}, FontId, TextStyle};
        s.text_styles.insert(TextStyle::Body, FontId::new(body, Proportional));
        s.text_styles.insert(TextStyle::Button, FontId::new(body, Proportional));
        s.text_styles.insert(TextStyle::Small, FontId::new(small, Proportional));
        s.text_styles.insert(TextStyle::Heading, FontId::new(head, Proportional));
        s.text_styles.insert(TextStyle::Monospace, FontId::new(mono, Monospace));
    });
    ctx.set_zoom_factor(look.ui_scale);
}

/// Inter (soft, professional) is the default font; Pixelify Sans when the pixel font is on.
/// Both are SIL OFL. Rebuilt only when the choice changes.
fn set_pixel_font(ctx: &egui::Context, pixel_font: bool) {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0); // 0 = never set, 1 = Inter, 2 = pixel
    let want = if pixel_font { 2 } else { 1 };
    if STATE.swap(want, Ordering::Relaxed) == want {
        return;
    }
    let mut fonts = egui::FontDefinitions::default();
    let (main, ext) = if pixel_font {
        (include_bytes!("../../../assets/fonts/PixelifySans-latin.ttf").as_slice(), include_bytes!("../../../assets/fonts/PixelifySans-latin-ext.ttf").as_slice())
    } else {
        (include_bytes!("../../../assets/fonts/Inter-latin.ttf").as_slice(), include_bytes!("../../../assets/fonts/Inter-latin-ext.ttf").as_slice())
    };
    fonts.font_data.insert("ui".into(), std::sync::Arc::new(egui::FontData::from_static(main)));
    fonts.font_data.insert("ui-ext".into(), std::sync::Arc::new(egui::FontData::from_static(ext)));
    let fam = fonts.families.entry(egui::FontFamily::Proportional).or_default();
    fam.insert(0, "ui-ext".into());
    fam.insert(0, "ui".into());
    ctx.set_fonts(fonts);
}

impl HinokoApp {
    pub fn new(cc: &eframe::CreationContext<'_>, engine: Engine, rt: tokio::runtime::Runtime) -> Self {
        let look = engine.settings().appearance.sanitized();
        apply_appearance(&cc.egui_ctx, &look);
        let active = engine.store().bots.first().map(|b| b.id.clone());
        let mut app = Self {
            engine,
            _rt: rt,
            cook: Cook::default(),
            chat_drafts: HashMap::new(),
            pending_delete: None,
            editing_machine: None,
            machine_msg: Arc::new(Mutex::new(String::new())),
            pubkey: Arc::new(Mutex::new(None)),
            mode: if active.is_some() { Mode::Bots } else { Mode::Chats },
            active_chat: None,
            active_code: None,
            confirm_delete_code: false,
            applied: serde_json::to_string(&look).unwrap_or_default(),
            look,
            theme_text: String::new(),
            theme_msg: String::new(),
            skill_spec: String::new(),
            sidebar_open: true,
            skill_msg: Arc::new(Mutex::new(String::new())),
            active,
            drafts: HashMap::new(),
            tab: Tab::Files,
            dialog: Dialog::None,
            new_bot: NewBot::default(),
            routine_form: RoutineForm::default(),
            editing: None,
            test_result: Arc::new(Mutex::new(None)),
            testing: false,
            term_input: String::new(),
            mem_input: String::new(),
            open_file: None,
            toast: String::new(),
            stick_bottom: true,
            confirm_delete_bot: false,
            md: Default::default(),
            compare_prompt: String::new(),
            compare_pick: [None, None],
            compare: None,
            ctx: cc.egui_ctx.clone(),
            git_diff: None,
            activity_open: false,
            update: Default::default(),
            git: Arc::new(Mutex::new(HashMap::new())),
        };
        if app.engine.settings().auto_update_check {
            app.engine.update_check(app.update.clone());
        }
        // developer aid for screenshots: HINOKO_OPEN=settings | settings-edit | routines
        match std::env::var("HINOKO_OPEN").as_deref() {
            Ok("settings") => app.open_settings(),
            Ok("settings-edit") => {
                app.open_settings();
                app.start_edit(&PRESETS[4]);
            }
            Ok("routines") => app.dialog = Dialog::Routines,
            Ok("chats") => app.mode = Mode::Chats,
            Ok("md-demo") => {
                let id = app.engine.new_chat();
                {
                    let mut st = app.engine.store();
                    st.push_chat_msg(&id, "user", "Which division does Rokas play in?");
                    st.push_chat_msg(&id, "bot", "## KTML\n**Rokas Smilgius** plays in the *men's* division [[1]](https://ktml.lt/a).\n\n- League: Kauno Tinklinio Mėgėjų Lyga\n- Season: 2026/2027\n\n```\nlet x = 1;\n```\n\n**Sources**\n1. [Kauno liga](https://ktml.lt/a)\n");
                }
                app.active_chat = Some(id);
                app.mode = Mode::Chats;
            }
            Ok("chats-long") => {
                // a very long title at a bigger zoom, to check the sidebar holds
                let id = app.engine.new_chat();
                if let Some(c) = app.engine.store().chat_mut(&id) {
                    c.title = "How about him in the LIK volleyball team on the KTML page Rokas Smilgius".into();
                }
                app.engine.new_chat();
                app.active_chat = Some(id);
                app.look.ui_scale = 1.25;
                app.mode = Mode::Chats;
            }
            Ok("code") => app.mode = Mode::Code,
            Ok("code-demo") => {
                // sample conversation for screenshots; no model needed
                let dir = std::env::temp_dir().join("hinoko-demo-project");
                let _ = std::fs::create_dir_all(&dir);
                if let Ok(id) = app.engine.code_new(&dir.to_string_lossy()) {
                    {
                        let mut st = app.engine.store();
                        st.push_code(&id, "user", "Add a --verbose flag to the CLI and run the tests", "", "");
                        st.push_code(&id, "tool", "search \"clap\" (4 matches)", "src/main.rs:3: use clap::Parser;\nsrc/main.rs:9: #[derive(Parser)]", "");
                        st.push_code(&id, "tool", "read_file src/main.rs (42 lines)", "", "");
                        st.push_code(&id, "bot", "The CLI is built with clap. I'll add a `verbose` field and print extra detail when it's set.", "", "");
                        st.push_code(&id, "edit", "Edited src/main.rs", "--- a/src/main.rs\n+++ b/src/main.rs\n@@ -9,5 +9,8 @@\n struct Args {\n     name: String,\n+    /// Print extra detail\n+    #[arg(short, long)]\n+    verbose: bool,\n }\n", "");
                        let mid = st.push_code(&id, "ask", "Run: cargo test", "cargo test", "pending");
                        if let Some(m) = st.code_mut(&id).and_then(|c| c.messages.iter_mut().find(|m| m.id == mid)) {
                            m.approval_id = Some("demo".into());
                        }
                    }
                    app.active_code = Some(id);
                    app.mode = Mode::Code;
                }
            }
            Ok("compare-demo") => {
                use hinoko_core::compare::{Compare, Lane};
                let lane = |m: &str, t: &str, first: u64, total: u64| Lane { model: m.into(), text: t.into(), done: true, first_ms: first, total_ms: total, tokens: (t.len() as u64) / 4, ..Default::default() };
                app.compare_prompt = "Write a Rust function that reverses the words in a sentence.".into();
                app.compare = Some(Compare::sample(
                    &app.compare_prompt,
                    vec![
                        lane("Local · Spark-X2.6-4B", "Here is a simple version:\n\n```rust\nfn reverse_words(s: &str) -> String {\n    s.split_whitespace().rev().collect::<Vec<_>>().join(\" \")\n}\n```\n\nIt splits on whitespace, reverses the order and joins with single spaces.", 1800, 9400),
                        lane("OpenRouter · big-model", "```rust\nfn reverse_words(s: &str) -> String {\n    s.split_whitespace().rev().collect::<Vec<_>>().join(\" \")\n}\n```\nCollapses repeated spaces. Use `split(' ')` if you need to keep them.", 600, 2100),
                    ],
                ));
                app.dialog = Dialog::Compare;
            }
            Ok("cookbook-demo") => {
                // sample data for screenshots; no network
                use hinoko_core::local::*;
                let gf = |n: &str, gb: f64| GgufFile { name: n.into(), size: (gb * 1e9) as u64, quant: quant_of(n) };
                let hw = Hardware { os: "windows".into(), arch: "x86_64".into(), cpu: "AMD Ryzen 5 5600".into(), threads: 12, ram_gb: 32.0, gpus: vec![Gpu { name: "NVIDIA GeForce RTX 3060".into(), vram_gb: 12.0 }], unified: false };
                let mk = |repo: &str, p: f64, f: GgufFile, moe: bool, vision: bool, dl: u64| {
                    let ft = fit(&hw, f.gb(), moe);
                    Candidate { repo: repo.into(), params_b: Some(p), downloads: dl, likes: 0, moe, vision, best: f, fit: ft, others: 5 }
                };
                let items = vec![
                    mk("bartowski/google_gemma-4-E4B-it-GGUF", 8.0, gf("google_gemma-4-E4B-it-Q6_K.gguf", 6.3), false, true, 46516),
                    mk("bartowski/google_gemma-4-26B-A4B-it-GGUF", 26.0, gf("google_gemma-4-26B-A4B-it-Q4_K_M.gguf", 16.8), true, true, 66479),
                    mk("unsloth/Qwen3.8-27B-GGUF", 27.0, gf("Qwen3.8-27B-UD-Q4_K_XL.gguf", 17.1), false, true, 6042413),
                    mk("unsloth/flint-GGUF", 33.0, gf("flint-UD-Q4_K_XL.gguf", 25.0), false, false, 178000),
                ];
                *app.cook.hw.lock().unwrap() = Some(hw);
                app.cook.list.lock().unwrap().items = items;
                app.cook.started = true;
                app.dialog = Dialog::Cookbook;
            }
            _ => {}
        }
        app
    }

    fn open_cookbook(&mut self) {
        self.dialog = Dialog::Cookbook;
        if !self.cook.started {
            self.cook.started = true;
            let (hw_slot, list) = (self.cook.hw.clone(), self.cook.list.clone());
            let engine = self.engine.clone();
            list.lock().unwrap().loading = true;
            self.engine.detect_hardware_async(move |hw| {
                *hw_slot.lock().unwrap() = Some(hw.clone());
                let hw = engine.engine_hw(&hw);
                engine.discover_async(hw, String::new(), move |r| {
                    let mut l = list.lock().unwrap();
                    l.loading = false;
                    match r {
                        Ok(items) => {
                            l.items = items;
                            l.error.clear();
                        }
                        Err(e) => l.error = e,
                    }
                });
            });
        }
    }

    fn cook_search(&mut self, query: String) {
        let Some(hw) = self.cook.hw.lock().unwrap().clone() else { return };
        let hw = self.engine.engine_hw(&hw);
        let list = self.cook.list.clone();
        list.lock().unwrap().loading = true;
        self.engine.discover_async(hw, query, move |r| {
            let mut l = list.lock().unwrap();
            l.loading = false;
            match r {
                Ok(items) => {
                    l.items = items;
                    l.error.clear();
                }
                Err(e) => l.error = e,
            }
        });
    }

    fn open_settings(&mut self) {
        self.dialog = Dialog::Settings;
        self.editing = None;
    }

    // ---- left: roster ----
    /// The window's own title bar: mark and name on the left, controls on the right,
    /// drag anywhere else to move, double-click to maximise.
    fn titlebar(&mut self, ui: &mut egui::Ui, maximized: bool) {
        let ctx = ui.ctx().clone();
        let rect = ui.max_rect();
        let drag = ui.interact(rect, egui::Id::new("titlebar-drag"), egui::Sense::click_and_drag());
        if drag.drag_started_by(egui::PointerButton::Primary) {
            ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
        }
        if drag.double_clicked() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized));
        }
        let painter = ui.painter();
        // the mark: the Hinoko star
        let m = rect.left_center() + egui::vec2(10.0, -16.0);
        paint_sprite(ui, m, &STAR, &STAR_PAL, 2.0);
        painter.text(egui::pos2(m.x + 40.0, rect.center().y), egui::Align2::LEFT_CENTER, "Hinoko", egui::FontId::proportional(15.0), egui::Color32::from_rgb(0xf3, 0xe9, 0xec));
        // show or hide the sidebar (Ctrl+B)
        let sb = egui::Rect::from_center_size(egui::pos2(m.x + 112.0, rect.center().y), egui::vec2(30.0, 24.0));
        let sr = ui.interact(sb, egui::Id::new("sidebar-toggle"), egui::Sense::click()).on_hover_text("Show or hide the sidebar (Ctrl+B)");
        if sr.hovered() {
            ui.painter().rect_filled(sb, 4.0, mix(panel(), egui::Color32::WHITE, 0.08));
        }
        let (ic, st) = (egui::Rect::from_center_size(sb.center(), egui::vec2(14.0, 11.0)), egui::Stroke::new(1.3, if sr.hovered() { egui::Color32::WHITE } else { muted() }));
        ui.painter().rect_stroke(ic, 2.0, st, egui::StrokeKind::Middle);
        ui.painter().line_segment([ic.left_top() + egui::vec2(5.0, 0.0), ic.left_bottom() + egui::vec2(5.0, 0.0)], st);
        if self.sidebar_open {
            ui.painter().rect_filled(egui::Rect::from_min_max(ic.left_top() + egui::vec2(1.0, 1.0), ic.left_bottom() + egui::vec2(4.5, -1.0)), 0.0, st.color);
        }
        if sr.clicked() {
            self.sidebar_open = !self.sidebar_open;
        }
        // window buttons, right to left: close, maximise, minimise
        let bw = 44.0;
        let mut x = rect.right();
        for (i, kind) in ["close", "max", "min"].iter().enumerate() {
            let b = egui::Rect::from_min_max(egui::pos2(x - bw, rect.top()), egui::pos2(x, rect.bottom()));
            x -= bw;
            let r = ui.interact(b, egui::Id::new(("wb", i)), egui::Sense::click());
            let hot = r.hovered();
            if hot {
                ui.painter().rect_filled(b, 0.0, if *kind == "close" { egui::Color32::from_rgb(0xc4, 0x2b, 0x3c) } else { mix(panel(), egui::Color32::WHITE, 0.08) });
            }
            let c = b.center();
            let stroke = egui::Stroke::new(1.3, if hot { egui::Color32::WHITE } else { muted() });
            let p = ui.painter();
            match *kind {
                "close" => {
                    p.line_segment([c + egui::vec2(-5.0, -5.0), c + egui::vec2(5.0, 5.0)], stroke);
                    p.line_segment([c + egui::vec2(-5.0, 5.0), c + egui::vec2(5.0, -5.0)], stroke);
                }
                "max" => {
                    if maximized {
                        p.rect_stroke(egui::Rect::from_center_size(c + egui::vec2(-1.0, 1.0), egui::vec2(8.0, 8.0)), 0.0, stroke, egui::StrokeKind::Middle);
                        p.line_segment([c + egui::vec2(-1.0, -5.0), c + egui::vec2(5.0, -5.0)], stroke);
                        p.line_segment([c + egui::vec2(5.0, -5.0), c + egui::vec2(5.0, 3.0)], stroke);
                    } else {
                        p.rect_stroke(egui::Rect::from_center_size(c, egui::vec2(10.0, 10.0)), 0.0, stroke, egui::StrokeKind::Middle);
                    }
                }
                _ => {
                    p.line_segment([c + egui::vec2(-5.0, 0.0), c + egui::vec2(5.0, 0.0)], stroke);
                }
            }
            if r.clicked() {
                match *kind {
                    "close" => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
                    "max" => ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized)),
                    _ => ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true)),
                }
            }
        }
    }

    fn roster(&mut self, ui: &mut egui::Ui) {
        ui.add_space(2.0);
        if let hinoko_core::update::Update::Available(rel) = self.update_state() {
            if ui.small_button(format!("\u{2191} Update available: {}", rel.tag)).clicked() {
                self.open_settings();
            }
        }
        ui.horizontal(|ui| {
            logo(ui, 2.0);
            ui.add_space(4.0);
            ui.label(RichText::new("Hinoko").heading().color(ember()));
        });
        ui.add_space(6.0);
        // Chats | Bots as one segmented control
        let w = (ui.available_width() - 12.0) / 3.0;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            for (m, label) in [(Mode::Chats, "Chats"), (Mode::Bots, "Bots"), (Mode::Code, "Code")] {
                if ui.add_sized([w, 30.0], egui::Button::selectable(self.mode == m, label)).clicked() {
                    self.mode = m;
                }
            }
        });
        ui.add_space(6.0);
        let list_h = (ui.available_height() - 120.0).max(60.0);
        match self.mode {
            Mode::Bots => self.bot_list(ui, list_h),
            Mode::Chats => self.chat_list(ui, list_h),
            Mode::Code => self.code_list(ui, list_h),
        }
        ui.add_space(6.0);
        if self.mode == Mode::Bots {
            if ui.add_sized([ui.available_width(), 32.0], egui::Button::new("+ New bot")).clicked() {
                self.new_bot = NewBot::default();
                self.dialog = Dialog::NewBot;
            }
        } else if self.mode == Mode::Code {
            if ui.add_sized([ui.available_width(), 32.0], egui::Button::new("+ Open a project")).clicked() {
                self.pick_project();
            }
        } else {
            if ui.add_sized([ui.available_width(), 32.0], egui::Button::new("+ New chat")).clicked() {
                self.active_chat = Some(self.engine.new_chat());
                self.stick_bottom = true;
            }
            if ui.add_sized([ui.available_width(), 26.0], egui::Button::new("⇄ Compare models")).on_hover_text("Ask two models the same question and watch both answer").clicked() {
                self.dialog = Dialog::Compare;
            }
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            let w = (ui.available_width() - 12.0 - 30.0) / 2.0;
            if ui.add_sized([w, 28.0], egui::Button::new("Routines")).clicked() {
                self.dialog = Dialog::Routines;
            }
            if ui.add_sized([w, 28.0], egui::Button::new("⚙ Settings")).clicked() {
                self.open_settings();
            }
            if ui.add_sized([30.0, 28.0], egui::Button::new("?")).clicked() {
                self.dialog = Dialog::About;
            }
        });
    }

    fn chat_list(&mut self, ui: &mut egui::Ui, height: f32) {
        let chats: Vec<(String, String)> = self.engine.store().chats.iter().map(|c| (c.id.clone(), c.title.clone())).collect();
        let mut delete: Option<String> = None;
        egui::ScrollArea::vertical().max_height(height).show(ui, |ui| {
            for (id, title) in &chats {
                let selected = self.active_chat.as_deref() == Some(id.as_str());
                let busy = self.engine.is_chat_busy(id);
                if self.pending_delete.as_deref() == Some(id.as_str()) {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Delete this chat?").color(ember()));
                        if ui.small_button("Delete").clicked() {
                            delete = Some(id.clone());
                        }
                        if ui.small_button("Keep").clicked() {
                            self.pending_delete = None;
                        }
                    });
                    continue;
                }
                let label = if busy { format!("{title}  …") } else { title.clone() };
                let w = ui.available_width();
                let r = ui
                    .horizontal(|ui| {
                        let r = ui.add_sized([(w - 36.0).max(40.0), 30.0], egui::Button::selectable(selected, label).truncate());
                        if ui.add_sized([24.0, 30.0], egui::Button::new(RichText::new("✕").color(muted())).frame(false)).on_hover_text("Delete chat").clicked() {
                            self.pending_delete = Some(id.clone());
                        }
                        r
                    })
                    .inner;
                if r.clicked() {
                    self.active_chat = Some(id.clone());
                    self.stick_bottom = true;
                }
                r.context_menu(|ui| {
                    if ui.button("Rename…").clicked() {
                        self.active_chat = Some(id.clone());
                        self.dialog = Dialog::ChatOptions;
                        ui.close();
                    }
                    if ui.button("Delete").clicked() {
                        self.pending_delete = Some(id.clone());
                        ui.close();
                    }
                });
            }
            if chats.is_empty() {
                ui.label(RichText::new("No chats yet. Start one — it's just you and a model.").color(muted()));
            }
        });
        if let Some(id) = delete {
            self.engine.delete_chat(&id);
            self.pending_delete = None;
            if self.active_chat.as_deref() == Some(id.as_str()) {
                self.active_chat = None;
            }
        }
    }

    fn bot_list(&mut self, ui: &mut egui::Ui, height: f32) {
        let bots: Vec<(String, String, String, String)> = self
            .engine
            .store()
            .bots
            .iter()
            .map(|b| (b.id.clone(), b.name.clone(), b.title.clone(), b.presence.clone()))
            .collect();
        egui::ScrollArea::vertical().max_height(height).show(ui, |ui| {
            for (id, name, title, presence) in &bots {
                let busy = self.engine.is_busy(id);
                let selected = self.active.as_deref() == Some(id.as_str());
                let (dot, label) = if busy {
                    (ember(), "working")
                } else {
                    match presence.as_str() {
                        "waiting" => (ember(), "needs you"),
                        "done" => (good(), "caught up"),
                        _ => (muted(), "idle"),
                    }
                };
                let text = format!("{name}\n{title} · {label}");
                let resp = ui.add_sized([ui.available_width(), 46.0], egui::Button::selectable(selected, text).truncate());
                let p = resp.rect.right_center() - egui::vec2(12.0, 0.0);
                ui.painter().circle_filled(p, 4.0, dot);
                if resp.clicked() {
                    self.active = Some(id.clone());
                    self.stick_bottom = true;
                    self.open_file = None;
                    self.confirm_delete_bot = false;
                }
            }
            if bots.is_empty() {
                ui.label(RichText::new("No bots yet.").color(muted()));
            }
        });
    }

    /// Draft box plus Send/Stop. The draft is kept per `key`. For chats (`chat`),
    /// a toolbar adds attach/tools on the left and thinking mode on the right.
    fn composer(&mut self, ui: &mut egui::Ui, key: &str, hint: &str, busy: bool, footer: &str, chat: bool) -> Composed {
        let draft_id = ui.make_persistent_id(("draft", key));
        let has_focus = ui.memory(|m| m.has_focus(draft_id));
        let enter_sends = self.look.enter_sends;
        let pressed = has_focus
            && ui.input_mut(|i| {
                if enter_sends {
                    !i.modifiers.shift && i.consume_key(Modifiers::NONE, Key::Enter)
                } else {
                    i.consume_key(Modifiers::COMMAND, Key::Enter)
                }
            });
        let mut out = Composed::Nothing;
        let mut cd = self.chat_drafts.remove(key).unwrap_or_default();
        if cd.effort.is_empty() {
            cd.effort = "auto".into();
        }
        let mut attach_error = String::new();
        card(ui, raised(), None, 10.0, self.look.rounding + 6.0, true, |ui| {
            if chat && !cd.files.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    let mut drop_at: Option<usize> = None;
                    for (i, f) in cd.files.iter().enumerate() {
                        if ui.small_button(format!("{}  ✕", f.name)).on_hover_text("Remove").clicked() {
                            drop_at = Some(i);
                        }
                    }
                    if let Some(i) = drop_at {
                        cd.files.remove(i);
                    }
                });
            }
            let draft = self.drafts.entry(key.to_string()).or_default();
            ui.add(
                egui::TextEdit::multiline(draft)
                    .id(draft_id)
                    .frame(egui::Frame::NONE)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY)
                    .hint_text(hint),
            );
            let can_send = !busy && (!draft.trim().is_empty() || (chat && !cd.files.is_empty()));
            ui.horizontal(|ui| {
                if chat {
                    ui.menu_button(RichText::new(" +  ").size(20.0).strong(), |ui| {
                        if ui.button("Attach files…").on_hover_text("Text files are read into the chat; images are shown to the model").clicked() {
                            if let Some(paths) = rfd::FileDialog::new().set_title("Attach files").pick_files() {
                                for p in paths {
                                    match read_attachment(&p) {
                                        Ok(a) => cd.files.push(a),
                                        Err(e) => attach_error = e,
                                    }
                                }
                            }
                            ui.close();
                        }
                        ui.separator();
                        ui.checkbox(&mut cd.web, "Web search")
                            .on_hover_text("Search the web first and let the answer cite it");
                        ui.checkbox(&mut cd.research, "Deep research")
                            .on_hover_text("Plans several searches, reads the pages, and writes a report with numbered sources. Slower.");
                    })
                    .response
                    .on_hover_text("Attach files, pick tools");
                    if cd.web && ui.small_button("Web  ✕").clicked() {
                        cd.web = false;
                    }
                    if cd.research && ui.small_button("Deep research  ✕").clicked() {
                        cd.research = false;
                    }
                }
                ui.label(RichText::new(footer).color(muted()).small());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if busy {
                        if ui.button("■ Stop").clicked() {
                            out = Composed::Stop;
                        }
                    } else if ui.add_enabled(can_send, egui::Button::new("Send ↑").fill(ember())).clicked() || (pressed && can_send) {
                        out = Composed::Send(std::mem::take(self.drafts.entry(key.to_string()).or_default()));
                    }
                    if chat {
                        let names = [("auto", "Auto"), ("off", "Off"), ("low", "Low"), ("medium", "Medium"), ("high", "High")];
                        let shown = names.iter().find(|(k, _)| *k == cd.effort).map(|(_, n)| *n).unwrap_or("Auto");
                        egui::ComboBox::from_id_salt(("effort", key)).width(96.0).selected_text(format!("Think: {shown}")).show_ui(ui, |ui| {
                            for (k, n) in names {
                                ui.selectable_value(&mut cd.effort, k.to_string(), n);
                            }
                        })
                        .response
                        .on_hover_text("How hard the model thinks before answering. Only models that support it react; Auto leaves it to the model.");
                    }
                });
            });
        });
        if chat {
            // drag a file onto the window to attach it
            let dropped: Vec<std::path::PathBuf> = ui.ctx().input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
            for p in dropped {
                match read_attachment(&p) {
                    Ok(a) => cd.files.push(a),
                    Err(e) => attach_error = e,
                }
            }
        }
        if !attach_error.is_empty() {
            self.toast = attach_error;
        }
        self.chat_drafts.insert(key.to_string(), cd);
        out
    }

    /// The reading column: centred, limited in width if the person asked for it.
    fn column<R>(chat_width: f32, ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
        let avail = ui.available_width();
        let w = if chat_width > 0.0 { chat_width.min(avail) } else { avail };
        let pad = ((avail - w) / 2.0).max(0.0);
        ui.horizontal_top(|ui| {
            ui.add_space(pad);
            ui.vertical(|ui| {
                ui.set_width(w);
                add(ui)
            })
            .inner
        })
        .inner
    }

    /// Text with ``` fenced code blocks, selectable, with a Copy button on code.
    fn rich_text(&self, ui: &mut egui::Ui, text: &str) {
        // a trailing "Sources" list is drawn as real, clearly clickable links
        let (text, sources) = split_sources(text);
        self.rich_body(ui, &text);
        if !sources.is_empty() {
            ui.add_space(4.0);
            ui.label(RichText::new("Sources").color(muted()).small());
            for (n, title, url) in sources {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(format!("{n}.")).color(muted()));
                    ui.hyperlink_to(RichText::new(title).underline().color(egui::Color32::from_rgb(0xff, 0x8f, 0xa3)), url);
                });
            }
        }
    }

    fn rich_body(&self, ui: &mut egui::Ui, text: &str) {
        let mono = self.look.monospace_chat;
        let mut code = false;
        let mut buf = String::new();
        let flush = |ui: &mut egui::Ui, buf: &mut String, code: bool| {
            let body = buf.trim_end_matches('\n').to_string();
            buf.clear();
            if body.trim().is_empty() {
                return;
            }
            if code {
                card(ui, ui.visuals().extreme_bg_color, None, 10.0, 8.0, false, |ui| {
                    ui.set_width(ui.available_width());
                    ui.add(egui::Label::new(RichText::new(&body).monospace()).selectable(true));
                    if ui.small_button("Copy").clicked() {
                        ui.ctx().copy_text(body.clone());
                    }
                });
            } else if mono {
                ui.add(egui::Label::new(RichText::new(&body).monospace()).selectable(true));
            } else {
                // bold, lists, headings and clickable links
                egui_commonmark::CommonMarkViewer::new().show(ui, &mut self.md.borrow_mut(), &body);
            }
        };
        for line in text.split('\n') {
            if line.trim_start().starts_with("```") {
                flush(ui, &mut buf, code);
                code = !code;
                continue;
            }
            buf.push_str(line);
            buf.push('\n');
        }
        flush(ui, &mut buf, code);
    }

    // ---- code sessions ----
    fn pick_project(&mut self) {
        if let Some(dir) = rfd::FileDialog::new().set_title("Choose the project folder").pick_folder() {
            match self.engine.code_new(&dir.to_string_lossy()) {
                Ok(id) => {
                    self.active_code = Some(id);
                    self.stick_bottom = true;
                    self.toast.clear();
                }
                Err(e) => self.toast = e,
            }
        }
    }

    fn code_list(&mut self, ui: &mut egui::Ui, height: f32) {
        let items: Vec<(String, String, String)> = self.engine.store().code.iter().map(|c| (c.id.clone(), c.title.clone(), c.dir.clone())).collect();
        // sessions run side by side: which are working, and which are waiting for you
        let waiting_ids: Vec<String> = self.engine.store().code.iter().filter(|c| c.messages.iter().any(|m| m.kind == "ask" && m.state == "pending")).map(|c| c.id.clone()).collect();
        let working: Vec<String> = items.iter().filter(|i| self.engine.is_code_busy(&i.0)).map(|i| i.0.clone()).collect();
        if working.len() + waiting_ids.len() > 0 {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} working \u{b7} {} need you", working.len(), waiting_ids.len())).small().color(muted()));
                if working.len() > 1 && ui.small_button("Stop all").clicked() {
                    for id in &working {
                        self.engine.code_stop(id);
                    }
                }
            });
            ui.add_space(2.0);
        }
        egui::ScrollArea::vertical().max_height((height - 22.0).max(40.0)).show(ui, |ui| {
            for (id, title, dir) in &items {
                let selected = self.active_code.as_deref() == Some(id.as_str());
                let busy = self.engine.is_code_busy(id);
                let mark = if waiting_ids.contains(id) { "? " } else if busy { "\u{25cf} " } else { "" };
                let label = format!("{mark}{}\n{}", title, clip_path(dir, 30));
                let r = ui.add_sized([ui.available_width(), 44.0], egui::Button::selectable(selected, label).truncate());
                if r.clicked() {
                    self.active_code = Some(id.clone());
                    self.stick_bottom = true;
                    self.confirm_delete_code = false;
                }
                r.on_hover_text(dir);
            }
            if items.is_empty() {
                ui.label(RichText::new("No projects yet. Open a folder and tell the agent what to build or fix.").color(muted()));
            }
        });
    }

    fn code_view(&mut self, ui: &mut egui::Ui) {
        let Some(id) = self.active_code.clone() else {
            ui.add_space(90.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("Code").size(28.0).strong());
                ui.add_space(6.0);
                ui.label(RichText::new("A coding agent that works inside one project folder, with the model you pick: API, local or CLI.").color(muted()));
                ui.label(RichText::new("It reads and searches your code, and asks before it edits files or runs commands.").color(muted()));
                ui.add_space(16.0);
                if ui.button("+ Open a project").clicked() {
                    self.pick_project();
                }
                if !self.toast.is_empty() {
                    ui.colored_label(ember(), &self.toast);
                }
            });
            return;
        };
        let Some(sess) = self.engine.store().code_session(&id).cloned() else {
            self.active_code = None;
            return;
        };
        let busy = self.engine.is_code_busy(&id);
        let settings = self.engine.settings();
        let using = settings.provider_for(sess.provider.as_deref()).map(|p| (p.name.clone(), p.kind == Kind::Cli));

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new(&sess.title).size(18.0).strong());
                ui.label(RichText::new(&sess.dir).color(muted()).small());
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if !self.confirm_delete_code {
                    if ui.button("Remove").on_hover_text("Forget this session. Your files are not touched.").clicked() {
                        self.confirm_delete_code = true;
                    }
                } else {
                    if ui.button("Yes, remove").clicked() {
                        self.engine.code_delete(&id);
                        self.active_code = None;
                        self.confirm_delete_code = false;
                    }
                    if ui.button("Keep").clicked() {
                        self.confirm_delete_code = false;
                    }
                }
                if ui.selectable_label(self.activity_open, "Activity").on_hover_text("Every file touched and command run in this session").clicked() {
                    self.activity_open = !self.activity_open;
                }
                if ui.add_enabled(!busy, egui::Button::new("Compact")).on_hover_text("Summarize the older messages to free up room for the model").clicked() {
                    self.engine.compact_code_async(id.clone());
                }
                if ui.add_enabled(!busy && self.engine.code_can_undo(&id), egui::Button::new("Undo request")).on_hover_text("Revert every file the last request changed").clicked() {
                    if let Err(e) = self.engine.code_undo_run(&id) {
                        self.toast = e;
                    }
                }
                if ui.add_enabled(!busy && self.engine.code_can_undo(&id), egui::Button::new("Undo step")).on_hover_text("Revert only the newest change").clicked() {
                    if let Err(e) = self.engine.code_undo(&id) {
                        self.toast = e;
                    }
                }
                let mut changed = false;
                let mut s2 = sess.clone();
                let label = match s2.autonomy.as_str() {
                    "all" => "Auto-run everything",
                    "edits" => "Auto-accept edits",
                    _ => "Ask before changes",
                };
                egui::ComboBox::from_id_salt(("autonomy", &id)).selected_text(label).show_ui(ui, |ui| {
                    changed |= ui.selectable_value(&mut s2.autonomy, "ask".into(), "Ask before changes").changed();
                    changed |= ui.selectable_value(&mut s2.autonomy, "edits".into(), "Auto-accept edits (commands still ask)").changed();
                    changed |= ui.selectable_value(&mut s2.autonomy, "all".into(), "Auto-run everything (careful)").changed();
                });
                ui.add_space(4.0);
                for (m, name) in [("build", "Build"), ("plan", "Plan")] {
                    if ui.selectable_label(s2.mode == m, name).on_hover_text(if m == "plan" { "Read-only: explore the code and make a plan" } else { "Can edit files and run commands, with your OK" }).clicked() {
                        s2.mode = m.into();
                        changed = true;
                    }
                }
                ui.add_space(4.0);
                let current = using.as_ref().map(|u| u.0.clone()).unwrap_or_else(|| "no model".into());
                egui::ComboBox::from_id_salt(("code-model", &id)).selected_text(current).show_ui(ui, |ui| {
                    changed |= ui.selectable_value(&mut s2.provider, None, "Desk default").changed();
                    for p in &settings.providers {
                        changed |= ui.selectable_value(&mut s2.provider, Some(p.id.clone()), &p.name).changed();
                    }
                });
                self.cli_model_picker(ui, &id, s2.provider.as_deref());
                if changed {
                    if let Some(c) = self.engine.store().code_mut(&id) {
                        c.provider = s2.provider;
                        c.mode = s2.mode;
                        c.autonomy = s2.autonomy;
                    }
                    self.engine.save();
                }
            });
        });
        if let Some(g) = self.git_status(&sess.dir, busy) {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("⎇ {}", g.branch)).monospace().small().color(good()));
                let n = g.changed.len();
                let text = if n == 0 { "clean".to_string() } else { format!("{n} file{} changed", if n == 1 { "" } else { "s" }) };
                let resp = ui.label(RichText::new(text).small().color(muted()));
                if n > 0 {
                    let names: Vec<String> = g.changed.iter().take(12).map(|(st, f)| format!("{} {f}", st.trim())).collect();
                    resp.on_hover_text(names.join("\n"));
                    if ui.small_button("Diff").clicked() {
                        let d = hinoko_core::git::diff(&sess.dir, 200_000);
                        self.git_diff = Some((sess.title.clone(), d));
                    }
                }
            });
        }
        if sess.autonomy == "all" {
            ui.colored_label(ember(), "Auto-run everything: the agent can run any command in this folder without asking.");
        }
        if using.as_ref().is_some_and(|u| u.1) {
            ui.label(RichText::new("CLI agents use their own tools outside Hinoko's approvals.").color(muted()).small());
        }
        ui.separator();

        let area = (ui.available_height() - 110.0).max(80.0);
        let mut decide: Option<(String, u8)> = None;
        egui::ScrollArea::vertical().max_height(area).auto_shrink([false, false]).stick_to_bottom(self.stick_bottom).show(ui, |ui| {
            Self::column(self.look.chat_width.max(0.0), ui, |ui| {
                if sess.messages.is_empty() {
                    ui.add_space(24.0);
                    ui.label(RichText::new("Tell the agent what to build or fix. Try: \"Explain how this project is organised\" (Plan mode) or \"Add a --verbose flag and run the tests\" (Build mode).").color(muted()));
                }
                let last = sess.messages.len().saturating_sub(1);
                for (i, m) in sess.messages.iter().enumerate() {
                    match m.kind.as_str() {
                        "user" => self.user_bubble(ui, &m.text),
                        "bot" => {
                            ui.add_space(4.0);
                            self.rich_text(ui, &m.text);
                            if !m.detail.is_empty() {
                                ui.label(RichText::new(&m.detail).small().color(muted()));
                            }
                        }
                        "error" => {
                            ui.colored_label(ember(), &m.text);
                        }
                        "event" => {
                            ui.label(RichText::new(format!("· {}", m.text)).monospace().color(muted()).small());
                        }
                        "tool" => {
                            let failed = m.state == "failed";
                            let line = RichText::new(if m.detail.is_empty() { format!("▸ {}", m.text) } else { m.text.clone() }).monospace().small().color(if failed { ember() } else { muted() });
                            if m.detail.is_empty() {
                                ui.label(line);
                            } else {
                                egui::CollapsingHeader::new(line).id_salt(("tool", &m.id)).show(ui, |ui| {
                                    ui.add(egui::Label::new(RichText::new(&m.detail).monospace().small().color(muted())).selectable(true));
                                });
                            }
                        }
                        "edit" => {
                            egui::CollapsingHeader::new(RichText::new(format!("✎ {}", m.text)).monospace().small().color(good())).id_salt(("edit", &m.id)).show(ui, |ui| {
                                diff_view(ui, &m.detail);
                            });
                        }
                        "cmd" => {
                            let ok = m.state == "0";
                            egui::CollapsingHeader::new(RichText::new(format!("$ {}", clip_path(&m.text, 90))).monospace().small().color(if ok { good() } else { ember() }))
                                .id_salt(("cmd", &m.id))
                                .show(ui, |ui| {
                                    ui.add(egui::Label::new(RichText::new(&m.detail).monospace().small()).selectable(true));
                                });
                        }
                        "ask" => {
                            let pending = m.state == "pending";
                            card(ui, raised(), Some((if pixel_on() { 2.0 } else { 1.0 }, if pending { ember() } else { raised() })), 10.0, self.look.rounding + 4.0, true, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.label(RichText::new(&m.text).strong());
                                    if m.text.starts_with("Run:") {
                                        ui.add(egui::Label::new(RichText::new(&m.detail).monospace()).selectable(true));
                                    } else if pending {
                                        diff_view(ui, &m.detail);
                                    }
                                    if pending {
                                        ui.horizontal(|ui| {
                                            if let Some(a) = &m.approval_id {
                                                if ui.add(egui::Button::new("Allow").fill(ember())).clicked() {
                                                    decide = Some((a.clone(), 1));
                                                }
                                                let always = if m.text.starts_with("Run:") { "Always allow this command" } else { "Allow all edits" };
                                                if ui.button(always).clicked() {
                                                    decide = Some((a.clone(), 2));
                                                }
                                                if ui.button("Deny").clicked() {
                                                    decide = Some((a.clone(), 0));
                                                }
                                            }
                                        });
                                    } else {
                                        let (t, c) = match m.state.as_str() {
                                            "denied" => ("Denied", ember()),
                                            "always" => ("Allowed (always)", good()),
                                            _ => ("Allowed", good()),
                                        };
                                        ui.label(RichText::new(t).color(c).small());
                                    }
                                });
                        }
                        _ => {}
                    }
                    if busy && i == last && !matches!(m.kind.as_str(), "ask") {
                        ui.horizontal(|ui| {
                            spin(ui);
                            ui.label(RichText::new("working…").color(muted()).small());
                        });
                    }
                }
            });
        });
        if let Some((a, c)) = decide {
            self.engine.code_decide(&a, c);
        }
        ui.add_space(4.0);
        let hint = if sess.mode == "plan" { "Ask about the code or request a plan…" } else { "Describe what to build or fix…" };
        let footer = using.map(|u| format!("{} · {}", u.0, if sess.mode == "plan" { "Plan" } else { "Build" })).unwrap_or_else(|| "no model set up".into());
        let key = format!("code:{id}");
        match self.composer(ui, &key, hint, busy, &footer, false) {
            Composed::Send(text) => match self.engine.code_send(&id, &text) {
                Ok(()) => {
                    self.stick_bottom = true;
                    self.toast.clear();
                }
                Err(e) => {
                    self.drafts.insert(key, text);
                    self.toast = e.to_string();
                }
            },
            Composed::Stop => self.engine.code_stop(&id),
            Composed::Nothing => {}
        }
        if !self.toast.is_empty() {
            ui.colored_label(ember(), &self.toast);
        }
    }

    // ---- centre: a plain chat ----
    /// For command-line agents that take a model name (Claude Code): a Haiku / Sonnet / Opus menu.
    fn cli_model_picker(&self, ui: &mut egui::Ui, salt: &str, provider: Option<&str>) {
        let settings = self.engine.settings();
        let Some(p) = settings.provider_for(provider).filter(|p| p.kind == Kind::Cli) else { return };
        let list = hinoko_core::settings::cli_models(&p.command);
        if list.is_empty() {
            return;
        }
        let shown = list.iter().find(|(v, _)| *v == p.model).map_or(p.model.clone(), |(_, l)| l.to_string());
        let mut pick = p.model.clone();
        egui::ComboBox::from_id_salt(("cli-model", salt)).selected_text(shown).show_ui(ui, |ui| {
            for (v, l) in list {
                ui.selectable_value(&mut pick, v.to_string(), *l);
            }
        });
        if pick != p.model {
            let mut s = settings.clone();
            if let Some(q) = s.providers.iter_mut().find(|q| q.id == p.id) {
                q.model = pick;
            }
            self.engine.set_settings(s);
        }
    }

    fn chat_view(&mut self, ui: &mut egui::Ui) {
        let Some(id) = self.active_chat.clone() else {
            self.chat_welcome(ui);
            return;
        };
        let Some(chat) = self.engine.store().chat(&id).cloned() else {
            self.active_chat = None;
            return;
        };
        let busy = self.engine.is_chat_busy(&id);
        let settings = self.engine.settings();
        let using = settings.provider_for(chat.provider.as_deref()).map(|p| p.name.clone());

        ui.horizontal(|ui| {
            ui.label(RichText::new(&chat.title).size(18.0).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Options").clicked() {
                    self.dialog = Dialog::ChatOptions;
                    self.confirm_delete_bot = false;
                }
                let current = using.clone().unwrap_or_else(|| "no model".into());
                let mut pick = chat.provider.clone();
                egui::ComboBox::from_id_salt(("chat-model", &id)).selected_text(current).show_ui(ui, |ui| {
                    ui.selectable_value(&mut pick, None, "Desk default");
                    for p in &settings.providers {
                        ui.selectable_value(&mut pick, Some(p.id.clone()), &p.name);
                    }
                });
                self.cli_model_picker(ui, &id, chat.provider.as_deref());
                if pick != chat.provider {
                    if let Some(c) = self.engine.store().chat_mut(&id) {
                        c.provider = pick;
                    }
                    self.engine.save();
                }
            });
        });
        ui.separator();

        let area = (ui.available_height() - 150.0).max(80.0);
        egui::ScrollArea::vertical().max_height(area).auto_shrink([false, false]).stick_to_bottom(self.stick_bottom).show(ui, |ui| {
            Self::column(self.look.chat_width, ui, |ui| {
                if chat.messages.is_empty() {
                    ui.add_space(30.0);
                    ui.label(RichText::new("Ask anything. Pick a different model from the menu above at any time.").color(muted()));
                }
                let last = chat.messages.len().saturating_sub(1);
                for (i, m) in chat.messages.iter().enumerate() {
                    match m.kind.as_str() {
                        "user" => {
                            self.user_bubble(ui, &m.text);
                            if !m.files.is_empty() {
                                ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
                                    ui.label(RichText::new(format!("Attached: {}", m.files.join(", "))).color(muted()).small());
                                });
                            }
                        }
                        "error" => {
                            ui.colored_label(ember(), &m.text);
                        }
                        "event" => {
                            ui.label(RichText::new(format!("· {}", m.text)).monospace().color(muted()).small());
                        }
                        _ => {
                            ui.add_space(4.0);
                            if !m.think.is_empty() {
                                egui::CollapsingHeader::new(RichText::new(if m.text.is_empty() && busy && i == last { "Thinking…" } else { "Thought process" }).color(muted()).small())
                                    .id_salt(("think", &m.id))
                                    .show(ui, |ui| {
                                        ui.add(egui::Label::new(RichText::new(&m.think).color(muted()).small()).selectable(true));
                                    });
                            }
                            if m.text.is_empty() && busy && i == last {
                                if m.think.is_empty() {
                                    spin(ui);
                                }
                            } else {
                                self.rich_text(ui, &m.text);
                                ui.horizontal(|ui| {
                                    if ui.small_button("Copy").clicked() {
                                        ui.ctx().copy_text(m.text.clone());
                                    }
                                    if !m.detail.is_empty() {
                                        ui.label(RichText::new(&m.detail).small().color(muted())).on_hover_text("Writing speed, measured from the first piece of output. A ~ means Hinoko estimated the token count because the server didn't report one.");
                                    }
                                });
                            }
                        }
                    }
                }
            });
        });
        ui.add_space(4.0);
        let hint = if self.look.enter_sends { "Message  (Enter sends, Shift+Enter for a new line)" } else { "Message  (Ctrl+Enter sends)" };
        let footer = using.map(|n| format!("answering with {n}")).unwrap_or_else(|| "no model set up".into());
        let key = format!("chat:{id}");
        match self.composer(ui, &key, hint, busy, &footer, true) {
            Composed::Send(text) => {
                let (opts, files_back) = {
                    let cd = self.chat_drafts.entry(key.clone()).or_default();
                    let files = std::mem::take(&mut cd.files);
                    let opts = hinoko_core::ChatOpts { research: cd.research, web: cd.web, effort: cd.effort.clone(), files: files.clone() };
                    (opts, files)
                };
                match self.engine.chat_send_with(&id, &text, opts) {
                    Ok(()) => {
                        self.stick_bottom = true;
                        self.toast.clear();
                        if let Some(cd) = self.chat_drafts.get_mut(&key) {
                            cd.research = false;
                        }
                    }
                    Err(e) => {
                        self.drafts.insert(key.clone(), text);
                        if let Some(cd) = self.chat_drafts.get_mut(&key) {
                            cd.files = files_back;
                        }
                        self.toast = e.to_string();
                    }
                }
            }
            Composed::Stop => self.engine.stop_chat(&id),
            Composed::Nothing => {}
        }
        if !self.toast.is_empty() {
            ui.colored_label(ember(), &self.toast);
        }
    }

    fn chat_welcome(&mut self, ui: &mut egui::Ui) {
        ui.add_space(90.0);
        ui.vertical_centered(|ui| {
            logo(ui, 6.0);
            ui.add_space(10.0);
            ui.label(RichText::new("What's on your mind?").size(28.0).strong());
            ui.add_space(6.0);
            ui.label(RichText::new("Chat with any model you've connected: API, local, or CLI.").color(muted()));
            ui.add_space(16.0);
            if self.engine.settings().provider_for(None).is_none() {
                ui.label("Connect a model first.");
                ui.horizontal(|ui| {
                    ui.add_space((ui.available_width() / 2.0 - 150.0).max(0.0));
                    if ui.button("Run a model on this PC").on_hover_text("Hinoko checks your hardware, suggests models that fit and runs them for you. Nothing else to install.").clicked() {
                        self.open_cookbook();
                    }
                    if ui.button("⚙ Add an API key").clicked() {
                        self.open_settings();
                    }
                });
            } else if ui.button("+ New chat").clicked() {
                self.active_chat = Some(self.engine.new_chat());
            }
        });
    }

    fn user_bubble(&self, ui: &mut egui::Ui, text: &str) {
        ui.add_space(6.0);
        ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
            let max = (ui.available_width() * 0.82).max(120.0);
            card(ui, mix(raised(), ember(), 0.22), None, 10.0, self.look.rounding + 6.0, false, |ui| {
                ui.set_max_width(max);
                ui.add(egui::Label::new(text).selectable(true));
            });
        });
        ui.add_space(2.0);
    }

    // ---- centre: a bot ----
    fn conversation(&mut self, ui: &mut egui::Ui) {
        let Some(id) = self.active.clone() else {
            self.welcome(ui);
            return;
        };
        let (bot, msgs, approvals): (_, Vec<Message>, _) = {
            let st = self.engine.store();
            let Some(bot) = st.bot(&id).cloned() else {
                drop(st);
                self.active = None;
                return;
            };
            (bot, st.messages.get(&id).cloned().unwrap_or_default(), st.approvals.clone())
        };
        let busy = self.engine.is_busy(&id);
        let settings = self.engine.settings();
        let provider = settings.provider_for(bot.provider.as_deref()).map(|p| p.name.clone());

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new(&bot.name).size(20.0).strong());
                ui.label(RichText::new(&bot.title).color(muted()));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Options").clicked() {
                    self.dialog = Dialog::BotOptions;
                    self.confirm_delete_bot = false;
                }
                let tag = match (&provider, bot.read_only) {
                    (Some(p), true) => format!("{p} · read-only"),
                    (Some(p), false) => p.clone(),
                    (None, _) => "no model".into(),
                };
                ui.label(RichText::new(tag).color(muted()).small());
            });
        });
        ui.label(RichText::new(&bot.mandate).color(muted()).small());
        ui.separator();

        let area = (ui.available_height() - 150.0).max(80.0);
        egui::ScrollArea::vertical().max_height(area).auto_shrink([false, false]).stick_to_bottom(self.stick_bottom).show(ui, |ui| {
            Self::column(self.look.chat_width, ui, |ui| {
                if msgs.is_empty() {
                    ui.add_space(24.0);
                    ui.label(RichText::new(format!("{} is ready. Ask for something that needs the computer: write a note, read a page, tidy a file.", bot.name)).color(muted()));
                }
                for m in &msgs {
                    self.message(ui, &bot.name, m, &approvals);
                }
                if busy {
                    ui.horizontal(|ui| {
                        spin(ui);
                        ui.label(RichText::new("working…").color(muted()));
                    });
                }
            });
        });

        ui.add_space(4.0);
        let first = bot.name.split(' ').next().unwrap_or("the bot").to_string();
        let hint = format!("Ask {first} to use the computer");
        let left = self.engine.store().calls_left(settings.calls_per_hour);
        match self.composer(ui, &id, &hint, busy, &format!("{left} model calls left this hour"), false) {
            Composed::Send(text) => match self.engine.chat(&id, &text) {
                Ok(()) => {
                    self.stick_bottom = true;
                    self.toast.clear();
                }
                Err(e) => {
                    self.drafts.insert(id.clone(), text);
                    self.toast = e.to_string();
                }
            },
            Composed::Stop | Composed::Nothing => {}
        }
        if !self.toast.is_empty() {
            ui.colored_label(ember(), &self.toast);
        }
    }

    fn welcome(&mut self, ui: &mut egui::Ui) {
        ui.add_space(80.0);
        ui.vertical_centered(|ui| {
            logo(ui, 6.0);
            ui.add_space(10.0);
            ui.label(RichText::new("Welcome to Hinoko").size(26.0).strong());
            ui.add_space(6.0);
            ui.label(RichText::new("Each bot keeps its own computer, memory and routines.").color(muted()));
            ui.add_space(18.0);
            let have_model = self.engine.settings().provider_for(None).is_some();
            if !have_model {
                ui.label("1. Connect a model: an API key, a local server, or a CLI agent.");
                if ui.button("⚙ Open Settings").clicked() {
                    self.open_settings();
                }
                ui.add_space(10.0);
            }
            ui.label(if have_model { "Create your first bot." } else { "2. Create your first bot." });
            if ui.button("+ New bot").clicked() {
                self.new_bot = NewBot::default();
                self.dialog = Dialog::NewBot;
            }
        });
    }

    fn message(&mut self, ui: &mut egui::Ui, bot_name: &str, m: &Message, approvals: &[hinoko_core::model::Approval]) {
        match m.kind.as_str() {
            "user" => self.user_bubble(ui, &m.text),
            "bot" => {
                ui.add_space(4.0);
                ui.horizontal_top(|ui| {
                    let (r, _) = ui.allocate_exact_size(egui::vec2(3.0, 18.0), egui::Sense::hover());
                    ui.painter().rect_filled(r, 1.0, ember());
                    ui.vertical(|ui| {
                        ui.label(RichText::new(bot_name).color(muted()).small());
                        self.rich_text(ui, &m.text);
                    });
                });
            }
            "event" => {
                if self.look.show_events {
                    ui.label(RichText::new(format!("· {}", m.text)).monospace().color(muted()).small());
                }
            }
            "approval" => {
                let ap = m.approval_id.as_ref().and_then(|id| approvals.iter().find(|a| &a.id == id));
                card(ui, if pixel_on() { ink() } else { Color32::TRANSPARENT }, Some((if pixel_on() { 2.0 } else { 1.0 }, ember())), 10.0, self.look.rounding + 2.0, true, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(RichText::new("Needs your OK").color(ember()).small());
                    ui.label(RichText::new(&m.text).strong());
                    if let Some(a) = ap {
                        ui.label(RichText::new(&a.detail).color(muted()));
                        if a.status == "pending" {
                            ui.horizontal_wrapped(|ui| {
                                if ui.button("Allow").clicked() {
                                    let _ = self.engine.decide(&a.id, true);
                                }
                                if ui.button("Deny").clicked() {
                                    let _ = self.engine.decide(&a.id, false);
                                }
                                // a standing rule for exactly this command or file, for this bot
                                let key = if a.action == "ssh" { a.command.clone() } else { a.path.clone() };
                                if !key.trim().is_empty() {
                                    if ui.small_button("Always allow this").on_hover_text("Adds a standing rule for this exact request and bot. Remove it any time in Settings → Standing rules.").clicked() {
                                        self.add_rule(&a.bot_id, &a.action, &key, true);
                                        let _ = self.engine.decide(&a.id, true);
                                    }
                                    if ui.small_button("Always deny this").clicked() {
                                        self.add_rule(&a.bot_id, &a.action, &key, false);
                                        let _ = self.engine.decide(&a.id, false);
                                    }
                                }
                            });
                        } else {
                            ui.label(RichText::new(a.status.to_uppercase()).small().color(muted()));
                        }
                    }
                });
            }
            _ => {}
        }
    }

    // ---- right: the bot's computer ----
    fn desk(&mut self, ui: &mut egui::Ui) {
        let Some(id) = self.active.clone() else {
            ui.add_space(8.0);
            ui.label(RichText::new("Computer").color(muted()));
            return;
        };
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            for (t, label) in [(Tab::Files, "Files"), (Tab::Terminal, "Terminal"), (Tab::Browser, "Browser"), (Tab::Memory, "Memory")] {
                ui.selectable_value(&mut self.tab, t, label);
            }
        });
        ui.separator();
        match self.tab {
            Tab::Files => self.files_tab(ui, &id),
            Tab::Terminal => self.terminal_tab(ui, &id),
            Tab::Browser => {
                let st = self.engine.store();
                if let Some(c) = st.computers.get(&id) {
                    ui.label(RichText::new(&c.browser_title).strong());
                    ui.label(RichText::new(&c.browser_url).color(muted()).small());
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| ui.label(&c.browser_excerpt));
                }
            }
            Tab::Memory => self.memory_tab(ui, &id),
        }
    }

    fn files_tab(&mut self, ui: &mut egui::Ui, id: &str) {
        let files: Vec<(String, String)> = self
            .engine
            .store()
            .computers
            .get(id)
            .map(|c| c.files.iter().map(|(p, b)| (p.clone(), b.clone())).collect())
            .unwrap_or_default();
        if files.is_empty() {
            ui.label(RichText::new("No files yet.").color(muted()));
            return;
        }
        let home = home_of(id);
        egui::ScrollArea::vertical().max_height(ui.available_height() * 0.4).show(ui, |ui| {
            for (path, body) in &files {
                let short = path.strip_prefix(&format!("{home}/")).unwrap_or(path);
                let sel = self.open_file.as_deref() == Some(path.as_str());
                if ui.selectable_label(sel, format!("{short}  ({} chars)", body.chars().count())).clicked() {
                    self.open_file = Some(path.clone());
                }
            }
        });
        ui.separator();
        if let Some(open) = &self.open_file {
            if let Some((_, body)) = files.iter().find(|(p, _)| p == open) {
                ui.label(RichText::new(open).monospace().small().color(muted()));
                egui::ScrollArea::both().show(ui, |ui| {
                    ui.add(egui::Label::new(RichText::new(body).monospace()).selectable(true));
                });
            }
        } else {
            ui.label(RichText::new("Pick a file to read it.").color(muted()));
        }
    }

    fn terminal_tab(&mut self, ui: &mut egui::Ui, id: &str) {
        let (lines, cwd) = {
            let st = self.engine.store();
            st.computers
                .get(id)
                .map(|c| (c.terminal.clone(), c.cwd.clone()))
                .unwrap_or_default()
        };
        let h = ui.available_height() - 40.0;
        egui::ScrollArea::vertical().max_height(h).stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            for l in &lines {
                ui.label(RichText::new(format!("$ {}", l.cmd)).monospace().color(ember()));
                if !l.out.is_empty() {
                    ui.label(RichText::new(&l.out).monospace());
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new(cwd).monospace().small().color(muted()));
            let r = ui.add(egui::TextEdit::singleline(&mut self.term_input).desired_width(f32::INFINITY).hint_text("help"));
            if r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                let cmd = clean(&self.term_input, 200);
                self.term_input.clear();
                if !cmd.is_empty() {
                    let home = home_of(id);
                    if let Some(c) = self.engine.store().computers.get_mut(id) {
                        let _ = computer::exec(c, &home, &cmd);
                    }
                    self.engine.save();
                }
                r.request_focus();
            }
        });
    }

    fn memory_tab(&mut self, ui: &mut egui::Ui, id: &str) {
        let items: Vec<(String, String)> = self
            .engine
            .store()
            .memories
            .get(id)
            .map(|v| v.iter().map(|m| (m.id.clone(), m.text.clone())).collect())
            .unwrap_or_default();
        ui.label(RichText::new(format!("{} of 24 notes. The bot reads these every turn.", items.len())).color(muted()).small());
        egui::ScrollArea::vertical().max_height(ui.available_height() - 70.0).show(ui, |ui| {
            for (mid, text) in &items {
                ui.horizontal_top(|ui| {
                    if ui.small_button("✕").on_hover_text("Forget this").clicked() {
                        if let Some(list) = self.engine.store().memories.get_mut(id) {
                            list.retain(|m| &m.id != mid);
                        }
                        self.engine.save();
                    }
                    ui.label(text);
                });
            }
        });
        ui.separator();
        ui.horizontal(|ui| {
            let r = ui.add(egui::TextEdit::singleline(&mut self.mem_input).desired_width(ui.available_width() - 60.0).hint_text("Add a note"));
            let go = ui.button("Add").clicked() || (r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)));
            let text = clean(&self.mem_input, 280);
            if go && !text.is_empty() {
                let mut st = self.engine.store();
                let list = st.memories.entry(id.to_string()).or_default();
                if list.len() < 24 {
                    list.push(hinoko_core::model::Memory { id: hinoko_core::model::nid("mem"), text, at_ms: hinoko_core::model::now_ms() });
                    self.mem_input.clear();
                }
                drop(st);
                self.engine.save();
            }
        });
    }

    /// Branch and changed files for a project, from a cache that is refreshed on a
    /// background thread (never blocks drawing). While the agent works it refreshes faster.
    fn git_status(&self, dir: &str, busy: bool) -> Option<hinoko_core::git::GitStatus> {
        let every = std::time::Duration::from_secs(if busy { 3 } else { 10 });
        let mut g = self.git.lock().ok()?;
        let e = g.entry(dir.to_string()).or_insert(GitEntry { at: std::time::Instant::now() - std::time::Duration::from_secs(3600), status: None, busy: false });
        if !e.busy && e.at.elapsed() > every {
            e.busy = true;
            let (cache, d, ctx) = (self.git.clone(), dir.to_string(), self.ctx.clone());
            std::thread::spawn(move || {
                let st = hinoko_core::git::status(&d);
                if let Ok(mut g) = cache.lock() {
                    g.insert(d, GitEntry { at: std::time::Instant::now(), status: st, busy: false });
                }
                ctx.request_repaint();
            });
        }
        e.status.clone()
    }

    fn compare_dialog(&mut self, ctx: &egui::Context) {
        let mut open = true;
        let settings = self.engine.settings();
        let ready: Vec<&Provider> = settings.providers.iter().filter(|p| p.ready()).collect();
        let name_of = |id: &Option<String>| -> String {
            settings.provider_for(id.as_deref()).map(|p| if p.model.is_empty() { p.name.clone() } else { format!("{} · {}", p.name, p.model) }).unwrap_or_else(|| "pick a model".into())
        };
        egui::Window::new("Compare models").open(&mut open).default_size([940.0, 560.0]).resizable(true).collapsible(false).show(ctx, |ui| {
            if ready.len() < 2 {
                ui.label(RichText::new("Set up at least two models in Settings to compare them. A local model counts as one; only one local model runs at a time.").color(muted()));
            }
            let running = self.compare.as_ref().is_some_and(|c| !c.finished());
            ui.add(egui::TextEdit::multiline(&mut self.compare_prompt).desired_rows(3).desired_width(f32::INFINITY).hint_text("Ask both models the same thing"));
            ui.horizontal(|ui| {
                for i in 0..2 {
                    egui::ComboBox::from_id_salt(("cmp-model", i)).width(260.0).selected_text(name_of(&self.compare_pick[i])).show_ui(ui, |ui| {
                        for p in &ready {
                            let label = if p.model.is_empty() { p.name.clone() } else { format!("{} · {}", p.name, p.model) };
                            ui.selectable_value(&mut self.compare_pick[i], Some(p.id.clone()), label);
                        }
                    });
                }
                let can = !running && !self.compare_prompt.trim().is_empty() && self.compare_pick.iter().all(|p| p.is_some());
                if running {
                    if ui.button("■ Stop").clicked() {
                        if let Some(c) = &self.compare {
                            c.stop();
                        }
                    }
                } else if ui.add_enabled(can, egui::Button::new("Run ↑").fill(ember())).clicked() {
                    self.compare = Some(self.engine.compare_start(self.compare_prompt.trim(), self.compare_pick.to_vec()));
                }
            });
            ui.separator();
            let Some(cmp) = self.compare.clone() else {
                ui.label(RichText::new("Pick two models, type a prompt and press Run. Both answer at once, so you can judge quality and speed side by side. Nothing here is saved to a chat.").color(muted()));
                return;
            };
            let lanes = cmp.run.lock().map(|r| r.lanes.clone()).unwrap_or_default();
            let w = (ui.available_width() - 12.0) / lanes.len().max(1) as f32;
            ui.horizontal_top(|ui| {
                for (i, lane) in lanes.iter().enumerate() {
                    ui.vertical(|ui| {
                        ui.set_width(w);
                        ui.label(RichText::new(&lane.model).strong());
                        let stats = if lane.done {
                            format!("{}{:.1} tok/s · {}{} tokens · first token {:.1}s · total {:.1}s", if lane.estimated { "~" } else { "" }, lane.tokens_per_sec(), if lane.estimated { "~" } else { "" }, lane.tokens, lane.first_ms as f32 / 1000.0, lane.total_ms as f32 / 1000.0)
                        } else if lane.text.is_empty() {
                            "waiting…".to_string()
                        } else {
                            format!("~{:.1} tok/s · writing…", lane.tokens_per_sec())
                        };
                        ui.label(RichText::new(stats).small().color(muted()));
                        egui::ScrollArea::vertical().id_salt(("cmp-scroll", i)).max_height(ui.available_height().max(260.0)).auto_shrink([false, false]).show(ui, |ui| {
                            if !lane.think.is_empty() {
                                egui::CollapsingHeader::new(RichText::new("Thought process").small().color(muted())).id_salt(("cmp-think", i)).show(ui, |ui| {
                                    ui.add(egui::Label::new(RichText::new(&lane.think).small().color(muted())).wrap());
                                });
                            }
                            if !lane.error.is_empty() {
                                ui.colored_label(ember(), &lane.error);
                            }
                            self.rich_body(ui, &lane.text);
                        });
                    });
                }
            });
        });
        if !open {
            if let Some(c) = &self.compare {
                c.stop();
            }
            self.dialog = Dialog::None;
        }
    }

    fn activity_window(&mut self, ctx: &egui::Context) {
        if !self.activity_open {
            return;
        }
        let Some(id) = self.active_code.clone() else { return };
        // files edited (with how many times) and commands run, across the whole session
        let (mut files, mut cmds, mut failed): (Vec<(String, usize)>, Vec<(String, String)>, usize) = (Vec::new(), Vec::new(), 0);
        if let Some(c) = self.engine.store().code_session(&id) {
            for m in &c.messages {
                match m.kind.as_str() {
                    "edit" => {
                        let name = m.text.split_once(' ').map(|x| x.1).unwrap_or(&m.text).to_string();
                        match files.iter_mut().find(|f| f.0 == name) {
                            Some(f) => f.1 += 1,
                            None => files.push((name, 1)),
                        }
                    }
                    "cmd" => cmds.push((m.text.clone(), m.state.clone())),
                    "tool" if m.state == "failed" => failed += 1,
                    _ => {}
                }
            }
        }
        let mut open = true;
        egui::Window::new("Activity").open(&mut open).default_size([460.0, 480.0]).resizable(true).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.label(RichText::new(format!("{} file{} changed \u{b7} {} command{} run \u{b7} {} tool call{} failed", files.len(), if files.len() == 1 { "" } else { "s" }, cmds.len(), if cmds.len() == 1 { "" } else { "s" }, failed, if failed == 1 { "" } else { "s" })).color(muted()));
                ui.add_space(6.0);
                ui.strong("Files");
                if files.is_empty() {
                    ui.label(RichText::new("None yet.").color(muted()));
                }
                for (f, n) in &files {
                    ui.label(if *n > 1 { format!("{f}  \u{d7}{n}") } else { f.clone() });
                }
                ui.add_space(8.0);
                ui.strong("Commands");
                if cmds.is_empty() {
                    ui.label(RichText::new("None yet.").color(muted()));
                }
                for (c, st) in cmds.iter().rev() {
                    let bad = !st.is_empty() && st != "0" && st != "ok";
                    ui.add(egui::Label::new(RichText::new(format!("$ {c}")).monospace().color(if bad { ember() } else { ui.visuals().text_color() })).wrap());
                }
            });
        });
        if !open {
            self.activity_open = false;
        }
    }

    fn git_diff_window(&mut self, ctx: &egui::Context) {
        let Some((title, diff)) = self.git_diff.clone() else { return };
        let mut open = true;
        egui::Window::new(format!("Changes in {title}")).open(&mut open).default_size([760.0, 520.0]).resizable(true).show(ctx, |ui| {
            egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                if diff.trim().is_empty() {
                    ui.label(RichText::new("No tracked changes. New files are listed in the hover text.").color(muted()));
                } else {
                    diff_view(ui, &diff);
                }
            });
        });
        if !open {
            self.git_diff = None;
        }
    }

    // ---- dialogs ----
    fn dialogs(&mut self, ctx: &egui::Context) {
        self.git_diff_window(ctx);
        self.activity_window(ctx);
        match self.dialog {
            Dialog::None => {}
            Dialog::NewBot => self.new_bot_dialog(ctx),
            Dialog::Settings => self.settings_dialog(ctx),
            Dialog::Routines => self.routines_dialog(ctx),
            Dialog::BotOptions => self.bot_options_dialog(ctx),
            Dialog::ChatOptions => self.chat_options_dialog(ctx),
            Dialog::Cookbook => self.cookbook_dialog(ctx),
            Dialog::Compare => self.compare_dialog(ctx),
            Dialog::About => {
                let mut open = true;
                egui::Window::new("About Hinoko").open(&mut open).collapsible(false).default_width(420.0).show(ctx, |ui| {
                    ui.label("Each bot has its own computer, memory, routines and approval queue.");
                    ui.add_space(6.0);
                    ui.label("• The shell is a simulation inside the bot's own folder. It never touches your machine.");
                    ui.label("• Deleting and sending always wait for your OK.");
                    ui.label("• Pages are public https only.");
                    ui.label("• Model calls are capped per rolling hour (Settings).");
                    ui.label("• CLI-agent providers run with their own tools on your machine.");
                    ui.add_space(6.0);
                    ui.label(RichText::new(format!("Data folder: {}", self.engine.home().display())).monospace().small().color(muted()));
                });
                if !open {
                    self.dialog = Dialog::None;
                }
            }
        }
    }

    fn new_bot_dialog(&mut self, ctx: &egui::Context) {
        let mut open = true;
        let mut created: Option<String> = None;
        egui::Window::new("New bot").open(&mut open).collapsible(false).default_width(420.0).show(ctx, |ui| {
            ui.label("Name");
            ui.text_edit_singleline(&mut self.new_bot.name);
            ui.label("Role (a few words, e.g. Research)");
            ui.text_edit_singleline(&mut self.new_bot.title);
            ui.label("Mandate: what it does and what it must not do");
            ui.add(egui::TextEdit::multiline(&mut self.new_bot.mandate).desired_rows(4).desired_width(f32::INFINITY));
            if !self.new_bot.error.is_empty() {
                ui.colored_label(ember(), &self.new_bot.error);
            }
            if ui.button("Create").clicked() {
                let res = self.engine.store().add_bot(&self.new_bot.name, &self.new_bot.title, &self.new_bot.mandate);
                match res {
                    Ok(id) => created = Some(id),
                    Err(e) => self.new_bot.error = e.to_string(),
                }
            }
        });
        if let Some(id) = created {
            self.engine.save();
            self.active = Some(id);
            self.dialog = Dialog::None;
        } else if !open {
            self.dialog = Dialog::None;
        }
    }

    fn bot_options_dialog(&mut self, ctx: &egui::Context) {
        let Some(id) = self.active.clone() else {
            self.dialog = Dialog::None;
            return;
        };
        let providers = self.engine.settings().providers;
        let mut open = true;
        let mut remove = false;
        egui::Window::new("Bot options").open(&mut open).collapsible(false).default_width(380.0).show(ctx, |ui| {
            let mut changed = false;
            {
                let mut st = self.engine.store();
                let Some(bot) = st.bot_mut(&id) else { return };
                ui.label(RichText::new(&bot.name).strong());
                ui.add_space(4.0);
                ui.label("Model for this bot");
                let current = bot
                    .provider
                    .as_ref()
                    .and_then(|pid| providers.iter().find(|p| &p.id == pid))
                    .map(|p| p.name.clone())
                    .unwrap_or_else(|| "Desk default".into());
                egui::ComboBox::from_id_salt("bot-provider").selected_text(current).show_ui(ui, |ui| {
                    changed |= ui.selectable_value(&mut bot.provider, None, "Desk default").changed();
                    for p in &providers {
                        changed |= ui.selectable_value(&mut bot.provider, Some(p.id.clone()), &p.name).changed();
                    }
                });
                changed |= ui.checkbox(&mut bot.read_only, "Read-only (plan mode): can look and read, cannot change anything").changed();
                let machines = self.engine.settings().machines;
                if !machines.is_empty() {
                    ui.add_space(4.0);
                    ui.label("Machines this bot may reach over SSH");
                    for m in &machines {
                        let mut on = bot.machines.contains(&m.id);
                        if ui.checkbox(&mut on, format!("{} ({}@{})", m.name, m.user, m.host)).changed() {
                            bot.machines.retain(|x| x != &m.id);
                            if on {
                                bot.machines.push(m.id.clone());
                            }
                            changed = true;
                        }
                    }
                }
            }
            if changed {
                self.engine.save();
            }
            ui.separator();
            if !self.confirm_delete_bot {
                if ui.button("Delete this bot…").clicked() {
                    self.confirm_delete_bot = true;
                }
            } else {
                ui.colored_label(ember(), "This removes its chat, files, memory and routines.");
                ui.horizontal(|ui| {
                    if ui.button("Yes, delete").clicked() {
                        remove = true;
                    }
                    if ui.button("Cancel").clicked() {
                        self.confirm_delete_bot = false;
                    }
                });
            }
        });
        if remove {
            self.engine.store().remove_bot(&id);
            self.engine.save();
            self.active = self.engine.store().bots.first().map(|b| b.id.clone());
            self.dialog = Dialog::None;
        } else if !open {
            self.dialog = Dialog::None;
        }
    }

    fn cookbook_dialog(&mut self, ctx: &egui::Context) {
        use hinoko_core::local::{Fit, Phase};
        let mut open = true;
        let hw = self.cook.hw.lock().unwrap().clone();
        let info = self.engine.local_info();
        let screen = ctx.content_rect();
        let mut search: Option<String> = None;
        let mut run: Option<(String, String)> = None;
        let mut start_file: Option<std::path::PathBuf> = None;
        egui::Window::new("Local models")
            .open(&mut open)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width((screen.width() * 0.7).clamp(480.0, 780.0))
            .default_height(screen.height() * 0.82)
            .max_height(screen.height() * 0.9)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.label(RichText::new("Run models on your own computer. Hinoko checks what your PC can handle, downloads the model and starts it. Nothing else to install.").color(muted()).small());
                    ui.add_space(4.0);
                    match &hw {
                        Some(h) => ui.label(RichText::new(h.summary()).strong()),
                        None => ui.label(RichText::new("Checking your computer…").color(muted())),
                    };
                    ui.add_space(6.0);

                    // what is running
                    if info.starting {
                        ui.horizontal(|ui| {
                            spin(ui);
                            ui.label(format!("Loading {} … big models take a minute", info.model));
                        });
                    } else if info.running {
                        ui.horizontal(|ui| {
                            ui.colored_label(good(), format!("● Running: {}", info.model));
                            if ui.button("Stop").clicked() {
                                self.engine.local_stop_async();
                            }
                        });
                        ui.label(RichText::new("It is set as your default model. Pick it in any chat from the model menu. It also starts by itself when a chat needs it.").color(muted()).small());
                    }
                    if !info.error.is_empty() && !info.running && !info.starting {
                        ui.colored_label(ember(), &info.error);
                    }
                    // which engine runs the models
                    {
                        let mut st = self.engine.settings();
                        let before = st.local_engine.clone();
                        let label = |v: &str| match v {
                            "builtin" => "Built-in (experimental: CPU only, tiny models)",
                            "llama" => "llama.cpp (fast, GPU or CPU, downloads ~40 MB)",
                            _ => "Automatic (llama.cpp)",
                        };
                        ui.horizontal(|ui| {
                            ui.label("Engine");
                            egui::ComboBox::from_id_salt("local_engine").selected_text(label(&st.local_engine)).show_ui(ui, |ui| {
                                for v in ["", "builtin", "llama"] {
                                    if cfg!(not(feature = "builtin")) && v != "llama" {
                                        continue;
                                    }
                                    ui.selectable_value(&mut st.local_engine, v.to_string(), label(v));
                                }
                            });
                        });
                        if st.local_engine != before {
                            self.engine.set_settings(st);
                            self.cook.started = false; // fits depend on the engine
                            *self.cook.list.lock().unwrap() = Default::default();
                            self.open_cookbook();
                        }
                    }
                    let builtin_engine = hw.as_ref().map_or(false, |h| self.engine.uses_builtin(h));
                    if builtin_engine {
                        ui.label(RichText::new("Experimental built-in engine: CPU only and much slower than llama.cpp. Only for tiny models (about 1-2B). Pick \u{201c}Automatic\u{201d} or \u{201c}llama.cpp\u{201d} above for normal use.").color(muted()).small());
                    } else if let Some(p) = self.engine.local_progress("runner") {
                        match p.phase {
                            Phase::Running => {
                                let f = if p.total > 0 { p.done as f32 / p.total as f32 } else { 0.0 };
                                ui.add(egui::ProgressBar::new(f).text(format!("{} {:.0} MB", p.label, p.done as f64 / 1e6)));
                            }
                            Phase::Failed(e) => {
                                ui.add(egui::Label::new(RichText::new(format!("Runner: {e}")).color(ember())).selectable(true));
                                if let Some(h) = hw.clone() {
                                    if ui.button("Retry runner download").clicked() {
                                        self.engine.install_runner_async(h);
                                    }
                                }
                            }
                            Phase::Done => {}
                        }
                    } else if self.engine.runner_path().is_none() {
                        ui.label(RichText::new("The model runner (llama.cpp, about 40 MB) is downloaded the first time you run a model.").color(muted()).small());
                    } else {
                        let v = self.engine.runner_version();
                        ui.label(RichText::new(format!("Runner ready{}", if v.is_empty() { String::new() } else { format!(" ({v})") })).color(muted()).small());
                    }
                    if let Some(p) = self.engine.local_progress("server") {
                        if let Phase::Failed(e) = p.phase {
                            ui.colored_label(ember(), e);
                        }
                    }

                    // already downloaded
                    let have = self.engine.downloaded_models();
                    if !have.is_empty() {
                        ui.separator();
                        ui.label(RichText::new("On this PC").size(16.0).strong());
                        for (name, path, bytes) in &have {
                            ui.horizontal(|ui| {
                                ui.label(name);
                                ui.label(RichText::new(format!("{:.1} GB", *bytes as f64 / 1e9)).color(muted()).small());
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("Delete").clicked() {
                                        self.engine.delete_model(path);
                                    }
                                    let busy = info.starting;
                                    if ui.add_enabled(!busy, egui::Button::new("Run")).clicked() {
                                        start_file = Some(path.clone());
                                    }
                                });
                            });
                        }
                    }

                    // the cookbook
                    ui.separator();
                    ui.label(RichText::new("Cookbook").size(16.0).strong());
                    ui.label(RichText::new("Models from Hugging Face, ranked for this PC. Pick one and press Download & run.").color(muted()).small());
                    ui.horizontal(|ui| {
                        let r = ui.add(egui::TextEdit::singleline(&mut self.cook.query).hint_text("Search models (e.g. qwen, gemma, coder)").desired_width(260.0));
                        let enter = r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                        if ui.button("Search").clicked() || enter {
                            search = Some(self.cook.query.clone());
                        }
                        if ui.button("Trending").clicked() {
                            self.cook.query.clear();
                            search = Some(String::new());
                        }
                        ui.checkbox(&mut self.cook.show_big, "Show models too big for this PC");
                    });
                    let list = self.cook.list.lock().unwrap();
                    if list.loading {
                        ui.horizontal(|ui| {
                            spin(ui);
                            ui.label(RichText::new("Asking Hugging Face…").color(muted()));
                        });
                    }
                    if !list.error.is_empty() {
                        ui.colored_label(ember(), &list.error);
                    }
                    let mut shown = 0;
                    for c in list.items.iter().filter(|c| self.cook.show_big || c.fit != Fit::TooBig) {
                        shown += 1;
                        ui.add_space(4.0);
                        card(ui, raised(), None, 8.0, self.look.rounding + 4.0, true, |ui| {
                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.label(RichText::new(&c.repo).strong());
                                    let mut bits = Vec::new();
                                    if let Some(p) = c.params_b {
                                        bits.push(format!("{p:.0}B params"));
                                    }
                                    bits.push(format!("{} · {:.1} GB", c.best.quant, c.best.gb()));
                                    if c.moe {
                                        bits.push("mixture-of-experts".into());
                                    }
                                    if c.vision {
                                        bits.push("sees images".into());
                                    }
                                    bits.push(format!("{} downloads", c.downloads));
                                    ui.label(RichText::new(bits.join("  ·  ")).color(muted()).small());
                                    let colour = match c.fit {
                                        Fit::Fast | Fit::Good => good(),
                                        Fit::Mixed | Fit::Slow => egui::Color32::from_rgb(222, 170, 80),
                                        Fit::TooBig => muted(),
                                    };
                                    ui.colored_label(colour, c.fit.label());
                                });
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    let key = format!("{}/{}", c.repo, c.best.name);
                                    let dest = self.engine.models_dir().join(c.best.name.replace(['/', '\\'], "_"));
                                    match self.engine.local_progress(&key) {
                                        Some(p) if p.phase == Phase::Running => {
                                            let f = if p.total > 0 { p.done as f32 / p.total as f32 } else { 0.0 };
                                            ui.add(egui::ProgressBar::new(f).desired_width(160.0).text(format!("{:.0}%  {:.1} GB", f * 100.0, p.done as f64 / 1e9)));
                                        }
                                        other => {
                                            if let Some(p) = &other {
                                                if let Phase::Failed(e) = &p.phase {
                                                    ui.colored_label(ember(), e);
                                                }
                                            }
                                            let label = if dest.exists() { "Run" } else { "Download & run" };
                                            let enabled = hw.is_some() && !info.starting && c.fit != Fit::TooBig;
                                            if ui.add_enabled(enabled, egui::Button::new(label).fill(ember())).clicked() {
                                                if dest.exists() {
                                                    start_file = Some(dest.clone());
                                                } else {
                                                    run = Some((c.repo.clone(), c.best.name.clone()));
                                                }
                                            }
                                        }
                                    }
                                });
                            });
                        });
                    }
                    if shown == 0 && !list.loading && list.error.is_empty() {
                        ui.label(RichText::new("Nothing suitable found. Try another search, or tick \"Show models too big\".").color(muted()));
                    }
                });
            });
        if let (Some(h), Some((repo, file))) = (hw.clone(), run) {
            self.engine.local_run_async(repo, file, h);
        }
        if let (Some(h), Some(path)) = (hw, start_file) {
            self.engine.local_start_async(path, h);
        }
        if let Some(q) = search {
            self.cook_search(q);
        }
        if !open {
            self.dialog = Dialog::None;
        }
    }

    fn chat_options_dialog(&mut self, ctx: &egui::Context) {
        let Some(id) = self.active_chat.clone() else {
            self.dialog = Dialog::None;
            return;
        };
        let mut open = true;
        let mut remove = false;
        egui::Window::new("Chat options").open(&mut open).collapsible(false).default_width(420.0).show(ctx, |ui| {
            let mut changed = false;
            if let Some(c) = self.engine.store().chat_mut(&id) {
                ui.label("Title");
                changed |= ui.text_edit_singleline(&mut c.title).changed();
                ui.add_space(4.0);
                ui.label("System prompt for this chat (optional)");
                changed |= ui.add(egui::TextEdit::multiline(&mut c.system).desired_rows(5).desired_width(f32::INFINITY).hint_text("e.g. You are a patient Rust tutor.")).changed();
            }
            if changed {
                self.engine.save();
            }
            ui.separator();
            let (summarized, busy_chat) = (self.engine.store().chat(&id).map_or(false, |c| !c.summary.is_empty()), self.engine.is_chat_busy(&id));
            ui.horizontal(|ui| {
                if ui.add_enabled(!busy_chat, egui::Button::new("Compact now")).on_hover_text("Summarize the older messages so long chats stay fast. The full conversation stays here.").clicked() {
                    self.engine.compact_chat_async(id.clone());
                }
                if summarized {
                    ui.label(RichText::new("Older messages are summarized for the model").color(muted()).small());
                }
            });
            ui.separator();
            if !self.confirm_delete_bot {
                if ui.button("Delete this chat…").clicked() {
                    self.confirm_delete_bot = true;
                }
            } else {
                ui.colored_label(ember(), "This removes the whole conversation.");
                ui.horizontal(|ui| {
                    if ui.button("Yes, delete").clicked() {
                        remove = true;
                    }
                    if ui.button("Cancel").clicked() {
                        self.confirm_delete_bot = false;
                    }
                });
            }
        });
        if remove {
            self.engine.store().chats.retain(|c| c.id != id);
            self.engine.save();
            self.active_chat = None;
            self.dialog = Dialog::None;
        } else if !open {
            self.dialog = Dialog::None;
        }
    }

    fn routines_dialog(&mut self, ctx: &egui::Context) {
        let mut open = true;
        let names: HashMap<String, String> = self.engine.store().bots.iter().map(|b| (b.id.clone(), b.name.clone())).collect();
        let routines: Vec<Routine> = self.engine.store().routines.clone();
        egui::Window::new("Routines").open(&mut open).collapsible(false).default_width(460.0).show(ctx, |ui| {
            ui.label(RichText::new("Routines run when you press Run. A bot can propose one, but it starts paused.").color(muted()).small());
            for r in &routines {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new(&r.name).strong());
                        ui.label(RichText::new(format!("{} · {}{}", names.get(&r.bot_id).cloned().unwrap_or_default(), r.schedule, if r.enabled { "" } else { " · paused" })).color(muted()).small());
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("✕").clicked() {
                            self.engine.store().routines.retain(|x| x.id != r.id);
                            self.engine.save();
                        }
                        let label = if r.enabled { "Pause" } else { "Enable" };
                        if ui.small_button(label).clicked() {
                            if let Some(x) = self.engine.store().routines.iter_mut().find(|x| x.id == r.id) {
                                x.enabled = !x.enabled;
                            }
                            self.engine.save();
                        }
                        if ui.add_enabled(r.enabled, egui::Button::new("Run")).clicked() {
                            self.toast = self.engine.run_routine(&r.id).err().map(String::from).unwrap_or_default();
                            self.active = Some(r.bot_id.clone());
                            self.stick_bottom = true;
                        }
                    });
                });
                ui.separator();
            }
            if self.active.is_some() {
                ui.label(RichText::new("Add a routine for the selected bot").strong());
                ui.text_edit_singleline(&mut self.routine_form.name).on_hover_text("Name");
                ui.add(egui::TextEdit::singleline(&mut self.routine_form.schedule).hint_text("Schedule note, e.g. Weekdays at 8:00 AM"));
                ui.add(egui::TextEdit::multiline(&mut self.routine_form.prompt).hint_text("What should the bot do?").desired_rows(3).desired_width(f32::INFINITY));
                let f = &self.routine_form;
                let ok = !clean(&f.name, 48).is_empty() && !clean(&f.prompt, 500).is_empty();
                if ui.add_enabled(ok && routines.len() < 12, egui::Button::new("Add")).clicked() {
                    let bot_id = self.active.clone().unwrap_or_default();
                    self.engine.store().routines.push(Routine {
                        id: hinoko_core::model::nid("rt"),
                        bot_id,
                        name: clean(&f.name, 48),
                        schedule: if clean(&f.schedule, 48).is_empty() { "On demand".into() } else { clean(&f.schedule, 48) },
                        prompt: clean(&f.prompt, 500),
                        enabled: true,
                    });
                    self.routine_form = RoutineForm::default();
                    self.engine.save();
                }
            }
        });
        if !open {
            self.dialog = Dialog::None;
        }
    }

    fn settings_dialog(&mut self, ctx: &egui::Context) {
        let mut open = true;
        let mut s = self.engine.settings();
        let before = serde_json::to_string(&s).unwrap_or_default();
        let screen = ctx.content_rect();
        egui::Window::new("Settings")
            .open(&mut open)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(620.0_f32.min(screen.width() - 40.0))
            .default_height((screen.height() * 0.85).max(420.0))
            .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(RichText::new("Models").size(16.0).strong());
                ui.label(RichText::new("Add an API key, a local server (Ollama, LM Studio, llama.cpp), or a CLI agent. The default is used by every bot unless you pick another in its Options.").color(muted()).small());
                if s.providers.is_empty() {
                    ui.label(RichText::new("No models yet. Pick a starting point below.").color(ember()));
                }
                let mut delete: Option<String> = None;
                let mut edit: Option<String> = None;
                for p in &s.providers.clone() {
                    ui.horizontal(|ui| {
                        if ui.radio(s.active == p.id, "").on_hover_text("Use as default").clicked() {
                            s.active = p.id.clone();
                        }
                        ui.vertical(|ui| {
                            ui.label(RichText::new(&p.name).strong());
                            ui.label(RichText::new(format!("{} · {}", p.kind.label(), p.summary())).color(muted()).small());
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Delete").clicked() {
                                delete = Some(p.id.clone());
                            }
                            if ui.small_button("Edit").clicked() {
                                edit = Some(p.id.clone());
                            }
                        });
                    });
                }
                if let Some(id) = delete {
                    s.remove(&id);
                    if self.editing.as_ref().is_some_and(|e| e.provider.id == id) {
                        self.editing = None;
                    }
                }
                if let Some(id) = edit {
                    if let Some(p) = s.provider(&id) {
                        self.editing = Some(Editing { args: p.args.join("\n"), provider: p.clone(), is_new: false });
                        self.testing = false;
                        *self.test_result.lock().unwrap() = None;
                    }
                }

                ui.add_space(6.0);
                if ui.button("Local models: run one on this PC…").on_hover_text("Hinoko checks your hardware, suggests models that fit, downloads and runs them. No Ollama needed.").clicked() {
                    self.open_cookbook();
                }
                ui.add_space(4.0);
                ui.label("Add from a starting point:");
                ui.horizontal_wrapped(|ui| {
                    for preset in PRESETS {
                        if ui.button(preset.label).clicked() {
                            self.start_edit(preset);
                        }
                    }
                    if ui.button("Custom API").clicked() {
                        self.editing = Some(Editing { provider: Provider::blank(Kind::OpenAi), args: String::new(), is_new: true });
                    }
                    if ui.button("Custom CLI").clicked() {
                        self.editing = Some(Editing { provider: Provider::blank(Kind::Cli), args: String::new(), is_new: true });
                    }
                });

                if self.editing.is_some() {
                    ui.separator();
                    self.provider_editor(ui, &mut s);
                }

                ui.separator();
                ui.label(RichText::new("Machines (SSH)").size(16.0).strong());
                ui.label(RichText::new("Let a bot work on another computer. Hinoko uses its own SSH key, never a password. Each bot only reaches the machines you tick in its Options, and every command except plain read-only ones waits for your Allow.").color(muted()).small());
                let mut del_machine: Option<String> = None;
                for m in &s.machines.clone() {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.label(RichText::new(&m.name).strong());
                            ui.label(RichText::new(format!("{}@{}:{}{}", m.user, m.host, m.port, if m.auto_readonly { " · read-only commands run without asking" } else { "" })).color(muted()).small());
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Delete").clicked() {
                                del_machine = Some(m.id.clone());
                            }
                            if ui.small_button("Edit").clicked() {
                                self.editing_machine = Some(m.clone());
                            }
                            if ui.small_button("Test").clicked() {
                                *self.machine_msg.lock().unwrap() = format!("Testing {}…", m.name);
                                let slot = self.machine_msg.clone();
                                let name = m.name.clone();
                                self.engine.test_machine(m.clone(), move |r| {
                                    *slot.lock().unwrap() = match r {
                                        Ok(_) => format!("✓ {name}: connected."),
                                        Err(e) => format!("{name}: could not connect. {}", e.lines().last().unwrap_or("")),
                                    };
                                });
                            }
                        });
                    });
                }
                if let Some(id) = del_machine {
                    s.machines.retain(|m| m.id != id);
                    for b in self.engine.store().bots.iter_mut() {
                        b.machines.retain(|x| *x != id);
                    }
                    self.engine.save();
                }
                let msg = self.machine_msg.lock().unwrap().clone();
                if !msg.is_empty() {
                    ui.colored_label(if msg.starts_with('✓') { good() } else { muted() }, msg);
                }
                if self.editing_machine.is_none() && ui.button("+ Add machine").clicked() {
                    self.editing_machine = Some(Machine::blank());
                }
                let mut done_edit = false;
                let mut cancel_edit = false;
                if let Some(m) = self.editing_machine.as_mut() {
                    egui::Grid::new("machine-grid").num_columns(2).spacing([10.0, 8.0]).show(ui, |ui| {
                        ui.label("Name");
                        ui.add(egui::TextEdit::singleline(&mut m.name).hint_text("Workshop PC"));
                        ui.end_row();
                        ui.label("Address");
                        ui.add(egui::TextEdit::singleline(&mut m.host).hint_text("192.168.1.20 or pc.local"));
                        ui.end_row();
                        ui.label("User");
                        ui.text_edit_singleline(&mut m.user);
                        ui.end_row();
                        ui.label("Port");
                        ui.add(egui::DragValue::new(&mut m.port).range(1..=65535));
                        ui.end_row();
                        ui.label("Notes");
                        ui.add(egui::TextEdit::singleline(&mut m.notes).hint_text("what it is, so the bot knows"));
                        ui.end_row();
                    });
                    ui.checkbox(&mut m.auto_readonly, "Run plain read-only commands (ls, df, uptime, ps…) without asking");
                    let ok = !m.name.trim().is_empty() && hinoko_core::ssh::validate(m).is_ok();
                    ui.horizontal(|ui| {
                        if ui.add_enabled(ok, egui::Button::new("Save machine")).clicked() {
                            done_edit = true;
                        }
                        if ui.button("Cancel").clicked() {
                            cancel_edit = true;
                        }
                    });
                    if !ok && !m.host.is_empty() {
                        ui.colored_label(ember(), "Address and user may only use letters, digits, . - _ (and : in addresses).");
                    }
                }
                if done_edit {
                    if let Some(m) = self.editing_machine.take() {
                        s.upsert_machine(m);
                    }
                }
                if cancel_edit {
                    self.editing_machine = None;
                }
                ui.add_space(4.0);
                ui.label(RichText::new("To let Hinoko into a computer: show the key, then run the setup line once on that computer.").color(muted()).small());
                ui.horizontal(|ui| {
                    if ui.button("Show Hinoko's SSH key").clicked() {
                        let slot = self.pubkey.clone();
                        self.engine.public_key_async(move |r| {
                            *slot.lock().unwrap() = Some(r.unwrap_or_else(|e| format!("error: {e}")));
                        });
                    }
                });
                if let Some(k) = self.pubkey.lock().unwrap().clone() {
                    if k.starts_with("error") {
                        ui.colored_label(ember(), &k);
                    } else {
                        let line = hinoko_core::ssh::authorize_command(&k);
                        ui.add(egui::Label::new(RichText::new(&line).monospace().small()).selectable(true));
                        if ui.button("Copy setup line").clicked() {
                            ui.ctx().copy_text(line);
                        }
                        ui.label(RichText::new("Linux/macOS: paste it in a terminal on that computer. Windows: turn on the OpenSSH Server feature, then add the key to C:\\ProgramData\\ssh\\administrators_authorized_keys (admin) or ~\\.ssh\\authorized_keys.").color(muted()).small());
                    }
                }

                ui.separator();
                ui.label(RichText::new("Web search").size(16.0).strong());
                ui.label(RichText::new("Used by bots (web_search) and Deep research. Point this at your own SearXNG for reliable results; blank falls back to scraping DuckDuckGo, which can break.").color(muted()).small());
                ui.add(egui::TextEdit::singleline(&mut s.search_url).desired_width(f32::INFINITY).hint_text("http://192.168.1.20:8888   (SearXNG with JSON format enabled)"));
                ui.separator();
                ui.label(RichText::new("Instructions").size(16.0).strong());
                ui.label(RichText::new("Added to every chat and bot, like a global system prompt.").color(muted()).small());
                ui.add(egui::TextEdit::multiline(&mut s.instructions).desired_rows(3).desired_width(f32::INFINITY).hint_text("e.g. Answer briefly. Prefer Rust examples."));
                ui.checkbox(&mut s.builtin_prompt, "Start plain chats with Hinoko's built-in prompt")
                    .on_hover_text("Gives the model Hinoko's identity, today's date, and guidance on searching, sources and tone. Small local models get a shortened version. Your instructions come after it.");

                ui.separator();
                ui.label(RichText::new("Appearance").size(16.0).strong());
                let a = &mut s.appearance;
                ui.horizontal(|ui| {
                    ui.label("Theme");
                    let current = THEMES.iter().find(|t| t.0 == a.theme).map(|t| t.1).unwrap_or("Custom");
                    egui::ComboBox::from_id_salt("theme").selected_text(current).show_ui(ui, |ui| {
                        for (id, label) in THEMES {
                            if ui.selectable_label(a.theme == *id, *label).clicked() && *id != "custom" {
                                *a = a.with_theme(id);
                            }
                        }
                    });
                });
                let mut recolored = false;
                egui::Grid::new("colors").num_columns(4).spacing([14.0, 6.0]).show(ui, |ui| {
                    ui.label("Accent");
                    recolored |= ui.color_edit_button_srgb(&mut a.accent).changed();
                    ui.label("Background");
                    recolored |= ui.color_edit_button_srgb(&mut a.background).changed();
                    ui.end_row();
                    ui.label("Panels");
                    recolored |= ui.color_edit_button_srgb(&mut a.panel).changed();
                    ui.label("Text");
                    recolored |= ui.color_edit_button_srgb(&mut a.text).changed();
                    ui.end_row();
                });
                if recolored {
                    a.theme = "custom".into();
                }
                ui.horizontal(|ui| {
                    ui.label("Zoom");
                    ui.add(egui::Slider::new(&mut a.ui_scale, 0.8..=1.6).step_by(0.05));
                });
                ui.horizontal(|ui| {
                    ui.label("Spacing");
                    ui.add(egui::Slider::new(&mut a.spacing, 2.0..=16.0));
                    ui.label("Corners");
                    ui.add(egui::Slider::new(&mut a.rounding, 0.0..=20.0));
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut a.pixel, "Pixel art interface").on_hover_text("Notched pixel corners, hard 2px borders, hard drop shadows and pixel sprites. Works with any colours.");
                    ui.checkbox(&mut a.pixel_font, "Pixel font").on_hover_text("Pixelify Sans instead of the default soft sans (Inter).");
                });
                ui.horizontal(|ui| {
                    ui.label("Reading width");
                    ui.add(egui::DragValue::new(&mut a.chat_width).range(0.0..=2000.0).speed(10.0));
                    ui.label(RichText::new("(0 = fill the window)").color(muted()).small());
                });
                ui.checkbox(&mut a.enter_sends, "Enter sends (Shift+Enter = new line). Off: Ctrl+Enter sends");
                ui.checkbox(&mut a.monospace_chat, "Monospace chat text");
                ui.checkbox(&mut a.show_events, "Show tool events in bot conversations");
                ui.checkbox(&mut a.show_computer_panel, "Show the computer panel next to bots");
                ui.horizontal(|ui| {
                    if ui.button("Copy theme").on_hover_text("Copies this look as JSON so you can share it").clicked() {
                        ui.ctx().copy_text(a.to_json());
                        self.theme_msg = "Copied to the clipboard.".into();
                    }
                    if ui.button("Reset look").clicked() {
                        *a = Appearance::default();
                    }
                });
                ui.add(egui::TextEdit::multiline(&mut self.theme_text).desired_rows(2).desired_width(f32::INFINITY).hint_text("Paste a theme here, then press Apply theme"));
                ui.horizontal(|ui| {
                    if ui.button("Apply theme").clicked() {
                        match Appearance::from_json(&self.theme_text) {
                            Ok(t) => {
                                *a = t;
                                self.theme_text.clear();
                                self.theme_msg = "Theme applied.".into();
                            }
                            Err(e) => self.theme_msg = e,
                        }
                    }
                    ui.label(RichText::new(&self.theme_msg).color(muted()).small());
                });

                ui.separator();
                ui.label(RichText::new("Updates").size(16.0).strong());
                self.update_section(ui, &mut s.auto_update_check);

                ui.separator();
                ui.label(RichText::new("Usage").size(16.0).strong());
                self.usage_section(ui, &s);

                ui.separator();
                ui.label(RichText::new("Skills").size(16.0).strong());
                self.skills_section(ui);

                ui.separator();
                ui.label(RichText::new("Bots: attention").size(16.0).strong());
                ui.checkbox(&mut s.quiet_routines, "Quiet routines")
                    .on_hover_text("A routine that finds nothing leaves one muted line instead of a message. Bots speak when there is something new or you must act.");
                ui.label(RichText::new("Standing rules").strong());
                if s.rules.is_empty() {
                    ui.label(RichText::new("None. Use \u{201c}Always allow this\u{201d} on a bot's approval card to make one. Deny rules win over allow rules.").color(muted()).small());
                }
                let bot_names: Vec<(String, String)> = self.engine.store().bots.iter().map(|b| (b.id.clone(), b.name.clone())).collect();
                let mut drop_rule: Option<String> = None;
                for r in &s.rules {
                    ui.horizontal(|ui| {
                        let who = if r.bot_id.is_empty() { "any bot".to_string() } else { bot_names.iter().find(|b| b.0 == r.bot_id).map(|b| b.1.clone()).unwrap_or_else(|| "removed bot".into()) };
                        let kind = if r.action.is_empty() { "any request".to_string() } else { r.action.clone() };
                        ui.label(RichText::new(if r.allow { "ALLOW" } else { "DENY" }).color(if r.allow { good() } else { ember() }).strong());
                        ui.add(egui::Label::new(RichText::new(format!("{who} \u{b7} {kind} \u{b7} {}", clip_path(&r.contains, 48))).monospace().small()).truncate());
                        if ui.small_button("Remove").clicked() {
                            drop_rule = Some(r.id.clone());
                        }
                    });
                }
                if let Some(id) = drop_rule {
                    s.rules.retain(|r| r.id != id);
                }

                ui.separator();
                ui.label(RichText::new("Limits").size(16.0).strong());
                ui.horizontal(|ui| {
                    ui.label("Model calls per hour");
                    ui.add(egui::DragValue::new(&mut s.calls_per_hour).range(1..=2000));
                });
                ui.horizontal(|ui| {
                    ui.label("Steps per turn");
                    ui.add(egui::DragValue::new(&mut s.max_steps).range(1..=12));
                    ui.label(RichText::new("(model calls a single request may chain)").color(muted()).small());
                });
                ui.horizontal(|ui| {
                    ui.label("Temperature");
                    ui.add(egui::Slider::new(&mut s.temperature, 0.0..=1.5));
                });
                ui.horizontal(|ui| {
                    ui.label("Max reply length");
                    let fmt = |v: f64, _: std::ops::RangeInclusive<usize>| if v < 256.0 { "auto".to_string() } else { format!("{} tokens", v as u32) };
                    ui.add(egui::DragValue::new(&mut s.max_reply_tokens).range(0..=200_000).speed(256.0).custom_formatter(fmt));
                    ui.label(RichText::new("chat; thinking counts. auto: 8,192, or 16,384 while Think is on").color(muted()).small());
                });
                ui.horizontal(|ui| {
                    ui.label("Max Code step length");
                    let fmt = |v: f64, _: std::ops::RangeInclusive<usize>| if v < 256.0 { "auto".to_string() } else { format!("{} tokens", v as u32) };
                    ui.add(egui::DragValue::new(&mut s.max_code_tokens).range(0..=200_000).speed(256.0).custom_formatter(fmt));
                    ui.label(RichText::new("one model turn in the Code tab. auto: 4,096").color(muted()).small());
                });
                ui.horizontal(|ui| {
                    ui.label("Code reviewer");
                    let fmt = |v: f64, _: std::ops::RangeInclusive<usize>| if v < 1.0 { "off".to_string() } else { format!("up to {} round{}", v as u32, if v as u32 == 1 { "" } else { "s" }) };
                    ui.add(egui::DragValue::new(&mut s.code_review).range(0..=3).custom_formatter(fmt));
                    ui.label(RichText::new("a second pass checks what the Code agent changed and sends it back with problems").color(muted()).small());
                });
                if s.code_review > 0 {
                    ui.horizontal(|ui| {
                        ui.label("Reviewer model");
                        let shown = s.providers.iter().find(|p| p.id == s.review_provider).map(|p| p.name.clone()).unwrap_or_else(|| "Same as the Code model".into());
                        let opts: Vec<(String, String)> = s.providers.iter().map(|p| (p.id.clone(), p.name.clone())).collect();
                        egui::ComboBox::from_id_salt("review_model").selected_text(shown).show_ui(ui, |ui| {
                            ui.selectable_value(&mut s.review_provider, String::new(), "Same as the Code model");
                            for (pid, pname) in &opts {
                                ui.selectable_value(&mut s.review_provider, pid.clone(), pname);
                            }
                        });
                        ui.label(RichText::new("a different, stronger model catches more").color(muted()).small());
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Summarize history after");
                    ui.add(egui::DragValue::new(&mut s.compact_tokens).range(0..=200_000).speed(250.0).custom_formatter(|v, _| if v < 1000.0 { "auto".into() } else { format!("{} tokens", v as u32) }));
                    ui.label(RichText::new("(auto: 5,000 for local models, 20,000 for others)").color(muted()).small());
                });
                ui.checkbox(&mut s.traces, "Write run traces")
                    .on_hover_text("Saves every model call and tool run as JSON lines in the data folder, under traces/. For debugging agents and building datasets from your own use.");
                ui.checkbox(&mut s.web_allow_proxy, "Let page reads use the system proxy (takes effect after restart)")
                    .on_hover_text("Off by default: a proxy resolves names itself, which skips Hinoko's public-address check.");
                ui.separator();
                ui.label(RichText::new("Data").size(16.0).strong());
                ui.label(RichText::new(self.engine.home().display().to_string()).monospace().small().color(muted()));
                ui.label(RichText::new("API keys are stored in settings.json in that folder (owner-only on Linux/macOS). Prefer an environment variable name to keep a key off disk.").color(muted()).small());
            });
        });
        if serde_json::to_string(&s).unwrap_or_default() != before {
            self.engine.set_settings(s);
        }
        if !open {
            self.dialog = Dialog::None;
            self.editing = None;
        }
    }

    fn start_edit(&mut self, preset: &Preset) {
        let p = preset.to_provider();
        self.editing = Some(Editing { args: p.args.join("\n"), provider: p, is_new: true });
        self.testing = false;
        *self.test_result.lock().unwrap() = None;
    }

    fn provider_editor(&mut self, ui: &mut egui::Ui, s: &mut hinoko_core::Settings) {
        let Some(is_new) = self.editing.as_ref().map(|e| e.is_new) else { return };
        ui.label(RichText::new(if is_new { "New model" } else { "Edit model" }).strong());
        let result = self.test_result.lock().unwrap().take();
        if let Some(r) = result {
            self.testing = false;
            self.toast_provider_test(r);
        }
        // keep the parsed argument list in step with the text box
        if let Some(ed) = self.editing.as_mut() {
            ed.provider.args = ed.args.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect();
        }
        let (ready, snapshot) = match self.editing.as_ref() {
            Some(ed) => (ed.provider.ready(), ed.provider.clone()),
            None => return,
        };
        let (mut test, mut save, mut cancel) = (false, false, false);
        ui.horizontal(|ui| {
            test = ui.add_enabled(ready && !self.testing, egui::Button::new(if self.testing { "Testing…" } else { "Test" })).clicked();
            save = ui.add_enabled(ready, egui::Button::new("Save")).clicked();
            cancel = ui.button("Cancel").clicked();
        });
        if test {
            self.testing = true;
            self.toast.clear();
            let slot = self.test_result.clone();
            self.engine.test_provider(snapshot.clone(), move |r| {
                *slot.lock().unwrap() = Some(r);
            });
        }
        if save {
            s.upsert(snapshot);
            self.editing = None;
            self.toast.clear();
            return;
        }
        if cancel {
            self.editing = None;
            self.toast.clear();
            return;
        }
        if !self.toast.is_empty() {
            let ok = self.toast.starts_with('✓');
            ui.colored_label(if ok { good() } else { ember() }, &self.toast);
        }
        let Some(ed) = self.editing.as_mut() else { return };
        egui::Grid::new("prov-grid").num_columns(2).spacing([10.0, 8.0]).show(ui, |ui| {
            ui.label("Name");
            ui.text_edit_singleline(&mut ed.provider.name);
            ui.end_row();
            ui.label("Type");
            egui::ComboBox::from_id_salt("prov-kind").selected_text(ed.provider.kind.label()).show_ui(ui, |ui| {
                ui.selectable_value(&mut ed.provider.kind, Kind::OpenAi, Kind::OpenAi.label());
                ui.selectable_value(&mut ed.provider.kind, Kind::Cli, Kind::Cli.label());
            });
            ui.end_row();
            match ed.provider.kind {
                Kind::OpenAi => {
                    ui.label("Base URL");
                    ui.add(egui::TextEdit::singleline(&mut ed.provider.base_url).desired_width(340.0).hint_text("https://api.openai.com/v1"));
                    ui.end_row();
                    ui.label("Model");
                    ui.text_edit_singleline(&mut ed.provider.model);
                    ui.end_row();
                    ui.label("API key");
                    ui.add(egui::TextEdit::singleline(&mut ed.provider.api_key).password(true).desired_width(340.0).hint_text("blank for local servers"));
                    ui.end_row();
                    ui.label("…or env variable");
                    ui.add(egui::TextEdit::singleline(&mut ed.provider.api_key_env).hint_text("e.g. OPENAI_API_KEY"));
                    ui.end_row();
                    ui.label("Tools");
                    ui.checkbox(&mut ed.provider.native_tools, "Native tool calling (turn off for models without it)");
                    ui.end_row();
                }
                Kind::Cli => {
                    ui.label("Command");
                    ui.add(egui::TextEdit::singleline(&mut ed.provider.command).hint_text("claude"));
                    ui.end_row();
                    ui.label("Arguments");
                    ui.add(egui::TextEdit::multiline(&mut ed.args).desired_rows(3).hint_text("one per line; the prompt arrives on stdin\n(or put {prompt} in an argument)"));
                    ui.end_row();
                }
            }
            ui.label("Timeout (s)");
            ui.add(egui::DragValue::new(&mut ed.provider.timeout_secs).range(5..=900));
            ui.end_row();
            if ed.provider.kind != Kind::Cli {
                ui.label("Temperature");
                let mut on = ed.provider.temperature.is_some();
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut on, "Set for this model").on_hover_text("Off = use the global temperature. Some models recommend their own, e.g. 1.0.").changed() {
                        ed.provider.temperature = if on { Some(0.7) } else { None };
                    }
                    if let Some(t) = ed.provider.temperature.as_mut() {
                        ui.add(egui::DragValue::new(t).range(0.0..=2.0).speed(0.05));
                    }
                });
                ui.end_row();
                ui.label("Top-p");
                let mut on = ed.provider.top_p.is_some();
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut on, "Set for this model").on_hover_text("Off = the server's default.").changed() {
                        ed.provider.top_p = if on { Some(0.95) } else { None };
                    }
                    if let Some(t) = ed.provider.top_p.as_mut() {
                        ui.add(egui::DragValue::new(t).range(0.05..=1.0).speed(0.01));
                    }
                });
                ui.end_row();
            }
            ui.label("Backup model");
            let others: Vec<(String, String)> = self.engine.settings().providers.iter().filter(|p| p.id != ed.provider.id).map(|p| (p.id.clone(), p.name.clone())).collect();
            let shown = others.iter().find(|o| o.0 == ed.provider.backup).map(|o| o.1.clone()).unwrap_or_else(|| "None".into());
            egui::ComboBox::from_id_salt("backup_model").selected_text(shown).show_ui(ui, |ui| {
                ui.selectable_value(&mut ed.provider.backup, String::new(), "None");
                for (pid, pname) in &others {
                    ui.selectable_value(&mut ed.provider.backup, pid.clone(), pname);
                }
            }).response.on_hover_text("Used automatically if this model is down, rate-limited or out of quota");
            ui.end_row();
            ui.label("Price per 1M tokens");
            ui.horizontal(|ui| {
                ui.label("in $");
                ui.add(egui::DragValue::new(&mut ed.provider.price_in).range(0.0..=1000.0).speed(0.05));
                ui.label("out $");
                ui.add(egui::DragValue::new(&mut ed.provider.price_out).range(0.0..=1000.0).speed(0.05));
            }).response.on_hover_text("Optional. Used for the cost estimate in Settings → Usage. Leave 0 for free or local models.");
            ui.end_row();
        });
        if ed.provider.kind == Kind::Cli {
            ui.colored_label(ember(), "A CLI agent runs with its own tools and your account's permissions, outside Hinoko's sandbox.");
        }
    }

    fn usage_section(&mut self, ui: &mut egui::Ui, s: &hinoko_core::settings::Settings) {
        use hinoko_core::usage;
        let rows = usage::all();
        ui.label(RichText::new("Counted on this computer. Tokens are the server's numbers when it reports them, otherwise an estimate; command-line agents are estimated.").color(muted()).small());
        for (label, n) in [("Today", 1), ("7 days", 7), ("30 days", 30)] {
            let from = usage::since(n);
            let mut by: std::collections::BTreeMap<String, usage::Row> = Default::default();
            for (day, pid, model, r) in rows.iter().filter(|r| r.0 >= from) {
                let _ = day;
                by.entry(format!("{pid}\u{1}{model}")).or_default().add(r);
            }
            let mut total = usage::Row::default();
            let mut dollars = 0.0;
            for (k, r) in &by {
                let pid = k.split('\u{1}').next().unwrap_or("");
                let (pi, po) = s.provider(pid).map(|p| (p.price_in, p.price_out)).unwrap_or((0.0, 0.0));
                dollars += usage::cost(r, pi, po);
                total.add(r);
            }
            egui::CollapsingHeader::new(format!("{label}: {} calls, {} in, {} out, about ${dollars:.2}", total.calls, total.prompt, total.completion)).id_salt(("usage", n)).show(ui, |ui| {
                if by.is_empty() {
                    ui.label(RichText::new("Nothing yet.").color(muted()));
                }
                for (k, r) in &by {
                    let (pid, model) = k.split_once('\u{1}').unwrap_or(("", k));
                    let (pi, po) = s.provider(pid).map(|p| (p.price_in, p.price_out)).unwrap_or((0.0, 0.0));
                    ui.label(format!("{model}: {} calls, {} in ({} cached), {} out, ${:.3}", r.calls, r.prompt, r.cached, r.completion, usage::cost(r, pi, po)));
                }
            });
        }
        ui.label(RichText::new("Set each model's price in its settings (Price per 1M tokens) to see dollars.").color(muted()).small());
    }

    fn skills_section(&mut self, ui: &mut egui::Ui) {
        let dir = self.engine.home().join("skills");
        ui.label(RichText::new("A skill is a folder with a SKILL.md (name and description, then instructions). The model sees only the list and loads one when a task matches. Code tab only. Also reads .claude/skills and .hinoko/skills inside a project.").color(muted()).small());
        let found = hinoko_core::skills::find(None, self.engine.home());
        if found.is_empty() {
            ui.label(RichText::new("No skills yet.").color(muted()));
        }
        for sk in &found {
            ui.horizontal(|ui| {
                ui.label(format!("{}: {}", sk.name, sk.description));
                if sk.file.starts_with(&dir) && ui.small_button("Remove").clicked() {
                    hinoko_core::skills::remove(self.engine.home(), &sk.name);
                }
            });
        }
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.skill_spec).hint_text("owner/repo, owner/repo/folder or GitHub link").desired_width(320.0));
            if ui.button("Install").clicked() && !self.skill_spec.trim().is_empty() {
                let (e, msg, spec) = (self.engine.clone(), self.skill_msg.clone(), self.skill_spec.trim().to_string());
                *msg.lock().unwrap() = "Installing…".into();
                self._rt.spawn(async move {
                    *msg.lock().unwrap() = match e.skill_install(&spec).await {
                        Ok(n) => format!("Installed: {}", n.join(", ")),
                        Err(err) => err,
                    };
                });
            }
        });
        ui.label(RichText::new(self.skill_msg.lock().unwrap().clone()).color(muted()).small());
        ui.horizontal(|ui| {
            if ui.button("Open skills folder").clicked() {
                let _ = std::fs::create_dir_all(&dir);
                ui.ctx().open_url(egui::OpenUrl::new_tab(format!("file://{}", dir.display())));
            }
            ui.label(RichText::new(dir.display().to_string()).color(muted()).small());
        });
    }

    fn update_state(&self) -> hinoko_core::update::Update {
        self.update.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn update_section(&mut self, ui: &mut egui::Ui, auto: &mut bool) {
        use hinoko_core::update::Update;
        ui.horizontal(|ui| {
            ui.label(format!("Hinoko {}", env!("CARGO_PKG_VERSION")));
            let idle = !matches!(self.update_state(), Update::Checking | Update::Downloading);
            if ui.add_enabled(idle, egui::Button::new("Check for updates")).clicked() {
                self.engine.update_check(self.update.clone());
            }
        });
        ui.checkbox(auto, "Check for updates when Hinoko starts").on_hover_text("It only tells you. Installing is always your click.");
        match self.update_state() {
            Update::Checking | Update::Downloading => {
                ui.horizontal(|ui| {
                    spin(ui);
                    ui.label(RichText::new("working\u{2026}").color(muted()).small());
                });
            }
            Update::UpToDate => {
                ui.label(RichText::new("You're up to date.").color(muted()).small());
            }
            Update::Failed(e) => {
                ui.add(egui::Label::new(RichText::new(e).color(ember()).small()).wrap());
            }
            Update::Available(rel) => {
                ui.label(RichText::new(format!("{} is available", rel.tag)).strong());
                if !rel.notes.is_empty() {
                    ui.add(egui::Label::new(RichText::new(&rel.notes).color(muted()).small()).wrap());
                }
                if cfg!(windows) && !rel.url.is_empty() {
                    if ui.button("Download and install").on_hover_text("Downloads the installer, closes Hinoko, updates it and opens it again. Your chats, bots and settings stay.").clicked() {
                        self.engine.update_download(self.update.clone(), rel.clone());
                    }
                } else if !rel.page.is_empty() {
                    ui.hyperlink_to("Open the release page", &rel.page);
                }
            }
            Update::Idle | Update::Ready(_) => {}
        }
    }

    fn add_rule(&mut self, bot_id: &str, action: &str, contains: &str, allow: bool) {
        let mut s = self.engine.settings();
        if s.rules.iter().any(|r| r.bot_id == bot_id && r.action == action && r.contains == contains && r.allow == allow) {
            return;
        }
        s.rules.push(hinoko_core::Rule { id: hinoko_core::model::nid("rule"), bot_id: bot_id.to_string(), action: action.to_string(), contains: contains.to_string(), allow });
        self.engine.set_settings(s);
    }

    fn toast_provider_test(&mut self, r: Result<String, String>) {
        self.toast = match r {
            Ok(reply) => format!("✓ Connected. The model replied: {}", reply.trim()),
            Err(e) => format!("Could not connect: {e}"),
        };
    }
}

impl eframe::App for HinokoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // a downloaded update installs itself: the installer closes Hinoko, replaces it and reopens it
        if let hinoko_core::update::Update::Ready(path) = self.update_state() {
            match hinoko_core::update::run_installer(&path) {
                Ok(()) => std::process::exit(0),
                Err(e) => *self.update.lock().unwrap_or_else(|x| x.into_inner()) = hinoko_core::update::Update::Failed(e),
            }
        }
        let settings = self.engine.settings();
        let look = settings.appearance.clone().sanitized();
        let key = serde_json::to_string(&look).unwrap_or_default();
        if key != self.applied {
            apply_appearance(ui.ctx(), &look);
            self.applied = key;
        }
        self.look = look;
        let any_busy = self.engine.store().bots.iter().any(|b| self.engine.is_busy(&b.id))
            || self.engine.store().chats.iter().any(|c| self.engine.is_chat_busy(&c.id));
        if any_busy {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
        }
        let r = rad(self.look.rounding);
        let ctx0 = ui.ctx().clone();
        let maximized = ctx0.input(|i| i.viewport().maximized.unwrap_or(false));
        egui::Panel::top("titlebar")
            .exact_size(36.0)
            .frame(egui::Frame::new().fill(mix(ink(), Color32::BLACK, 0.4)))
            .show(ui, |ui| self.titlebar(ui, maximized));
        if ctx0.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::B)) {
            self.sidebar_open = !self.sidebar_open;
        }
        if self.sidebar_open {
            egui::Panel::left("roster").default_size(230.0).size_range(150.0..=560.0).resizable(true).frame(egui::Frame::new().fill(panel()).inner_margin(12.0)).show(ui, |ui| self.roster(ui));
        }
        if self.mode == Mode::Bots && self.look.show_computer_panel {
            egui::Panel::right("desk").default_size(380.0).size_range(280.0..=640.0).frame(egui::Frame::new().fill(panel()).inner_margin(12.0)).show(ui, |ui| self.desk(ui));
        }
        let _ = r;
        egui::CentralPanel::default().frame(egui::Frame::new().fill(ink()).inner_margin(16.0)).show(ui, |ui| match self.mode {
            Mode::Chats => self.chat_view(ui),
            Mode::Bots => self.conversation(ui),
            Mode::Code => self.code_view(ui),
        });
        let ctx = ui.ctx().clone();
        self.dialogs(&ctx);
        window_edges(&ctx, maximized);
    }
}

/// Resize grips and a thin border for the borderless window.
/// Show the end of a long path.
fn clip_path(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    let t: String = s.chars().skip(n - max + 1).collect();
    format!("…{t}")
}

/// A unified diff with added and removed lines coloured.
fn diff_view(ui: &mut egui::Ui, diff: &str) {
    card(ui, ui.visuals().extreme_bg_color, None, 8.0, 8.0, false, |ui| {
        ui.set_width(ui.available_width());
        egui::ScrollArea::vertical().max_height(260.0).id_salt(ui.next_auto_id()).show(ui, |ui| {
            for line in diff.lines() {
                let (c, bg) = if line.starts_with("+++") || line.starts_with("---") {
                    (muted(), None)
                } else if line.starts_with('+') {
                    (egui::Color32::from_rgb(120, 200, 140), Some(egui::Color32::from_rgba_unmultiplied(60, 160, 90, 28)))
                } else if line.starts_with('-') {
                    (egui::Color32::from_rgb(240, 120, 130), Some(egui::Color32::from_rgba_unmultiplied(200, 60, 70, 28)))
                } else if line.starts_with("@@") {
                    (egui::Color32::from_rgb(130, 150, 220), None)
                } else {
                    (ui.visuals().text_color(), None)
                };
                let mut t = RichText::new(line).monospace().small().color(c);
                if let Some(b) = bg {
                    t = t.background_color(b);
                }
                ui.add(egui::Label::new(t).wrap_mode(egui::TextWrapMode::Extend));
            }
        });
    });
}

fn window_edges(ctx: &egui::Context, maximized: bool) {
    use egui::viewport::ResizeDirection as D;
    let rect = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("window-border")));
    painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, mix(panel(), ember(), 0.25)), egui::StrokeKind::Inside);
    if maximized {
        return;
    }
    let Some(pos) = ctx.pointer_hover_pos() else { return };
    let g = 6.0;
    let (l, r, t, b) = (pos.x - rect.left() < g, rect.right() - pos.x < g, pos.y - rect.top() < g, rect.bottom() - pos.y < g);
    let dir = match (l, r, t, b) {
        (true, _, true, _) => Some(D::NorthWest),
        (_, true, true, _) => Some(D::NorthEast),
        (true, _, _, true) => Some(D::SouthWest),
        (_, true, _, true) => Some(D::SouthEast),
        (true, ..) => Some(D::West),
        (_, true, ..) => Some(D::East),
        (_, _, true, _) => Some(D::North),
        (_, _, _, true) => Some(D::South),
        _ => None,
    };
    if let Some(d) = dir {
        ctx.set_cursor_icon(match d {
            D::North | D::South => egui::CursorIcon::ResizeVertical,
            D::East | D::West => egui::CursorIcon::ResizeHorizontal,
            D::NorthWest | D::SouthEast => egui::CursorIcon::ResizeNwSe,
            D::NorthEast | D::SouthWest => egui::CursorIcon::ResizeNeSw,
        });
        if ctx.input(|i| i.pointer.primary_pressed()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::BeginResize(d));
        }
    }
}


/// Split a trailing `**Sources**` / `**Sources looked at**` list off a message.
/// Returns the text without it and the (number, title, url) entries.
fn split_sources(text: &str) -> (String, Vec<(String, String, String)>) {
    for marker in ["\n\n**Sources**\n", "\n\n**Sources looked at**\n"] {
        if let Some(i) = text.rfind(marker) {
            let list = &text[i + marker.len()..];
            let mut out = Vec::new();
            for line in list.lines().filter(|l| !l.trim().is_empty()) {
                let parsed = line.split_once(". [").and_then(|(n, rest)| {
                    let (title, url) = rest.split_once("](")?;
                    let url = url.strip_suffix(')')?;
                    (url.starts_with("http") && n.chars().all(|c| c.is_ascii_digit())).then(|| (n.to_string(), title.to_string(), url.to_string()))
                });
                match parsed {
                    Some(e) => out.push(e),
                    None => return (text.to_string(), Vec::new()), // not our list
                }
            }
            if !out.is_empty() {
                return (text[..i].to_string(), out);
            }
        }
    }
    (text.to_string(), Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_list_is_split_off() {
        let (body, src) = split_sources("Answer [[1]](https://a.b/c).\n\n**Sources**\n1. [A site](https://a.b/c)\n2. [Two](https://x.y/z)\n");
        assert_eq!(body, "Answer [[1]](https://a.b/c).");
        assert_eq!(src.len(), 2);
        assert_eq!(src[1], ("2".into(), "Two".into(), "https://x.y/z".into()));
        let (b, s) = split_sources("No list here.");
        assert_eq!((b.as_str(), s.len()), ("No list here.", 0));
    }
}
