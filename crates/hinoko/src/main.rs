#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod cli;

use std::sync::Arc;

use hinoko_core::{Engine, Paths};

fn icon() -> eframe::egui::IconData {
    // the Hinoko star on a navy tile (same 16x16 grid as app::STAR, 4 px per cell)
    let n = 64usize;
    let mut rgba = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let (u, v) = (x as f32 / 2.0, y as f32 / 2.0);
            let (cx, cy) = (u.clamp(7.0, 25.0), v.clamp(7.0, 25.0));
            if (u - cx).hypot(v - cy) > 7.0 {
                continue;
            }
            let c = match app::STAR[y / 4].as_bytes()[x / 4] {
                b'w' => [255, 255, 255],
                b'b' => [236, 236, 236],
                b'c' => [250, 250, 250],
                _ => [0x0e, 0x12, 0x30],
            };
            let i = (y * n + x) * 4;
            rgba[i..i + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    eframe::egui::IconData { rgba, width: n as u32, height: n as u32 }
}

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if cli::wants_cli(&args) {
        attach_parent_console();
        std::process::exit(cli::run(&args));
    }
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("runtime");
    let slot: Arc<std::sync::Mutex<Option<eframe::egui::Context>>> = Arc::new(std::sync::Mutex::new(None));
    let repaint = {
        let slot = slot.clone();
        Arc::new(move || {
            if let Some(ctx) = slot.lock().unwrap().as_ref() {
                ctx.request_repaint();
            }
        })
    };
    let engine = match Engine::new(Paths::discover(), rt.handle().clone(), repaint) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("hinoko: could not start: {e}");
            std::process::exit(1);
        }
    };
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Hinoko")
            .with_decorations(false) // Hinoko draws its own title bar
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 520.0])
            .with_icon(icon()),
        ..Default::default()
    };
    eframe::run_native(
        "Hinoko",
        options,
        Box::new(move |cc| {
            *slot.lock().unwrap() = Some(cc.egui_ctx.clone());
            Ok(Box::new(app::HinokoApp::new(cc, engine, rt)))
        }),
    )
}

/// Release builds on Windows have no console of their own; borrow the parent's so `hinoko bots` prints.
#[cfg(all(windows, not(debug_assertions)))]
fn attach_parent_console() {
    extern "system" {
        fn AttachConsole(pid: u32) -> i32;
    }
    unsafe {
        AttachConsole(u32::MAX);
    }
}

#[cfg(not(all(windows, not(debug_assertions))))]
fn attach_parent_console() {}
