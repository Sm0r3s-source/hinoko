fn main() {
    // Embed the icon in the Windows .exe so the file, taskbar and shortcuts show it.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../assets/hinoko.ico");
        res.set("ProductName", "Hinoko");
        res.set("FileDescription", "Hinoko");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed icon: {e}");
        }
    }
}
