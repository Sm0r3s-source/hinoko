#!/bin/sh
# Builds Hinoko and adds it to your app menu.  Run:  sh install-linux.sh
set -e
command -v cargo >/dev/null || { echo "Install Rust first: https://rustup.rs"; exit 1; }
cargo build --release -p hinoko
mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications" "$HOME/.local/share/icons"
cp target/release/hinoko "$HOME/.local/bin/hinoko"
cp assets/hinoko.png "$HOME/.local/share/icons/hinoko.png"
cat > "$HOME/.local/share/applications/hinoko.desktop" <<EOD
[Desktop Entry]
Type=Application
Name=Hinoko
Comment=Bots, chats and deep research in one app
Exec=$HOME/.local/bin/hinoko
Icon=$HOME/.local/share/icons/hinoko.png
Categories=Utility;
Terminal=false
EOD
echo "Done. Find Hinoko in your app menu (macOS: run 'cargo build --release' and drag target/release/hinoko anywhere)."
