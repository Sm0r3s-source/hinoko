# Publish the version in Cargo.toml: commit, tag it vX.Y.Z and push.
# GitHub Actions then builds Hinoko-Setup.exe and publishes the release; installed copies offer the update.
# First time only:  git init; git branch -M main; git remote add origin https://github.com/Sm0r3s-source/hinoko.git
$v = (Select-String -Path Cargo.toml -Pattern '^version = "(.*)"').Matches[0].Groups[1].Value
git add -A
git commit -m "Hinoko v$v"
git tag "v$v"
git push -u origin HEAD "v$v"
