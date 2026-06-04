# yt-dlp Skill

Codex skill for installing, configuring, using, and troubleshooting the `yt-dlp` CLI across Windows, macOS, Linux, and Android/Termux.

## Contents

- `SKILL.md`: skill interface and workflow guidance.
- `agents/openai.yaml`: UI metadata for Codex skill discovery.
- `references/installation.md`: cross-platform install and update commands based on official yt-dlp documentation.
- `references/command-patterns.md`: common download, format, subtitle, cookie, post-processing, and config command patterns.
- `references/windows-powershell.md`: Windows quoting, PATH, ffmpeg, and filename guidance.
- `references/troubleshooting.md`: stale extractor, auth, format, ffmpeg, TLS, and playlist diagnostics.
- `scripts/check_ytdlp.ps1`: Windows environment diagnostic with GitHub latest stable release comparison.
- `scripts/check_ytdlp.sh`: macOS/Linux shell diagnostic with GitHub latest stable release comparison.

## Validation

Validate the skill metadata with:

```powershell
python "$env:USERPROFILE\.codex\skills\.system\skill-creator\scripts\quick_validate.py" ".\yt-dlp"
```

On Unix-like systems, run the diagnostic script with:

```bash
sh ./yt-dlp/scripts/check_ytdlp.sh
```

On Windows, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\yt-dlp\scripts\check_ytdlp.ps1
```

Both diagnostics compare `yt-dlp --version` with the latest stable GitHub release from `https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest`. The check is non-blocking: if GitHub is unavailable, the script reports an unknown latest version instead of failing. Use `-SkipLatestCheck` in PowerShell, `--skip-latest-check` in Unix shells, or `YT_DLP_SKIP_LATEST_CHECK=1` to skip the network check.

## Notes

The install commands in `references/installation.md` intentionally point back to official yt-dlp routes and preserve the rule that each installation should be updated through the same route used to install it. Package-manager packages can lag behind upstream yt-dlp, so version checks are part of the workflow. The skill should recommend updates when needed, but not run update commands unless the user explicitly approves.
