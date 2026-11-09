#!/usr/bin/env sh
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for task_file in uv.lock pyproject.toml launcher.py app.py; do
    if [ ! -f "$task_root/$task_file" ]; then
        printf '%s\n' 'Extract the entire Whisper Pro archive before opening the launcher.' >&2
        exit 1
    fi
done
task_os=$(uname -s)
case "$(uname -m)" in
    arm64|aarch64) task_arch=aarch64 ;;
    x86_64|amd64) task_arch=x86_64 ;;
    *) printf '%s\n' 'Whisper Pro requires an x64 or ARM64 computer.' >&2; exit 1 ;;
esac
case "$task_os" in
    Darwin)
        task_target=apple-darwin
        task_version=$(sw_vers -productVersion)
        [ "${task_version%%.*}" -ge 14 ] || { printf '%s\n' 'Upgrade to macOS 14 or newer and reopen Start.command.' >&2; exit 1; }
        ;;
    Linux)
        task_target=unknown-linux-gnu
        task_glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
        if ! printf '%s\n' "$task_glibc" | awk '/^glibc / { split($2, v, "."); if (v[1] > 2 || (v[1] == 2 && v[2] >= 28)) ok=1 } END { exit !ok }'; then
            printf '%s\n' 'Whisper Pro needs a Linux distribution with glibc 2.28 or newer.' >&2
            exit 1
        fi
        ;;
    *) printf '%s\n' 'Use Start.sh on Linux or Start.command on macOS.' >&2; exit 1 ;;
esac
if [ -n "${WHISPER_PRO_DATA_DIR:-}" ]; then
    task_data="$WHISPER_PRO_DATA_DIR"
elif [ "$(uname -s)" = "Darwin" ]; then
    task_data="$HOME/Library/Application Support/WhisperPro"
else
    task_data="${XDG_DATA_HOME:-$HOME/.local/share}/whisper-pro"
fi
task_tools="$task_data/tools/uv-0.8.14"
task_uv="$task_tools/uv"
task_env="$task_data/app-environment-v2"
if command -v sha256sum >/dev/null 2>&1; then
    task_sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    task_sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    printf '%s\n' 'Install sha256sum (coreutils) or shasum, then reopen the launcher.' >&2
    exit 1
fi
if [ ! -x "$task_uv" ]; then
    for task_program in curl tar; do
        if ! command -v "$task_program" >/dev/null 2>&1; then
            printf 'Install %s with your system package manager, then reopen the launcher.\n' "$task_program" >&2
            exit 1
        fi
    done
fi
task_digest=$(task_sha256 "$task_root/uv.lock")
if [ ! -x "$task_env/bin/python" ] || [ ! -f "$task_env/.whisper-setup.json" ] || [ "$(cat "$task_env/.whisper-setup.json")" != "\"$task_digest\"" ]; then
    printf 'Setup will use Python 3.12 and install the application libraries in %s.\n' "$task_data"
    printf '%s\n' 'Keep your Internet connection on during setup. The app opens when setup finishes.'
    if [ "${WHISPER_PRO_SETUP_YES:-}" != 1 ]; then
        printf '%s' 'Set up and launch? [y/N] '
        read -r task_answer || task_answer=
        case "$task_answer" in y|Y|yes|YES|Yes) ;; *) printf '%s\n' 'Setup cancelled.'; exit 0 ;; esac
    fi
fi
if [ ! -x "$task_uv" ]; then
    mkdir -p "$task_tools"
    printf '%s\n' 'Downloading the setup tool...'
    task_asset="uv-$task_arch-$task_target.tar.gz"
    task_url="https://github.com/astral-sh/uv/releases/download/0.8.14/$task_asset"
    curl -fL --retry 3 "$task_url" -o "$task_tools/$task_asset"
    curl -fL --retry 3 "$task_url.sha256" -o "$task_tools/checksum.txt"
    task_expected=$(awk '{print $1}' "$task_tools/checksum.txt")
    task_actual=$(task_sha256 "$task_tools/$task_asset")
    [ "$task_actual" = "$task_expected" ] || { printf '%s\n' 'Download verification failed'; exit 1; }
    tar -xzf "$task_tools/$task_asset" -C "$task_tools"
    task_found=$(find "$task_tools" -type f -name uv | head -n 1)
    [ -n "$task_found" ] || exit 1
    [ "$task_found" = "$task_uv" ] || cp "$task_found" "$task_uv"
    chmod +x "$task_uv"
fi
export WHISPER_PRO_DATA_DIR="$task_data"
export UV_PROJECT_ENVIRONMENT="$task_env"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$task_data/cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export PYTHONUTF8=1
exec "$task_uv" run --project "$task_root" --locked --no-dev --python 3.12 python "$task_root/launcher.py"
