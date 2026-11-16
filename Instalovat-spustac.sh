#!/usr/bin/env bash
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/deploy/distribution/messages.sh"
validate_path() {
    case "$1" in
        *$'\n'*|*$'\r'*|*$'\t'*|*'"'*|*'\'*|*'$'*|*'%'*|*':'*|*','*)
            printf 'Unflatten W: %s\n' "$(message unsupported_path)" >&2
            exit 1 ;;
    esac
}
validate_path "${BASH_SOURCE[0]}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TARGET="${XDG_DATA_HOME:-$HOME/.local/share}/applications/unflatten-w.desktop"
validate_path "$ROOT"
validate_path "$TARGET"
mkdir -p -- "$(dirname -- "$TARGET")"
cat > "$TARGET" <<EOF
[Desktop Entry]
Type=Application
Name=Unflatten W 1.0
Comment=Documents to editable Word · EVOLIQ s.r.o.
Comment[sk]=Dokumenty do editovateľného Wordu · EVOLIQ s.r.o.
Exec="$ROOT/Spustit.sh"
Icon=$ROOT/web/public/apple-touch-icon.png
Terminal=true
Categories=Office;Utility;
StartupNotify=true
EOF
chmod 755 "$TARGET"
printf '%s\n' "$(message shortcut_ready)"
