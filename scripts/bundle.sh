#!/usr/bin/env bash
# scripts/bundle.sh — Build a self-contained offline installer
#
# Embeds repository sources as a base64 payload inside install.sh, producing a
# single script that can install without cloning.
#
# Requirements: git, gzip, base64, tar
#
# Usage:
#   bash scripts/bundle.sh              # outputs dist/harnessx-bundle.sh
#   bash scripts/bundle.sh --ref v1.2   # bundle a specific tag/branch/commit
#   bash scripts/bundle.sh --out /tmp/harnessx-installer.sh
#
# The generated bundle:
#   - Extracts sources to INSTALL_DIR without cloning (no git needed at runtime)
#   - Includes frontend/dist so `hx lab` can serve UI without building on target
#   - Runs install.sh logic (uv, Python, optional Node.js/frontend rebuild, etc.)
#   - Marks the install as a bundle install (.bundle-install) so update.sh
#     can give a helpful message instead of failing
#
# Note: uv (Python package manager) is still fetched from astral.sh if not
# already installed on the target machine.  For a truly air-gapped install,
# pre-install uv manually: https://docs.astral.sh/uv/getting-started/installation/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/dist"
OUT_FILE=""
GIT_REF="HEAD"

if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; DIM=''; NC=''
fi

info()    { echo -e "${BLUE}▸${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}!${NC} $*"; }
error()   { echo -e "${RED}✗${NC} $*" >&2; }
header()  { echo -e "\n${BOLD}$*${NC}"; }
b64_encode() { base64 | tr -d '\n'; }

# ── Args ──────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)   GIT_REF="$2"; shift 2 ;;
        --out)   OUT_FILE="$2"; shift 2 ;;
        *)       error "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Resolve output path ───────────────────────────────────────────────────────

SHORT_REF="$(cd "$REPO_ROOT" && git rev-parse --short "$GIT_REF" 2>/dev/null || echo "$GIT_REF")"

if [[ -z "$OUT_FILE" ]]; then
    mkdir -p "$OUT_DIR"
    OUT_FILE="$OUT_DIR/harnessx-bundle-${SHORT_REF}.sh"
fi

# ── Validate ──────────────────────────────────────────────────────────────────

if [[ ! -d "$REPO_ROOT/.git" ]]; then
    error "Not a git repository: $REPO_ROOT"
    exit 1
fi

for cmd in git gzip base64 tar; do
    if ! command -v "$cmd" &>/dev/null; then
        error "Required tool not found: $cmd"
        exit 1
    fi
done

# ── Build payload ─────────────────────────────────────────────────────────────

header "Building offline bundle"

COMMIT_HASH="$(cd "$REPO_ROOT" && git rev-parse --short "$GIT_REF")"
COMMIT_MSG="$(cd  "$REPO_ROOT" && git log -1 --format='%s' "$GIT_REF")"
TARGET_HASH="$(cd "$REPO_ROOT" && git rev-parse "$GIT_REF")"
HEAD_HASH="$(cd "$REPO_ROOT" && git rev-parse HEAD)"

info "Archiving $GIT_REF ($COMMIT_HASH) ..."
PAYLOAD="$(cd "$REPO_ROOT" && git archive "$GIT_REF" | gzip | b64_encode)"
PAYLOAD_KB=$(( ${#PAYLOAD} / 1024 ))
success "Payload: ${PAYLOAD_KB} KB (base64-encoded gzip tar)"

# Include frontend dist so bundle installs can run `hx lab` without Node.js.
FRONTEND_PAYLOAD=""
GATEWAY_CONSOLE_PAYLOAD=""

_embed_or_build_frontend() {
    local src_dir="$1" label="$2" var_name="$3"
    local dist_dir="$src_dir/dist"

    if [[ "$TARGET_HASH" != "$HEAD_HASH" ]]; then
        warn "Ref '$GIT_REF' is not HEAD; skipping $label dist embedding to avoid version mismatch"
        warn "Target machine can build via: bash \$INSTALL_DIR/scripts/build-frontend.sh"
        return
    fi

    if [[ -f "$dist_dir/index.html" ]]; then
        info "Embedding $label dist ..."
        local payload
        payload="$(cd "$src_dir" && tar -czf - dist | b64_encode)"
        local kb=$(( ${#payload} / 1024 ))
        success "$label payload: ${kb} KB"
        printf -v "$var_name" '%s' "$payload"
    elif command -v node &>/dev/null && command -v npm &>/dev/null; then
        info "$label dist missing; building it once for the bundle ..."
        if (cd "$src_dir" && npm ci --silent && npm run build); then
            local payload
            payload="$(cd "$src_dir" && tar -czf - dist | b64_encode)"
            local kb=$(( ${#payload} / 1024 ))
            success "$label payload: ${kb} KB"
            printf -v "$var_name" '%s' "$payload"
        else
            warn "Failed to build $label dist while bundling"
            warn "Target machine must run: bash \$INSTALL_DIR/scripts/build-frontend.sh"
        fi
    else
        warn "$label dist missing and node/npm not available on bundling machine"
        warn "Target machine must build $label for UI support"
    fi
}

_embed_or_build_frontend "$REPO_ROOT/frontend" "Harness Lab frontend" FRONTEND_PAYLOAD

if [[ -d "$REPO_ROOT/gateway/console" ]]; then
    _embed_or_build_frontend "$REPO_ROOT/gateway/console" "Gateway console" GATEWAY_CONSOLE_PAYLOAD
fi

# ── Generate bundle script ────────────────────────────────────────────────────

info "Writing bundle to $OUT_FILE ..."

{
    # ── Header ────────────────────────────────────────────────────────────────
    printf '#!/usr/bin/env bash\n'
    printf '# HarnessX self-extracting offline installer\n'
    printf '# Generated: %s\n'   "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '# Commit:    %s %s\n' "$COMMIT_HASH" "$COMMIT_MSG"
    printf '#\n'
    printf '# Usage:\n'
    printf '#   bash harnessx-bundle.sh          # interactive\n'
    printf '#   bash harnessx-bundle.sh --all    # non-interactive, install everything\n'
    printf '#\n'
    printf '# No git clone required on target machine.\n'
    printf '# uv (Python package manager) will be fetched from astral.sh if not present.\n'
    printf '\n'

    # ── Embedded payload ──────────────────────────────────────────────────────
    # Single-quoted to avoid any shell interpretation; base64 chars are safe.
    printf "BUNDLE_PAYLOAD='%s'\n" "$PAYLOAD"
    if [[ -n "$FRONTEND_PAYLOAD" ]]; then
        printf "BUNDLE_FRONTEND_DIST_PAYLOAD='%s'\n" "$FRONTEND_PAYLOAD"
    fi
    if [[ -n "$GATEWAY_CONSOLE_PAYLOAD" ]]; then
        printf "BUNDLE_GATEWAY_CONSOLE_PAYLOAD='%s'\n" "$GATEWAY_CONSOLE_PAYLOAD"
    fi
    printf '\n'

    # ── install.sh body (skip shebang line) ───────────────────────────────────
    tail -n +2 "$SCRIPT_DIR/install.sh"

} > "$OUT_FILE"

chmod +x "$OUT_FILE"

# ── Report ────────────────────────────────────────────────────────────────────

SIZE="$(du -sh "$OUT_FILE" | cut -f1)"

echo
echo -e "${BOLD}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  Bundle ready!${NC}"
echo -e "${BOLD}══════════════════════════════════════════${NC}"
echo
echo "  File:    $OUT_FILE"
echo "  Size:    $SIZE"
echo "  Commit:  $COMMIT_HASH  $COMMIT_MSG"
if [[ -n "$FRONTEND_PAYLOAD" ]]; then
    echo "  Frontend: embedded (frontend/dist)"
else
    echo "  Frontend: not embedded (target machine must run build-frontend.sh)"
fi
if [[ -n "$GATEWAY_CONSOLE_PAYLOAD" ]]; then
    echo "  Gateway console: embedded (gateway/console/dist)"
elif [[ -d "$REPO_ROOT/gateway/console" ]]; then
    echo "  Gateway console: not embedded (target machine must run build-frontend.sh --gateway)"
fi
echo
echo "  Distribute this single file.  Recipients run:"
echo "    bash harnessx-bundle-${SHORT_REF}.sh"
echo "    bash harnessx-bundle-${SHORT_REF}.sh --all   # non-interactive"
echo
echo "  Full uninstall (on target machine, after install):"
echo "    rm -f ~/.local/bin/hx ~/.local/bin/harnessx"
echo "    rm -rf ~/.local/share/harnessx ~/.harnessx"
echo "    # remove HarnessX PATH / HARNESSX_WORKSPACE lines from ~/.bashrc or ~/.zshrc"
echo
echo -e "  ${DIM}Note: uv is still downloaded at install time if not present.${NC}"
echo -e "  ${DIM}For fully air-gapped installs, pre-install uv on the target machine.${NC}"
echo -e "  ${DIM}  https://docs.astral.sh/uv/getting-started/installation/${NC}"
echo
