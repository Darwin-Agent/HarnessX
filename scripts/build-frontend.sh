#!/usr/bin/env bash
# scripts/build-frontend.sh — Build Harness Lab frontend (and optionally Gateway console)
#
# Use this when:
#   - Node.js was skipped during install and you want to enable the Lab UI
#   - You pulled new code and need to rebuild the frontend
#
# Usage:
#   bash scripts/build-frontend.sh            # build Harness Lab frontend only
#   bash scripts/build-frontend.sh --gateway  # build Gateway console only
#   bash scripts/build-frontend.sh --all      # build both
#   bash $HOME/.local/share/harnessx/scripts/build-frontend.sh

set -euo pipefail

INSTALL_DIR="${HARNESSX_INSTALL_DIR:-$HOME/.local/share/harnessx}"
NODE_MIN_VERSION=18

BUILD_LAB=true
BUILD_GATEWAY=false

for arg in "$@"; do
    case "$arg" in
        --gateway) BUILD_LAB=false; BUILD_GATEWAY=true ;;
        --all)     BUILD_LAB=true;  BUILD_GATEWAY=true ;;
    esac
done

if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'
    DIM='\033[2m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''
    DIM=''; NC=''
fi

info()    { echo -e "${BLUE}▸${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}!${NC} $*"; }
error()   { echo -e "${RED}✗${NC} $*" >&2; }

# ── Check Node.js ─────────────────────────────────────────────────────────────

check_node() {
    if ! command -v node &>/dev/null; then
        error "Node.js not found. Please install Node.js >= $NODE_MIN_VERSION first."
        echo "  macOS:   brew install node"
        echo "  Ubuntu:  sudo apt install nodejs npm"
        echo "  Any:     https://github.com/nvm-sh/nvm"
        exit 1
    fi
    local ver
    ver=$(node --version | sed 's/v//' | cut -d. -f1)
    if [[ "$ver" -lt $NODE_MIN_VERSION ]]; then
        error "Node.js version too old (v$ver), requires >= $NODE_MIN_VERSION"
        exit 1
    fi
    success "Node.js $(node --version) OK"
}

# ── Check install dir ─────────────────────────────────────────────────────────

check_repo() {
    if $BUILD_LAB && [[ ! -d "$INSTALL_DIR/frontend" ]]; then
        error "Install directory not found: $INSTALL_DIR"
        error "Please run the installer first: bash $INSTALL_DIR/scripts/install.sh"
        exit 1
    fi
    if $BUILD_GATEWAY && [[ ! -d "$INSTALL_DIR/gateway/console" ]]; then
        error "Gateway not found at $INSTALL_DIR/gateway/console"
        error "Make sure the Gateway was included when installing."
        exit 1
    fi
}

# ── Ensure Lab backend deps are installed ────────────────────────────────────

ensure_lab_backend_deps() {
    local venv="$INSTALL_DIR/.venv"
    if [[ ! -f "$venv/bin/python" ]]; then
        error "Virtual environment not found: $venv"
        error "Please run the installer first: bash $INSTALL_DIR/scripts/install.sh"
        exit 1
    fi

    # FastAPI is required by `hx lab`; if missing, reinstall the package.
    if ! "$venv/bin/python" -c "import fastapi" &>/dev/null; then
        info "Reinstalling package to ensure Lab backend dependencies ..."
        uv pip install --python "$venv/bin/python" -e "$INSTALL_DIR"
        success "Lab backend dependencies ready"
    fi
}

# ── Build ─────────────────────────────────────────────────────────────────────

build() {
    if $BUILD_LAB; then
        info "Building Harness Lab frontend ..."
        cd "$INSTALL_DIR/frontend"
        npm ci --silent
        npm run build
        success "Frontend built -> $INSTALL_DIR/frontend/dist/"
    fi

    if $BUILD_GATEWAY; then
        info "Building Gateway console ..."
        cd "$INSTALL_DIR/gateway/console"
        npm ci --silent
        npm run build
        success "Gateway console built -> $INSTALL_DIR/gateway/console/dist/"
    fi
}

# ── main ──────────────────────────────────────────────────────────────────────

print_banner() {
    echo
    printf "${CYAN}"
    cat << 'LOGO'
 _  _    _    ___  _  _  ___  ___  ___ __  __
| || |  /_\  | _ \| \| || __|/ __|/ __| \ \/ /
| __ | / _ \ |   /| .` || _| \__ \\__ \  >  <
|_||_|/_/ \_\|_|_\|_|\_||___||___/|___//_/\_\
LOGO
    printf "${NC}"
    echo
    echo -e "  ${DIM}Frontend Build  ·  Agent Harness for LLM Agents${NC}"
    echo
    echo    "────────────────────────────────────────────────────────────"
    echo
}

main() {
    print_banner
    check_node
    check_repo
    if $BUILD_LAB; then
        ensure_lab_backend_deps
    fi
    build
    echo
    echo "  You can now run:"
    if $BUILD_LAB;     then echo "    hx lab"; fi
    if $BUILD_GATEWAY; then echo "    hx-gateway"; fi
    echo
}

main "$@"
