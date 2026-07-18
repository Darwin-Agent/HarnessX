#!/usr/bin/env bash
# scripts/install.sh — HarnessX one-click installer
#
# Usage:
#   bash scripts/install.sh          # interactive mode
#   bash scripts/install.sh --all    # non-interactive, install everything
#   bash scripts/install.sh -y       # same as --all
#
# curl one-liner:
#   curl -sSf https://<host>/scripts/install.sh | bash
#   curl -sSf https://<host>/scripts/install.sh | bash -s -- --all

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

REPO_URL="${HARNESSX_REPO_URL:-https://github.com/Darwin-Agent/HarnessX.git}"
INSTALL_DIR="${HARNESSX_INSTALL_DIR:-$HOME/.local/share/harnessx}"
BIN_DIR="$HOME/.local/bin"
PYTHON_VERSION="3.12"
NODE_MIN_VERSION=18

# ── Colors ────────────────────────────────────────────────────────────────────

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
header()  { echo -e "\n${BOLD}$*${NC}"; }

decode_base64() {
    if base64 --help 2>/dev/null | grep -q -- "--decode"; then
        base64 --decode
    else
        base64 -d 2>/dev/null || base64 -D
    fi
}

has_frontend_dist() {
    [[ -f "$INSTALL_DIR/frontend/dist/index.html" ]]
}

# ── Args ──────────────────────────────────────────────────────────────────────

YES=false
for arg in "$@"; do
    case "$arg" in
        --all|-y) YES=true ;;
    esac
done

# Prompt user; with --all/-y return the default immediately without asking
ask() {
    local prompt="$1"
    local default="${2:-Y}"
    if $YES; then
        echo "$default"
        return
    fi
    local choices="[Y/n]"
    [[ "$default" == [nN] ]] && choices="[y/N]"
    local answer
    read -r -p "$prompt $choices " answer </dev/tty
    echo "${answer:-$default}"
}

# ── §1  uv ───────────────────────────────────────────────────────────────────

install_uv() {
    header "§1  uv package manager"
    if command -v uv &>/dev/null; then
        success "uv already installed ($(uv --version))"
        return
    fi
    info "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Reload PATH (uv installs to ~/.local/bin or ~/.cargo/bin)
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if command -v uv &>/dev/null; then
        success "uv installed ($(uv --version))"
    else
        error "uv installation failed. Please install manually and retry:"
        error "  https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
}

# ── §2  Python 3.12 via uv ───────────────────────────────────────────────────

setup_python() {
    header "§2  Python $PYTHON_VERSION"
    info "Setting up Python $PYTHON_VERSION via uv (independent of system Python) ..."
    uv python install "$PYTHON_VERSION"
    success "Python $PYTHON_VERSION ready"
}

# ── §3  Repository ────────────────────────────────────────────────────────────

setup_repo() {
    header "§3  Repository"
    # Already installed (either git clone or bundle extract)
    if [[ -f "$INSTALL_DIR/pyproject.toml" ]]; then
        success "Sources already present at $INSTALL_DIR, skipping"
        return
    fi
    # Offline bundle: payload variables are set by scripts/bundle.sh
    if [[ -n "${BUNDLE_PAYLOAD:-}" ]]; then
        info "Extracting bundled sources to $INSTALL_DIR ..."
        mkdir -p "$INSTALL_DIR"
        echo "$BUNDLE_PAYLOAD" | decode_base64 | tar -xzf - -C "$INSTALL_DIR"
        touch "$INSTALL_DIR/.bundle-install"   # marker for update.sh
        success "Sources extracted"

        if [[ -n "${BUNDLE_FRONTEND_DIST_PAYLOAD:-}" ]]; then
            info "Extracting bundled frontend dist ..."
            mkdir -p "$INSTALL_DIR/frontend"
            echo "$BUNDLE_FRONTEND_DIST_PAYLOAD" | decode_base64 | tar -xzf - -C "$INSTALL_DIR/frontend"
            success "Bundled frontend dist extracted"
        fi
        if [[ -n "${BUNDLE_GATEWAY_CONSOLE_PAYLOAD:-}" ]]; then
            info "Extracting bundled gateway console dist ..."
            mkdir -p "$INSTALL_DIR/gateway/console"
            echo "$BUNDLE_GATEWAY_CONSOLE_PAYLOAD" | decode_base64 | tar -xzf - -C "$INSTALL_DIR/gateway/console"
            success "Bundled gateway console dist extracted"
        fi
        return
    fi
    # Online: git clone
    info "Cloning into $INSTALL_DIR ..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    success "Clone complete"
}

# ── §4  Node.js ───────────────────────────────────────────────────────────────

NODE_OK=false

_node_version() {
    node --version 2>/dev/null | sed 's/v//' | cut -d. -f1
}

check_node() {
    command -v node &>/dev/null && [[ "$(_node_version)" -ge $NODE_MIN_VERSION ]]
}

_install_node_macos() {
    if command -v brew &>/dev/null; then
        brew install node
    else
        error "Homebrew not found. Please install it first: https://brew.sh"
        return 1
    fi
}

_install_node_nvm() {
    info "Installing Node.js 20 via nvm ..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    # shellcheck source=/dev/null
    [[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh"
    nvm install 20 && nvm use 20
}

_install_node_linux() {
    if command -v apt-get &>/dev/null; then
        # Remove all Ubuntu nodejs-related packages that conflict with NodeSource v20.
        # Suppress errors: some packages may not be installed on this machine.
        sudo apt-get remove -y nodejs nodejs-doc libnode-dev libnode108 npm 2>/dev/null || true
        sudo apt-get autoremove -y 2>/dev/null || true

        # Try NodeSource v20. Do NOT trust its exit code — the setup script can return 0
        # even when GPG key import fails. Verify the installed version with check_node instead.
        if curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - \
                && sudo apt-get install -y nodejs \
                && check_node; then
            return 0
        fi

        warn "NodeSource install did not produce Node.js >= $NODE_MIN_VERSION, falling back to nvm"
        _install_node_nvm
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y nodejs npm
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm nodejs npm
    else
        _install_node_nvm
    fi
}

install_node() {
    case "$(uname -s)" in
        Darwin) _install_node_macos ;;
        Linux)  _install_node_linux ;;
        *)
            error "Unsupported platform: $(uname -s)"
            return 1
            ;;
    esac
}

handle_node() {
    header "§4  Node.js >= $NODE_MIN_VERSION (build Harness Lab frontend)"

    if check_node; then
        success "Node.js v$(_node_version) satisfied"
        NODE_OK=true
        return
    fi

    echo
    warn "Node.js >= $NODE_MIN_VERSION not found. It's needed to build/update the Lab frontend."
    echo "  Install options:"
    echo "    macOS:   brew install node"
    echo "    Ubuntu:  sudo apt install nodejs npm"
    echo "    Any:     https://github.com/nvm-sh/nvm"
    echo

    local ans
    ans=$(ask "Install Node.js now?")
    if [[ "$ans" =~ ^[yY] ]]; then
        if install_node && check_node; then
            success "Node.js v$(_node_version) installed"
            NODE_OK=true
        else
            warn "Node.js installation failed, skipping frontend build"
            warn "After installing Node.js, run: bash $INSTALL_DIR/scripts/build-frontend.sh"
        fi
    else
        warn "Skipping Node.js frontend build step"
        if has_frontend_dist; then
            info "Prebuilt frontend dist detected from bundle; hx lab UI can still run."
        else
            info "To enable the UI later, install Node.js and run: bash $INSTALL_DIR/scripts/build-frontend.sh"
        fi
    fi
}

# ── §5  Python package ────────────────────────────────────────────────────────

GATEWAY_INSTALLED=false

install_package() {
    header "§5  Python package"
    cd "$INSTALL_DIR"

    info "Creating Python $PYTHON_VERSION virtual environment ..."
    uv venv --python "$PYTHON_VERSION" .venv

    # Lab backend dependencies are part of the main package dependencies.
    info "uv pip install -e '.' ..."
    uv pip install --python .venv/bin/python -e "."

    success "harnessx installed"
}

# ── §5.5  Gateway (optional) ──────────────────────────────────────────────────

has_gateway_dist() {
    [[ -f "$INSTALL_DIR/gateway/console/dist/index.html" ]]
}

install_gateway() {
    if [[ ! -d "$INSTALL_DIR/gateway" ]]; then
        return
    fi

    header "§5.5  IM Gateway (optional)"
    echo "  The IM Gateway lets you interact with your agent through Feishu, Slack,"
    echo "  Discord, Telegram, or DingTalk."
    echo

    local ans
    ans=$(ask "Install IM Gateway?")
    if [[ ! "$ans" =~ ^[yY] ]]; then
        info "Skipping Gateway install"
        return
    fi

    echo "  Available channel extras: feishu, telegram, slack, discord, dingtalk, all"
    local extras=""
    if $YES; then
        extras="all"
        info "Non-interactive mode: installing all channel extras"
    else
        read -r -p "  Extras to install (comma-separated, or press Enter for none): " extras </dev/tty
    fi

    local pkg="$INSTALL_DIR/gateway"
    if [[ -n "$extras" ]]; then
        info "uv pip install -e 'gateway[$extras]' ..."
        uv pip install --python "$INSTALL_DIR/.venv/bin/python" -e "$pkg[$extras]"
    else
        info "uv pip install -e 'gateway' ..."
        uv pip install --python "$INSTALL_DIR/.venv/bin/python" -e "$pkg"
    fi

    GATEWAY_INSTALLED=true
    success "harnessx-gateway installed"
}

# ── §6  Frontend build ────────────────────────────────────────────────────────

build_frontend() {
    if ! $NODE_OK; then
        if has_frontend_dist; then
            success "Using prebuilt frontend dist -> frontend/dist/"
        else
            warn "Frontend dist not available; hx lab will expose API only until frontend is built"
        fi
        if $GATEWAY_INSTALLED && has_gateway_dist; then
            success "Using prebuilt gateway console dist -> gateway/console/dist/"
        elif $GATEWAY_INSTALLED; then
            warn "Gateway console dist not available; run: bash $INSTALL_DIR/scripts/build-frontend.sh --gateway"
        fi
        return
    fi

    header "§6  Build Harness Lab frontend"
    cd "$INSTALL_DIR/frontend"
    info "npm ci ..."
    npm ci --silent
    info "npm run build ..."
    npm run build
    success "Frontend built -> frontend/dist/"

    if $GATEWAY_INSTALLED; then
        info "Building Gateway console ..."
        cd "$INSTALL_DIR/gateway/console"
        npm ci --silent
        npm run build
        success "Gateway console built -> gateway/console/dist/"
    fi
}

# ── §7  Environment (PATH + workspace) ───────────────────────────────────────

_rc_file() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "$HOME/.zshrc"
    else
        echo "$HOME/.bashrc"
    fi
}

setup_env() {
    header "§7  Environment"
    mkdir -p "$BIN_DIR"

    # Symlink both entry points: hx (primary) and harnessx (compat)
    ln -sf "$INSTALL_DIR/.venv/bin/hx"        "$BIN_DIR/hx"
    ln -sf "$INSTALL_DIR/.venv/bin/harnessx"  "$BIN_DIR/harnessx"
    success "hx / harnessx -> $BIN_DIR"

    # Gateway entry points (only if gateway was installed)
    if $GATEWAY_INSTALLED; then
        ln -sf "$INSTALL_DIR/.venv/bin/hx-gateway"        "$BIN_DIR/hx-gateway"
        ln -sf "$INSTALL_DIR/.venv/bin/harnessx-gateway"  "$BIN_DIR/harnessx-gateway"
        success "hx-gateway / harnessx-gateway -> $BIN_DIR"
    fi

    local rc_file need_source=false
    rc_file="$(_rc_file)"

    local ws_dir="$HOME/.harnessx/workspace"

    # PATH
    if ! echo "$PATH" | grep -q "$BIN_DIR"; then
        {
            echo ""
            echo "# HarnessX"
            echo "export PATH=\"$BIN_DIR:\$PATH\""
        } >> "$rc_file"
        need_source=true
        success "PATH written to $rc_file"
    fi
    # Apply to the current process so subsequent steps (e.g. hx invocations) work
    export PATH="$BIN_DIR:$PATH"

    # Pin workspace to ~/.harnessx/workspace.
    # Without this, the CLI defaults to cwd/.harnessx/workspace,
    # scattering sessions across directories.
    if ! grep -q "HARNESSX_WORKSPACE" "$rc_file" 2>/dev/null; then
        echo "export HARNESSX_WORKSPACE=\"$ws_dir\"" >> "$rc_file"
        need_source=true
        success "HARNESSX_WORKSPACE=$ws_dir written to $rc_file"
    fi
    export HARNESSX_WORKSPACE="$ws_dir"

    if $need_source; then
        warn "Open a new terminal or run the following to use hx right now:"
        warn "  source $rc_file"
    fi
}

# ── §8  API Key ───────────────────────────────────────────────────────────────

guide_apikey() {
    header "§8  API Key"
    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        success "ANTHROPIC_API_KEY is set"
        return
    fi
    warn "ANTHROPIC_API_KEY not set"
    echo "  Add to your shell profile for permanent use:"
    echo '    export ANTHROPIC_API_KEY="sk-ant-..."'
    echo "  Or set it inline at runtime:"
    echo '    ANTHROPIC_API_KEY=sk-ant-... hx "your task"'
}

# ── §9  Summary ───────────────────────────────────────────────────────────────

print_summary() {
    local rc_file ws_dir
    rc_file="$(_rc_file)"
    ws_dir="$HOME/.harnessx/workspace"

    echo
    echo -e "${BOLD}══════════════════════════════════${NC}"
    echo -e "${GREEN}  HarnessX installed!${NC}"
    echo -e "${BOLD}══════════════════════════════════${NC}"
    echo
    echo "  Quick start:"
    echo "    hx \"write fizzbuzz\"            # one-shot task"
    echo "    hx                             # interactive chat"
    echo "    hx -p deep_research            # deep research mode"
    if has_frontend_dist; then
        echo "    hx lab                         # launch Harness Lab UI"
    else
        echo "    hx lab                         # requires Node.js >= 18 first"
        echo "                                      # bash $INSTALL_DIR/scripts/build-frontend.sh"
    fi
    if $GATEWAY_INSTALLED; then
        echo "    hx-gateway                     # start IM Gateway service"
    fi
    echo
    echo "  Install dir: $INSTALL_DIR"
    echo "  Workspace:   $ws_dir"
    if [[ -n "${BUNDLE_PAYLOAD:-}" ]]; then
        echo "  Update:      Download a new bundle and re-run it"
    else
        echo "  Update:      bash $INSTALL_DIR/scripts/update.sh"
    fi
    echo
    echo "  Full uninstall (remove binaries + code + workspace data):"
    if $GATEWAY_INSTALLED; then
        echo "    rm -f \"$BIN_DIR/hx\" \"$BIN_DIR/harnessx\" \"$BIN_DIR/hx-gateway\" \"$BIN_DIR/harnessx-gateway\""
    else
        echo "    rm -f \"$BIN_DIR/hx\" \"$BIN_DIR/harnessx\""
    fi
    echo "    rm -rf \"$INSTALL_DIR\" \"$HOME/.harnessx\""
    echo "    # Then edit $rc_file and remove:"
    echo "    #   1) the '# HarnessX' block (PATH export)"
    echo "    #   2) the HARNESSX_WORKSPACE export line"
    echo
}

# ── Banner ────────────────────────────────────────────────────────────────────

print_banner() {
    echo
    printf "${CYAN}"
    cat << 'LOGO'
 _   _   ___  ______  _   _  _____  _____  _____ __   __
| | | | / _ \ | ___ \| \ | ||  ___|/  ___|/  ___|\ \ / /
| |_| |/ /_\ \| |_/ /|  \| || |__  \ `--. \ `--.  \ V /
|  _  ||  _  ||    / | . ` ||  __|  `--. \ `--. \ /   \
| | | || | | || |\ \ | |\  || |___ /\__/ //\__/ // /^\ \
\_| |_/\_| |_/\_| \_|\_| \_/\____/ \____/ \____/ \/   \/
LOGO
    printf "${NC}\n"
    echo -e "  ${DIM}Agent Harness for LLM Agents${NC}"
    echo
    echo    "────────────────────────────────────────────────────────────────────────────"
    echo
}

# ── main ──────────────────────────────────────────────────────────────────────

main() {
    print_banner
    if $YES; then
        info "Non-interactive mode (--all), all dependencies will be installed"
        echo
    fi

    install_uv
    setup_python
    setup_repo
    handle_node
    install_package
    install_gateway
    build_frontend
    setup_env
    guide_apikey
    print_summary
}

main "$@"
