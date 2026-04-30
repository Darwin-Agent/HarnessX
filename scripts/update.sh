#!/usr/bin/env bash
# scripts/update.sh — Update HarnessX to the latest version
#
# Runs: git pull + reinstall Python package + rebuild frontend (if Node.js available)
#
# Usage:
#   bash scripts/update.sh
#   bash $HOME/.local/share/harnessx/scripts/update.sh

set -euo pipefail

INSTALL_DIR="${HARNESSX_INSTALL_DIR:-$HOME/.local/share/harnessx}"
PYTHON_VERSION="3.12"
NODE_MIN_VERSION=18

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

# ── Check install dir ─────────────────────────────────────────────────────────

check_install() {
    if [[ -f "$INSTALL_DIR/.bundle-install" ]]; then
        warn "This installation was created from an offline bundle."
        warn "Git-based updates are not available."
        warn "To update, download a new bundle and re-run it."
        exit 0
    fi
    if [[ ! -d "$INSTALL_DIR/.git" ]]; then
        error "HarnessX install directory not found: $INSTALL_DIR"
        error "Please run the installer first: bash scripts/install.sh"
        exit 1
    fi
}

# ── §1  Pull latest ───────────────────────────────────────────────────────────

pull_latest() {
    header "§1  Pull latest"
    cd "$INSTALL_DIR"

    local before after
    before=$(git rev-parse HEAD)
    git pull --ff-only
    after=$(git rev-parse HEAD)

    if [[ "$before" == "$after" ]]; then
        success "Already up to date ($(git log -1 --format='%h %s'))"
    else
        success "Updated: $before -> $after"
        git log --oneline "$before..$after"
    fi
}

# ── §2  Reinstall Python package ──────────────────────────────────────────────

reinstall_package() {
    header "§2  Python package"
    cd "$INSTALL_DIR"

    local venv=".venv"
    if [[ ! -f "$venv/bin/python" ]]; then
        warn "Virtual environment not found, recreating ..."
        uv venv --python "$PYTHON_VERSION" "$venv"
    fi

    # Lab backend dependencies are in the main package; reinstall base package.
    info "uv pip install -e '.' ..."
    uv pip install --python "$venv/bin/python" -e "."

    success "Python package updated"

    # Update gateway if it was previously installed
    if [[ -f "$venv/bin/hx-gateway" ]] && [[ -d "$INSTALL_DIR/gateway" ]]; then
        info "Updating harnessx-gateway ..."
        uv pip install --python "$venv/bin/python" -e "$INSTALL_DIR/gateway"
        success "harnessx-gateway updated"
    fi
}

# ── §3  Rebuild frontend ──────────────────────────────────────────────────────

rebuild_frontend() {
    header "§3  Frontend"

    if ! command -v node &>/dev/null; then
        warn "Node.js not found, skipping frontend rebuild"
        return
    fi

    local ver
    ver=$(node --version | sed 's/v//' | cut -d. -f1)
    if [[ "$ver" -lt $NODE_MIN_VERSION ]]; then
        warn "Node.js version too old (v$ver), skipping frontend rebuild"
        return
    fi

    cd "$INSTALL_DIR"
    local changed
    changed=$(git diff HEAD@{1} --name-only 2>/dev/null || true)

    # Harness Lab frontend
    local lab_changed
    lab_changed=$(echo "$changed" | grep '^frontend/' || true)
    if [[ -z "$lab_changed" ]] && [[ -d "frontend/dist" ]]; then
        success "No changes in frontend/, skipping rebuild"
    else
        info "Rebuilding Harness Lab frontend ..."
        cd "$INSTALL_DIR/frontend"
        npm ci --silent
        npm run build
        success "Frontend rebuilt"
    fi

    # Gateway console (only if gateway is installed)
    local venv="$INSTALL_DIR/.venv"
    if [[ -f "$venv/bin/hx-gateway" ]] && [[ -d "$INSTALL_DIR/gateway/console" ]]; then
        local gw_changed
        gw_changed=$(echo "$changed" | grep '^gateway/console/' || true)
        if [[ -z "$gw_changed" ]] && [[ -d "$INSTALL_DIR/gateway/console/dist" ]]; then
            success "No changes in gateway/console/, skipping rebuild"
        else
            info "Rebuilding Gateway console ..."
            cd "$INSTALL_DIR/gateway/console"
            npm ci --silent
            npm run build
            success "Gateway console rebuilt"
        fi
    fi
}

# ── Summary ───────────────────────────────────────────────────────────────────

print_summary() {
    echo
    echo -e "${BOLD}══════════════════════════════════${NC}"
    echo -e "${GREEN}  HarnessX updated!${NC}"
    echo -e "${BOLD}══════════════════════════════════${NC}"
    echo
    echo "  Current version: $(cd "$INSTALL_DIR" && git log -1 --format='%h %s')"
    echo
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
    echo -e "  ${DIM}Updater  ·  Agent Harness for LLM Agents${NC}"
    echo
    echo    "────────────────────────────────────────────────────────────"
    echo
}

main() {
    print_banner
    check_install
    pull_latest
    reinstall_package
    rebuild_frontend
    print_summary
}

main "$@"
