#!/usr/bin/env bash
# scripts/uninstall.sh — HarnessX complete uninstaller
#
# Usage:
#   bash scripts/uninstall.sh          # interactive (asks before each step)
#   bash scripts/uninstall.sh --yes    # non-interactive, remove everything
#   hx uninstall                       # same as interactive mode
#   hx uninstall --yes                 # same as non-interactive

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

INSTALL_DIR="${HARNESSX_INSTALL_DIR:-$HOME/.local/share/harnessx}"
BIN_DIR="$HOME/.local/bin"

# ── Colors ────────────────────────────────────────────────────────────────────

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

# ── Args ──────────────────────────────────────────────────────────────────────

YES=false
for arg in "$@"; do
    case "$arg" in
        --yes|-y) YES=true ;;
    esac
done

ask_yes() {
    local prompt="$1"
    if $YES; then
        echo "y"
        return
    fi
    local answer
    read -r -p "$prompt [y/N] " answer </dev/tty
    echo "${answer:-n}"
}

# ── Helpers ───────────────────────────────────────────────────────────────────

_rc_file() {
    if [[ -n "${ZSH_VERSION:-}" ]] || [[ "$SHELL" == */zsh ]]; then
        echo "$HOME/.zshrc"
    elif [[ -f "$HOME/.bash_profile" ]]; then
        echo "$HOME/.bash_profile"
    else
        echo "$HOME/.bashrc"
    fi
}

_remove_rc_lines() {
    local rc_file="$1"
    [[ -f "$rc_file" ]] || return 0

    # Lines added by install.sh: the "# HarnessX" comment, PATH export, and HARNESSX_WORKSPACE
    local tmp
    tmp=$(mktemp)
    grep -v "# HarnessX" "$rc_file" \
        | grep -v 'export PATH=.*harnessx\|export PATH=.*\.local/bin' \
        | grep -v 'HARNESSX_WORKSPACE' \
        > "$tmp" || true
    mv "$tmp" "$rc_file"
}

# ── Uninstall steps ───────────────────────────────────────────────────────────

remove_binaries() {
    header "Step 1/4  Remove CLI binaries"
    local bins=("$BIN_DIR/hx" "$BIN_DIR/harnessx" "$BIN_DIR/hx-gateway" "$BIN_DIR/harnessx-gateway")
    local found=()
    for b in "${bins[@]}"; do
        [[ -e "$b" || -L "$b" ]] && found+=("$b")
    done

    if [[ ${#found[@]} -eq 0 ]]; then
        info "No binaries found in $BIN_DIR — skipping"
        return
    fi

    info "Will remove: ${found[*]}"
    local ans
    ans=$(ask_yes "Remove CLI binaries?")
    if [[ "${ans,,}" =~ ^y ]]; then
        rm -f "${found[@]}"
        success "Binaries removed"
    else
        warn "Skipped"
    fi
}

remove_install_dir() {
    header "Step 2/4  Remove install directory"
    if [[ ! -d "$INSTALL_DIR" ]]; then
        info "Install directory $INSTALL_DIR not found — skipping"
        return
    fi

    local size
    size=$(du -sh "$INSTALL_DIR" 2>/dev/null | cut -f1 || echo "?")
    info "Install directory: $INSTALL_DIR  ($size)"
    local ans
    ans=$(ask_yes "Delete $INSTALL_DIR (source code, venv, built assets)?")
    if [[ "${ans,,}" =~ ^y ]]; then
        rm -rf "$INSTALL_DIR"
        success "Install directory removed"
    else
        warn "Skipped — install directory kept at $INSTALL_DIR"
    fi
}

remove_data_dir() {
    header "Step 3/4  Remove workspace and config data"
    local data_dir="$HOME/.harnessx"
    if [[ ! -d "$data_dir" ]]; then
        info "Data directory $data_dir not found — skipping"
        return
    fi

    local size
    size=$(du -sh "$data_dir" 2>/dev/null | cut -f1 || echo "?")
    warn "This contains your agent workspace sessions, logs, and local config ($size)."
    local ans
    ans=$(ask_yes "Delete $data_dir (workspace sessions, logs, config)?")
    if [[ "${ans,,}" =~ ^y ]]; then
        rm -rf "$data_dir"
        success "Data directory removed"
    else
        warn "Skipped — workspace data kept at $data_dir"
    fi
}

remove_shell_env() {
    header "Step 4/4  Clean shell profile"
    local rc_file
    rc_file="$(_rc_file)"
    if [[ ! -f "$rc_file" ]]; then
        info "Shell profile $rc_file not found — skipping"
        return
    fi

    if ! grep -qE "# HarnessX|HARNESSX_WORKSPACE" "$rc_file" 2>/dev/null; then
        info "No HarnessX entries found in $rc_file — skipping"
        return
    fi

    info "Will remove HarnessX PATH and HARNESSX_WORKSPACE entries from $rc_file"
    local ans
    ans=$(ask_yes "Clean $rc_file?")
    if [[ "${ans,,}" =~ ^y ]]; then
        _remove_rc_lines "$rc_file"
        success "Shell profile cleaned"
        warn "Restart your shell or run: source $rc_file"
    else
        warn "Skipped — remove these lines manually from $rc_file:"
        warn "  # HarnessX"
        warn "  export PATH=\"$BIN_DIR:\$PATH\""
        warn "  export HARNESSX_WORKSPACE=..."
    fi
}

# ── Banner ────────────────────────────────────────────────────────────────────

print_banner() {
    echo
    echo -e "${BOLD}══════════════════════════════════${NC}"
    echo -e "${RED}  HarnessX Uninstaller${NC}"
    echo -e "${BOLD}══════════════════════════════════${NC}"
    echo
    echo -e "  ${DIM}This will remove HarnessX from your system.${NC}"
    echo -e "  ${DIM}You will be asked before each step.${NC}"
    echo
}

# ── main ──────────────────────────────────────────────────────────────────────

main() {
    print_banner

    if $YES; then
        warn "Non-interactive mode (--yes): all components will be removed without prompting"
        echo
    fi

    local ans
    ans=$(ask_yes "Proceed with HarnessX uninstall?")
    if [[ ! "${ans,,}" =~ ^y ]]; then
        info "Uninstall cancelled"
        exit 0
    fi

    remove_binaries
    remove_install_dir
    remove_data_dir
    remove_shell_env

    echo
    echo -e "${BOLD}══════════════════════════════════${NC}"
    echo -e "${GREEN}  Uninstall complete${NC}"
    echo -e "${BOLD}══════════════════════════════════${NC}"
    echo
    info "HarnessX has been removed from your system."
    echo
}

main "$@"
