# HarnessX agent base image
# Build: docker build -t harnessx/agent:latest container/
# Push:  docker push harnessx/agent:latest
#
# Provides the runtime environment for agent tool execution:
#   bash, git, python3, pip, ripgrep, node, curl, jq, file utilities
#   playwright + chromium (CDP remote debugging on port 9222)
#
# Usage with DockerSandboxProvider:
#   from harnessx.sandbox.docker import DockerSandboxProvider
#   sandbox_provider=DockerSandboxProvider(image="harnessx/agent:latest")
#
# Usage with browser support (CDP):
#   sandbox_provider=DockerSandboxProvider(
#       image="harnessx/agent:latest",
#       enable_browser=True,   # starts chromium --remote-debugging-port=9222
#   )

FROM python:3.12-slim

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core shell utilities
    bash \
    coreutils \
    findutils \
    procps \
    # Version control
    git \
    # Text search (used by grep_tool)
    ripgrep \
    # Network tools (agent web tasks)
    curl \
    wget \
    # Data utilities
    jq \
    # Node.js (for JS/TS tasks)
    nodejs \
    npm \
    # Office / document tools
    libreoffice-core-nogui \
    # Misc
    file \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    # Data science basics
    numpy \
    pandas \
    matplotlib \
    # Document generation (office skill)
    python-pptx \
    python-docx \
    openpyxl \
    # HTTP client
    httpx \
    requests \
    # Markdown / text processing
    beautifulsoup4 \
    markdownify \
    # Browser automation (CDP server exposed on port 9222 via enable_browser=True)
    playwright

# ── Playwright / Chromium ─────────────────────────────────────────────────────
# Install chromium and its system dependencies as root before switching user.
# PLAYWRIGHT_BROWSERS_PATH controls where binaries are stored (/ms-playwright
# is writable by all users so the agent user can launch the browser).
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install chromium --with-deps \
    && chmod -R 755 /ms-playwright

# CDP port (only used when DockerSandboxProvider(enable_browser=True))
EXPOSE 9222

# ── Workspace mount point ─────────────────────────────────────────────────────
RUN mkdir -p /workspace /skills
WORKDIR /workspace

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 agent && chown -R agent:agent /workspace /skills
USER agent

# Default: keep container alive for exec_run() calls
CMD ["sleep", "infinity"]
