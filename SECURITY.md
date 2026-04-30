# Security Policy

## Supported Versions

Only the latest release on `main` is actively maintained for security fixes.

| Version | Supported |
| ------- | --------- |
| latest  | ✓         |
| < latest | ✗        |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues privately via [GitHub Security Advisories](../../security/advisories/new).
This ensures the issue can be investigated and patched before public disclosure.

### What to include

- A clear description of the vulnerability and its potential impact
- Steps to reproduce or a minimal proof-of-concept
- Affected versions or components (e.g. `harnessx/sandbox/docker.py`)
- Any suggested mitigations, if available

### Response timeline

| Stage | Target |
|-------|--------|
| Acknowledgement | Within 3 business days |
| Initial assessment | Within 7 business days |
| Fix / patch | Dependent on severity |

We follow a **90-day coordinated disclosure** policy. If a fix cannot be delivered within 90 days,
we will notify you and discuss a mutually agreed extended timeline.

## Scope

The following are **in scope** for this policy:

- Remote code execution via the harness API or tool interface
- Sandbox escape (Docker, E2B, local sandbox)
- Credential or secret leakage through logs, trajectories, or the Lab UI
- Authentication bypass in the Lab API

The following are **out of scope**:

- Vulnerabilities in third-party dependencies (please report upstream)
- Issues requiring physical access to the host machine
- Social engineering or phishing
