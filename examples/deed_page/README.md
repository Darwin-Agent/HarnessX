# deed.page MCP

Publish an HTML page to a public `https://{slug}.deed.page/` URL through
HarnessX's `MCPToolRegistry`, without an account or API key. The
[deed.page](https://deed.page) MCP server is keyless: the first publish returns
an owner token, which you need later to update or delete the page.
Tool reference: https://deed.page/docs

From the repository root, install in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python examples/deed_page/publish.py
```

With no file argument the script publishes a small sample page that expires
after an hour (`--ttl 3600`). It prints the tool output, which includes `url`,
`slug` and `token`. To publish your own file with a chosen slug and keep it:

```bash
python examples/deed_page/publish.py report.html --slug <your-slug> --ttl 0
```

To update the page later, re-run with the same slug and the token from its
first publish:

```bash
DEEDPAGE_TOKEN=deed_tok_... python examples/deed_page/publish.py report.html --slug <your-slug> --ttl 0
```

Delete a page with the same token:

```bash
DEEDPAGE_TOKEN=deed_tok_... python examples/deed_page/publish.py --delete <your-slug>
```

Published pages are public, so never publish secrets or private data. This
example calls tools directly, so it needs no model credentials. To let an agent
publish its own output, register `registry.tools()` in its tool registry and
keep the MCP connection open for the run, as shown in `MCPToolRegistry`'s
lifecycle example.
