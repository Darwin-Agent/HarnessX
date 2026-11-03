# Parallel Search MCP

Search the web and fetch page excerpts through HarnessX's `MCPToolRegistry`,
without a Parallel account or API key. [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
uses Fast mode for anonymous searches. The free endpoint is intended for exploration
and light use and is subject to rate limits.

From the repository root, install in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python examples/parallel_search/search.py \
  "Find HarnessX's MCP setup instructions" \
  --query "HarnessX MCP server configuration" \
  --fetch-url https://github.com/Darwin-Agent/HarnessX
```

The script discovers the remote tools, executes `web_search`, optionally executes
`web_fetch` for the supplied URLs, prints their results, and closes the connection.
Repeat `--query` for related queries and `--fetch-url` for up to 20 pages.
Search results include source URLs and excerpts; fetching returns excerpts focused
on the same objective. Each invocation shares one anonymous session identifier
across both tools.

The default endpoint is `https://search.parallel.ai/mcp`. Set `MCP_SEARCH_URL`
only when you want to override it; no environment variables are required.

This example calls tools directly, so it needs no model credentials. It does not
change built-in search tools, model settings, or saved MCP server configuration.
To use the discovered tools in your own agent, register `registry.tools()` in its
tool registry and keep the MCP connection open for the run, as shown in
`MCPToolRegistry`'s lifecycle example.
