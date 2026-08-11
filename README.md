# hermes-tool-forge

Runtime tool forging plugin for Hermes Agent. Agents write their own tools mid-conversation, have them safety-reviewed by an LLM judge, tested in a sandbox, and registered for use — all without breaking prompt caching.

Inspired by [AgentOS (framerslab)](https://github.com/framerslab/agentos) runtime tool forging.

## Why

Hermes Agent has a fixed toolset at session start. When an agent discovers it needs a capability that no existing tool provides, it has to work around the gap or wait for a human to write a skill. This plugin lets the agent create the tool itself — safely, with judge review and sandbox testing.

## What it does

| Feature | Description |
|---|---|
| `forge_tool` | Agent writes a Python `execute()` function + JSON schema → static validation → LLM judge review → sandbox test → registered |
| `forge_call` | Call a previously forged tool by name |
| `forge_list` | List all forged tools with status and usage stats |
| `forge_promote` | Promote a forged tool to a SKILL.md file for permanent use |

## Architecture

```
Agent calls forge_tool(name, description, params_schema, python_code)
         │
         ▼
  ┌─────────────┐
  │ Static val   │  AST analysis — forbidden imports, os.system, etc.
  └──────┬──────┘
         │ pass
         ▼
  ┌─────────────┐
  │ LLM judge   │  Second model reviews code for safety risks
  └──────┬──────┘
         │ approved
         ▼
  ┌─────────────┐
  │ Sandbox test │  Subprocess with restricted env, 10s timeout
  └──────┬──────┘
         │ pass
         ▼
  ┌─────────────┐
  │ Register    │  Handler loaded in-process, persisted to SQLite
  └──────┬──────┘
         │
         ▼
  Agent calls forge_call(tool_name, args) → execute(args) → result
```

### Prompt caching invariant

Forged tools are registered in the Hermes tool registry but **NOT added to the system prompt**. The 4 forge tools (`forge_tool`, `forge_call`, `forge_list`, `forge_promote`) are in the system prompt; forged tools themselves are called via `forge_call`, which dispatches to the registered handler. This preserves Hermes's prompt caching invariant — no mid-conversation tool schema changes.

### Safety layers

1. **Static validation** (sandbox.py) — AST analysis blocks forbidden imports (socket, urllib, requests, subprocess, pickle, ctypes), os.system/popen/exec, and wildcard imports
2. **LLM judge** (judge.py) — a second model reviews the code with a structured safety prompt, returns a JSON verdict (approved/rejected + risks + confidence)
3. **Sandbox test** (sandbox.py) — code runs in a subprocess with restricted env (no PYTHONPATH, minimal PATH), 10s timeout, captured output
4. **Handler isolation** — forged tool's `execute()` function runs in a namespace with only `__builtins__`; the handler is called via `forge_call`, not directly

## Install

```bash
# Step 1 — copy plugin to Hermes plugins directory
mkdir -p ~/.hermes/plugins/tool_forge
cp -r tool_forge/* ~/.hermes/plugins/tool_forge/

# Step 2 — enable the plugin in config
hermes config set plugins.enabled '["tool_forge"]'

# Step 3 — restart Hermes
sudo systemctl restart hermes-webui

# Step 4 — verify in logs
grep -i "tool-forge" ~/.hermes/logs/agent.log
```

## Config

No config keys required. The plugin uses defaults:
- DB path: `~/.hermes/tool_forge/forge.db`
- Skills dir: `~/.hermes/skills/`
- Sandbox timeout: 10 seconds
- Auto-promote threshold: 3 uses

## Agent tools

### forge_tool

Create a new tool at runtime.

| Parameter | Type | Required | Description |
|---|---|---|---|
| name | string | Yes | Unique tool name (snake_case) |
| description | string | Yes | What the tool does |
| params_schema | object | Yes | JSON schema for parameters |
| python_code | string | Yes | Python defining `execute(args: dict) -> dict` |
| test_args | object | No | Arguments for sandbox test |

### forge_call

Call a previously forged tool.

| Parameter | Type | Required | Description |
|---|---|---|---|
| tool_name | string | Yes | Name of the forged tool |
| args | object | No | Arguments to pass to execute() |

### forge_list

List all forged tools. No parameters.

### forge_promote

Promote a forged tool to a SKILL.md file.

| Parameter | Type | Required | Description |
|---|---|---|---|
| tool_name | string | Yes | Name of the tool to promote |

## How it works

### The forge flow

1. Agent calls `forge_tool` with a name, description, JSON schema, and Python code
2. The code is statically validated (AST analysis for forbidden imports/calls)
3. An LLM judge reviews the code for safety risks (uses the host's model via `ctx.llm`)
4. The code is tested in a sandbox subprocess with restricted env and timeout
5. If all steps pass, the tool is registered in-process and persisted to SQLite
6. The agent can call the tool via `forge_call`

### Persistence

Forged tools are stored in `~/.hermes/tool_forge/forge.db` (SQLite). On plugin load, all previously approved+tested tools are reloaded from the store. This means forged tools survive `/reset` and Hermes restarts.

### Promotion to skills

After 3 uses (configurable), a forged tool is eligible for auto-promotion to a SKILL.md file. Once promoted, the tool loads automatically in future sessions — no mid-session forging needed. Manual promotion is available via `forge_promote`.

### Sandbox limitations

The sandbox is a safety net, not a full security boundary:
- No network access (forbidden imports)
- No subprocess access (forbidden imports)
- Restricted env (minimal PATH, no PYTHONPATH)
- 10-second timeout
- Captured output (5KB max)

The LLM judge is the primary safety gate. For production use with untrusted code, consider running the sandbox in a container or microVM.

## Security

- **exec() in _build_handler**: runs LLM-generated code in a namespace with `__builtins__` only. This is the core forge mechanism — it's inherently risky, which is why the static validation + LLM judge + sandbox test are required before this step.
- **SQL**: all queries use parameterized `?` placeholders
- **Thread safety**: RLock on all store operations
- **Subprocess**: restricted env, no shell=True, temp files cleaned up
- **Input validation**: name, description, code checked for emptiness; code size capped at 10KB

## Development

```bash
# Run tests
cd hermes-tool-forge
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_sandbox.py -v
```

## Project structure

```
hermes-tool-forge/
├── tool_forge/
│   ├── __init__.py       # Plugin entry point (register())
│   ├── forge_tool.py     # Main handler (forge, call, list, promote)
│   ├── judge.py          # LLM safety judge
│   ├── sandbox.py        # Restricted execution + static validation
│   ├── store.py          # SQLite persistence
│   ├── promote.py        # Promote to SKILL.md
│   └── plugin.yaml       # Plugin manifest
├── tests/
│   ├── test_sandbox.py   # 14 tests
│   ├── test_store.py     # 12 tests
│   ├── test_judge.py     # 9 tests
│   ├── test_forge_tool.py # 15 tests
│   └── test_promote.py   # 10 tests
├── pyproject.toml
├── pytest.ini
├── LICENSE
└── README.md
```

## Audit history

- v1.0.0 — Initial implementation, 60 tests passing, static security scan clean, independent reviewer dispatched

## License

MIT