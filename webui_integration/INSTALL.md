# WebUI Integration — Forge Tools Panel

The forge tools management panel is built into the Hermes WebUI (Memory panel → Forge Tools section).

## What it shows

- **Stats chips:** total / approved / tested / promoted tool counts
- **Tool list:** each forged tool with name, status badges (APPROVED/PENDING/TESTED/PROMOTED), use count, description, and code preview
- **Tool detail view:** click "View" to see the full code, JSON schema, judge verdict, and test output
- **Delete:** remove a forged tool from the store (with confirmation)
- **Filter:** search tools by name or description

## Backend

The bridge module `api/forge_bridge.py` in the hermes-webui repo provides:

- `GET /api/forge` — list all tools + stats
- `POST /api/forge` — actions: `delete`, `view`, `promote_test` (dry-run SKILL.md preview)

The bridge loads the plugin's `store.py` directly (same pattern as the cognitive memory bridge) — it never imports Hermes Agent internals into the WebUI process.

## Frontend

The panel JS is in `static/panels.js`:

- `MEMORY_SECTIONS` entry: `{ key: 'forge', label: 'Forge Tools', iconKey: 'wrench', readOnly: true }`
- Render dispatch in `_renderMemoryDetail()` → `_renderForgeToolsDetail()`
- Functions: `_loadForgeData`, `_renderForgeToolsDetail`, `_forgeCardHtml`, `_forgeToolDetailHtml`, `forgeViewTool`, `forgeDeleteTool`, `forgeCloseDetail`, `forgeSetQuery`
- Reuses existing `.cognitive-card`, `.cognitive-controls`, `.cognitive-stats`, `.cognitive-badge`, `.cognitive-btn` CSS classes — no new CSS needed