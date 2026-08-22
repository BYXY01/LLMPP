# Example Plugins

Ready-to-use plugin examples for LLMPP. Copy a file into `plugins/` (or point
`PluginManager` at this directory) to enable it.

| File | Shows |
|------|-------|
| `datetime_tool.py` | Basic tool (no args, returns text) |
| `calc.py` | Tool with typed parameters |
| `weather.py` | `__deps__` + external HTTP API (needs `AMAP_API_KEY` in `.env`) |
| `hooks.py` | Inbound/outbound hooks (incl. streaming `stream_chunk`) |
| `manager.py` | Manager function (`manager_plugin` auth; enable/disable/reload) |

## Usage

Copy what you need:

```bash
cp example_plugins/calc.py plugins/
```

Then reload/restart LLMPP. To use the hooks, enable them in `config.json`:

```json
"hooks": {"inbound": "inbound", "outbound": "outbound"}
```

To use the manager, authorize the function:

```json
"manager_plugin": "manage"
```

`manager.py` declares `manage` as a normal tool (`__tools__`). Authorized
via `manager_plugin`, LLMPP injects `manager` as its first argument and
**filters it out of the tool schema**, so the model calls it with just
`action`/`name` — e.g. "disable the weather plugin" ->
`manage(action="disable", name="weather")`. Actions: `list`, `enable`,
`disable`, `reload` (enable/disable/reload take effect immediately).

`weather.py` needs an AMAP (Gaode) key:

```bash
echo "AMAP_API_KEY=your-key" > .env
```
