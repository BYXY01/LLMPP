# LLMPP - LLM Plugin Proxy

An out-of-the-box, **OpenAI-compatible** middleware that extends any LLM backend with plugin capabilities through a single file.

```
LLM  -> LLM_Server  : proxy part, OpenAI in / OpenAI out
Plugin -> PluginManager : plugin part, drop .py files to add tools & hooks
```

## Features

- **Single file** — one `LLMPP.py`, zero project setup. Auto-installs missing core deps on boot.
- **OpenAI-compatible API** — `POST /v1/chat/completions`, works with any OpenAI SDK/client.
- **Drop-in plugins** — put `.py` files in `plugins/`, they auto-load on startup.
- **Two protocols**:
  - `native` — native OpenAI function/tool calling.
  - `compatible` — text-protocol fallback for models without tools support.
- **Hooks** — inbound/outbound message processors (1 each, chosen in config).
- **Plugin deps** — a plugin can declare `__deps__` and LLMPP auto-installs them.
- **Streaming** (experimental) — native mode supports SSE streaming, gated by `server.stream`.
- **Any LLM backend** — OpenAI, LM Studio, Ollama, vLLM, etc. (any OpenAI-compatible base URL).

## Quick Start

```bash
git clone https://github.com/BYXY01/LLMPP.git
cd LLMPP
python LLMPP.py
```

First run generates `config.json`, then exits. Edit it, then run again:

```bash
python LLMPP.py
```

## Configuration

`config.json` (auto-generated):

```json
{
    "server": {
        "host": "127.0.0.1",
        "port": 55677,
        "stream": false
    },
    "llm": {
        "api_base": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "timeout": 120
    },
    "mode": "native",
    "hooks": {
        "inbound": "",
        "outbound": ""
    }
}
```

| Field | Description |
|-------|-------------|
| `server.host` / `server.port` | Bind address and port. `port` is required. |
| `server.stream` | Enable SSE streaming (experimental, native mode only). |
| `llm.api_base` / `api_key` | Your OpenAI-compatible backend (LM Studio, Ollama, vLLM, ...). |
| `mode` | `native` (function calling) or `compatible` (text protocol). |
| `hooks.inbound` / `hooks.outbound` | Name of the inbound/outbound hook plugin to use. |

The model name is **not** configured here — clients pass it in each request, as with the standard OpenAI API.

## Plugins

Drop `.py` files into `plugins/`. Plugins need **no imports** — just define functions and declare them.

### Tools

```python
# plugins/my_tools.py
def get_time() -> str:
    """Get the current time.

    Returns:
        The current datetime string.
    """
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b.
    """
    return a + b


__tools__ = [get_time, add]
```

The docstring becomes the tool description the model sees. Python type hints drive the JSON Schema types.

### Hooks

```python
# plugins/my_hooks.py
def inbound(messages):
    """Runs before messages reach the LLM. May rewrite the list."""
    return messages


def outbound(response):
    """Runs on the LLM reply before returning to the user."""
    return response


__hooks__ = [inbound, outbound]
```

Select them in config: `"hooks": {"inbound": "inbound", "outbound": "outbound"}`.

### Plugin dependencies

Declare deps at the top of the file; LLMPP auto-installs them if missing:

```python
__deps__ = ["python-dotenv"]

import os
from dotenv import load_dotenv
```

## Usage (API)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:55677/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "What time is it?"}],
)
print(resp.choices[0].message.content)
```

## How it works

- `native` mode: LLMPP sends the plugin schemas as `tools`; when the model asks for a tool, LLMPP executes it and feeds the result back as a `tool` message.
- `compatible` mode: LLMPP injects a system prompt describing the tools and a JSON contract. If the model replies with `{"call_function": "...", "arg": {...}}`, LLMPP runs the tool and returns `tool_result:...`; a non-JSON reply is returned to the user directly.

## Status

Alpha (`v0.0.11`). Core features work; streaming (experimental), request auth, and tool-failure limits are in progress.
