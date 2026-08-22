"""
plugin_manager.py - PluginManager: plugin registry + management thread.

Part of LLMPP. Owns the shared plugin/hook registry, executes plugin tools,
and runs a daemon management thread that applies enable/disable commands.

LLMPP entry (LLMPP.py) creates a PluginManager, shares its registry with the
LLM_Server (in llm_server.py), and starts its management thread.
"""

import importlib.util
import inspect
import json
import logging
import os
import queue
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("LLMPP")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from mcp_bridge import MCPClient as _MCPClient
except ImportError:
    _MCPClient = None  # type: ignore[assignment,misc]


def ensure_deps(deps: List[Tuple[str, str]]):
    """Auto-install missing dependencies before importing them.

    Uses pip's Python API (not shell), so package names are never interpreted
    by a shell -> no command injection from untrusted package strings.

    Args:
        deps: List of (import_name, pip_package) pairs.
    """
    import importlib

    pipmain = importlib.import_module("pip._internal").main

    for _dep, _pkg in deps:
        try:
            __import__(_dep)
        except ImportError:
            print(f"[deps] installing missing dependency: {_pkg}")
            pipmain(["install", _pkg])


class PluginManager:
    """Discover plugins and manage the tool/hook registry.

    Plugins declare their tools/hooks via `__tools__` / `__hooks__` lists,
    so plugin files never need to import LLMPP.

    The registry is shared across threads (read-heavy). State changes
    (enable/disable/unload) go through a management queue processed by a
    daemon thread, keeping plugin mutations off the request path.
    """

    def __init__(self, plugins_dir: str = "./plugins", manager_plugin: str = ""):
        self.plugins_dir = os.path.abspath(plugins_dir)
        self.plugins: Dict[str, Callable] = {}
        self.hooks: Dict[str, Callable] = {}
        self.manager_fn: Optional[Callable] = None
        self.manager_plugin = manager_plugin
        self.state_path = os.path.join(BASE_DIR, "plugins.json")
        self.plugin_states: Dict[str, str] = self._read_states()
        self.disabled: set = {n for n, s in self.plugin_states.items() if s == "disabled"}
        self.mcp_client = _MCPClient() if _MCPClient is not None else None
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}
        # management queue + lock
        self._mgmt_queue: "queue.Queue[Optional[Tuple[str, str, queue.Queue[Any]]]]" = queue.Queue()  # type: ignore[valid-type]
        self._mgmt_thread: Optional[threading.Thread] = None
        self._registry_lock = threading.RLock()

    # --- plugin state persistence -----------------------------------------

    def _read_states(self) -> Dict[str, str]:
        if not os.path.exists(self.state_path):
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "plugins" in data and isinstance(data["plugins"], dict):
                data = data["plugins"]
            states: Dict[str, str] = {}
            for name, meta in data.items():
                if isinstance(meta, dict):
                    states[str(name)] = str(meta.get("status", "enabled"))
                elif isinstance(meta, str):
                    states[str(name)] = meta
            return states
        except Exception as e:
            log.error(f"Failed to read {self.state_path}: {e}")
            return {}

    def _write_states(self) -> None:
        data = {name: {"status": status} for name, status in sorted(self.plugin_states.items())}
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # --- management thread ------------------------------------------------

    def start(self) -> None:
        """Start the daemon management thread (idempotent)."""
        if self._mgmt_thread is not None and self._mgmt_thread.is_alive():
            return
        self._mgmt_thread = threading.Thread(target=self._mgmt_loop, name="llmpp-manager", daemon=True)
        self._mgmt_thread.start()
        log.info("PluginManager management thread started")
        if self.mcp_client:
            self.mcp_client.start()
            log.info("MCP client thread started")

    def stop(self) -> None:
        """Signal the management thread to exit."""
        if self.mcp_client:
            self.mcp_client.stop()
        if self._mgmt_thread is not None and self._mgmt_thread.is_alive():
            self._mgmt_queue.put(None)
            self._mgmt_thread.join(timeout=5)

    def _mgmt_loop(self) -> None:
        """Process management commands on the daemon thread."""
        while True:
            item = self._mgmt_queue.get()
            if item is None:
                return
            action, name, resp = item
            try:
                resp.put(self.manager(action, name, lock=False))
            except Exception as e:
                resp.put(f"[error] management command failed: {e}")

    def _mgmt_request(self, action: str, name: str) -> str:
        """Submit a management command and wait for its result."""
        resp: "queue.Queue[str]" = queue.Queue()
        self._mgmt_queue.put((action, name, resp))
        return resp.get(timeout=30)

    # --- public management API -------------------------------------------

    def manager(self, action: str, name: str = "", lock: bool = True) -> Any:
        """Unified plugin management entry point.

        Actions:
            list                -> list all discovered plugins with state
            enable <name>       -> enable a plugin (takes effect on reload)
            disable <name>      -> disable a plugin (persist + unload now)

        `lock` guards registry mutation; callers already on the management
        thread pass lock=False.
        """
        if action == "list":
            discovered = set()
            if os.path.isdir(self.plugins_dir):
                for n in os.listdir(self.plugins_dir):
                    if n.startswith("_") or not n.endswith(".py"):
                        continue
                    discovered.add(os.path.splitext(n)[0])
            return [
                {
                    "name": n,
                    "enabled": n not in self.disabled,
                    "loaded": n in self.plugins,
                }
                for n in sorted(discovered)
            ]
        if action == "enable":
            if name in self.disabled:
                with self._registry_lock if lock else _nullcontext():
                    self.disabled.discard(name)
                    self.plugin_states[name] = "enabled"
                    self._write_states()
                    ok = self._load_plugin_file(name)
                return f"[ok] plugin '{name}' enabled{' and loaded' if ok else ' (load failed)'}"
            return f"[ok] plugin '{name}' already enabled"
        if action == "disable":
            with self._registry_lock if lock else _nullcontext():
                self.disabled.add(name)
                self.plugin_states[name] = "disabled"
                self._write_states()
                self._unload(name)
            return f"[ok] plugin '{name}' disabled"
        if action == "reload":
            with self._registry_lock if lock else _nullcontext():
                self._unload(name)
                ok = self._load_plugin_file(name)
            return f"[ok] plugin '{name}' reloaded" if ok else f"[error] reload failed: {name}"
        return f"[error] Unknown action: {action}"

    def _unload(self, name: str) -> None:
        """Immediately remove a plugin's tools/hooks/manager entries."""
        changed = False
        for tool in list(self.plugins):
            if getattr(self.plugins[tool], "__module__", "") == name:
                del self.plugins[tool]
                changed = True
        for key in list(self.hooks):
            if getattr(self.hooks[key], "__module__", "") == name:
                del self.hooks[key]
                changed = True
        if self.manager_fn is not None and getattr(self.manager_fn, "__module__", "") == name:
            self.manager_fn = None
            changed = True
        if changed:
            log.info(f"Plugin '{name}' unloaded")

    def load(self) -> None:
        """Scan and load all .py files in the plugins directory."""
        if not os.path.isdir(self.plugins_dir):
            os.makedirs(self.plugins_dir, exist_ok=True)
            log.info(f"Plugin directory created: {self.plugins_dir}")
            return

        sys.path.insert(0, self.plugins_dir)
        for name in sorted(os.listdir(self.plugins_dir)):
            if name.startswith("_") or not name.endswith(".py"):
                continue
            module_name = os.path.splitext(name)[0]
            if module_name in self.disabled:
                log.info(f"Skipping disabled plugin: {module_name}")
                continue
            self._load_plugin_file(module_name)

        log.info(f"Loaded plugins: {sorted(self.plugins)}")
        log.info(f"Registered hooks: {sorted(self.hooks)}")
        if self.manager_fn is not None:
            log.info(f"Registered manager: {self.manager_fn.__name__}")
        if self.mcp_client:
            try:
                self.mcp_tools = {t["function"]["name"]: t for t in self.mcp_client.load()}
                if self.mcp_tools:
                    log.info(f"Loaded MCP tools: {sorted(self.mcp_tools)}")
            except Exception as e:
                log.warning(f"MCP tools load failed: {e}")
        self._sync_discovered_states()

    def _load_plugin_file(self, module_name: str) -> bool:
        """Load a single plugin file by module name; True on success."""
        module_path = os.path.join(self.plugins_dir, f"{module_name}.py")
        if not os.path.exists(module_path):
            log.error(f"Plugin file not found: {module_path}")
            return False
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create spec for {module_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except ImportError:
                # Plugin declares __deps__ before imports; if an import
                # failed, install the declared deps and retry.
                deps = getattr(module, "__deps__", [])
                if deps:
                    ensure_deps([(d, d) for d in deps])
                    spec.loader.exec_module(module)
                else:
                    raise
            self._collect(module)
            log.info(f"Loading plugin file: {module_name}.py")
            return True
        except Exception as e:
            log.error(f"Failed to load plugin {module_name}: {e}")
            return False


    def _sync_discovered_states(self) -> None:
        """Auto-register discovered plugins into plugins.json (default enabled).

        Newly found plugins are added as enabled; ones already present keep
        their persisted state.
        """
        changed = False
        for n in self.manager("list", lock=False):
            name = n["name"]
            if name not in self.plugin_states:
                self.plugin_states[name] = "enabled"
                changed = True
        if changed:
            self._write_states()

    def _collect(self, module) -> None:
        """Collect tools/hooks declared via __tools__ / __hooks__ in a plugin module."""
        for fn in getattr(module, "__tools__", []):
            name = fn.__name__
            if name in self.plugins:
                log.warning(f"Duplicate tool name: {name}, overwritten")
            self.plugins[name] = fn
        for fn in getattr(module, "__hooks__", []):
            name = fn.__name__
            if name in self.hooks:
                log.warning(f"Duplicate hook name: {name}, overwritten")
            self.hooks[name] = fn
        if self.manager_plugin:
            for fn in list(getattr(module, "__tools__", [])) + list(getattr(module, "__hooks__", [])):
                if fn.__name__ == self.manager_plugin:
                    if self.manager_fn is not None:
                        log.warning(
                            f"Duplicate manager function '{self.manager_plugin}'; "
                            f"keeping the first ({self.manager_fn.__name__})"
                        )
                    else:
                        self.manager_fn = fn

    def _tool_schema(self, name: str, func: Callable) -> Dict[str, Any]:
        """Generate the OpenAI tool schema from a function's signature and docstring."""
        type_map = {int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
        sig = inspect.signature(func)
        description = inspect.getdoc(func) or ""

        properties: Dict[str, Any] = {}
        required: List[str] = []
        skip_first = func is self.manager_fn
        for idx, (pname, param) in enumerate(sig.parameters.items()):
            if pname in ("self", "cls"):
                continue
            if skip_first and idx == 0:
                continue
            ptype = "string"
            if param.annotation is not inspect.Parameter.empty:
                ptype = type_map.get(param.annotation, "string")
            properties[pname] = {"type": ptype}
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def tools(self) -> List[Dict[str, Any]]:
        """Build the tool list for the OpenAI API (plugins + MCP)."""
        tools = [
            self._tool_schema(name, func)
            for name, func in self.plugins.items()
        ]
        tools.extend(self.mcp_tools.values())
        return tools

    def has_tool(self, name: str) -> bool:
        """True if `name` is a plugin tool or an MCP tool."""
        return name in self.plugins or name in self.mcp_tools

    def call(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool (plugin or MCP) and return its result as a string."""
        if name in self.mcp_tools and self.mcp_client:
            return self.mcp_client.call(name, args)
        if name not in self.plugins:
            return f"[error] Function not found: {name}"
        try:
            fn = self.plugins[name]
            result = fn(self.manager, **args) if fn is self.manager_fn else fn(**args)
            if result is None:
                return "[ok] Function executed, no return value"
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as e:
            log.error(f"Tool {name} failed: {e}")
            return f"[error] Function execution failed: {e}"

    async def call_async(self, name: str, args: Dict[str, Any]) -> str:
        """Async tool execution; MCP uses the async client, plugins stay sync."""
        if name in self.mcp_tools and self.mcp_client:
            return await self.mcp_client.call_async(name, args)
        return self.call(name, args)

    def run_inbound(self, cfg: Dict[str, Any], messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the inbound hook (before messages reach the LLM)."""
        name = cfg["hooks"].get("inbound")
        if name and name in self.hooks:
            try:
                hook_fn = self.hooks[name]
                if hook_fn is self.manager_fn:
                    result = hook_fn(self.manager, messages)
                else:
                    result = hook_fn(messages)
                if isinstance(result, list):
                    return result
            except Exception as e:
                log.error(f"Inbound hook {name} error: {e}")
        return messages

    def run_outbound(self, cfg: Dict[str, Any], messages: List[Dict[str, Any]], stream_chunk: Optional[str] = None) -> Any:
        """Run the outbound hook (after the LLM replies, before returning).

        Non-streaming: hook(messages) returns a list -> the new message list.

        Streaming: hook(messages, stream_chunk) returns a (list, chunk) tuple
        -> (processed messages, processed stream chunk). Hooks without a
        stream_chunk parameter raise TypeError, which is caught and logged;
        the original messages are returned unchanged (try/except/continue).
        """
        name = cfg["hooks"].get("outbound")
        if name and name in self.hooks:
            hook_fn = self.hooks[name]
            try:
                if stream_chunk:
                    if hook_fn is self.manager_fn:
                        r_msg, r_chunk = hook_fn(self.manager, messages=messages, stream_chunk=stream_chunk)
                    else:
                        r_msg, r_chunk = hook_fn(messages=messages, stream_chunk=stream_chunk)
                    r_msg = r_msg if isinstance(r_msg, list) else messages
                    return r_msg, r_chunk
                if hook_fn is self.manager_fn:
                    result = hook_fn(self.manager, messages)
                else:
                    result = hook_fn(messages)
                if isinstance(result, list):
                    return result
            except Exception as e:
                log.error(f"Outbound hook {name} error: {e}")
                if stream_chunk:
                    return messages, None
        return messages


class _nullcontext:
    """Minimal nullcontext (avoids importing contextlib at class scope)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
