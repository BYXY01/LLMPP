"""
LLMPP - LLM Plugin Proxy
An out-of-the-box OpenAI-compatible API middleware that extends LLM capabilities through a plugin system.

Two core parts, matching the name (now split into separate modules):
    LLM      -> LLM_Server (llm_server.py): proxy part, forwards OpenAI-compatible requests to any LLM backend
    Plugin   -> PluginManager (plugin_manager.py): plugin part, loads/registers/executes plugins

main() coordinates two threads:
    - PluginManager management thread (plugin state changes)
    - LLM_Server waitress thread (serves requests)

Usage:
    python LLMPP.py              # Start server (auto-generates config.json)
    python LLMPP.py --gen-config # Generate default config only
"""

import argparse
import json
import logging
import os
import sys
import threading
from typing import Any, Dict, List, Tuple


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


ensure_deps([("flask", "flask"), ("waitress", "waitress"), ("openai", "openai")])

from llm_server import LLM_Server  # noqa: E402
from plugin_manager import PluginManager  # noqa: E402

VERSION = "0.0.18-alpha"

BANNER = r"""   ______         __    __    __  _______  ____ 
  /     /|       / /   / /   /  |/  / __ \/ __ \
 /_____/ |      / /   / /   / /|_/ / /_/ / /_/ /
 |     | |     / /___/ /___/ /  / / ____/ ____/ 
 |_____|/     /_____/_____/_/  /_/_/   /_/      """

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("LLMPP")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        # "host": "0.0.0.0",  # expose to network
        # "port": 55677,  # required: LLMPP phone-keypad encoding, uncomment and set
        "stream": False,
        # Empty list = LLMPP auth disabled, always call provider with llm.api_key.
        # Up to 5 keys. Include the sentinel "_PASSTHROUGH_API_KEY" to enable
        # passthrough: caller keys that miss the list go to the provider as-is.
        # Fallback when empty: LLMPP_API_KEYs env var (comma-separated).
        "api_keys": [],
    },
    "llm": {
        "api_base": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "timeout": 120,
        "provider": "openai",  # openai | anthropic (backend format)
    },
    "mode": "native",  # native | compatible
    "routes": {
        # full_v1: proxy the whole /v1/* namespace (models, embeddings, etc.)
        # to the backend. False keeps only /v1/chat/completions + /v1/messages.
        "full_v1": False,
    },
    "tools": {
        "max_rounds": 10,
    },
    "hooks": {
        "inbound": "",
        "outbound": "",
    },
    "manager_plugin": "",
}


def load_config() -> Dict[str, Any]:
    """Load config.json, merging any missing top-level keys from DEFAULT_CONFIG."""
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        log.info(f"Default config generated: {CONFIG_PATH}")
        log.info("Please edit config.json and restart.")
        sys.exit(0)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse config.json: {e}")
        sys.exit(1)

    for key, value in DEFAULT_CONFIG.items():
        cfg.setdefault(key, value)
    return cfg


def save_config(cfg: Dict[str, Any]):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="LLMPP - LLM Plugin Proxy")
    parser.add_argument("--gen-config", action="store_true", help="Generate default config only")
    args = parser.parse_args()

    cfg = load_config()
    if args.gen_config:
        save_config(cfg)
        log.info(f"Config generated: {CONFIG_PATH}")
        return

    host = cfg["server"].get("host", "0.0.0.0")
    port = cfg["server"].get("port")
    if not port:
        log.error("server.port is required. Set it in config.json (e.g. 55677) and restart.")
        sys.exit(1)

    # PluginManager: load plugins (main thread), then start its management thread.
    manager = PluginManager(manager_plugin=cfg.get("manager_plugin", ""))
    manager.load()
    manager.start()

    # LLM_Server: serve requests on a dedicated waitress thread.
    server = LLM_Server(cfg, manager, version=VERSION)
    print(BANNER, flush=True)
    log.info(f"LLMPP v{VERSION} starting")
    log.info(f"OpenAI-compatible endpoint: http://{host}:{port}/v1/chat/completions")
    if cfg["mode"] == "compatible":
        log.info("Compatible mode enabled")

    server_thread = threading.Thread(target=server.run, args=(host, port), name="llmpp-server", daemon=True)
    server_thread.start()

    try:
        server_thread.join()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        manager.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
