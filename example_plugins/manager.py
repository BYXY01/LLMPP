"""Example plugin: manager (list/enable/disable/reload plugins).

Copy to `plugins/` to use, then authorize this function in config.json:
    "manager_plugin": "manage"

The authorized manager function receives `manager` (the plugin manager's
single entry point) as its first argument, injected by LLMPP. That first
argument is filtered out of the tool schema the model sees, so the model
calls it with just `action`/`name`:
    "disable the weather plugin" -> manage(action="disable", name="weather")
"""


def manage(manager, action, name=None):
    """Manage plugins (list/enable/disable/reload).

    Args:
        manager: The PluginManager.manager callable, injected as the first
            argument (filtered from the tool schema shown to the model).
        action: One of "list", "enable", "disable", "reload".
        name: Plugin name for enable/disable/reload.

    Returns:
        A human-readable result.
    """
    result = manager(action, name or "")
    if action == "list":
        return "\n".join(
            f"- {p['name']}: {'enabled' if p['enabled'] else 'disabled'}"
            for p in result
        )
    return result


__tools__ = [manage]
