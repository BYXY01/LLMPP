# LLMPP - LLM 插件代理

```
   ______         __    __    __  _______  ____ 
  /     /|       / /   / /   /  |/  / __ \/ __ \
 /_____/ |      / /   / /   / /|_/ / /_/ / /_/ /
 |     | |     / /___/ /___/ /  / / ____/ ____/ 
 |_____|/     /_____/_____/_/  /_/_/   /_/      
```

一个**开箱即用**的 OpenAI 兼容中间件，通过插件系统为任意 LLM 后端扩展能力。

```
LLM    -> LLM_Server    : 代理部分，OpenAI 进 / OpenAI 出
Plugin -> PluginManager : 插件部分，丢 .py 文件即可添加工具与 hooks
```

## 特性

- **零配置运行** — 启动时自动安装缺失的核心依赖。
- **OpenAI 兼容 API** — `POST /v1/chat/completions`，兼容任何 OpenAI SDK / 客户端。
- **即插即用插件** — 把 `.py` 文件丢进 `plugins/`，启动时自动加载。
- **双协议**：
  - `native` — 原生 OpenAI function/tool calling。
  - `compatible` — 不支持 tools 模型的回退，含双层降级（结构化 JSON schema，再提示词 + 容错解析）。
- **消息处理器** — 入站 / 出站处理器各 1 个，在配置中指定；出站支持可选流式块。
- **插件依赖** — 插件可声明 `__deps__`，LLMPP 自动安装。
- **流式输出**（实验）— native 模式支持 SSE 流式（逐 token），由 `server.stream` 开关控制。
- **异步与线程化** — 异步请求管线（AsyncOpenAI）；PluginManager 管理运行于独立线程，共享加锁注册表。
- **双协议** — 入站支持 OpenAI `/v1/chat/completions` 与 Anthropic `/v1/messages`；`llm.provider` 选择 OpenAI 或 Anthropic 后端。
- **完整 `/v1/*` 透传** — 可选 `routes.full_v1` 将 models、embeddings 等透传到后端。
- **模块化文件** — `plugin_manager.py`、`llm_server.py` 与 `LLMPP.py` 入口拆分两大主类。
- **调用者工具** — 合并客户端提供的 `tools`；调用者自有工具回传给客户端执行（标准 agentic 循环）。
- **鉴权与密钥穿透** — 通过 `LLMPP_API_KEYs` 环境变量设置可选鉴权；调用者密钥可穿透给后端供应方。
- **生产 WSGI** — 由 waitress 托管，非 Flask 开发服务器。
- **MCP 客户端**（实验）— 通过独立的 `mcp_bridge.py` 连接外部 MCP 服务器（stdio / HTTP），其工具并入工具体系。
- **插件管理** — 配置授权的管理函数可列出/启用/禁用/重载插件；状态持久化于 `plugins.json`。
- **任意 LLM 后端** — OpenAI、LM Studio、Ollama、vLLM 等（任何 OpenAI 兼容 base URL）。

## 快速开始

```bash
git clone https://github.com/BYXY01/LLMPP.git
cd LLMPP
python LLMPP.py
```

首次运行会生成 `config.json` 并退出。编辑后再运行：

```bash
python LLMPP.py
```

## 配置

`config.json`（自动生成）：

```json
{
    "server": {
        "host": "127.0.0.1",
        "port": 55677,
        "stream": false,
        "api_keys": []
    },
    "llm": {
        "api_base": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "timeout": 120
    },
    "mode": "native",
    "tools": {
        "max_rounds": 10
    },
    "hooks": {
        "inbound": "",
        "outbound": ""
    }
}
```

| 字段 | 说明 |
|------|------|
| `server.host` / `server.port` | 监听地址与端口。`port` 必填。 |
| `server.stream` | 启用 SSE 流式（实验，仅 native 模式）。 |
| `server.api_keys` | LLMPP 鉴权 key（最多 5 个）；可含 `_PASSTHROUGH_API_KEY`。见"鉴权与密钥穿透"。 |
| `llm.api_base` / `api_key` | 你的 OpenAI 兼容后端（LM Studio、Ollama、vLLM 等）。 |
| `llm.provider` | 后端格式：`openai`（默认）或 `anthropic`（Claude）。 |
| `mode` | `native`（函数调用）或 `compatible`（文本协议）。 |
| `routes.full_v1` | 将完整 `/v1/*` 命名空间（models、embeddings 等）透传到后端。 |
| `tools.max_rounds` | 单次请求内工具调用轮数上限（防死循环安全阀）。 |
| `hooks.inbound` / `hooks.outbound` | 使用的入站 / 出站处理器插件名。 |
| `manager_plugin` | 唯一管理授权函数名（列出/启用/禁用/重载）。 |

模型名**不在**这里配置——由客户端在每次请求中传入，与标准 OpenAI API 一致。

## 鉴权与密钥穿透

通过 `config.json` 的 `server.api_keys` 列表（**最多 5 个 key**）或环境变量 `LLMPP_API_KEYs`（逗号分隔，config 列表为空时回退）配置：

```json
"server": {
    "api_keys": ["my-key-1", "my-key-2"]
}
```

```bash
LLMPP_API_KEYs="my-key-1,my-key-2" python LLMPP.py
```

- **未设置 / 为空** — LLMPP 鉴权关闭；请求始终用 `llm.api_key` 调用供应方。
- **严格模式**（无哨兵标记）— 客户端必须带 `Authorization: Bearer <其中一个key>`。命中则用 `llm.api_key` 调用供应方；未命中返回 **401**。
- **穿透模式** — 在列表中加上哨兵 `_PASSTHROUGH_API_KEY`。命中 LLMPP key 的调用者密钥被改写为 `llm.api_key`；未命中的密钥**原样透传**给供应方（调用者可用自己的供应方密钥）。无 key 请求按空 key 透传。

```json
"server": {
    "api_keys": ["my-key-1", "_PASSTHROUGH_API_KEY"]
}
```

```bash
LLMPP_API_KEYs="my-key-1,_PASSTHROUGH_API_KEY" python LLMPP.py
```

## MCP 客户端（实验）

LLMPP 可连接外部 **MCP 服务器**（stdio 或 streamable HTTP），其工具与插件工具一同使用。此功能为**实验性质**且完全可选——若 `mcp` 包（或 `mcp_bridge.py`）缺失，LLMPP 照常运行不启用 MCP。

MCP 服务器在**独立**的 `mcp_config.json`（同目录，不在 `config.json`）中配置：

```json
{
    "servers": [
        {"name": "math", "command": ["python", "/path/to/mcp_server.py"]},
        {"name": "remote", "url": "http://127.0.0.1:8000/mcp"}
    ]
}
```

`command` 条目走 stdio 传输；`url` 条目走 HTTP。安装 SDK 以启用：

```bash
python -m pip install mcp
```

独立列出已连接的工具：

```bash
python mcp_bridge.py
```

MCP 工具名与插件工具进入同一工具注册表——模型可像调用插件工具一样调用它们。

## 插件

把 `.py` 文件丢进 `plugins/`。插件**无需任何 import**——只需定义函数并声明。

### 工具插件

```python
# plugins/my_tools.py
def get_time() -> str:
    """获取当前时间。

    Returns:
        当前日期时间字符串。
    """
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add(a: int, b: int) -> int:
    """两个整数相加。

    Args:
        a: 第一个数。
        b: 第二个数。

    Returns:
        a 与 b 的和。
    """
    return a + b


__tools__ = [get_time, add]
```

docstring 会成为模型看到的工具描述；Python 类型标注决定 JSON Schema 类型。

### 消息处理器

```python
# plugins/my_hooks.py
def inbound(messages):
    """在消息到达 LLM 前执行，可改写列表。"""
    return messages


def outbound(messages, stream_chunk=None):
    """在 LLM 回复返回用户前执行。

    接收完整回复消息列表。流式时 `stream_chunk` 携带当前文本块，
    hook 应返回 (处理后的消息列表, 处理后的块)；否则返回列表。
    不带 stream_chunk 参数的 hook 在流式时会被跳过。
    """
    return messages


__hooks__ = [inbound, outbound]
```

在配置中指定：`"hooks": {"inbound": "inbound", "outbound": "outbound"}`。

### 插件依赖

在文件顶部声明，LLMPP 缺失时自动安装：

```python
__deps__ = ["python-dotenv"]

import os
from dotenv import load_dotenv
```

### 管理函数

在 `config.json` 中按函数名授权**唯一的管理函数**：

```json
"manager_plugin": "manage"
```

被授权函数将 `manager`（插件管理的单一入口）作为**第一个参数**接收，由 LLMPP 注入。该参数会在模型看到的工具 schema 中被过滤，因此模型只需传 `action`/`name`：

```python
def manage(manager, action, name=None):
    """列出 / 启用 / 禁用 / 重载插件。"""
    return manager(action, name or "")


__tools__ = [manage]
```

任何函数——无论是通过 `__tools__` 还是 `__hooks__` 声明——都可作为管理函数；只要名字匹配 `manager_plugin`，其第一个参数就会注入 `manager`（作为工具时该参数对模型 schema 隐藏）。

## 使用示例（API）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:55677/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "现在几点钟了？"}],
)
print(resp.choices[0].message.content)
```

## 工作原理

- **native 模式**：LLMPP 合并插件 schema（及客户端提供的 `tools`）作为 `tools` 发送。LLMPP 插件工具在内部执行；调用者自有工具回传给客户端执行（标准 agentic 循环）。
- **compatible 模式**：双层降级——先结构化 JSON schema 输出（无提示词注入），再提示词注入 + 容错 JSON 提取。工具结果以 `tool_result:...` / JSON 消息回填。
- 响应包含**完整消息列表**（`messages`），客户端可直接替换历史续轮，不丢上下文。
- **流式**（仅 native）：文本逐 token 转发，工具调用内部累积执行。

## 状态

**Beta**（`v0.0.19`）。核心功能、鉴权与密钥穿透、插件管理、实验性 MCP 客户端可用；异步管线 + 线程化架构，双 OpenAI/Anthropic 协议与后端，完整 `/v1/*` 透传。自动化测试位于 `tests/`（运行 `pytest tests/`）。

## 许可证

[MIT](LICENSE) © 2026 BYXY01 (XY001)
