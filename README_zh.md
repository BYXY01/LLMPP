# LLMPP - LLM 插件代理

一个**开箱即用**的 OpenAI 兼容中间件，通过单个文件为任意 LLM 后端扩展插件能力。

```
LLM    -> LLM_Server    : 代理部分，OpenAI 进 / OpenAI 出
Plugin -> PluginManager : 插件部分，丢 .py 文件即可添加工具与处理器
```

## 特性

- **单文件** — 只有一个 `LLMPP.py`，零项目搭建成本。启动时自动安装缺失的核心依赖。
- **OpenAI 兼容 API** — `POST /v1/chat/completions`，兼容任何 OpenAI SDK / 客户端。
- **即插即用插件** — 把 `.py` 文件丢进 `plugins/`，启动时自动加载。
- **双协议**：
  - `native` — 原生 OpenAI function/tool calling。
  - `compatible` — 文本协议回退，适用于不支持 tools 的模型。
- **消息处理器** — 入站 / 出站处理器各 1 个，在配置中指定。
- **插件依赖** — 插件可声明 `__deps__`，LLMPP 自动安装。
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
        "port": 55677
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

| 字段 | 说明 |
|------|------|
| `server.host` / `server.port` | 监听地址与端口。`port` 必填。 |
| `llm.api_base` / `api_key` | 你的 OpenAI 兼容后端（LM Studio、Ollama、vLLM 等）。 |
| `mode` | `native`（函数调用）或 `compatible`（文本协议）。 |
| `hooks.inbound` / `hooks.outbound` | 使用的入站 / 出站处理器插件名。 |

模型名**不在**这里配置——由客户端在每次请求中传入，与标准 OpenAI API 一致。

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


def outbound(response):
    """在 LLM 回复返回用户前执行。"""
    return response


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

- **native 模式**：LLMPP 把插件 schema 作为 `tools` 发送；模型要求调用工具时，LLMPP 执行并以 `tool` 消息回填结果。
- **compatible 模式**：LLMPP 注入描述工具与 JSON 约定的 system 提示。若模型回复 `{"call_function": "...", "arg": {...}}`，LLMPP 执行工具并返回 `tool_result:...`；非 JSON 回复则直接返回给用户。

## 状态

Alpha（`v0.0.10`）。核心功能可用；流式输出、请求鉴权、工具失败上限为计划项。
