"""模型客户端初始化。

两套 provider：
- DashScope（OpenAI 兼容端点）：裸模型名（如 `qwen3.5-plus`）。
- 本地 Gemini 聚合代理：模型名以 `gemini/` 前缀开头。通过 `GeminiAdapter`
  包装 `google-generativeai`，对外暴露与 OpenAI SDK 相同的 `.chat.completions.create(...)`
  接口，使下游调用点零改动。
"""

import base64
import os
import threading
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI

from .config import is_gemini_model

load_dotenv()


# ══════════════════════════════════════════════════════════════════════
# Gemini 代理适配器
# ══════════════════════════════════════════════════════════════════════

_DEFAULT_GEMINI_BASE_URL = "http://127.0.0.1:8045"
_gemini_configure_lock = threading.Lock()
_gemini_configured_for: tuple[str, str] | None = None


def _ensure_gemini_configured(api_key: str, base_url: str) -> None:
    """幂等地初始化 google-generativeai 的全局状态。

    `genai.configure()` 是进程级全局状态；相同 (key, base_url) 只配一次。
    """
    global _gemini_configured_for
    import google.generativeai as genai

    with _gemini_configure_lock:
        if _gemini_configured_for == (api_key, base_url):
            return
        genai.configure(
            api_key=api_key,
            transport="rest",
            client_options={"api_endpoint": base_url},
        )
        _gemini_configured_for = (api_key, base_url)


def _extract_system_text(system_content) -> str:
    """把 OpenAI 风格的 system content（str 或 list[dict]）扁平化为纯字符串。

    会忽略 DashScope/Anthropic 特有的 `cache_control` 字段。
    """
    if system_content is None:
        return ""
    if isinstance(system_content, str):
        return system_content
    if isinstance(system_content, list):
        parts = []
        for item in system_content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(system_content)


def _openai_content_to_gemini_parts(content) -> list:
    """把 OpenAI 风格的 user/assistant content 转成 Gemini `parts` 列表。

    支持：
    - 纯字符串 → 单个文本 part
    - list[dict]：`{"type":"text","text":...}` / `{"type":"image_url","image_url":{"url":"data:<mime>;base64,<b64>"}}`
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [content] if content else []

    parts: list = []
    if not isinstance(content, list):
        return [str(content)]

    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            if text:
                parts.append(text)
        elif item_type == "image_url":
            image_url = item.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not url:
                continue
            # 仅支持 data URL（项目内部都用 encode_data_url 生成）
            if url.startswith("data:"):
                try:
                    header, b64_payload = url.split(",", 1)
                    # header 形如 "data:image/png;base64"
                    mime = header[5:].split(";", 1)[0] or "image/png"
                    raw = base64.b64decode(b64_payload)
                    parts.append({"mime_type": mime, "data": raw})
                except Exception:
                    continue
            # 远程 URL 暂不支持——本项目未用到
        elif "text" in item:
            parts.append(str(item["text"]))

    return parts


class _GeminiChatCompletions:
    """模拟 `openai.resources.chat.completions.Completions` 的最小接口。"""

    def __init__(self, adapter: "GeminiAdapter"):
        self._adapter = adapter

    def create(
        self,
        model: str,
        messages: list,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        **_ignored,
    ):
        """调用 Gemini 代理，返回 OpenAI 形状的响应对象。

        自动忽略 Qwen/DashScope/Anthropic 专属参数：
        - `extra_body={"enable_thinking": ...}`
        - 系统提示中的 `cache_control`
        """
        import google.generativeai as genai

        _ensure_gemini_configured(self._adapter.api_key, self._adapter.base_url)

        # 剥掉 "gemini/" 前缀
        real_model = model[len("gemini/"):] if model.startswith("gemini/") else model

        system_text = ""
        contents: list[dict] = []

        for msg in messages or []:
            role = msg.get("role", "user")
            content = msg.get("content")

            if role == "system":
                system_text_piece = _extract_system_text(content)
                if system_text_piece:
                    system_text = (
                        system_text_piece if not system_text
                        else f"{system_text}\n{system_text_piece}"
                    )
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts = _openai_content_to_gemini_parts(content)
            if not parts:
                continue
            contents.append({"role": gemini_role, "parts": parts})

        generation_config: dict = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        model_kwargs = {}
        if system_text:
            model_kwargs["system_instruction"] = system_text

        gm = genai.GenerativeModel(real_model, **model_kwargs)
        response = gm.generate_content(
            contents,
            generation_config=generation_config or None,
        )

        # 提取文本：优先 response.text；失败时遍历 candidates/parts
        text = ""
        try:
            text = response.text or ""
        except Exception:
            try:
                cand = (response.candidates or [None])[0]
                if cand and getattr(cand, "content", None):
                    parts = getattr(cand.content, "parts", []) or []
                    text = "".join(getattr(p, "text", "") or "" for p in parts)
            except Exception:
                text = ""

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text or ""),
                    finish_reason="stop",
                    index=0,
                )
            ]
        )


class _GeminiChat:
    def __init__(self, adapter: "GeminiAdapter"):
        self.completions = _GeminiChatCompletions(adapter)


class GeminiAdapter:
    """对外暴露 OpenAI 风格 `.chat.completions.create(...)` 的 Gemini 代理适配器。"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = _GeminiChat(self)


# ══════════════════════════════════════════════════════════════════════
# 工厂
# ══════════════════════════════════════════════════════════════════════

def _get_dashscope_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "  Windows:  set DASHSCOPE_API_KEY=sk-xxxxx\n"
            "  Linux:    export DASHSCOPE_API_KEY=sk-xxxxx\n"
            "或在项目根目录的 .env 文件中写入 DASHSCOPE_API_KEY=sk-xxxxx"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _get_gemini_client() -> GeminiAdapter:
    api_key = os.environ.get("GEMINI_PROXY_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "请设置环境变量 GEMINI_PROXY_API_KEY（本地 Gemini 聚合代理 key）\n"
            "  Windows:  set GEMINI_PROXY_API_KEY=sk-xxxxx\n"
            "  Linux:    export GEMINI_PROXY_API_KEY=sk-xxxxx\n"
            "或在项目根目录的 .env 文件中写入 GEMINI_PROXY_API_KEY=sk-xxxxx"
        )
    base_url = os.environ.get("GEMINI_PROXY_BASE_URL", _DEFAULT_GEMINI_BASE_URL)
    return GeminiAdapter(api_key=api_key, base_url=base_url)


def get_client(model: str | None = None):
    """按模型名返回对应 provider 的客户端。

    - `model` 以 `gemini/` 开头 → 本地 Gemini 聚合代理（`GeminiAdapter`）。
    - 其他（含 `None`）→ 阿里云 DashScope OpenAI 兼容客户端。

    两个 provider 的 API Key 仅在被选中时才校验。
    """
    if model and is_gemini_model(model):
        return _get_gemini_client()
    return _get_dashscope_client()
