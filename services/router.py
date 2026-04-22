import logging
from typing import Any, Optional

import httpx

from config import settings
from services.vertex_auth import call_vertex_ai, VertexAuthError

log = logging.getLogger("router")


class RouterError(Exception):
    pass


PROVIDER_BASE_URLS: dict[str, str] = {
    "openai":    "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "deepseek":  "https://api.deepseek.com/v1",
}

MODEL_PREFIX_MAP: dict[str, str] = {
    "gemini-":   "vertex",
    "claude-":   "anthropic",
    "gpt-":      "openai",
    "o1-":       "openai",
    "o3-":       "openai",
    "deepseek":  "deepseek",
}


def infer_provider(model: str) -> Optional[str]:
    """根据模型名称前缀推断 provider。"""
    model_lower = model.lower()
    for prefix, provider in MODEL_PREFIX_MAP.items():
        if model_lower.startswith(prefix.lower()):
            return provider
    return None


def build_vertex_payload(
    messages: list[dict],
    generation_config: Optional[dict],
) -> tuple[list[dict], dict]:
    """将 OpenAI 格式 messages 转换为 Vertex AI 格式。"""
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            parts = [{"text": content}]
        else:
            parts = content
        contents.append({"role": role, "parts": parts})

    gen_config = generation_config or {}
    return contents, gen_config


def build_openai_payload(
    messages: list[dict],
    model: str,
    generation_config: Optional[dict],
) -> dict:
    """构建 OpenAI 兼容的请求 payload。"""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if generation_config:
        if "temperature" in generation_config:
            payload["temperature"] = generation_config["temperature"]
        if "max_tokens" in generation_config or "maxOutputTokens" in generation_config:
            payload["max_tokens"] = generation_config.get(
                "max_tokens", generation_config.get("maxOutputTokens", 4096)
            )
        if "top_p" in generation_config:
            payload["top_p"] = generation_config["top_p"]
        if "stream" in generation_config:
            payload["stream"] = generation_config["stream"]
    return payload


async def route_chat_completion(
    provider: str,
    model: str,
    messages: list[dict],
    generation_config: Optional[dict] = None,
    **kwargs,
) -> dict:
    """
    统一路由入口，调用对应 provider 并返回 OpenAI 兼容格式。

    Args:
        provider: 强制指定 provider（vertex | openai | anthropic | deepseek | custom）
        model:    模型名称
        messages: OpenAI 格式消息列表
        generation_config: 可选生成参数

    Returns:
        OpenAI 格式的 chat completion 响应
    """
    gen_config = generation_config or {}

    if provider == "vertex":
        return await _route_vertex(model, messages, gen_config, **kwargs)

    if provider in PROVIDER_BASE_URLS:
        return await _route_rest(
            provider, PROVIDER_BASE_URLS[provider], model, messages, gen_config
        )

    raise RouterError(f"不支持的 provider: {provider}")


async def _route_vertex(
    model: str,
    messages: list[dict],
    gen_config: dict,
    **kwargs,
) -> dict:
    """路由到 Vertex AI。"""
    project_id = kwargs.get("project_id")
    location = kwargs.get("location", "us-central1")

    if not project_id:
        raise RouterError("vertex provider 需要提供 project_id")

    contents, _ = build_vertex_payload(messages, gen_config)

    try:
        data = await call_vertex_ai(
            project_id=project_id,
            location=location,
            model=model,
            contents=contents,
            generation_config=gen_config,
        )
    except VertexAuthError as e:
        raise RouterError(f"Vertex AI 调用失败: {e}") from e

    return _vertex_response_to_openai(data)


async def _route_rest(
    provider: str,
    base_url: str,
    model: str,
    messages: list[dict],
    gen_config: dict,
) -> dict:
    """路由到 OpenAI / Anthropic / DeepSeek 等 REST API。"""
    api_key = _get_api_key(provider)
    if not api_key:
        raise RouterError(f"{provider} API key 未配置")

    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if provider == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        payload = _build_anthropic_payload(messages, model, gen_config)
    else:
        payload = build_openai_payload(messages, model, gen_config)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

    if resp.status_code != 200:
        log.error(f"[router] {provider} 错误 {resp.status_code}: {resp.text}")
        raise RouterError(f"{provider} 返回 {resp.status_code}: {resp.text}")

    return resp.json()


def _get_api_key(provider: str) -> str:
    mapping = {
        "openai":    settings.OPENAI_API_KEY,
        "anthropic": settings.ANTHROPIC_API_KEY,
        "deepseek":  settings.DEEPSEEK_API_KEY,
    }
    return mapping.get(provider, "")


def _build_anthropic_payload(
    messages: list[dict],
    model: str,
    gen_config: dict,
) -> dict:
    """将 OpenAI 格式 messages 转换为 Anthropic 格式。"""
    system_msg = ""
    anthropic_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_msg = content
        elif role == "user":
            anthropic_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            anthropic_messages.append({"role": "assistant", "content": content})

    payload: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
    }
    if system_msg:
        payload["system"] = system_msg
    if "temperature" in gen_config:
        payload["temperature"] = gen_config["temperature"]
    if "max_tokens" in gen_config or "maxOutputTokens" in gen_config:
        payload["max_tokens"] = gen_config.get(
            "max_tokens", gen_config.get("maxOutputTokens", 4096)
        )
    return payload


def _vertex_response_to_openai(data: dict) -> dict:
    """将 Vertex AI 响应转换为 OpenAI 兼容格式。"""
    candidates = data.get("candidates", [])
    if not candidates:
        return {"choices": [{"message": {"role": "assistant", "content": ""}}]}

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)

    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": candidates[0].get("finishReason", "STOP"),
            }
        ],
        "usage": data.get("usageMetadata", {}),
        "model": data.get("modelVersion", ""),
    }
