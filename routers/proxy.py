import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from services.router import route_chat_completion, infer_provider, RouterError, PROVIDER_BASE_URLS, _get_api_key
from services.rate_limit import check_rate_limit

log = logging.getLogger("proxy")

router = APIRouter(prefix="/v1/relay", tags=["relay"])


def _verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """验证 relay API key。"""
    if not settings.RELAY_API_KEY:
        return
    if not x_api_key or x_api_key != settings.RELAY_API_KEY:
        raise HTTPException(status_code=401, detail="无效或缺失 API Key")


# ── Request / Response 模型 ──────────────────────────────────────────────────

class ChatCompletionRequest(BaseModel):
    provider: str = Field(..., description="vertex | openai | anthropic | deepseek")
    model: str = Field(..., description="模型名称，如 gemini-3.1-pro-preview")
    api_key: Optional[str] = Field(None, description="provider API key（ECS 请求方传入，优先使用）")
    project_id: Optional[str] = Field(None, description="Vertex AI 必需：GCP 项目 ID")
    location: str = Field("us-central1", description="区域，默认 us-central1")
    messages: list[dict] = Field(..., description="OpenAI 格式消息列表")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(4096, ge=1, le=32768)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    stream: bool = Field(False)

    model_config = {"extra": "allow"}


class EmbeddingsRequest(BaseModel):
    provider: str = Field(..., description="openai")
    model: str = Field(default="text-embedding-3-small")
    input: str | list[str]


# ── 端点实现 ─────────────────────────────────────────────────────────────────

@router.post("/chat/completions")
async def relay_chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    x_api_key: Optional[str] = Header(None),
):
    """
    统一聊天补全代理端点。

    请求示例（Vertex AI）:
    ```json
    {
      "provider": "vertex",
      "model": "gemini-3.1-pro-preview",
      "project_id": "gen-lang-client-0568442340",
      "messages": [{"role": "user", "content": "Hello!"}]
    }
    ```
    """
    _verify_api_key(x_api_key)

    if rate_limited := check_rate_limit(request, "chat/completions"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    generation_config = {
        "temperature": body.temperature,
        "maxOutputTokens": body.max_tokens,
    }
    if body.top_p is not None:
        generation_config["top_p"] = body.top_p

    try:
        provider = body.provider
        if provider == "auto":
            inferred = infer_provider(body.model)
            if not inferred:
                raise HTTPException(400, "无法从 model 推断 provider，请显式指定")
            provider = inferred

        result = await route_chat_completion(
            provider=provider,
            model=body.model,
            messages=body.messages,
            generation_config=generation_config,
            api_key=body.api_key,
            project_id=body.project_id,
            location=body.location,
        )
        return result

    except RouterError as e:
        log.error(f"[proxy] chat/completions 错误: {e}")
        raise HTTPException(502, str(e))


@router.post("/embeddings")
async def relay_embeddings(
    request: Request,
    body: EmbeddingsRequest,
    x_api_key: Optional[str] = Header(None),
):
    """向量嵌入代理端点（目前仅透传 OpenAI 兼容接口）。"""
    _verify_api_key(x_api_key)

    if rate_limited := check_rate_limit(request, "embeddings"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    import httpx
    from services.router import PROVIDER_BASE_URLS, _get_api_key

    base_url = PROVIDER_BASE_URLS.get(body.provider)
    if not base_url:
        raise HTTPException(400, f"embeddings 不支持 provider: {body.provider}")

    api_key = _get_api_key(body.provider)
    if not api_key:
        raise HTTPException(502, f"{body.provider} API key 未配置")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": body.model, "input": body.input}

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        resp = await client.post(
            f"{base_url}/embeddings",
            json=payload,
            headers=headers,
        )

    if resp.status_code != 200:
        raise HTTPException(502, f"嵌入服务返回 {resp.status_code}: {resp.text}")

    return resp.json()


@router.get("/models")
async def list_models(
    x_api_key: Optional[str] = Header(None),
):
    """查询可用模型列表。"""
    _verify_api_key(x_api_key)
    return {
        "object": "list",
        "data": [
            {"id": "gemini-3.1-pro-preview", "object": "model", "provider": "vertex"},
            {"id": "gemini-2.0-flash", "object": "model", "provider": "vertex"},
            {"id": "gemini-2.5-pro-preview", "object": "model", "provider": "vertex"},
            {"id": "gpt-4o", "object": "model", "provider": "openai"},
            {"id": "gpt-4o-mini", "object": "model", "provider": "openai"},
            {"id": "claude-3-5-sonnet-20241022", "object": "model", "provider": "anthropic"},
            {"id": "deepseek-chat", "object": "model", "provider": "deepseek"},
        ],
    }


@router.get("/health")
async def health_check():
    """健康检查端点。"""
    return {"status": "ok", "service": "muraguchi-relay"}


@router.post("/validate-project")
async def validate_project(
    project_id: str,
    x_api_key: Optional[str] = Header(None),
):
    """验证 project_id 是否在白名单中。"""
    _verify_api_key(x_api_key)
    from services.vertex_auth import validate_project_id
    allowed = validate_project_id(project_id)
    return {"project_id": project_id, "allowed": allowed}


# ── RSS / HTTP Fetch 代理 ───────────────────────────────────────────────────────

class FetchRequest(BaseModel):
    url: str = Field(..., description="要抓取的 URL")
    method: str = Field("GET", description="HTTP 方法: GET | POST | PUT | DELETE")
    timeout: int = Field(30, ge=5, le=120, description="超时秒数，默认30秒")
    headers: Optional[dict[str, str]] = Field(
        default=None,
        description="可选的额外请求头",
    )
    body: Optional[str | dict] = Field(
        default=None,
        description="请求体，字符串或 dict（用于 POST/PUT）",
    )


class FetchResponse(BaseModel):
    url: str
    status: int
    content: str
    error: Optional[str] = None


@router.post("/fetch", response_model=FetchResponse)
async def relay_fetch(
    request: Request,
    body: FetchRequest,
    x_api_key: Optional[str] = Header(None),
):
    """
    HTTP GET 代理端点，供 ECS 抓取外网 RSS 源使用。
    EC2 可以访问外网，承担所有出站 HTTP 请求。

    请求示例：
    ```json
    {
      "url": "https://nitter.net/karpathy/rss",
      "timeout": 15,
      "headers": {"User-Agent": "Mozilla/5.0 ..."}
    }
    ```

    响应示例：
    ```json
    {
      "url": "https://nitter.net/karpathy/rss",
      "status": 200,
      "content": "<?xml ...",
      "error": null
    }
    ```
    """
    _verify_api_key(x_api_key)

    if rate_limited := check_rate_limit(request, "fetch"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    headers = body.headers or {}
    if "User-Agent" not in headers:
        headers["User-Agent"] = headers.get(
            "User-Agent", "Mozilla/5.0 (compatible; AI-Daily/1.0)"
        )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(body.timeout)) as client:
            req_kwargs: dict = {"headers": headers, "follow_redirects": True}
            method = body.method.upper()
            if method in ("POST", "PUT", "PATCH") and body.body:
                if isinstance(body.body, dict):
                    req_kwargs["json"] = body.body
                else:
                    req_kwargs["content"] = body.body
            resp = await client.request(method, body.url, **req_kwargs)

        return FetchResponse(
            url=body.url,
            status=resp.status_code,
            content=resp.text,
            error=None,
        )
    except httpx.TimeoutException:
        log.warning(f"[relay_fetch] 超时 {body.url}")
        return FetchResponse(url=body.url, status=0, content="", error="timeout")
    except httpx.RequestError as e:
        log.warning(f"[relay_fetch] 请求错误 {body.url}: {e}")
        return FetchResponse(url=body.url, status=0, content="", error=str(e))
    except Exception as e:
        log.error(f"[relay_fetch] 未知错误 {body.url}: {e}")
        return FetchResponse(url=body.url, status=0, content="", error=str(e))


# ── 图片生成专用代理 ────────────────────────────────────────────────────────────

class ImageGenerateRequest(BaseModel):
    model: str
    prompt: str
    n: int = Field(default=1, ge=1, le=10)
    size: str = Field(default="1024x1024")
    api_key: Optional[str] = Field(default=None)  # ECS DB 中的 key，通过 body 传入

    model_config = {"extra": "allow"}


class ImageGenerateGoogleRequest(BaseModel):
    model: str = Field(default="gemini-3-pro-image-preview")
    contents: list[dict]  # Google generateContent 格式
    api_key: Optional[str] = Field(default=None)  # ECS DB 中的 key，通过 body 传入

    model_config = {"extra": "allow"}


@router.post("/openai/image", response_model=dict)
async def relay_openai_image(
    request: Request,
    body: ImageGenerateRequest,
    x_api_key: Optional[str] = Header(None),
):
    """
    OpenAI / GPT-Image 代理端点（走 /v1/images/generations）。
    通过 relay 代为转发，EC2 可访问 OpenAI API。
    """
    _verify_api_key(x_api_key)

    if rate_limited := check_rate_limit(request, "openai/image"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    # 优先用请求 body 中的 api_key（ECS DB 里的 key）
    # 其次用 relay .env 配置的 key
    request_api_key = getattr(body, "api_key", None) or None
    api_key = _get_api_key("openai", request_api_key)
    if not api_key:
        raise HTTPException(status_code=502, detail="OpenAI API key 未配置（请求方未传且 relay 未配置）")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": body.model,
        "prompt": body.prompt,
        "n": body.n,
        "size": body.size,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(
            f"{PROVIDER_BASE_URLS['openai']}/images/generations",
            json=payload,
            headers=headers,
        )

    if resp.status_code != 200:
        # 返回原始错误 body（可能是 JSON 或纯文本）
        try:
            err_json = resp.json()
            raise HTTPException(status_code=502, detail=f"OpenAI image error: {err_json}")
        except Exception:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI image error HTTP {resp.status_code}: {resp.text[:500]}",
            )

    return resp.json()


@router.post("/google/image", response_model=dict)
async def relay_google_image(
    request: Request,
    body: ImageGenerateGoogleRequest,
    x_api_key: Optional[str] = Header(None),
):
    """
    Google Imagen / Gemini Image 代理端点（走 generateContent）。
    通过 relay 代为转发，EC2 可访问 Google API。
    """
    _verify_api_key(x_api_key)

    if rate_limited := check_rate_limit(request, "google/image"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    # 优先用请求 body 中的 api_key（ECS DB 里的 key）
    # 其次用 relay .env 配置的 key
    request_api_key = getattr(body, "api_key", None) or None
    env_api_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key = request_api_key or env_api_key
    if not api_key:
        raise HTTPException(status_code=502, detail="Google API key 未配置（请求方未传且 relay 未配置）")

    # 从请求 body 中取 model（默认 gemini-3-pro-image-preview）
    model = getattr(body, "model", "gemini-3-pro-image-preview") or "gemini-3-pro-image-preview"

    base_url = "https://generativelanguage.googleapis.com"
    url = f"{base_url}/v1beta/models/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
    }
    params = {"key": api_key}

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(
            url,
            json=body.contents,
            headers=headers,
            params=params,
        )

    if resp.status_code != 200:
        try:
            err_json = resp.json()
            raise HTTPException(status_code=502, detail=f"Google image error: {err_json}")
        except Exception:
            raise HTTPException(
                status_code=502,
                detail=f"Google image error HTTP {resp.status_code}: {resp.text[:500]}",
            )

    return resp.json()
