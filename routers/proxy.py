import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from services.router import route_chat_completion, infer_provider, RouterError
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
    timeout: int = Field(30, ge=5, le=120, description="超时秒数，默认30秒")
    headers: Optional[dict[str, str]] = Field(
        default=None,
        description="可选的额外请求头",
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
            resp = await client.get(body.url, headers=headers, follow_redirects=True)

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
