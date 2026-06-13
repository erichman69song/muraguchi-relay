import logging
import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional

import httpx
from fastapi import APIRouter, Request, HTTPException, Header
from pydantic import BaseModel, Field

from config import settings
from services.router import route_chat_completion, infer_provider, RouterError, PROVIDER_BASE_URLS, _get_api_key
from services.rate_limit import check_rate_limit

log = logging.getLogger("proxy")
router = APIRouter(prefix="/v1/relay", tags=["relay"])

# ── Async Job Store ──────────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="img_gen_")


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ImageJob:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.result: Optional[dict] = None
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


# In-memory job store (survives within a single relay process)
_jobs: dict[str, ImageJob] = {}
_jobs_lock = asyncio.Lock()


def _cleanup_old_jobs(max_age_seconds: int = 3600):
    """Remove jobs older than max_age_seconds."""
    import time
    now = time.time()
    expired = []
    for jid in list(_jobs.keys()):
        parts = jid.split("_")
        # job_id format: img_{timestamp_ms}_{uuid}
        if len(parts) >= 2:
            try:
                ts = float(parts[1]) / 1000  # ms → seconds
                if now - ts > max_age_seconds:
                    expired.append(jid)
            except ValueError:
                pass
    for jid in expired:
        _jobs.pop(jid, None)


async def _run_openai_image(job_id: str, api_key: str, payload: dict):
    """后台线程池执行 OpenAI image 生成。"""
    import httpx as _hx
    from services.router import PROVIDER_BASE_URLS as _PBU

    async with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].status = JobStatus.PROCESSING

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        with _hx.Client(timeout=_hx.Timeout(300.0)) as client:
            resp = client.post(
                f"{_PBU['openai']}/images/generations",
                json=payload,
                headers=headers,
            )
        if resp.status_code != 200:
            error_detail = f"OpenAI image error HTTP {resp.status_code}: {resp.text[:500]}"
            async with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id].status = JobStatus.FAILED
                    _jobs[job_id].error = error_detail
            log.error(f"[relay_openai_image] job={job_id} failed: {error_detail}")
            return
        result = resp.json()
        async with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].result = result
                _jobs[job_id].status = JobStatus.COMPLETED
        log.info(f"[relay_openai_image] job={job_id} completed")
    except Exception as e:
        async with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].status = JobStatus.FAILED
                _jobs[job_id].error = str(e)
        log.error(f"[relay_openai_image] job={job_id} exception: {e}")


def _verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """验证 relay API key。"""
    if not settings.RELAY_API_KEY:
        return
    if not x_api_key or x_api_key != settings.RELAY_API_KEY:
        raise HTTPException(status_code=401, detail="无效或缺失 API Key")


# ── Request / Response 模型 ──────────────────────────────────────────────────

class ChatCompletionRequest(BaseModel):
    provider: str = Field(..., description="vertex | openai | anthropic | deepseek")
    model: str = Field(..., description="模型名称")
    api_key: Optional[str] = Field(None, description="provider API key（ECS 请求方传入，优先使用）")
    project_id: Optional[str] = Field(None, description="Vertex AI 必需：GCP 项目 ID")
    location: str = Field("us-central1")
    messages: list[dict] = Field(...)
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
    _verify_api_key(x_api_key)

    if rate_limited := check_rate_limit(request, "embeddings"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

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
    return {"status": "ok", "service": "muraguchi-relay"}


@router.post("/validate-project")
async def validate_project(
    project_id: str,
    x_api_key: Optional[str] = Header(None),
):
    _verify_api_key(x_api_key)
    from services.vertex_auth import validate_project_id
    allowed = validate_project_id(project_id)
    return {"project_id": project_id, "allowed": allowed}


# ── RSS / HTTP Fetch 代理 ───────────────────────────────────────────────────────

class FetchRequest(BaseModel):
    url: str = Field(...)
    method: str = Field("GET")
    timeout: int = Field(30, ge=5, le=120)
    headers: Optional[dict[str, str]] = Field(default=None)
    body: Optional[str | dict] = Field(default=None)


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
    _verify_api_key(x_api_key)

    if rate_limited := check_rate_limit(request, "fetch"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    headers = body.headers or {}
    if "User-Agent" not in headers:
        headers["User-Agent"] = "Mozilla/5.0 (compatible; AI-Daily/1.0)"

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
        return FetchResponse(url=body.url, status=0, content="", error="timeout")
    except httpx.RequestError as e:
        return FetchResponse(url=body.url, status=0, content="", error=str(e))
    except Exception as e:
        return FetchResponse(url=body.url, status=0, content="", error=str(e))


# ── 图片生成：异步 Job 模式 ───────────────────────────────────────────────────

class ImageGenerateRequest(BaseModel):
    model: str
    prompt: str
    n: int = Field(default=1, ge=1, le=10)
    size: str = Field(default="1024x1024")
    api_key: Optional[str] = Field(default=None)
    style: Optional[str] = Field(default=None)

    model_config = {"extra": "allow"}


class ImageJobResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


@router.post("/openai/image", response_model=dict)
async def relay_openai_image_submit(
    request: Request,
    body: ImageGenerateRequest,
    x_api_key: Optional[str] = Header(None),
):
    """
    OpenAI / GPT-Image 代理端点 — 异步 Job 模式。
    立即返回 job_id，后台线程池执行 OpenAI 调用。
    ECS 通过 GET /v1/relay/openai/image/{job_id} 轮询结果。
    """
    _verify_api_key(x_api_key)

    if rate_limited := check_rate_limit(request, "openai/image"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    request_api_key = getattr(body, "api_key", None) or None
    api_key = _get_api_key("openai", request_api_key)
    if not api_key:
        raise HTTPException(status_code=502, detail="OpenAI API key 未配置")

    import time
    job_id = f"img_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    job = ImageJob(job_id)
    async with _jobs_lock:
        _jobs[job_id] = job
    _cleanup_old_jobs()

    payload = {
        "model": body.model,
        "prompt": body.prompt,
        "n": body.n,
        "size": body.size,
    }
    if getattr(body, "style", None):
        payload["style"] = body.style

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, lambda: asyncio.run(_run_openai_image(job_id, api_key, payload)))

    log.info(f"[relay_openai_image] submitted job={job_id} model={body.model} n={body.n}")
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Job 已提交，请通过 GET /v1/relay/openai/image/{job_id} 轮询结果",
    }


@router.get("/openai/image/{job_id}", response_model=ImageJobResponse)
async def relay_openai_image_status(
    job_id: str,
    x_api_key: Optional[str] = Header(None),
):
    """
    轮询 OpenAI image job 状态。
    """
    _verify_api_key(x_api_key)

    async with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} 不存在或已过期")

    return job.to_dict()


# ── Google Image 代理（保持原有逻辑） ───────────────────────────────────────

class ImageGenerateGoogleRequest(BaseModel):
    model: str = Field(default="gemini-3-pro-image-preview")
    contents: Optional[list[dict]] = Field(default=None)
    input: Optional[list[dict]] = Field(default=None)
    api_revision: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)

    model_config = {"extra": "allow"}


@router.post("/google/image", response_model=dict)
async def relay_google_image(
    request: Request,
    body: ImageGenerateGoogleRequest,
    x_api_key: Optional[str] = Header(None),
):
    _verify_api_key(x_api_key)

    if rate_limited := check_rate_limit(request, "google/image"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    request_api_key = getattr(body, "api_key", None) or None
    env_api_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key = request_api_key or env_api_key
    if not api_key:
        raise HTTPException(status_code=502, detail="Google API key 未配置")

    model = getattr(body, "model", "gemini-3-pro-image-preview") or "gemini-3-pro-image-preview"

    has_input = hasattr(body, "input") and body.input is not None
    has_contents = hasattr(body, "contents") and body.contents is not None

    if has_input:
        base_url = "https://generativelanguage.googleapis.com"
        url = f"{base_url}/v1beta/interactions"

        google_headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        if hasattr(body, "api_revision") and body.api_revision:
            google_headers["Api-Revision"] = body.api_revision

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(
                url,
                json={"model": model, "input": body.input},
                headers=google_headers,
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

        resp_data = resp.json()
        steps = resp_data.get("steps", [])
        for step in steps:
            contents = step.get("content", [])
            for item in contents:
                if isinstance(item, dict) and item.get("mime_type", "").startswith("image/"):
                    return {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "inlineData": {
                                                "data": item["data"],
                                                "mimeType": item["mime_type"],
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
        return resp_data

    elif has_contents:
        base_url = "https://generativelanguage.googleapis.com"
        url = f"{base_url}/v1beta/models/{model}:generateContent"

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(
                url,
                json={"contents": body.contents},
                headers={"Content-Type": "application/json"},
                params={"key": api_key},
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

    else:
        raise HTTPException(status_code=400, detail="请求 body 必须包含 input 或 contents 字段")
