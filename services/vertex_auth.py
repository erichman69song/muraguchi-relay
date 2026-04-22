import logging
from typing import Optional

import google.auth
import google.auth.transport.requests
import httpx

from config import settings

log = logging.getLogger("vertex_auth")


class VertexAuthError(Exception):
    pass


def validate_project_id(project_id: str) -> bool:
    """检查 project_id 是否在白名单中。"""
    if not project_id:
        return False
    allowed = settings.allowed_projects_set
    if not allowed:
        log.warning("[vertex_auth] VERTEX_ALLOWED_PROJECTS 未配置，跳过白名单检查")
        return True
    return project_id in allowed


def get_vertex_access_token() -> str:
    """使用 GCP Application Default Credentials 获取访问令牌。"""
    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except google.auth.exceptions.DefaultCredentialsError as e:
        raise VertexAuthError(f"GCP 凭据未找到: {e}") from e

    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    if not credentials.token:
        raise VertexAuthError("获取 token 失败，credentials.token 为空")

    return credentials.token


async def call_vertex_ai(
    project_id: str,
    location: str,
    model: str,
    contents: list[dict],
    generation_config: Optional[dict] = None,
    stream: bool = False,
) -> dict:
    """
    向 Vertex AI 发起 generateContent 请求。

    Args:
        project_id: GCP 项目 ID
        location:   区域，如 us-central1
        model:      模型名称，如 gemini-3.1-pro-preview
        contents:   Vertex AI 格式的内容列表
        generation_config: 可选，覆盖默认生成参数
        stream:     是否使用流式响应

    Returns:
        Vertex AI 返回的完整 JSON 响应字典
    """
    if not validate_project_id(project_id):
        raise VertexAuthError(f"project_id '{project_id}' 不在白名单中")

    token = get_vertex_access_token()

    base_url = "https://aiplatform.googleapis.com/v1beta1"
    endpoint = f"{base_url}/projects/{project_id}/locations/{location}/publishers/google/models/{model}:generateContent"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload: dict = {
        "contents": contents,
        "generationConfig": generation_config or {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
        },
    }
    if stream:
        payload["stream"] = True

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)

    if resp.status_code != 200:
        log.error(f"[vertex_auth] Vertex AI 错误 {resp.status_code}: {resp.text}")
        raise VertexAuthError(f"Vertex AI 返回 {resp.status_code}: {resp.text}")

    return resp.json()
