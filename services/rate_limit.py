import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Optional

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import settings

log = logging.getLogger("rate_limit")

_in_memory_store: dict[str, list[float]] = defaultdict(list)
_store_lock = Lock()


def _client_key(request) -> str:
    """从请求中提取客户端标识，默认用 IP。"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return f"key:{auth_header[7:]}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(
    key_func=_client_key,
    storage_uri="memory://",
)


def check_rate_limit(request, endpoint: str) -> Optional[str]:
    """
    手动速率检查接口。

    Returns:
        None           — 通过
        "rate_limited" — 被限流

    Raises:
        RateLimitExceeded — 由 FastAPI 异常处理器处理
    """
    limit = settings.RATE_LIMIT_PER_MINUTE
    key = _client_key(request)
    now = time.time()
    window = 60.0

    with _store_lock:
        timestamps = _in_memory_store[key]
        cutoff = now - window
        timestamps[:] = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= limit:
            log.warning(f"[rate_limit] 客户端 {key} 达到限制 {limit}/min (endpoint={endpoint})")
            return "rate_limited"

        timestamps.append(now)

    return None


def clear_client_key(key: str) -> None:
    """手动清除某个 key 的记录（管理接口用）。"""
    with _store_lock:
        _in_memory_store.pop(key, None)
