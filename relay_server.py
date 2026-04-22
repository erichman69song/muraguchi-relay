"""
muraguchi-relay — AI API Relay Server
FastAPI main entry point
"""
import logging
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import settings
from routers.proxy import router as proxy_router
from services.rate_limit import limiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("relay_server")


app = FastAPI(
    title="muraguchi-relay",
    description="AI API Relay Server — 多厂商 AI 接口统一代理（Vertex / OpenAI / Anthropic / DeepSeek）",
    version="1.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
if settings.ALLOWED_CLIENT_IPS:
    allow_origins = ["https://muraguchi.club"]
else:
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiter ─────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(proxy_router)


@app.get("/")
async def root():
    return {"service": "muraguchi-relay", "version": "1.0.0", "status": "running"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"[relay_server] 未捕获异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "relay_server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
