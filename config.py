from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # GCP
    VERTEX_ALLOWED_PROJECTS: str = ""
    GCP_SERVICE_ACCOUNT_JSON_PATH: Optional[str] = None

    # Providers
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""

    # Access Control
    ALLOWED_CLIENT_IPS: str = ""
    RELAY_API_KEY: str = ""

    # FastAPI
    HOST: str = "0.0.0.0"
    PORT: int = 8080

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_FETCH_PER_MINUTE: int = 300

    @property
    def allowed_projects_set(self) -> set[str]:
        if not self.VERTEX_ALLOWED_PROJECTS:
            return set()
        return {p.strip() for p in self.VERTEX_ALLOWED_PROJECTS.split(",") if p.strip()}

    @property
    def allowed_ips_set(self) -> set[str]:
        if not self.ALLOWED_CLIENT_IPS:
            return set()
        return {ip.strip() for ip in self.ALLOWED_CLIENT_IPS.split(",") if ip.strip()}


settings = Settings()
