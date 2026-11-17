"""從環境變數讀取後端的設定,變數清單見 repo 根目錄的 .env.example。"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "info"
    finmind_token: str | None = None


settings = Settings()
