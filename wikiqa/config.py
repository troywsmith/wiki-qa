from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="WIKIQA_", extra="ignore")

    anthropic_api_key: str = ""
    model: str = "claude-sonnet-4-6"  # answerer
    judge_model: str = "claude-opus-4-8"  # eval judge — must differ from the answerer
    max_agent_steps: int = 6
    max_tokens: int = 1024
    wikipedia_lang: str = "en"
    request_timeout: float = 15.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
