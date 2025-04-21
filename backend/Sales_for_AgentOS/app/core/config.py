from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Mongo
    SALES_MONGODB_URI: str
    SALES_MONGO_DB_NAME: str = "agentos_sales"

    # Auth
    AUTH_JWT_SECRET: str
    AUTH_ALGORITHM: str = "HS256"

    # Celery / Redis
    SALES_CELERY_BROKER: str
    SALES_CELERY_BACKEND: str

    # Integration URLs / Keys
    PEOPLE_SERVICE_URL: str | None = None
    PEOPLE_SERVICE_S2S_TOKEN: str | None = None
    BANKING_SERVICE_URL: str | None = None
    BANKING_API_KEY: str | None = None
    DELIVERY_SERVICE_URL: str | None = None
    DELIVERY_API_KEY: str | None = None

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

@lru_cache
def get_settings() -> Settings:
    return Settings()