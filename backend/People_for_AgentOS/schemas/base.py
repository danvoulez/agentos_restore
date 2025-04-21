# agentos-pessoas/schemas/base.py
from pydantic import BaseModel, Field
# Import BaseSettings and SettingsConfigDict for proper config loading
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from loguru import logger  # Import logger for potential warnings

class MsgDetail(BaseModel):
    msg: str = Field(..., description="Mensagem descritiva.")

# Inherit from BaseSettings and add required config variables
class Settings(BaseSettings):
    APP_NAME: str = "AgentOS Pessoas"
    LOG_LEVEL: str = "INFO"
    # MongoDB configuration
    MONGODB_URI: str  # MUST be set in environment
    MONGO_DB_NAME: str | None = None  # Optional: derive from URI if not set

    # JWT configuration - MUST match the backend issuing the tokens
    JWT_SECRET: str  # MUST be set in environment
    ALGORITHM: str = "HS256"  # Default, ensure matches issuer

    # Optional: Add S2S secret if using static token approach (INSECURE - plan replacement)
    INTERNAL_S2S_SHARED_SECRET: str | None = None

    # Optional: Host/Port/Reload for local dev via uvicorn
    HOST: str = "0.0.0.0"
    PORT: int = 8001  # Use a different default port than main backend
    RELOAD: bool = True  # Enable reload for local dev

    model_config = SettingsConfigDict(
        env_file=str(Path.cwd() / ".env.pessoas"),  # Look for .env.pessoas first
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars not defined in the model
        case_sensitive=False  # Environment variables are typically case-insensitive
    )

    # --- Post-validation/Defaults ---
    def __init__(self, **values):
        super().__init__(**values)
        if self.MONGO_DB_NAME is None:
            try:
                db_name_from_uri = self.MONGODB_URI.split('/')[-1].split('?')[0]
                if db_name_from_uri:
                    self.MONGO_DB_NAME = db_name_from_uri
                else:
                    self.MONGO_DB_NAME = "agentos_pessoas"  # Fallback default
                    logger.warning(f"Could not derive MONGO_DB_NAME from URI, defaulting to: {self.MONGO_DB_NAME}")
            except Exception:
                self.MONGO_DB_NAME = "agentos_pessoas"  # Fallback default on error
                logger.warning(f"Could not derive MONGO_DB_NAME from URI, defaulting to: {self.MONGO_DB_NAME}")

        if not self.JWT_SECRET:
            logger.critical("FATAL: JWT_SECRET is not set in environment/config!")
            raise ValueError("JWT_SECRET environment variable is required.")
        if not self.MONGODB_URI:
            logger.critical("FATAL: MONGODB_URI is not set in environment/config!")
            raise ValueError("MONGODB_URI environment variable is required.")

try:
    settings = Settings()
except ValueError as e:
    logger.critical(f"Missing required configuration: {e}")
    import sys
    sys.exit(f"Configuration Error: {e}")
except Exception as e:
    logger.critical(f"Unexpected error loading settings: {e}")
    import sys
    sys.exit(f"Unexpected Settings Error: {e}")
