"""Configuration settings for the application."""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=("settings_",),
        extra="ignore",
    )

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # Model Settings
    model_version: str = "v1"
    model_path: Path = Path("models/v1")

    # Data Settings
    data_processed_path: Path = Path("data/processed")

    # Logging
    log_level: str = "INFO"

    # AI Agent Settings
    huggingface_token: Optional[str] = None
    hf_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    enable_ai_agent: bool = True
    
    # Explainability Settings
    enable_shap: bool = True
    shap_top_features: int = 3


# Global settings instance
settings = Settings()

