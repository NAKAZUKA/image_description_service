from pydantic_settings import BaseSettings
from typing import ClassVar
import os

class Settings(BaseSettings):
    MODEL_NAME: str
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models_cache/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit")
    DEVICE: str
    TRANSLATOR_URL: str
    PORT: int
    HOST: str

    class Config:
        env_file = ".env"

settings = Settings()
