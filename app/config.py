
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    debug: bool = True
    port: int = 4321

    app_version: str = "1.0.0"
    app_title: str = "Vanna SQL Assistant"
    app_url: str = "http://localhost:4321"
    origin_url: str = "http://localhost:8000"

    model_name: str
    chroma_folder: Optional[str] = "database"
    static_folder: str = "static"
    postgres_conn: Optional[str] = os.getenv("POSTGRES_CONN")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
