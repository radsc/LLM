from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    OLLAMA_URL:str
    MODEL:str
    EMBEDDING_MODEL:str
    PERSIST_DIR:str

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"), 
        env_file_encoding='utf-8', 
        case_sensitive=False
    )


settings = Settings()