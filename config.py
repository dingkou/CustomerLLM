# ---------- config.py ----------
# from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

class Settings(object):
    DEEPSEEK_API_KEY: str = DEEPSEEK_API_KEY
    EMBEDDING_MODEL: str = EMBEDDING_MODEL
    # model_config = SettingsConfigDict(
    #     env_file=".env",
    #     env_file_encoding="utf-8",
    #     extra="ignore"  # 添加此项允许忽略额外字段
    # )

    # class Config:
    #     env_file = ".env"


settings = Settings()

# if __name__ == "__main__":
#     print(settings.DEEPSEEK_API_KEY)

