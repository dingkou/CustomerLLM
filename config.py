# ---------- config.py ----------
# from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(object):
    DEEPSEEK_API_KEY: str = "sk-nfgrlybhbvmabswhysnhysuhyaknndjhuhmadkuzojulrloo"
    EMBEDDING_MODEL: str = "E:\huggingface_model_file\\all-MiniLM-L6-v2"
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

