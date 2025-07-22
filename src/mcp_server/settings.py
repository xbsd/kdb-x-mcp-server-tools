from pydantic_settings import BaseSettings
from pydantic import SecretStr

from typing import Optional, Literal

class KDBConfig(BaseSettings):
    host: str = "localhost"
    port: int = 5000
    username: Optional[str] = ""
    password: Optional[SecretStr] = ""
    timeout: Optional[int] = 1
    retry: Optional[int] = 2

    class Config:
        env_prefix = 'KDBX_'

class ServerConfig(BaseSettings):
    server_name: str = "KDB-X_Demo"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    transport: Literal["stdio", "streamable-http"] = "streamable-http"
    port: int = 8000
    host: str = '127.0.0.1'

    class Config:
        env_prefix = 'KDBX_MCP_'

server_config = ServerConfig()
default_kdb_config = KDBConfig()