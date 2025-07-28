import logging
from functools import lru_cache
import pykx as kx
from typing import Optional
from mcp_server.settings import KDBConfig, default_kdb_config
from mcp_server.server import config

logger = logging.getLogger(__name__)

@lru_cache()
def get_kdb_connection(config: Optional[KDBConfig] = None) -> kx.QConnection:
    if config is None:
        config = default_kdb_config

    logger.debug(f"KDBConfig: {config=}")
    logger.info(f"Connecting to KDB at {config.host}:{config.port}")
    retry = config.retry
    success = False  # flag to track if connection was successful
    for attempt in range(1, retry + 1):
        try:
            conn = kx.SyncQConnection(
                host=config.host,
                port=config.port,
                username=config.username,
                password=config.password.get_secret_value(),
                timeout=config.timeout,
                reconnection_attempts=2
            )
            logger.info("Connected to Q/KDB-X")
            return conn
        except Exception as e:
            logger.warning(f"KDB-X connectivity attempt {attempt}/{retry} failed: {str(e)}")

    logger.error(f"Failed to connect to KDB")
    raise

def run_kdbx_sql_query(query:str, max_rows:int) -> any:
    try:
        conn = get_kdb_connection(config.kdbx_config)
        result = conn('{r:.s.e x;`rowCount`data!(count r;y sublist r)}', kx.CharVector(query), max_rows)
        return result
    except Exception as e:
        if "Attempted to use a closed IPC connection" in str(e):
            logger.warning("KDB-X connection was closed. Reinitializing...")
            cleanup_kdb_connection()
            conn = get_kdb_connection(config.kdbx_config)
            return conn('{r:.s.e x;`rowCount`data!(count r;y sublist r)}', kx.CharVector(query), max_rows)
        else:
            logger.error(f"Error running KDB query: {e}")
            raise

def cleanup_kdb_connection():
    get_kdb_connection.cache_clear()
    logger.info("KDB connection cache cleared")
