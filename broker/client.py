"""Redis client for the broker service."""

import redis as redis_lib
from typing import Optional

import config
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def is_redis_connection_error(exc: Exception) -> bool:
    """Return True when an exception indicates a transient Redis connection failure."""
    return isinstance(
        exc,
        (
            redis_lib.exceptions.ConnectionError,
            redis_lib.exceptions.TimeoutError,
        ),
    )


def reset_redis_connection(reason: Optional[str] = None) -> None:
    """Force-close pooled sockets so the next command reconnects with fresh connections."""
    if not redis_client:
        return

    try:
        redis_client.connection_pool.disconnect()

        if reason:
            logger.warning(f"Redis connection pool reset: {reason}")

    except Exception as exc:
        logger.warning(f"Failed to reset Redis connection pool: {exc}")


try:
    # Define the Redis client
    redis_client = redis_lib.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        password=config.REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=config.REDIS_SOCKET_CONNECT_TIMEOUT,
        socket_timeout=config.REDIS_SOCKET_TIMEOUT,
        socket_keepalive=True,
        health_check_interval=config.REDIS_HEALTH_CHECK_INTERVAL,
        retry_on_timeout=False
    )

    # Check the connection
    redis_client.ping()
    logger.info(f"✅ Redis connected: {config.REDIS_HOST}:{config.REDIS_PORT}")

except Exception as redis_exc:
    logger.error(f"❌ Redis failed: {redis_exc}")
    redis_client = None
