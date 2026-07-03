"""Redis connection keepalive daemon."""

import time

import config
from broker.client import is_redis_connection_error, redis_client, reset_redis_connection
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def redis_keepalive_daemon() -> None:
    """Periodically ping Redis to reduce idle-connection drops by network middleboxes."""
    if not redis_client:
        logger.warning("Redis client not initialized; keepalive daemon will not start")
        return

    logger.info(f"Redis keepalive daemon started (interval={config.REDIS_KEEPALIVE_INTERVAL_SECONDS}s)")

    while True:
        try:
            # Sleep before the next keepalive ping.
            time.sleep(config.REDIS_KEEPALIVE_INTERVAL_SECONDS)

            # Ping Redis to keep the connection alive.
            redis_client.ping()

        except Exception as exc:
            # Handle Redis connection errors separately to attempt a reset.
            if is_redis_connection_error(exc):
                reset_redis_connection("keepalive ping connectivity issue")

            logger.debug(f"Keepalive ping failed (will retry): {exc}")
