"""Utility functions for cleaning up temporary files and old Redis Stream entries."""

import glob
import os
import time

from broker.queue import cleanup_old_stream_entries
import config
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def cleanup_task_files(uid: str):
    """
    Delete all local temp files.

    Args:
        uid: The unique identifier for the task.
    """
    logger.info(f"🧹 Cleanup temporary files for {uid}")

    pattern = os.path.join(config.SAVE_DIR, f"{uid}*")
    deleted = 0

    for file_path in glob.glob(pattern):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted += 1

        except Exception:
            pass

    logger.info(f"✨ Cleaned {deleted} files for {uid}")


def auto_cleanup_daemon(interval_seconds: int = 60, file_age_limit: int = 300):
    """
    Periodically delete local temp files older than a certain age, and trim old Redis Stream entries.

    Args:
        interval_seconds: Sleep between runs (default 60 s).
        file_age_limit: Delete files older than this many seconds (default 300 s).
    """
    logger.info(f"🧹 Auto-cleanup daemon started: interval={interval_seconds}s, age_limit={file_age_limit}s")

    while True:
        try:
            now = time.time()

            files = glob.glob(os.path.join(config.SAVE_DIR, "*"))
            deleted = 0

            for file_path in files:
                try:
                    if now - os.path.getmtime(file_path) > file_age_limit:
                        os.remove(file_path)
                        deleted += 1

                except Exception:
                    pass

            if deleted > 0:
                logger.info(f"🧹 Auto-cleanup: deleted {deleted} files")

            # Remove stream entries older than the configured retention window.
            cleanup_old_stream_entries()

        except Exception as exc:
            logger.error(f"Cleanup error: {exc}")

        time.sleep(interval_seconds)


def stream_cleanup_daemon(interval_seconds: int = 60) -> None:
    """
    Periodically trim old stream entries based on retention window.

    Args:
        interval_seconds: Sleep between XTRIM cycles (default 60 s).
    """
    logger.info(f"🧹 Stream cleanup daemon started: interval={interval_seconds}s")

    while True:
        try:
            # Remove stream entries older than the configured retention window.
            cleanup_old_stream_entries()

        except Exception as exc:
            logger.error(f"Stream cleanup error: {exc}")

        time.sleep(interval_seconds)
