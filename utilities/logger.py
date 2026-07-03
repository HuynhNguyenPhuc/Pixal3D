"""Logging utilities."""

import logging
import sys


logger = None


def setup_logger(save_dir: str = "gradio_cache") -> None:
    """
    Setup logging configuration for the application.

    Logs are emitted to console only.

    Args:
        save_dir (str): Unused legacy argument kept for compatibility.
    """
    del save_dir

    # Define one formatter for the whole process tree so logs remain consistent
    # across API handlers, worker supervision, and cleanup threads.
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # Configure the root logger instead of per-module handlers so repeated calls
    # to get_logger never create duplicated output streams.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear handlers first because startup hooks and notebook-like execution can
    # call setup more than once inside the same interpreter.
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Keep logging console-only. File handlers are intentionally omitted so the
    # runtime has no local log-file cleanup burden.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party libraries so queue, worker, and watchdog events
    # remain readable under load.
    for lib in ["httpx", "urllib3", "tqdm", "google.api_core", "timm"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for the given module name.

    Args:
        name (str): Logger name, typically ``__name__``.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Lazily bootstrap logging for modules imported in isolation during tests or
    # ad-hoc scripts.
    if not logging.getLogger().handlers:
        setup_logger()

    return logging.getLogger(name)


def build_logger(logger_name: str, logger_filename=None) -> logging.Logger:
    """
    Backward-compatible wrapper around :func:`get_logger`.

    Args:
        logger_name (str): Logger name.
        logger_filename: Unused legacy file path argument.

    Returns:
        logging.Logger: Configured logger instance.
    """
    del logger_filename
    return get_logger(logger_name)


def init_logger(save_dir: str = "gradio_cache") -> None:
    """
    Backward-compatible wrapper for startup logging.

    Args:
        save_dir (str): Unused legacy argument kept for compatibility.
    """
    global logger

    # Preserve the compatibility entrypoint while still routing everything to
    # the shared root logger configuration above.
    setup_logger(save_dir)
    logger = get_logger("controller")
