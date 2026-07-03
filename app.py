"""Pixal3D Generation API Server."""

import argparse
import multiprocessing as mp
import os
import sys
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app_state
import config
from broker.client import redis_client
from broker.keepalive import redis_keepalive_daemon
from broker.queue import ensure_consumer_group
from constants import (
    API_CONTACT,
    API_DESCRIPTION,
    API_LICENSE_INFO,
    API_TAGS_METADATA,
    API_TITLE,
    API_VERSION,
)
from routes.generation import router as generation_router
from routes.health import router as health_router
from utilities.cleanup import auto_cleanup_daemon, stream_cleanup_daemon
from utilities.logger import get_logger, setup_logger
from utilities.signal_handlers import register_signal_handlers
from worker.process import start_worker_process
from worker.watchdog import worker_process_watchdog


# --- Logger --- #
# Configure logging once at process startup
setup_logger()

# Keep a main logger for startup and lifecycle messages.
logger = get_logger("main")


# ── FastAPI Application ──────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    license_info=API_LICENSE_INFO,
    openapi_tags=API_TAGS_METADATA,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main() -> None:
    """Main entry point for the Pixal3D Generation API server."""
    parser = argparse.ArgumentParser(description="Pixal3D Generation API Server")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default="TencentARC/Pixal3D")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-path", type=str, default="./gradio_cache")
    # api    — HTTP only (gateway), no GPU worker subprocess
    # worker — HTTP health probes only + GPU worker subprocess
    # all    — HTTP + GPU worker subprocess
    parser.add_argument("--mode", type=str, default="all", choices=["all", "api", "worker"])

    # Parse the command-line arguments.
    args = parser.parse_args()
    logger.info(f"Args: {args}")
    
    # Set the global run mode.
    app_state.run_mode = args.mode


    # ── Routes ───────────────────────────────────────────────────────────────
    # All modes include the health router for Kubernetes probes.
    app.include_router(health_router)

    # If running in API mode or all mode, include the generation router.
    # If running in worker mode, skip registering generation router.
    if args.mode in ["all", "api"]:
        app.include_router(generation_router)


    # ── Multiprocessing ──────────────────────────────────────────────────────
    # Force spawn so the worker subprocess boots with a clean interpreter state
    # instead of inheriting partially initialized CUDA or model state.
    if args.mode in ["all", "worker"]:
        # If running in worker mode, spawn a new process for the worker subprocess.
        try:
            mp.set_start_method("spawn", force=True)

        except RuntimeError:
            pass


    # ── Working directory ────────────────────────────────────────────────────
    if args.mode in ["all", "worker"]:
        # If running in worker mode, ensure the cache directory exists for model caching.
        config.SAVE_DIR = args.cache_path
        os.makedirs(config.SAVE_DIR, exist_ok=True)


    # ── Redis Streams setup ──────────────────────────────────────────────────
    # Ensure the consumer group exists before tasks are pushed.
    # Run in all modes since all pods interact with Redis:
    # - API mode: API pod pushes tasks into Redis Streams.
    # - Worker mode: Worker pod pulls tasks from Redis Streams.
    if redis_client:
        if ensure_consumer_group():
            logger.info(f"✅ Redis consumer group exists: {config.CONSUMER_GROUP}")
        else:
            logger.error("Failed to initialise Redis Streams consumer group")
    else:
        logger.error("❌ Redis unavailable — cannot start task consumer")


    # ── Worker process ───────────────────────────────────────────────────────
    if args.mode in ["all", "worker"]:
        # If running in worker mode, start the GPU worker subprocess for handling generation tasks.
        app_state.worker_initialization_args = {
            "model_path": args.model_path,
            "device": args.device,
            "cache_path": args.cache_path,
        }

        if not start_worker_process():
            logger.error("❌ Failed to start process backend worker")
            sys.exit(1)

        # ── Watchdog thread ──────────────────────────────────────────────────
        # The watchdog supervises the subprocess and restarts it if it crashes. 
        watchdog_thread = threading.Thread(target=worker_process_watchdog, daemon=True)
        watchdog_thread.start()
        logger.warning("✅ Process backend enabled with watchdog")
    else:
        # If running in API mode, skip starting the GPU worker subprocess.
        logger.info("ℹ️  Running in api mode — GPU worker subprocess skipped")


    # ── Redis keepalive thread ───────────────────────────────────────────────
    # The keepalive daemon maintains a continuous connection to Redis and periodically sends PINGs to prevent idle timeouts.
    keepalive_thread = threading.Thread(target=redis_keepalive_daemon, daemon=True)
    keepalive_thread.start()
    logger.info("✅ Redis keepalive daemon started")


    # ── Auto-cleanup thread ──────────────────────────────────────────────────
    if args.mode in ["all", "worker"]:
        # If running in worker mode, start the auto-cleanup daemon for cleaning up old files and trimming Redis Streams.
        cleanup_thread = threading.Thread(
            target=auto_cleanup_daemon,
            args=(300, 3600),  # interval=300s, age_limit=3600s
            daemon=True,
        )

        # Start the cleanup thread.
        cleanup_thread.start()

        logger.info("✅ Auto-cleanup daemon started")
    
    else:
        # If running in API mode, start a lightweight stream-only cleanup daemon.
        cleanup_thread = threading.Thread(
            target=stream_cleanup_daemon,
            args=(60,),  # interval=60s
            daemon=True,
        )

        # Start the stream cleanup thread.
        cleanup_thread.start()

        logger.info("✅ Stream cleanup daemon started")


    # ── Signal handlers  ─────────────────────────────────────────────────────
    # Marks in-flight tasks interrupted and terminates the worker subprocess.
    if args.mode in ["all", "worker"]:
        # If running in worker mode, register signal handlers for graceful shutdown.
        register_signal_handlers()


    # ── Server ───────────────────────────────────────────────────────────────
    logger.info(f"🚀 Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        timeout_keep_alive=3600,
        access_log=False,
    )


if __name__ == "__main__":
    main()
