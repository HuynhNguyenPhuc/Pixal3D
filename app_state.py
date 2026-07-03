"""Application state shared across the main process and worker subprocesses."""

import multiprocessing as mp
import threading
import uuid
from collections import deque
from typing import Optional


# ── Node and Worker IDs ───────────────────────────────────────────────────────
# Node ID for each container instance
node_id: str = str(uuid.uuid4())[:6]    

# Worker ID for the current worker subprocess
worker_id: Optional[str] = None         


# ── Worker runtime ────────────────────────────────────────────────────────────
worker = None
consumer_id: Optional[str] = None


# ── Process management ────────────────────────────────────────────────────────
worker_process: Optional[mp.Process] = None
worker_process_lock = threading.Lock()
worker_initialization_args: Optional[dict] = None
worker_ready: bool = False
worker_ready_event = None


# ── Restart rate limiting ─────────────────────────────────────────────────────
_restart_timestamps: deque = deque()


# ── Run mode ──────────────────────────────────────────────────────────────────
# "all"    — HTTP server + GPU worker subprocess
# "api"    — HTTP server only; no GPU worker subprocess
# "worker" — HTTP health probes only + GPU worker subprocess
run_mode: str = "all"


# ── CUDA health ───────────────────────────────────────────────────────────────
# Flag indicating whether the worker subprocess needs to be restarted due to consecutive CUDA OOM failures.
needs_subprocess_restart: bool = False

# Tracks consecutive CUDA OOM failures across tasks.
consecutive_cuda_oom_count: int = 0
