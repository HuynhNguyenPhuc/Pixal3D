"""Configuration for Pixal3D API and worker processes."""

import os
import warnings

# Suppress Python warnings (FutureWarning, DeprecationWarning, UserWarning, etc.)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_SAVE_DIR = "gradio_cache"
SAVE_DIR: str = DEFAULT_SAVE_DIR
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "8388608"))  # 8 MB


# ── GCS ───────────────────────────────────────────────────────────────────────
GOOGLE_SERVICE_ACCOUNT = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GCS_BUCKET = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")
GCS_DESTINATION_PREFIX = "stickerPBR/{folder_hash}/"


# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
REDIS_HEALTH_CHECK_INTERVAL = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
REDIS_KEEPALIVE_INTERVAL_SECONDS = int(os.getenv("REDIS_KEEPALIVE_INTERVAL_SECONDS", "60"))
REDIS_RETRY_BACKOFF_MAX_SECONDS = float(os.getenv("REDIS_RETRY_BACKOFF_MAX_SECONDS", "30"))
REQUIRE_REDIS = os.getenv("REQUIRE_REDIS", "false").lower() == "true"
STATUS_TTL = int(os.getenv("STATUS_TTL", "7200"))
STATUS_DELETE_ON_READ = os.getenv("STATUS_DELETE_ON_READ", "false").lower() in {"1", "true", "yes"}
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "100"))


# ── Execution timeouts ────────────────────────────────────────────────────────
EXECUTION_TIMEOUT_SECONDS = int(
    os.getenv("GENERATION_TIMEOUT_SECONDS", os.getenv("GENERATION_TIMEOUT", "1800"))
)
SHUTDOWN_GRACE_PERIOD = int(os.getenv("SHUTDOWN_GRACE_PERIOD", "90"))


# ── Watchdog / process supervision ───────────────────────────────────────────
WORKER_WATCHDOG_INTERVAL_SECONDS = int(os.getenv("WORKER_WATCHDOG_INTERVAL_SECONDS", "5"))
WORKER_RESTART_WINDOW_SECONDS = int(os.getenv("WORKER_RESTART_WINDOW_SECONDS", "300"))
WORKER_MAX_RESTARTS_PER_WINDOW = int(os.getenv("WORKER_MAX_RESTARTS_PER_WINDOW", "3"))
WORKER_RESTART_BACKOFF_SECONDS = int(os.getenv("WORKER_RESTART_BACKOFF_SECONDS", "30"))
EXECUTION_TIMEOUT_GRACE_SECONDS = int(os.getenv("EXECUTION_TIMEOUT_GRACE_SECONDS", "5"))
RUNNING_TASK_KEY_TTL_SECONDS = int(
    os.getenv("RUNNING_TASK_KEY_TTL_SECONDS", str(max(EXECUTION_TIMEOUT_SECONDS * 3, 300)))
)
RUNNING_HEARTBEAT_SECONDS = int(os.getenv("RUNNING_HEARTBEAT_SECONDS", "5"))
RUNNING_HEARTBEAT_STALE_SECONDS = int(
    os.getenv("RUNNING_HEARTBEAT_STALE_SECONDS", str(max(RUNNING_HEARTBEAT_SECONDS * 3, 15)))
)
EXEC_LOCK_TTL_SECONDS = int(
    os.getenv("EXEC_LOCK_TTL_SECONDS", str(max(EXECUTION_TIMEOUT_SECONDS * 4, 240)))
)


# ── Pending reclaim ───────────────────────────────────────────────────────────
PENDING_RECLAIM_MIN_IDLE_MS = int(
    os.getenv("PENDING_RECLAIM_MIN_IDLE_MS", str(max(EXECUTION_TIMEOUT_SECONDS * 1000, 60000)))
)
PENDING_RECLAIM_BATCH_SIZE = int(os.getenv("PENDING_RECLAIM_BATCH_SIZE", "10"))


# ── Redis Streams ─────────────────────────────────────────────────────────────
STREAM_KEY = "task:queue"
ACTIVE_TASK_HEARTBEAT_KEY = "task:heartbeat"
CONSUMER_GROUP = "pixal3d-workers"
CONSUMER_GROUP_INIT_LOCK_KEY = os.getenv("CONSUMER_GROUP_INIT_LOCK_KEY", f"lock:{STREAM_KEY}:group_init")
CONSUMER_GROUP_INIT_LOCK_TTL_SECONDS = int(os.getenv("CONSUMER_GROUP_INIT_LOCK_TTL_SECONDS", "15"))
STREAM_RETENTION_SECONDS = int(os.getenv("STREAM_RETENTION_SECONDS", str(6 * 3600)))

# ── Queue depth counter ───────────────────────────────────────────────────────
# Independent Redis integer key incremented on enqueue and decremented on ACK.
# Decouples autoscaling signal and backpressure checks from Redis Streams lag
# estimation, which can return null after long idle gaps.
QUEUE_DEPTH_KEY = os.getenv("QUEUE_DEPTH_KEY", "queue:depth")
QUEUE_DEPTH_RECONCILE_INTERVAL_SECONDS = int(os.getenv("QUEUE_DEPTH_RECONCILE_INTERVAL_SECONDS", "60"))


# ── Worker lifecycle ──────────────────────────────────────────────────────────
MAX_TASK_RETRIES = int(os.getenv("MAX_TASK_RETRIES", "3"))
MAX_JOBS_PER_WORKER = int(os.getenv("MAX_JOBS_PER_WORKER", "100"))
MAX_CONSECUTIVE_CUDA_OOM = int(os.getenv("MAX_CONSECUTIVE_CUDA_OOM", "2"))
VRAM_FRAGMENTATION_THRESHOLD = float(os.getenv("VRAM_FRAGMENTATION_THRESHOLD", "0.85"))
