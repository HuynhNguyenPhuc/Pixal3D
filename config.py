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

# Caps OpenMP/MKL/OpenBLAS/OpenCV thread pools inside the GPU worker subprocess
# to the pod's actual cgroup CPU quota. GKE nodes expose their full physical
# core count via sched_getaffinity/nproc even though the pod's
# `resources.limits.cpu` caps it far lower; libraries that size thread pools
# off the visible core count then oversubscribe the CFS quota, which throttles
# the *entire* container -- including the main process's /health handler --
# for the rest of the accounting period. Default matches the StatefulSet's
# requests.cpu (see k8s/pixal3d-deployment.yaml), leaving the limits.cpu
# headroom for the main API process.
WORKER_CPU_THREAD_LIMIT = int(os.getenv("WORKER_CPU_THREAD_LIMIT", "6"))


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
# A still-running task is protected from reclaim by its Redis heartbeat
# (RUNNING_HEARTBEAT_STALE_SECONDS), not by this value -- this only bounds how
# long a genuinely orphaned entry (e.g. left behind by a hard process crash)
# sits unclaimed before another worker cleans it up. Previously tied to
# EXECUTION_TIMEOUT_SECONDS (30 min default), which left crash orphans
# unclaimed for up to 30 minutes with no added safety benefit. A margin over
# the heartbeat-stale window, with a low floor, gives fast orphan cleanup
# while still comfortably outlasting transient heartbeat-write delays.
PENDING_RECLAIM_MIN_IDLE_MS = int(
    os.getenv("PENDING_RECLAIM_MIN_IDLE_MS", str(max(RUNNING_HEARTBEAT_STALE_SECONDS * 1000 * 4, 180000)))
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

# ── Proactive Mesh Simplification (VRAM Safeguards) ──────────────────────────
# Threshold: If mesh exceeds this number of faces, proactively decimate it on CPU
# first to prevent cuMesh BVH building from crashing the GPU.
# Lowered from 8,000,000: a 9.2M-face mesh reaching mesh export reproducibly
# corrupted the CUDA context (illegal memory access) inside the UV-atlas/BVH
# step. Triggering simplification earlier costs nothing for normal meshes --
# it only changes behavior for meshes that are already this dense.
SIMPLIFICATION_THRESHOLD_FACES = int(os.getenv("SIMPLIFICATION_THRESHOLD_FACES", "5000000"))
# Target: The optimized face density to target when proactively decimating.
# 4,000,000 faces is mathematically the ideal detail cap for a 1536^3 voxel grid.
SIMPLIFICATION_TARGET_FACES = int(os.getenv("SIMPLIFICATION_TARGET_FACES", "4000000"))
# Hard ceiling: no mesh may reach the mesh-export call above this face count,
# regardless of SIMPLIFICATION_THRESHOLD_FACES/TARGET_FACES configuration or
# whether proactive simplification succeeded. This is deliberately NOT purely
# env-driven -- it's clamped so a misconfigured or missing threshold upstream
# can never silently let an oversized, crash-prone mesh through unchecked.
# See worker/model.py's pre-export face-count gate.
MESH_HARD_FACE_CEILING = min(
    int(os.getenv("MESH_HARD_FACE_CEILING", "6000000")),
    9_000_000,
)

