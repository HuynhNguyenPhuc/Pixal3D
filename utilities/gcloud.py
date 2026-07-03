"""Cloud storage helper utilities."""

import io
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from PIL import Image
from google.cloud.storage import Client, transfer_manager

from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def parse_gcs_url(gs_url: str) -> tuple[str, str]:
    """Parse a canonical ``gs://`` URL.

    Args:
        gs_url (str): Canonical GCS URL.

    Returns:
        tuple[str, str]: Bucket name and blob path.
    """
    # Reject non-canonical schemes early so callers do not accidentally pass
    # HTTP URLs or local paths into logic that assumes object storage semantics.
    if not gs_url.startswith("gs://"):
        raise ValueError("Invalid GCS URL format. URL must start with 'gs://'.")

    parsed = urlparse(gs_url)
    bucket_name = parsed.netloc
    blob_name = unquote(parsed.path.lstrip("/"))

    if not bucket_name:
        raise ValueError("Missing bucket name in GCS URL.")
    if not blob_name:
        raise ValueError("Missing object path in GCS URL.")

    return bucket_name, blob_name


def convert_to_gcs_url(bucket_name: str, blob_name: str) -> str:
    """Build a canonical ``gs://`` URL.

    Args:
        bucket_name (str): Bucket name.
        blob_name (str): Object path inside the bucket.

    Returns:
        str: Canonical ``gs://`` URL.
    """
    # Normalize both pieces so storage URLs produced by different call sites
    # still resolve to one canonical form.
    bucket_name = bucket_name.strip("/")
    blob_name = quote(blob_name.strip("/"), safe="/")
    return f"gs://{bucket_name}/{blob_name}"


def convert_pil_image_to_bytes(image: Image.Image, mime_type: str = "image/jpeg") -> bytes:
    """Convert a PIL image to bytes.

    Args:
        image (Image.Image): PIL image instance.
        mime_type (str): Output MIME type.

    Returns:
        bytes: Encoded image bytes.
    """
    img_format = mime_type.split("/")[1].upper()
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=img_format)
    return image_bytes.getvalue()


def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str) -> None:
    """Upload a local file to GCS.

    Args:
        bucket_name (str): Target bucket name.
        source_file_name (str): Local file path.
        destination_blob_name (str): Destination blob path.
    """
    # Resolve the bucket and blob on demand so uploads always use the current
    # environment credentials rather than any cached global client state.
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    logger.info(
        f"Uploaded {source_file_name} to {destination_blob_name} in bucket {bucket_name}"
    )


def upload_directory(
    bucket_name: str,
    source_directory: str,
    destination_prefix: str = "",
    workers: int = 8,
) -> None:
    """Upload every file in a directory to GCS.

    Args:
        bucket_name (str): Target bucket name.
        source_directory (str): Local directory to upload.
        destination_prefix (str): Prefix applied to each uploaded object.
        workers (int): Max worker count for the transfer manager.
    """
    # Enumerate files first so failures are reported against deterministic
    # relative paths instead of opaque transfer-manager task handles.
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    directory_path = Path(source_directory)
    file_paths = [path for path in directory_path.rglob("*") if path.is_file()]
    relative_paths = [str(path.relative_to(directory_path)) for path in file_paths]

    results = transfer_manager.upload_many_from_filenames(
        bucket,
        relative_paths,
        source_directory=source_directory,
        blob_name_prefix=destination_prefix,
        max_workers=workers,
    )

    for name, result in zip(relative_paths, results):
        if isinstance(result, Exception):
            raise RuntimeError(f"Failed to upload {name}: {result}") from result


def download_blob_as_bytes(bucket_name: str, source_blob_name: str) -> bytes:
    """Download a blob from GCS as bytes.

    Args:
        bucket_name (str): Source bucket name.
        source_blob_name (str): Blob path to download.

    Returns:
        bytes: Blob payload.
    """
    # Use a fresh client here for the same reason as uploads: download helpers
    # should follow the active runtime credentials and project selection.
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return blob.download_as_bytes()