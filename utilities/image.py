"""Image utilities for Pixal3D API server."""

import io
import os
from typing import Any, Union

import httpx
from PIL import Image, UnidentifiedImageError

from utilities.gcloud import download_blob_as_bytes, parse_gcs_url
from utilities.logger import get_logger
from utilities.validation import validate_http_url


# --- Logger --- #
logger = get_logger(__name__)

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/90.0.4430.93 Safari/537.36"
    )
}


def convert_pil_image_to_bytes(image: Image.Image, img_format: str = "JPEG") -> bytes:
    """
    Convert a PIL image to raw bytes.

    Args:
        image (Image.Image): Source image.
        img_format (str): Output encoding format.

    Returns:
        bytes: Encoded image payload.
    """
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=img_format)
    return image_bytes.getvalue()


def convert_bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """
    Convert raw bytes into a PIL image.

    Args:
        image_bytes (bytes): Encoded image payload.

    Returns:
        Image.Image: Parsed image, or ``None`` when parsing fails.
    """
    if not image_bytes:
        logger.warning("Attempted to convert empty bytes to an image.")
        return None

    try:
        return Image.open(io.BytesIO(image_bytes))

    except UnidentifiedImageError:
        logger.warning("Cannot identify image file from raw bytes.")
        return None


def fetch_image_from_url(url: str) -> Image.Image:
    """
    Fetch an image from a URL.

    Args:
        url (str): HTTP or HTTPS URL.

    Returns:
        Image.Image: Parsed image, or ``None`` when download/parsing fails.
    """
    try:
        with httpx.Client() as client:
            # Perform a GET request to fetch the image content.
            response = client.get(url, headers=_HTTP_HEADERS, follow_redirects=True, timeout=30.0)
            
            # Raise an exception for non-200 status codes to trigger retry logic in the caller.
            response.raise_for_status()

            return Image.open(io.BytesIO(response.content))

    except httpx.RequestError as exc:
        logger.warning(f"Error fetching image from URL '{url}': {exc}")
        return None

    except UnidentifiedImageError:
        logger.warning(f"Could not identify image downloaded from URL '{url}'.")
        return None


def convert_to_pil_image(image_source: Any, content_type: str = None) -> Image.Image:
    """
    Convert a supported image source into a PIL image.

    Args:
        image_source (Any): PIL image, bytes, gs:// URL, HTTP(S) URL, or local file path.
        content_type (str): Unused compatibility argument.

    Returns:
        Image.Image: Parsed image, or ``None`` when conversion fails.
    """
    del content_type

    if isinstance(image_source, Image.Image):
        return image_source

    if isinstance(image_source, bytes):
        return convert_bytes_to_pil_image(image_source)

    if isinstance(image_source, str):
        if image_source.startswith("gs://"):
            try:
                bucket_name, blob_name = parse_gcs_url(image_source)
                image_bytes = download_blob_as_bytes(bucket_name, blob_name)
                return convert_bytes_to_pil_image(image_bytes)

            except Exception as exc:
                logger.warning(f"Error downloading image from GCS URL '{image_source}': {exc}")
                return None

        if validate_http_url(image_source):
            return fetch_image_from_url(image_source)

        if os.path.exists(image_source):
            try:
                return Image.open(image_source)

            except IOError as exc:
                logger.warning(f"Error opening local file '{image_source}': {exc}")
                return None

        logger.warning("Unsupported image string source. Use gs://, http(s) URL, or local file path.")
        return None

    logger.warning(f"Unsupported image source type: {type(image_source)}")
    return None


def get_image_size(data: Union[str, bytes]) -> int:
    """
    Get the size of the image data in bytes.

    Args:
        data: Image bytes, gs:// URL, HTTP(S) URL, or local file path.

    Returns:
        int: Size of the image payload in bytes.
    """
    if isinstance(data, bytes):
        return len(data)

    if isinstance(data, str):
        if data.startswith("gs://"):
            # Downloading to measure size is acceptable here because queueing
            # already needs a hard payload-size gate before work enters Redis.
            bucket_name, blob_name = parse_gcs_url(data)
            return len(download_blob_as_bytes(bucket_name, blob_name))

        if validate_http_url(data):
            with httpx.Client() as client:
                # Prefer HEAD first so normal remote objects do not need a full
                # download just to enforce the size limit.
                response = client.head(data, follow_redirects=True, timeout=10.0)
                response.raise_for_status()

                size = response.headers.get("content-length")
                if size:
                    return int(size)

                response = client.get(data, follow_redirects=True, timeout=10.0)
                response.raise_for_status()
                return len(response.content)

        if os.path.exists(data):
            return os.path.getsize(data)

        raise ValueError("Unsupported string image source. Use gs:// or http(s) URL.")

    return 0
