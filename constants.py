"""
Constants and error messages for Pixal3D API server.
"""

# Error messages
SERVER_ERROR_MSG = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
MODERATION_MSG = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

# Default values
DEFAULT_WORKER_ID = None  # Will be generated if None

# API metadata
API_TITLE = "Pixal3D API Server"
API_DESCRIPTION = """
# Pixal3D API Server

This API server provides endpoints for generating 3D models from 2D images using the Pixal3D model.

## Features

- **3D Shape Generation**: Convert 2D images to 3D meshes with PBR materials
- **Multiple Resolutions**: Support for 512, 1024, and 1536 resolution outputs
- **Background Removal**: Automatic background removal from input images
- **GLB Export**: Export with decimation and texture baking
- **Async Processing**: Background task processing with status tracking

## Usage

1. Use `/generate` for immediate 3D model generation from images
2. Use `/send` for asynchronous processing with status tracking
3. Use `/status/{uid}` to check task progress and retrieve results
4. Use `/health` to verify service status

## Model Information

- **Model**: Pixal3D-4B by Microsoft
- **License**: MIT License
- **Capabilities**: Image-to-3D with PBR materials
"""
API_VERSION = "2.0.0"
API_CONTACT = {
    "name": "Pixal3D Team",
    "url": "https://github.com/microsoft/Pixal3D",
}
API_LICENSE_INFO = {
    "name": "MIT License",
    "url": "https://github.com/microsoft/Pixal3D/blob/main/LICENSE",
}

# API tags metadata
API_TAGS_METADATA = [
    {
        "name": "generation",
        "description": "3D model generation endpoints. Generate 3D models from 2D images with PBR materials.",
    },
    {
        "name": "status",
        "description": "Task status and health check endpoints. Monitor generation progress and service health.",
    },
]
