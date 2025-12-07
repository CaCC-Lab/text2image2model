"""
External API clients for cloud-based generation services.

Image Generation APIs:
- DALL-E 3 (OpenAI): https://platform.openai.com/docs/guides/images
- Gemini (Google): https://ai.google.dev/gemini-api/docs/image-generation

3D Generation APIs:
- Tripo3D: https://platform.tripo3d.ai/docs/generation
"""
import base64
import io
import logging
import tempfile
import time
from typing import Optional, Tuple

import requests
from PIL import Image

from .config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Image Generation APIs
# =============================================================================

class DalleAPIClient:
    """
    Client for OpenAI DALL-E API (image generation)

    API Reference: https://platform.openai.com/docs/guides/images

    Models:
    - dall-e-3: 1024x1024, 1792x1024, 1024x1792
    - dall-e-2: 256x256, 512x512, 1024x1024

    Parameters:
    - quality: "standard" or "hd"
    - style: "vivid" or "natural"
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid"
    ) -> Image.Image:
        """
        Generate image from text prompt using DALL-E 3.

        Args:
            prompt: Text prompt
            size: "1024x1024", "1792x1024", "1024x1792"
            quality: "standard" or "hd"
            style: "vivid" or "natural"

        Returns:
            PIL Image
        """
        logger.info(f"[DALL-E 3] Generating image: {prompt[:50]}...")

        response = requests.post(
            f"{self.BASE_URL}/images/generations",
            headers=self.headers,
            json={
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": size,
                "quality": quality,
                "style": style,
                "response_format": "b64_json"
            }
        )
        response.raise_for_status()

        data = response.json()["data"][0]
        image_data = data["b64_json"]
        revised_prompt = data.get("revised_prompt", prompt)

        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        logger.info(f"[DALL-E 3] Generated! Revised: {revised_prompt[:50]}...")

        return image


class GeminiAPIClient:
    """
    Client for Google Gemini API (image generation)

    API Reference: https://ai.google.dev/gemini-api/docs/image-generation

    Models:
    - gemini-2.5-flash-image (Nano Banana - fast image generation)
    - gemini-3-pro-image-preview (Nano Banana Pro - high quality)

    Features:
    - Text to Image
    - Image editing
    - Multi-view generation from single image
    """

    # Default model for image generation
    DEFAULT_MODEL = "gemini-2.5-flash-image"
    # High quality model (Nano Banana Pro)
    PRO_MODEL = "gemini-3-pro-image-preview"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Get or create the Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _extract_image_from_response(self, response) -> Optional[Image.Image]:
        """Extract image from Gemini response."""
        if not response.parts:
            return None

        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                # Decode base64 image data
                image_data = part.inline_data.data
                if isinstance(image_data, str):
                    image_data = base64.b64decode(image_data)
                return Image.open(io.BytesIO(image_data))
        return None

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        use_pro_model: bool = False
    ) -> Image.Image:
        """
        Generate image from text prompt using Gemini.

        Args:
            prompt: Text prompt
            aspect_ratio: "1:1", "16:9", "4:3", "3:4", "9:16"
            use_pro_model: Use Nano Banana Pro for higher quality

        Returns:
            PIL Image
        """
        from google.genai import types

        model = self.PRO_MODEL if use_pro_model else self.DEFAULT_MODEL
        logger.info(f"[Gemini] Generating image with {model}: {prompt[:50]}...")

        client = self._get_client()

        # Configure generation
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            ),
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            image = self._extract_image_from_response(response)
            if image is None:
                raise Exception("No image generated by Gemini")

            logger.info("[Gemini] Image generated!")
            return image

        except Exception as e:
            logger.error(f"[Gemini] Error: {e}")
            raise

    def generate_multiview(
        self,
        image: Image.Image,
        views: list[str] = None
    ) -> dict[str, Image.Image]:
        """
        Generate multiple views from a single image using Gemini.

        This uses Gemini's image editing capability to generate consistent
        multi-angle views of the same subject for 3D reconstruction.

        Args:
            image: Input PIL Image (front view)
            views: List of views to generate. Default: ["front", "left", "back"]

        Returns:
            Dictionary mapping view names to PIL Images
            Example: {"front": <Image>, "left": <Image>, "back": <Image>}
        """
        from google.genai import types

        if views is None:
            views = ["front", "left", "back"]

        logger.info(f"[Gemini MultiView] Generating views: {views}")

        client = self._get_client()
        result_images = {}

        # The input image is assumed to be the front view
        result_images["front"] = image

        # Convert input image to bytes for the SDK
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # Generate other views
        for view in views:
            if view == "front":
                continue

            view_prompt = self._get_view_prompt(view)
            logger.info(f"[Gemini MultiView] Generating {view} view...")

            try:
                # Create image part using SDK types
                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png"
                )

                # Configure for image output
                config = types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                )

                response = client.models.generate_content(
                    model=self.DEFAULT_MODEL,
                    contents=[image_part, view_prompt],
                    config=config,
                )

                generated_image = self._extract_image_from_response(response)
                if generated_image:
                    result_images[view] = generated_image
                    # Log image hash for debugging
                    import hashlib
                    img_hash = hashlib.md5(generated_image.tobytes()).hexdigest()[:8]
                    logger.info(f"[Gemini MultiView] {view} view generated! (hash: {img_hash})")
                else:
                    logger.warning(f"[Gemini MultiView] No image returned for {view} view")

            except Exception as e:
                logger.error(f"[Gemini MultiView] Failed to generate {view} view: {e}")

        # Debug: check if any views are identical
        import hashlib
        hashes = {}
        for view, img in result_images.items():
            h = hashlib.md5(img.tobytes()).hexdigest()
            hashes[view] = h
        logger.info(f"[Gemini MultiView] Image hashes: {hashes}")

        # Check for duplicates
        hash_values = list(hashes.values())
        if len(hash_values) != len(set(hash_values)):
            logger.warning("[Gemini MultiView] WARNING: Some views have identical images!")

        return result_images

    def _get_view_prompt(self, view: str) -> str:
        """Get the prompt for generating a specific view."""
        prompts = {
            "left": (
                "Show this exact same subject from the LEFT SIDE (90° counter-clockwise rotation). "
                "The camera is positioned to the LEFT of the subject, capturing the LEFT PROFILE. "
                "If this is a character/robot, show its LEFT arm, LEFT shoulder, LEFT side of face. "
                "Maintain exact same subject, colors, lighting, style. White/neutral background."
            ),
            "right": (
                "Show this exact same subject from the RIGHT SIDE (90° clockwise rotation). "
                "The camera is positioned to the RIGHT of the subject, capturing the RIGHT PROFILE. "
                "If this is a character/robot, show its RIGHT arm, RIGHT shoulder, RIGHT side of face. "
                "Maintain exact same subject, colors, lighting, style. White/neutral background."
            ),
            "back": (
                "Show this exact same subject from the BACK (180° rotation). "
                "The camera is positioned BEHIND the subject, showing the rear view. "
                "If this is a character/robot, show the back of its head, back panel, rear details. "
                "Maintain exact same subject, colors, lighting, style. White/neutral background."
            ),
            "top": (
                "Generate the TOP VIEW of the same subject shown in this image. "
                "View the subject from directly above. "
                "Keep the subject, lighting, style, and background consistent."
            ),
            "bottom": (
                "Generate the BOTTOM VIEW of the same subject shown in this image. "
                "View the subject from directly below. "
                "Keep the subject, lighting, style, and background consistent."
            ),
        }
        return prompts.get(view, f"Generate the {view} view of this subject.")


# =============================================================================
# 3D Generation APIs
# =============================================================================

class TripoAPIClient:
    """
    Client for Tripo3D API (image to 3D)

    API Reference: https://platform.tripo3d.ai/docs/generation
    Python SDK: https://github.com/VAST-AI-Research/tripo-python-sdk

    Parameters:
    - model_version: "v2.0-20240919"
    - face_limit: Max polygon count
    - texture_alignment: "original_image" or "geometry"
    """

    BASE_URL = "https://api.tripo3d.ai/v2/openapi"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
        }

    def _upload_image(self, image: Image.Image) -> str:
        """Upload image and return file token."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        files = {"file": ("image.png", buf, "image/png")}
        response = requests.post(
            f"{self.BASE_URL}/upload",
            headers=self.headers,
            files=files
        )
        response.raise_for_status()
        return response.json()["data"]["image_token"]

    def image_to_3d(
        self,
        image: Image.Image,
        face_limit: int = 100000,
        texture_alignment: str = "original_image"
    ) -> Tuple[str, str]:
        """
        Generate 3D model from image.

        Args:
            image: PIL Image (RGBA recommended)
            face_limit: Max polygon count
            texture_alignment: "original_image" or "geometry"

        Returns:
            Tuple of (obj_path, glb_path)
        """
        logger.info("[Tripo API] Uploading image...")

        file_token = self._upload_image(image)
        logger.info(f"[Tripo API] Uploaded, token: {file_token[:20]}...")

        # Create task
        task_payload = {
            "type": "image_to_model",
            "file": {"type": "image", "file_token": file_token},
            "model_version": "v2.0-20240919",
            "face_limit": face_limit,
            "texture_alignment": texture_alignment,
        }

        logger.info("[Tripo API] Creating task...")
        task_response = requests.post(
            f"{self.BASE_URL}/task",
            headers={**self.headers, "Content-Type": "application/json"},
            json=task_payload
        )

        # Log detailed error info before raising
        if task_response.status_code != 200:
            logger.error(f"[Tripo API] Error {task_response.status_code}: {task_response.text}")
            # Check common issues
            if task_response.status_code == 403:
                error_detail = task_response.json() if task_response.text else {}
                error_msg = error_detail.get("message", error_detail.get("error", "Forbidden"))
                raise Exception(
                    f"Tripo API access denied (403): {error_msg}. "
                    "Please check: 1) API key is valid (should start with 'tsk_'), "
                    "2) Account has sufficient credits, 3) API key has correct permissions."
                )

        task_response.raise_for_status()
        task_id = task_response.json()["data"]["task_id"]
        logger.info(f"[Tripo API] Task: {task_id}")

        # Poll for completion
        start_time = time.time()
        while time.time() - start_time < 600:
            status_response = requests.get(
                f"{self.BASE_URL}/task/{task_id}",
                headers=self.headers
            )
            status_response.raise_for_status()
            status_data = status_response.json()["data"]

            status = status_data.get("status", "unknown")
            progress = status_data.get("progress", 0)

            if status == "success":
                logger.info("[Tripo API] Complete!")
                break
            elif status == "failed":
                raise Exception(f"Tripo failed: {status_data.get('message')}")

            logger.info(f"[Tripo API] {status} ({progress}%)")
            time.sleep(5)
        else:
            raise Exception("Tripo API timeout")

        # Download - Get GLB URL from response
        output = status_data.get("output", {})

        # Current API format: output.pbr_model is the direct GLB URL
        glb_url = output.get("pbr_model")

        # Fallback: check result.pbr_model.url (alternative format)
        if not glb_url:
            result = status_data.get("result", {})
            pbr_model = result.get("pbr_model", {})
            if isinstance(pbr_model, dict):
                glb_url = pbr_model.get("url")
            elif isinstance(pbr_model, str):
                glb_url = pbr_model

        # Legacy fallback: output.model.url
        if not glb_url:
            model_data = output.get("model", {})
            glb_url = model_data.get("url") if isinstance(model_data, dict) else model_data

        if not glb_url:
            logger.error(f"[Tripo API] Could not find model URL in response: {status_data}")
            raise Exception(f"No model URL in response. Output keys: {list(output.keys()) if output else 'empty'}")

        logger.info(f"[Tripo API] Found GLB URL: {glb_url[:80]}...")

        glb_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        glb_response = requests.get(glb_url)
        glb_response.raise_for_status()
        glb_file.write(glb_response.content)
        glb_file.close()

        obj_file = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        obj_file.write(glb_response.content)
        obj_file.close()

        logger.info(f"[Tripo API] Downloaded: {len(glb_response.content)} bytes")
        return obj_file.name, glb_file.name


class HunyuanAPIClient:
    """
    Client for Tencent Cloud Hunyuan 3D API

    API Reference: https://www.tencentcloud.com/products/ai3d
    SDK: pip install tencentcloud-sdk-python-hunyuan

    Requires:
    - SecretID: Tencent Cloud API access key ID
    - SecretKey: Tencent Cloud API secret access key
    """

    def __init__(self, secret_id: str, secret_key: str):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self._client = None

    def _get_client(self):
        """Lazy load Tencent Cloud client."""
        if self._client is None:
            try:
                from tencentcloud.common import credential
                from tencentcloud.hunyuan.v20230901 import hunyuan_client
                cred = credential.Credential(self.secret_id, self.secret_key)
                self._client = hunyuan_client.HunyuanClient(cred, "ap-guangzhou")
            except ImportError:
                raise ImportError(
                    "tencentcloud-sdk-python-hunyuan not installed. "
                    "Run: pip install tencentcloud-sdk-python-hunyuan"
                )
        return self._client

    def image_to_3d(
        self,
        image: Image.Image,
        quality: str = "balanced"
    ) -> Tuple[str, str]:
        """
        Generate 3D model from image using Hunyuan Cloud API.

        Args:
            image: PIL Image (RGBA recommended)
            quality: "fast", "balanced", or "high"

        Returns:
            Tuple of (obj_path, glb_path)
        """
        logger.info("[Hunyuan API] Starting cloud 3D generation...")

        # Convert image to base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            from tencentcloud.hunyuan.v20230901 import models as hunyuan_models

            client = self._get_client()

            # Create request
            req = hunyuan_models.Submit3DModelGenerationRequest()
            req.ImageBase64 = image_b64

            logger.info("[Hunyuan API] Submitting task...")
            resp = client.Submit3DModelGeneration(req)
            task_id = resp.TaskId

            logger.info(f"[Hunyuan API] Task ID: {task_id}")

            # Poll for completion
            start_time = time.time()
            while time.time() - start_time < 1800:  # 30 min timeout
                query_req = hunyuan_models.Query3DModelGenerationRequest()
                query_req.TaskId = task_id
                query_resp = client.Query3DModelGeneration(query_req)

                status = query_resp.Status
                if status == "SUCCESS":
                    logger.info("[Hunyuan API] Generation complete!")
                    model_url = query_resp.ModelUrl
                    break
                elif status == "FAILED":
                    raise Exception(f"Hunyuan API failed: {query_resp.Message}")

                logger.info(f"[Hunyuan API] Status: {status}")
                time.sleep(10)
            else:
                raise Exception("Hunyuan API timeout")

            # Download model
            glb_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
            model_response = requests.get(model_url)
            model_response.raise_for_status()
            glb_file.write(model_response.content)
            glb_file.close()

            obj_file = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
            obj_file.write(model_response.content)
            obj_file.close()

            logger.info(f"[Hunyuan API] Downloaded: {len(model_response.content)} bytes")
            return obj_file.name, glb_file.name

        except ImportError as e:
            raise e
        except Exception as e:
            logger.error(f"[Hunyuan API] Error: {e}")
            raise


# =============================================================================
# Factory Functions
# =============================================================================

def get_dalle_client() -> Optional[DalleAPIClient]:
    """Get DALL-E client if configured."""
    if settings.openai_api_key:
        return DalleAPIClient(settings.openai_api_key)
    return None


def get_gemini_client() -> Optional[GeminiAPIClient]:
    """Get Gemini client if configured."""
    if settings.gemini_api_key:
        return GeminiAPIClient(settings.gemini_api_key)
    return None


def get_tripo_client() -> Optional[TripoAPIClient]:
    """Get Tripo client if configured."""
    if settings.tripo_api_key:
        return TripoAPIClient(settings.tripo_api_key)
    return None


def get_hunyuan_client() -> Optional[HunyuanAPIClient]:
    """Get Hunyuan API client if configured."""
    if settings.hunyuan_secret_id and settings.hunyuan_secret_key:
        return HunyuanAPIClient(settings.hunyuan_secret_id, settings.hunyuan_secret_key)
    return None
