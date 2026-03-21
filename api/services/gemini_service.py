"""
Gemini API integration for scene description.

Uses gemini-1.5-flash to generate human-readable security scene descriptions.
"""

import os
import base64
import cv2
from dotenv import load_dotenv

load_dotenv()

_client = None
_model_name = "gemini-2.0-flash"

SECURITY_PROMPT = (
    "You are a professional security camera analyst. "
    "Analyze this surveillance camera frame and describe what you see in ONE clear sentence. "
    "Focus on: people and their actions, any suspicious or threatening behaviour, "
    "objects being carried or concealed, and the general scene context. "
    "If someone appears to be stealing, concealing merchandise, acting aggressively, "
    "or behaving suspiciously, describe that specifically. "
    "If nothing suspicious is happening, say the scene appears normal. "
    "Keep your response to a single sentence, no more than 40 words."
)


def _get_client():
    """Initialize the Gemini client once."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI: No API key configured — scene descriptions disabled")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _client = genai.GenerativeModel(_model_name)
        print(f"GEMINI: Initialized with {_model_name}")
        return _client
    except Exception as e:
        print(f"GEMINI: Init failed — {e}")
        return None


def describe_frame(frame_bgr):
    """
    Send a BGR frame to Gemini and get a scene description.

    Returns dict with description and model name, or None if unavailable.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        # Encode frame as JPEG base64
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64_image = base64.b64encode(buf).decode("utf-8")

        # Send to Gemini with image
        response = client.generate_content([
            SECURITY_PROMPT,
            {
                "mime_type": "image/jpeg",
                "data": b64_image,
            },
        ])

        description = response.text.strip() if response.text else "Unable to analyze scene."

        return {
            "description": description,
            "model": _model_name,
        }
    except Exception as e:
        print(f"GEMINI: Analysis failed — {e}")
        return {
            "description": f"Scene analysis unavailable: {str(e)[:80]}",
            "model": _model_name,
        }


def get_status():
    """Return Gemini service status."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    return {
        "model": _model_name,
        "configured": bool(api_key),
        "initialized": _client is not None,
    }
