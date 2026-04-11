"""
Gemini Vision Service — Falantir v2.1

Uses Gemini 2.5 Flash with structured JSON output to perform ALL vision tasks
in a single API call per frame:

  - Scene description (one sentence)
  - Object detection with bounding boxes (replaces YOLO)
  - Threat classification (safe / suspicious / critical)
  - Confidence + probability breakdown
  - Reasoning (why this threat level was chosen)

This module is called by the VisionProvider abstraction — it does not
implement fallback itself; that happens one layer up in vision_provider.py.
"""

import os
import base64
import json
import time
import cv2
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
REQUEST_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT_S", "15"))
JPEG_QUALITY = int(os.getenv("GEMINI_JPEG_QUALITY", "70"))

# Singleton client — initialized on first use
_client = None
_client_model_name = None


# ─── Structured Output Schema ─────────────────────────────
# Gemini will return JSON matching this schema. This replaces having
# to parse free-form text and removes any ambiguity.

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_description": {
            "type": "string",
            "description": "One clear sentence (max 40 words) describing what is happening in the frame.",
        },
        "threat_label": {
            "type": "string",
            "enum": ["safe", "suspicious", "critical"],
            "description": "Overall threat classification.",
        },
        "threat_level": {
            "type": "integer",
            "description": "0=safe, 1=suspicious, 2=critical",
        },
        "confidence": {
            "type": "number",
            "description": "Overall confidence in the threat_label, 0.0 to 1.0",
        },
        "probabilities": {
            "type": "object",
            "properties": {
                "safe": {"type": "number"},
                "suspicious": {"type": "number"},
                "critical": {"type": "number"},
            },
            "required": ["safe", "suspicious", "critical"],
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence explaining WHY this threat level was chosen.",
        },
        "detected_objects": {
            "type": "array",
            "description": "People and notable objects visible in the frame.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Object name, e.g. 'person', 'bag', 'bottle', 'phone'",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Detection confidence 0.0 to 1.0",
                    },
                    "bbox": {
                        "type": "array",
                        "description": "Bounding box [ymin, xmin, ymax, xmax], normalized 0-1000",
                        "items": {"type": "integer"},
                    },
                    "action": {
                        "type": "string",
                        "description": "What this object/person is doing, if notable. Empty string if not relevant.",
                    },
                },
                "required": ["label", "confidence", "bbox", "action"],
            },
        },
    },
    "required": [
        "scene_description",
        "threat_label",
        "threat_level",
        "confidence",
        "probabilities",
        "reasoning",
        "detected_objects",
    ],
}


SYSTEM_PROMPT = (
    "You are a professional security camera AI analyst. Analyze this surveillance "
    "frame and return structured JSON.\n\n"
    "Instructions:\n"
    "1. scene_description — ONE sentence, max 40 words, describing what is happening.\n"
    "2. detected_objects — list people and notable items (bags, phones, merchandise, weapons). "
    "Use bbox format [ymin, xmin, ymax, xmax] with integer values 0-1000 (normalized). "
    "For each object, the 'action' field describes notable behavior (e.g. 'concealing item in bag', "
    "'reaching for shelf', 'standing idle'). Use empty string if nothing notable.\n"
    "3. threat_label — pick ONE:\n"
    "   - 'safe': normal activity, browsing, paying, walking.\n"
    "   - 'suspicious': concealing merchandise, furtive glances, lingering near exits, "
    "tampering with packaging, acting erratically.\n"
    "   - 'critical': direct theft, violent behavior, weapon visible, physical confrontation, "
    "or emergency (fire, medical).\n"
    "4. threat_level — 0 for safe, 1 for suspicious, 2 for critical.\n"
    "5. confidence — how sure you are (0.0 to 1.0).\n"
    "6. probabilities — {safe, suspicious, critical} summing to ~1.0.\n"
    "7. reasoning — ONE sentence explaining WHY you chose this threat level. "
    "Be specific about what you observed.\n\n"
    "Be conservative: only flag 'critical' if you are very confident. "
    "If the frame is empty or shows nothing unusual, return 'safe' with high confidence."
)


# ─── Client Initialization ────────────────────────────────

def _get_client(model_name=None):
    """Lazy-initialize the Gemini client. Returns None if no API key."""
    global _client, _client_model_name

    target_model = model_name or DEFAULT_MODEL

    if _client is not None and _client_model_name == target_model:
        return _client

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI: No API key configured — vision analysis disabled")
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _client = genai.GenerativeModel(target_model)
        _client_model_name = target_model
        print(f"GEMINI: Initialized with {target_model}")
        return _client
    except Exception as e:
        print(f"GEMINI: Init failed — {e}")
        return None


# ─── Core Analysis ────────────────────────────────────────

def _empty_result(reason="analysis unavailable"):
    """Fallback shape returned when Gemini is unreachable."""
    return {
        "scene_description": f"Scene analysis unavailable ({reason}).",
        "threat_label": "safe",
        "threat_level": 0,
        "confidence": 0.0,
        "probabilities": {"safe": 1.0, "suspicious": 0.0, "critical": 0.0},
        "reasoning": reason,
        "detected_objects": [],
        "model": DEFAULT_MODEL,
        "error": reason,
    }


def analyze_frame(frame_bgr):
    """
    Send a BGR frame to Gemini 2.5 Flash and get the full vision analysis
    as structured JSON.

    Returns a dict with:
      scene_description, threat_label, threat_level, confidence,
      probabilities, reasoning, detected_objects, model, inference_time_ms

    On any error, returns a safe fallback dict (never raises).
    """
    start = time.time()
    client = _get_client()
    if client is None:
        return _empty_result("no API key")

    try:
        # Encode frame → JPEG → base64
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return _empty_result("frame encoding failed")
        b64_image = base64.b64encode(buf).decode("utf-8")

        import google.generativeai as genai

        response = client.generate_content(
            [
                SYSTEM_PROMPT,
                {"mime_type": "image/jpeg", "data": b64_image},
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.2,
            ),
            request_options={"timeout": REQUEST_TIMEOUT_S},
        )

        if not response or not response.text:
            return _empty_result("empty response")

        data = json.loads(response.text)

        # Normalize / sanitize — Gemini may return partial fields
        threat_label = data.get("threat_label", "safe")
        if threat_label not in ("safe", "suspicious", "critical"):
            threat_label = "safe"

        threat_level = int(data.get("threat_level", 0))
        if threat_level not in (0, 1, 2):
            threat_level = {"safe": 0, "suspicious": 1, "critical": 2}[threat_label]

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        probs = data.get("probabilities") or {}
        probabilities = {
            "safe": float(probs.get("safe", 0.0)),
            "suspicious": float(probs.get("suspicious", 0.0)),
            "critical": float(probs.get("critical", 0.0)),
        }
        # Renormalize if Gemini returned values that don't sum to 1
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: round(v / total, 4) for k, v in probabilities.items()}
        else:
            probabilities = {"safe": 1.0, "suspicious": 0.0, "critical": 0.0}

        detected_objects = []
        for obj in data.get("detected_objects") or []:
            try:
                bbox = obj.get("bbox") or [0, 0, 0, 0]
                bbox = [int(v) for v in bbox][:4]
                while len(bbox) < 4:
                    bbox.append(0)
                detected_objects.append({
                    "label": str(obj.get("label", "object"))[:40],
                    "confidence": round(float(obj.get("confidence", 0.0)), 4),
                    "bbox": bbox,  # [ymin, xmin, ymax, xmax] normalized 0-1000
                    "action": str(obj.get("action", ""))[:120],
                })
            except (TypeError, ValueError):
                continue

        elapsed_ms = round((time.time() - start) * 1000, 1)

        return {
            "scene_description": str(data.get("scene_description", "Scene described.")).strip()[:300],
            "threat_label": threat_label,
            "threat_level": threat_level,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "reasoning": str(data.get("reasoning", "")).strip()[:300],
            "detected_objects": detected_objects,
            "model": _client_model_name or DEFAULT_MODEL,
            "inference_time_ms": elapsed_ms,
            "error": None,
        }

    except json.JSONDecodeError as e:
        print(f"GEMINI: JSON parse failed — {e}")
        return _empty_result(f"json parse error: {str(e)[:60]}")
    except Exception as e:
        print(f"GEMINI: Analysis failed — {e}")
        return _empty_result(f"api error: {str(e)[:60]}")


# ─── Status ───────────────────────────────────────────────

def get_status():
    """Return Gemini service status for the /models endpoint."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    return {
        "model": _client_model_name or DEFAULT_MODEL,
        "configured": bool(api_key),
        "initialized": _client is not None,
        "capabilities": [
            "scene_description",
            "object_detection",
            "threat_classification",
            "reasoning",
        ],
    }


# ─── Backward-compat shim ─────────────────────────────────
# Older code imported describe_frame() — keep it working, but route it
# through the new analyze_frame() and return the old shape.

def describe_frame(frame_bgr):
    """DEPRECATED — use analyze_frame(). Kept for backward compatibility."""
    result = analyze_frame(frame_bgr)
    return {
        "description": result.get("scene_description", ""),
        "model": result.get("model", DEFAULT_MODEL),
    }
