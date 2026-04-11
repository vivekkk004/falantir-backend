"""
Vision Provider Abstraction — Falantir v2.1

This module wraps the different ways Falantir can analyze a frame, and
provides an automatic fallback chain so the system never goes down.

Provider priority (configurable via VISION_PROVIDER env var):

    1. GeminiProvider        — Gemini 2.5 Flash via AI Studio / Vertex AI
                                (best quality, requires API key / credit)

    2. MobileNetV3Provider   — Your trained student model
                                (free, offline, requires .pt file)

    3. SafeFallbackProvider  — Always returns "safe"
                                (last resort — system stays up)

Usage:
    from api.services.vision_provider import analyze_frame
    result = analyze_frame(frame_bgr)

The result dict always has the same shape regardless of which provider ran,
and includes a `provider_used` field for observability.
"""

import os
import time
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()


# ─── Canonical Result Shape ───────────────────────────────

def _safe_result(reason="no provider available"):
    """The shape every provider must return. Used as a safe default."""
    return {
        "scene_description": "Scene analysis unavailable.",
        "threat_label": "safe",
        "threat_level": 0,
        "confidence": 0.0,
        "probabilities": {"safe": 1.0, "suspicious": 0.0, "critical": 0.0},
        "reasoning": reason,
        "detected_objects": [],
        "model": "none",
        "inference_time_ms": 0.0,
        "provider_used": "safe_fallback",
        "error": reason,
    }


# ─── Base Provider ────────────────────────────────────────

class VisionProvider(ABC):
    """Base class for all vision providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short id for logs and the /models endpoint."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can currently run."""

    @abstractmethod
    def analyze(self, frame_bgr) -> dict:
        """Run inference on a frame. Must return the canonical result dict."""


# ─── Provider 1: Gemini ───────────────────────────────────

class GeminiProvider(VisionProvider):
    """Gemini 2.5 Flash with structured output — primary provider."""

    @property
    def name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY", "").strip())

    def analyze(self, frame_bgr) -> dict:
        from api.services.gemini_service import analyze_frame as gemini_analyze
        result = gemini_analyze(frame_bgr)
        result["provider_used"] = self.name
        return result


# ─── Provider 2: Trained MobileNetV3 ──────────────────────

class MobileNetV3Provider(VisionProvider):
    """
    Custom-trained MobileNetV3 threat classifier — offline fallback.

    Only available once a .pt file exists at MODEL_PATH. Until then,
    is_available() returns False and we skip to the next provider.
    """

    @property
    def name(self) -> str:
        return "mobilenetv3"

    def is_available(self) -> bool:
        model_path = os.getenv("MODEL_PATH", "models/threat_classifier.pt")
        if not os.path.exists(model_path):
            return False
        # Try loading — returns (None, None) if weights are broken
        try:
            from api.models.threat_classifier import load_model
            model, _ = load_model()
            return model is not None
        except Exception:
            return False

    def analyze(self, frame_bgr) -> dict:
        start = time.time()
        try:
            from api.models.threat_classifier import classify_frame
            result = classify_frame(frame_bgr)
            if result is None:
                return _safe_result("mobilenetv3 not loaded")

            elapsed_ms = round((time.time() - start) * 1000, 1)
            return {
                "scene_description": "Local model classification (no description available).",
                "threat_label": result["threat_label"],
                "threat_level": result["threat_level"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "reasoning": f"Classified as {result['threat_label']} by local MobileNetV3.",
                "detected_objects": [],  # MobileNetV3 is classification-only, no bboxes
                "model": "mobilenetv3_large_student",
                "inference_time_ms": elapsed_ms,
                "provider_used": self.name,
                "error": None,
            }
        except Exception as e:
            print(f"MOBILENETV3: Analysis failed — {e}")
            return _safe_result(f"mobilenetv3 error: {str(e)[:60]}")


# ─── Provider 3: Safe Fallback ────────────────────────────

class SafeFallbackProvider(VisionProvider):
    """Always returns 'safe'. Guarantees the system stays up."""

    @property
    def name(self) -> str:
        return "safe_fallback"

    def is_available(self) -> bool:
        return True

    def analyze(self, frame_bgr) -> dict:
        return _safe_result("all upstream providers unavailable")


# ─── Registry & Active Provider Selection ────────────────

_PROVIDERS = {
    "gemini": GeminiProvider(),
    "mobilenetv3": MobileNetV3Provider(),
    "safe_fallback": SafeFallbackProvider(),
}

# Order in which we try providers. First available wins.
# Override via env: VISION_PROVIDER_CHAIN="gemini,mobilenetv3,safe_fallback"
_DEFAULT_CHAIN = ["gemini", "mobilenetv3", "safe_fallback"]


def _get_chain():
    raw = os.getenv("VISION_PROVIDER_CHAIN", "").strip()
    if not raw:
        return _DEFAULT_CHAIN
    chain = [p.strip() for p in raw.split(",") if p.strip() in _PROVIDERS]
    return chain or _DEFAULT_CHAIN


def get_active_provider() -> VisionProvider:
    """Return the first available provider in the chain."""
    for name in _get_chain():
        provider = _PROVIDERS.get(name)
        if provider and provider.is_available():
            return provider
    # Should be unreachable — SafeFallback is always available
    return _PROVIDERS["safe_fallback"]


def analyze_frame(frame_bgr) -> dict:
    """
    Run the active vision provider on a frame.

    This is the main entry point the rest of the app should call.
    """
    provider = get_active_provider()
    try:
        return provider.analyze(frame_bgr)
    except Exception as e:
        print(f"VISION PROVIDER [{provider.name}]: Unhandled error — {e}")
        # Auto-fallback on unhandled crash
        return _PROVIDERS["safe_fallback"].analyze(frame_bgr)


def get_providers_status() -> dict:
    """Return availability status of all providers for /models endpoint."""
    chain = _get_chain()
    active = get_active_provider()
    return {
        "active_provider": active.name,
        "chain": chain,
        "providers": {
            name: {
                "available": provider.is_available(),
                "priority": chain.index(name) if name in chain else -1,
            }
            for name, provider in _PROVIDERS.items()
        },
    }


def warmup():
    """Called at app startup — eagerly initializes the active provider."""
    active = get_active_provider()
    print(f"VISION: Active provider = {active.name}")
    if active.name == "gemini":
        # Eager client init so first request is fast
        from api.services.gemini_service import _get_client
        _get_client()
    elif active.name == "mobilenetv3":
        from api.models.threat_classifier import load_model
        load_model()
    return active
