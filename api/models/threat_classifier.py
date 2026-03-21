"""
Custom Threat Classifier — MobileNetV3-Large backbone with dual heads.

Outputs:
  - threat_label: "safe" | "suspicious" | "critical"
  - threat_level: 0 (safe) | 1 (suspicious) | 2 (critical)
  - confidence: float 0–1
  - probabilities: {safe: float, suspicious: float, critical: float}
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from dotenv import load_dotenv

load_dotenv()

THREAT_LABELS = ["safe", "suspicious", "critical"]


class ThreatClassifierModel(nn.Module):
    """MobileNetV3-Large with frozen early layers and two output heads."""

    def __init__(self, num_classes=3):
        super().__init__()

        # Backbone — MobileNetV3-Large pretrained on ImageNet
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features

        # Freeze early layers (first 10 of 16 blocks)
        for i, layer in enumerate(self.features):
            if i < 10:
                for param in layer.parameters():
                    param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head — threat level
        self.classifier = nn.Sequential(
            nn.Linear(960, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Confidence head — single scalar 0–1
        self.confidence_head = nn.Sequential(
            nn.Linear(960, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        logits = self.classifier(pooled)
        confidence = self.confidence_head(pooled)
        return logits, confidence


# ─── Preprocessing ────────────────────────────────────────

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ─── Singleton Loader ─────────────────────────────────────

_model = None
_device = None


def _download_from_huggingface():
    """Download model from HuggingFace Hub if not present locally."""
    model_path = os.getenv("MODEL_PATH", "models/threat_classifier.pt")
    repo = os.getenv("HUGGINGFACE_MODEL_REPO", "")
    token = os.getenv("HUGGINGFACE_TOKEN", "")

    if os.path.exists(model_path):
        return model_path

    if not repo:
        print("THREAT MODEL: No HuggingFace repo configured, skipping download")
        return None

    try:
        from huggingface_hub import hf_hub_download
        print(f"THREAT MODEL: Downloading from {repo}...")
        downloaded = hf_hub_download(
            repo_id=repo,
            filename="threat_classifier.pt",
            token=token or None,
            local_dir=os.path.dirname(model_path) or "models",
        )
        print(f"THREAT MODEL: Downloaded to {downloaded}")
        return downloaded
    except Exception as e:
        print(f"THREAT MODEL: Download failed — {e}")
        return None


def load_model():
    """Load the threat classifier model once. Returns (model, device) or (None, None)."""
    global _model, _device

    if _model is not None:
        return _model, _device

    model_path = os.getenv("MODEL_PATH", "models/threat_classifier.pt")

    # Try downloading if not present
    if not os.path.exists(model_path):
        model_path = _download_from_huggingface()

    if model_path is None or not os.path.exists(model_path):
        print("THREAT MODEL: No model file found — classifier disabled")
        return None, None

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = ThreatClassifierModel(num_classes=3)

    try:
        state = torch.load(model_path, map_location=_device, weights_only=True)
        _model.load_state_dict(state)
    except Exception:
        # Try loading full checkpoint (with optimizer etc.)
        try:
            checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
            if "model_state_dict" in checkpoint:
                _model.load_state_dict(checkpoint["model_state_dict"])
            else:
                _model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"THREAT MODEL: Failed to load weights — {e}")
            _model = None
            return None, None

    _model.to(_device)
    _model.eval()
    print(f"THREAT MODEL: Loaded on {_device}")
    return _model, _device


def classify_frame(frame_bgr):
    """
    Classify a single BGR frame.

    Returns dict with threat_label, threat_level, confidence, probabilities
    or None if model is not loaded.
    """
    model, device = load_model()
    if model is None:
        return None

    import cv2
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, confidence = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        conf = confidence.squeeze().cpu().item()
        pred = int(logits.argmax(dim=1).item())

    return {
        "threat_label": THREAT_LABELS[pred],
        "threat_level": pred,
        "confidence": round(conf, 4),
        "probabilities": {
            "safe": round(float(probs[0]), 4),
            "suspicious": round(float(probs[1]), 4),
            "critical": round(float(probs[2]), 4),
        },
    }
