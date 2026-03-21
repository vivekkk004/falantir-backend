"""
Falantir v2 — Custom Threat Classifier Training Script.

Run on Google Colab with free T4 GPU.
Paste this into a Colab notebook or run as a script.

Dataset structure:
    dataset/
    ├── train/
    │   ├── safe/
    │   ├── suspicious/
    │   └── critical/
    └── val/
        ├── safe/
        ├── suspicious/
        └── critical/

Sources:
    - UCF-Crime dataset (1,900 surveillance videos, 13 anomaly categories)
    - UCSD Anomaly Detection dataset (pedestrian anomaly CCTV footage)
    - Custom recorded footage (retail theft, loitering, suspicious behavior)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# ─── Config ───────────────────────────────────────────────

DATASET_DIR = "dataset"  # Change to your dataset path
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_CLASSES = 3
LABELS = ["safe", "suspicious", "critical"]
SAVE_PATH = "threat_classifier.pt"
BEST_MODEL_PATH = "threat_classifier_best.pt"


# ─── Model Definition ────────────────────────────────────

class ThreatClassifierModel(nn.Module):
    """MobileNetV3-Large with frozen early layers and two output heads."""

    def __init__(self, num_classes=3):
        super().__init__()

        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features

        # Freeze early layers (first 10 of 16 blocks)
        for i, layer in enumerate(self.features):
            if i < 10:
                for param in layer.parameters():
                    param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(960, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Confidence head
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


# ─── Data Transforms ─────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─── Training Loop ────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = ThreatClassifierModel(num_classes=NUM_CLASSES).to(device)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, confidence = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, confidence = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total

        print(
            f"Epoch {epoch + 1}/{EPOCHS} — "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ New best model saved ({val_acc:.4f})")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
            }, f"checkpoint_epoch_{epoch + 1}.pt")

    # Save final model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Final model: {SAVE_PATH}")

    return model


def upload_to_huggingface(model_path=None):
    """Upload trained model to HuggingFace Hub."""
    from huggingface_hub import HfApi

    model_path = model_path or BEST_MODEL_PATH
    repo_id = os.getenv("HUGGINGFACE_MODEL_REPO", "")
    token = os.getenv("HUGGINGFACE_TOKEN", "")

    if not repo_id or not token:
        print("Set HUGGINGFACE_MODEL_REPO and HUGGINGFACE_TOKEN to upload")
        return

    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="threat_classifier.pt",
        repo_id=repo_id,
        token=token,
    )
    print(f"Model uploaded to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    model = train()
    # Uncomment to upload after training:
    # upload_to_huggingface()
