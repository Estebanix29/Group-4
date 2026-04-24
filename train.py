"""
Transfer learning pipeline for IIVP 2026 digit classification.
Uses EfficientNet-B0 pretrained on ImageNet, fine-tuned on 32x32 grayscale digit images.

Usage:
    python train.py
"""

import os
import copy
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_DIR = os.path.join("iivp-2026-challenge", "train", "train")
MODEL_PATH = "best_model.pth"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
NUM_CLASSES = 10
VAL_SPLIT = 0.15
SEED = 42

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
PIN_MEMORY = DEVICE.type == "cuda"   # pin_memory only works on CUDA

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # L → RGB
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def main():
    # -----------------------------------------------------------------------
    # Dataset — ImageFolder reads category from subfolder name (0-9)
    # -----------------------------------------------------------------------
    full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)

    # Sorted folder names '0'..'9' → class_to_idx = {'0':0, ..., '9':9}
    assert list(full_dataset.class_to_idx.keys()) == [str(i) for i in range(10)], \
        "Unexpected class ordering — check folder names."

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Val subset uses val_transform (no augmentation)
    val_subset.dataset = copy.deepcopy(full_dataset)
    val_subset.dataset.transform = val_transform

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=PIN_MEMORY)

    # -----------------------------------------------------------------------
    # Model — EfficientNet-B0 with replaced classifier head
    # -----------------------------------------------------------------------
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    model = model.to(DEVICE)

    print(f"Device: {DEVICE} | Train: {train_size} | Val: {val_size}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]  "):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch:>2}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → New best model saved (val acc: {best_val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()