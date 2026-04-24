"""
Inference script for IIVP 2026 digit classification.
Loads the trained EfficientNet-B0 model and generates submission.csv.

Usage:
    python predict.py
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_CSV = os.path.join("iivp-2026-challenge", "test.csv")
TEST_DIR = os.path.join("iivp-2026-challenge", "test", "test")
MODEL_PATH = "best_model.pth"
SUBMISSION_PATH = "iivp-2026-challenge/sample_submission.csv"

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 10

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
PIN_MEMORY = DEVICE.type == "cuda"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class TestDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["Id"]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_id


def main():
    test_dataset = TestDataset(TEST_CSV, TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=PIN_MEMORY)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_ids.extend(ids.tolist() if isinstance(ids, torch.Tensor) else ids)
            all_preds.extend(preds)

    # -----------------------------------------------------------------------
    # Submission
    # -----------------------------------------------------------------------
    submission = pd.DataFrame({"Id": all_ids, "Category": all_preds})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {SUBMISSION_PATH} ({len(submission)} rows)")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()