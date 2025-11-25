import os
from sys import exit
import pandas as pd
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from train import train_model

def build_video_index(csv_path):
    df = pd.read_csv(csv_path)

    video_to_frames = defaultdict(list)
    video_to_label = {}

    for _, row in df.iterrows():
        img_path = row['image_path']
        label = row['label']

        # Extract video folder name
        # Example: './data/Dataset/fake/178_598/face_0416.jpg'
        parts = img_path.split('/')
        video_id = parts[-2]   # e.g., '178_598'

        # Store frame
        video_to_frames[video_id].append(img_path)

        # Store label once per video
        video_to_label[video_id] = label

    # Sort frames numerically by frame index
    for vid in video_to_frames:
        video_to_frames[vid].sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    return video_to_frames, video_to_label

# Example usage:
#train_video_index, train_labels = build_video_index(
#    "./data/Dataset/labels/train_labels.csv"
#)
def evaluate(model, loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

class DeepfakeSequenceDataset(Dataset):
    def __init__(self, video_to_frames, video_to_label, transform=None, seq_len=16):
        self.video_ids = list(video_to_frames.keys())
        self.video_to_frames = video_to_frames
        self.video_to_label = video_to_label
        self.transform = transform
        self.seq_len = seq_len

    def sample_or_pad(self, frames):
        """Return exactly seq_len frames."""
        n = len(frames)

        if n == self.seq_len:
            return frames

        if n > self.seq_len:
            # Evenly pick seq_len frames
            idxs = np.linspace(0, n-1, self.seq_len).astype(int)
            return [frames[i] for i in idxs]

        # n < seq_len → pad last frame
        last = frames[-1]
        padded = frames + [last] * (self.seq_len - n)
        return padded

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frame_paths = self.video_to_frames[video_id]
        label = self.video_to_label[video_id]

        # Get exactly 16 frames
        selected_frames = self.sample_or_pad(frame_paths)

        # Load images
        imgs = []
        for p in selected_frames:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs)   # (16, 3, 224, 224)

        return imgs, torch.tensor(label, dtype=torch.long)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Build indices
train_video_index, train_labels = build_video_index("./data/Dataset/labels/train_labels.csv")
val_video_index, val_labels = build_video_index("./data/Dataset/labels/val_labels.csv")
test_video_index, test_labels = build_video_index("./data/Dataset/labels/test_labels.csv")

# Create datasets
train_ds = DeepfakeSequenceDataset(train_video_index, train_labels, transform)
val_ds   = DeepfakeSequenceDataset(val_video_index, val_labels, transform)
test_ds  = DeepfakeSequenceDataset(test_video_index, test_labels, transform)

# DataLoaders
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4)



device = "cuda" if torch.cuda.is_available() else exit()

model = train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    lr=1e-4,
    device=device
)

evaluate(model, test_loader, device)
