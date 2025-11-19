import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split



# Path to your main dataset folder
# Example: dataset/real/... and dataset/fake/...
root_dir = "./data/Dataset"
output_csv = "./data/labels/labels.csv"

# Create a CSV file with headers
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])

    # Loop through "real" and "fake" directories
    for label_name in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        # Assign numeric label
        label = 0 if label_name.lower() == "fake" else 1

        # Loop through all video folders inside (e.g. 001_001, 002_045, etc.)
        for video_folder in os.listdir(label_path):
            video_path = os.path.join(label_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            # Loop through face images inside the video folder
            for img_file in os.listdir(video_path):
                if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(video_path, img_file)
                    writer.writerow([img_path, label])

print(f"✅ Label file created: {output_csv}")



# Load labels
df = pd.read_csv("./data/labels/labels.csv")

# Split into train (80%) and test (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Further split train into train (80% of total) and val (10% of total)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

# Save to separate files
train_df.to_csv("./data/labels/train_labels.csv", index=False)
val_df.to_csv("./data/labels/val_labels.csv", index=False)
test_df.to_csv("./data/labels/test_labels.csv", index=False)

print(f"✅ Split complete!")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Testing samples: {len(test_df)}")
