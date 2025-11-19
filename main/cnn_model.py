import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim




class FaceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
#prepare data for CNN ####################################
#THIS STUFF IS ALREADY DONE IN THE PRIOR SCRIPTS
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # already 224x224, but harmless
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = FaceDataset("./data/labels/train_labels.csv", transform=transform)
    val_dataset   = FaceDataset("./data/labels/val_labels.csv", transform=transform)
    test_dataset  = FaceDataset("./data/labels/test_labels.csv", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    
    # load in resnet ##################################
    resnet = models.resnet50(pretrained=True)
    
    for param in resnet.parameters():
        param.requires_grad = False
    
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)  
    )
    
    # Define the training loop ##############################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.fc.parameters(), lr=1e-4)
    
    for epoch in range(2):  # start small, like 5–10 epochs
        resnet.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    
    # Evaluate on the Validation set ######################
    resnet.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")
    
    # save features for LSTM
    cnn_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # remove last FC
