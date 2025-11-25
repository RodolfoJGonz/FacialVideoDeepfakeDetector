import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ResNet50_LSTM


def train_model(
    train_loader,
    val_loader,
    num_epochs=10,
    lr=1e-4,
    device="cuda"
):
    best_val_acc = 0.0
    model = ResNet50_LSTM().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for videos, labels in pbar:
            videos = videos.to(device)      # [B, 16, 3, 224, 224]
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(videos)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * videos.size(0)

            # accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.4f}"
            })

        avg_loss = train_loss / total
        avg_acc = correct / total

        # ----------------------
        # Validation
        # ----------------------
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)

                outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * videos.size(0)

                #_, preds = torch.max(outputs, 1)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total

        print(f"\nEpoch {epoch}: "
              f"Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}")
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), "./models/best.pt")
            print("Saved new best model!")
    torch.save(model.state_dict(), "./models/last.pt")
    print("Saved LAST model (end of epoch)")
    return model

