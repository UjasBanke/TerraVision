import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

DATA_DIR   = "data/eurosat"
MODEL_PATH = "model.pth"
IMG_SIZE   = 96
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 0.001
TEST_SPLIT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading EuroSAT dataset...")
full_dataset = datasets.EuroSAT(root=DATA_DIR, transform=None, download=True)
class_names  = full_dataset.classes
num_classes  = len(class_names)
print(f"Classes ({num_classes}): {class_names}")
print(f"Total images: {len(full_dataset)}")

test_size  = int(TEST_SPLIT * len(full_dataset))
train_size = len(full_dataset) - test_size
train_indices, test_indices = random_split(range(len(full_dataset)), [train_size, test_size])


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = list(indices)
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


train_dataset = SplitDataset(full_dataset, train_indices, train_transform)
test_dataset  = SplitDataset(full_dataset, test_indices,  test_transform)
print(f"Train: {train_size} | Test: {test_size}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


#  RESNET18 
# Load pretrained ResNet18 
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# keep pretrained knowledge intact
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer for our 10 classes
# will be trained in phase 1
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes)
)

model = model.to(device)
print(f"ResNet18 loaded. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Phase 1: train only the new head (fast, 10 epochs)
print("\nPhase 1: Training classifier head only...\n")
optimizer = optim.AdamW(model.fc.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
best_acc  = 0.0

for epoch in range(10):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total        += labels.size(0)
        correct      += (predicted == labels).sum().item()

    scheduler.step()

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted    = torch.max(model(images), 1)
            val_total      += labels.size(0)
            val_correct    += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    marker  = " ✓ best" if val_acc > best_acc else ""
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"model_state_dict": model.state_dict(), "class_names": class_names, "num_classes": num_classes, "arch": "resnet18"}, MODEL_PATH)

    print(f"Epoch [{epoch+1:2d}/10] Loss: {running_loss/len(train_loader):.4f} | Train: {100*correct/total:.1f}% | Val: {val_acc:.1f}%{marker}")

# Phase 2:  fine-tune everything together
print("\nPhase 2: Fine-tuning full network...\n")

for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.AdamW([
    {"params": model.layer3.parameters(), "lr": LR * 0.1},
    {"params": model.layer4.parameters(), "lr": LR * 0.1},
    {"params": model.fc.parameters(),     "lr": LR},
], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total        += labels.size(0)
        correct      += (predicted == labels).sum().item()

    scheduler.step()

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted    = torch.max(model(images), 1)
            val_total      += labels.size(0)
            val_correct    += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    marker  = " ✓ best" if val_acc > best_acc else ""
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"model_state_dict": model.state_dict(), "class_names": class_names, "num_classes": num_classes, "arch": "resnet18"}, MODEL_PATH)

    print(f"Epoch [{epoch+1:2d}/10] Loss: {running_loss/len(train_loader):.4f} | Train: {100*correct/total:.1f}% | Val: {val_acc:.1f}%{marker}")

print(f"\nBest Accuracy: {best_acc:.2f}%")
print(f"Model saved to '{MODEL_PATH}'")
print("Done!")