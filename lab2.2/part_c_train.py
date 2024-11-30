import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from trainloader import CustomImageDataset
import time 

# Define the improved CNN model
class ImprovedMultiClassCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(ImprovedMultiClassCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 11))  # Adaptive pooling to fixed output size
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 11, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Training function
def train_model(train_dataset_root, save_weights_path):
    start_time = time.time()
    transform = transforms.Compose([
        transforms.Resize((50, 100)),
        transforms.ToTensor()
    ])

    # Load dataset
    csv_path = os.path.join(train_dataset_root, "public_train.csv")
    train_dataset = CustomImageDataset(root_dir=train_dataset_root, csv=csv_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss function, and optimizer
    model = ImprovedMultiClassCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        current_time = time.time()
        epoch_time = current_time - start_time
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        print(f'{epoch_time} sec')
        # Adjust learning rate
        scheduler.step(epoch_loss)

        # Save weights after each epoch
        torch.save(model.state_dict(), save_weights_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_root', required=True, help='Path to the train dataset root')
    parser.add_argument('--save_weights_path', required=True, help='Path to save weights')
    args = parser.parse_args()

    times = []
    train_model(args.train_dataset_root, args.save_weights_path)