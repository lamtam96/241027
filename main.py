import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.enc_conv1(x)
        x2 = self.pool(x1)
        x3 = self.enc_conv2(x2)
        x4 = self.pool(x3)
        x5 = self.upsample(x4)
        x6 = self.dec_conv1(x5)
        x7 = self.upsample(x6)
        x_out = self.dec_conv2(x7)
        return x_out


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_mask_pairs = [
            (img, img.replace('.jpg', '.png'))
            for img in os.listdir(image_dir) if img.endswith('.jpg')
        ]

        missing_masks = [img for img, mask in self.image_mask_pairs if not os.path.exists(os.path.join(mask_dir, mask))]
        if missing_masks:
            raise ValueError(f"Not match: {missing_masks}")

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_mask_pairs[idx][0])
        mask_path = os.path.join(self.mask_dir, self.image_mask_pairs[idx][1])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


image_dir = "../G1020/Images"
mask_dir = "../G1020/Masks"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = CustomDataset(image_dir, mask_dir, transform=transform)

train_size = 0.8
train_indices, val_indices = train_test_split(range(len(dataset)), train_size=train_size, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


def train_model():
    model = UNet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 10
    model.train()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        avg_val_loss = evaluate_model(model, val_loader, criterion, device)
        val_loss_history.append(avg_val_loss)

        print(f"Training Loss in epoch {epoch + 1}: {avg_train_loss}")
        print(f"Validation Loss in epoch {epoch + 1}: {avg_val_loss}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss over Epochs")
    plt.savefig("loss_curve.png")
    plt.show()


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks.float())
            val_loss += loss.item()

            outputs = (outputs > 0.5).float()
            intersection = (outputs * masks).sum()
            dice = (2. * intersection) / (outputs.sum() + masks.sum())
            total_dice += dice.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    print(f"\nValidation Loss: {avg_val_loss}, Average Dice Coefficient: {avg_dice}")
    model.train()
    return avg_val_loss


def visualize_predictions(model, val_loader, device):
    model.eval()
    images, masks = next(iter(val_loader))
    images, masks = images.to(device), masks.to(device)
    with torch.no_grad():
        outputs = model(images)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        axes[i, 0].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title("Original Image")
        axes[i, 1].imshow(masks[i].cpu().squeeze(), cmap='gray')
        axes[i, 1].set_title("True Mask")
        axes[i, 2].imshow(outputs[i].cpu().squeeze() > 0.5, cmap='gray')
        axes[i, 2].set_title("Predicted Mask")

    plt.tight_layout()
    plt.savefig("prediction_examples.png")
    plt.show()
    model.train()


train_model()
visualize_predictions(UNet().to('cuda' if torch.cuda.is_available() else 'cpu'), val_loader,
                      'cuda' if torch.cuda.is_available() else 'cpu')
