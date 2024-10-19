import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
import time
import os

# Privacy parameters
EPSILON = 3.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.5
SECURE_MODE = False  # Enable secure mode for better privacy guarantees

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Image preprocessing modules
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    normalize
])
    
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# Load datasets with smaller batch size for DP-SGD
BATCH_SIZE = 512  # Larger batch size for better privacy/utility trade-off
MAX_PHYSICAL_BATCH_SIZE = 64  # Maximum batch size that can fit in memory

train_dataset = torchvision.datasets.CIFAR10(root='data/', train=True, transform=train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='data/', train=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Hyper-parameters
num_epochs = 40
learning_rate = 0.001



# For updating learning rate
def update_lr(optimizer, lr):
    """
    This method update learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def conv3x3(in_channels, out_channels, stride=1):
    """
    return 3x3 Conv2d
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    """
    Initialize basic ResidualBlock with forward propogation
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    Initialize  ResNet with forward propogation
    """
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Initialize model
model = ResNet(ResidualBlock, [3, 3, 3]).to(device)

# Ensure the model is compatible with Opacus
errors = ModuleValidator.validate(model, strict=False)
if errors:
    model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# Initialize privacy engine
privacy_engine = PrivacyEngine()

# Make the model private
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=num_epochs,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
    poisson_sampling=True,  # Enable Poisson sampling for better privacy guarantees
)

# Print model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

def evaluate(model):
    """
    Evaluate accuracy of test set and save weight of model
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        # Save the model checkpoint
    torch.save(model.state_dict(), 'model_weight/'+str(int(100 * correct / total))+'resnet.ckpt')

# Training loop with privacy tracking
total_step = len(train_loader)
curr_lr = learning_rate

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer
    ) as memory_safe_data_loader:
        
        for i, (images, labels) in enumerate(memory_safe_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Step [{i+1}/{total_step}], "
                      f"Loss: {loss.item():.4f}, "
                      f"ε: {epsilon:.2f}")

        # Print epoch statistics
        avg_loss = epoch_loss / len(memory_safe_data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Evaluate model every 5 epochs
        if (epoch + 1) % 5 == 0:
            evaluate(model)
        
        # Learning rate decay
        if (epoch + 1) % 6 == 0:
            curr_lr /= 4
            update_lr(optimizer, curr_lr)

# Final privacy accounting
final_epsilon = privacy_engine.get_epsilon(DELTA)
print(f"\nFinal privacy guarantee: (ε = {final_epsilon:.2f}, δ = {DELTA})")