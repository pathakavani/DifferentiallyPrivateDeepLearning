import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
import os
import time

def setup(rank, world_size):
    """
    Initialize the distributed environment
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment
    """
    dist.destroy_process_group()

# Privacy parameters
EPSILON = 3.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.5
SECURE_MODE = True

# Training parameters
BATCH_SIZE = 4096
MAX_PHYSICAL_BATCH_SIZE = 512
num_epochs = 140
learning_rate = 0.001

def prepare_dataloader(rank, world_size):
    """
    Prepare distributed dataloaders
    """
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

    train_dataset = torchvision.datasets.CIFAR10(
        root='data/',
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='data/',
        train=False,
        transform=test_transform
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE // world_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE // world_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader, train_sampler

def update_lr(optimizer, lr):
    """
    Update learning rate
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



def train(rank, world_size):
    """
    Training function for each process
    """
    setup(rank, world_size)
    
    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Create model and move to GPU
    model = ResNet(ResidualBlock, [3, 3, 3]).to(device)
    
    # Ensure model is compatible with Opacus
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=True)

    # Create optimizer
    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=learning_rate,
        #momentum=0.9,
        weight_decay=1e-5
        #nesterov=True
    )

    # Get distributed data loaders
    train_loader, test_loader, train_sampler = prepare_dataloader(rank, world_size)

    # Initialize privacy engine
    privacy_engine = PrivacyEngine()

    # Make model private using DPDDP
    model = DPDDP(model)
    
    # Attach privacy engine to the model
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=num_epochs,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
        poisson_sampling=True
    )

    criterion = nn.CrossEntropyLoss()
    curr_lr = learning_rate

    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
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

                if rank == 0 and (i + 1) % 50 == 0:
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Step [{i+1}/{len(memory_safe_data_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"ε: {epsilon:.2f}")

        # Only print metrics from rank 0
        if rank == 0:
            avg_loss = epoch_loss / len(memory_safe_data_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                evaluate(model, test_loader, device, epoch)

        # Learning rate decay
        # if (epoch + 1) % 6 == 0:
        #     curr_lr /= 4
        #     update_lr(optimizer, curr_lr)

        # Synchronize processes
        dist.barrier()

    if rank == 0:
        final_epsilon = privacy_engine.get_epsilon(DELTA)
        print(f"\nFinal privacy guarantee: (ε = {final_epsilon:.2f}, δ = {DELTA})")

    cleanup()

def evaluate(model, test_loader, device, epoch):
    """
    Evaluate the model on the test set
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Save model checkpoint (only from rank 0)
    torch.save(model.module.state_dict(), f'model_weight/resnet_epoch_{epoch}_acc_{accuracy:.2f}.ckpt')


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs!")
    
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )