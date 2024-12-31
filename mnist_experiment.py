import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import matplotlib.pyplot as plt

from models import MLP, MLPSharedWeights
from utils import evaluate_model, train_model

# Load datasets (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split MNIST dataset
train_size = int(0.8 * len(mnist))
val_size = len(mnist) - train_size
train_mnist, val_mnist = random_split(mnist, [train_size, val_size])

# Dataloaders
batch_size = 64
train_loader = DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_mnist, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_mnist, batch_size=batch_size, shuffle=False)

# Model parameters
input_size = 28 * 28  # MNIST input size
hidden_size = 128
num_classes = 10
epochs = 20
device = torch.device("mps")
# Initialize models
mlp_a = MLP(input_size, hidden_size, num_classes)
mlp_b = MLPSharedWeights(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_a = optim.Adam(mlp_a.parameters(), lr=0.001)
optimizer_b = optim.Adam(mlp_b.parameters(), lr=0.001)

# Train models
start_time_a = time.time()
train_loss_a, val_loss_a = train_model(mlp_a, train_loader, val_loader, criterion, optimizer_a, epochs, device)
training_time_a = time.time() - start_time_a

start_time_b = time.time()
train_loss_b, val_loss_b = train_model(mlp_b, train_loader, val_loader, criterion, optimizer_b, epochs, device)
training_time_b = time.time() - start_time_b

# Evaluate models
test_accuracy_a = evaluate_model(mlp_a, test_loader, device)
test_accuracy_b = evaluate_model(mlp_b, test_loader, device)

# Count trainable parameters
trainable_params_a = sum(p.numel() for p in mlp_a.parameters() if p.requires_grad)
trainable_params_b = sum(p.numel() for p in mlp_b.parameters() if p.requires_grad)

# Plot losses
plt.plot(train_loss_a, label="Train Loss (MLP A)")
plt.plot(val_loss_a, label="Validation Loss (MLP A)")
plt.plot(train_loss_b, label="Train Loss (MLP B)")
plt.plot(val_loss_b, label="Validation Loss (MLP B)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("MNIST_losses.png")

# Print results
print(f"Simple MLP - Test Accuracy: {test_accuracy_a}, Trainable Parameters: {trainable_params_a}, Training Time: {training_time_a}s")
print(f"Shared Weights MLP - Test Accuracy: {test_accuracy_b}, Trainable Parameters: {trainable_params_b}, Training Time: {training_time_b}s")
