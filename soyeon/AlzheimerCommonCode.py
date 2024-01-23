from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torchvision.utils import make_grid

# transformation to the image dataset:
transform = transforms.Compose([transforms.Resize((227, 227)),
                                transforms.ToTensor()
                                ])
train_dir = "/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/"
test_dir = "/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/"

# Applying the already definded transformation to the dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Loading the training dataset in a way that optimizes the training process
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get one batch of images
images, labels = next(iter(train_loader))

# number of images you want to display
num_images = 4

class_names = train_dataset.classes

class_names = train_dataset.classes
class_counts = Counter([label for _, label in train_dataset])

# split the dataset into training and validation sets(80-20 split)

train_size = int(0.8*len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(
    train_dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = models.AlexNet()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training & eval
best_val_loss = float('inf')
patience = 100
num_epochs = 500
patience_counter = 0

train_losses = []
val_losses = []
val_accuracies = []
all_preds = []
all_labels = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_corrects.double() / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy} patience counter: {patience_counter}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset counter
        # Optionally save the model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    # Early stopping check
    if patience_counter >= patience:
        print("Stopping early due to no improvement in validation loss.")
        break
print("progress:", epoch)

# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Training and Validation Losses per Epoch")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
val_accuracies_cpu = [acc.cpu().numpy() for acc in val_accuracies]
plt.plot(val_accuracies_cpu, label='Validation Accuracy')
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# save the trained model
torch.save(model.state_dict(), 'AlexNet_cnn_model.pth')


# Storage for predictions and actual labels
all_preds = []
all_labels = []
f1_scores = []

# Evaluation loop
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to arrays for metric calculation
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate metrics for each class
for class_index in range(4):  # Replace num_classes with the actual number of classes
    class_preds = (all_preds == class_index)
    class_labels = (all_labels == class_index)
    accuracy = accuracy_score(class_labels, class_preds)
    precision = precision_score(class_labels, class_preds, zero_division=0)
    recall = recall_score(class_labels, class_preds, zero_division=0)
    f1 = f1_score(class_labels, class_preds, zero_division=0)
    f1_scores.append(f1)

    class_name = class_names[class_index]

    print(f"Class name {class_name}({class_index}) - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Calculate overall metrics
overall_accuracy = accuracy_score(all_labels, all_preds)
overall_precision = precision_score(
    all_labels, all_preds, average='macro', zero_division=0)
overall_recall = recall_score(
    all_labels, all_preds, average='macro', zero_division=0)
overall_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")

print("------------ micro ------------")
overall_accuracy_micro = accuracy_score(all_labels, all_preds)
overall_precision_micro = precision_score(
    all_labels, all_preds, average="micro", zero_division=0)
overall_recall_micro = recall_score(
    all_labels, all_preds, average='micro', zero_division=0)
overall_f1_micro = f1_score(all_labels, all_preds,
                            average='micro', zero_division=0)

print(f"Overall Accuracy: {overall_accuracy_micro}")
print(f"Overall Precision: {overall_precision_micro}")
print(f"Overall Recall: {overall_recall_micro}")
print(f"Overall F1 Score: {overall_f1_micro}")

cm = confusion_matrix(all_labels, all_preds)
print(cm)

# calculate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


def roc_auc_score_multiclass(actual_class, pred_class):

    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    total_roc = 0.0
    for per_class in unique_class:

        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(
            new_actual_class, new_pred_class, average="macro")
        roc_auc_dict[per_class] = roc_auc
        total_roc += roc_auc

    return total_roc/4, roc_auc_dict


roc_auc_dict = roc_auc_score_multiclass(all_labels, all_preds)
roc_auc_dict
