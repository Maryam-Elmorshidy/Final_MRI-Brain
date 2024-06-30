# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# We use Alzheimer_s Dataset to demonstrate the classification method

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import torch
import torchvision
from torchvision import datasets
from torchvision import models
from torchvision import transforms 
import torchvision.transforms as T# for simplifying the transforms
from torch.cuda.amp import autocast, GradScaler

from torch import nn, optim
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, sampler, random_split
import timm
from timm.loss import LabelSmoothingCrossEntropy # This is better than normal nn.CrossEntropyLoss

# remove warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
#%matplotlib inline

import sys
from tqdm import tqdm
import time
import copy

def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes


def get_data_loaders(data_dir, batch_size, batch_size_test,train_percentage=0.8):
    # Define data transformations for data augmentation
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
        T.RandomErasing(p=0.1, value='random')
    ])


    # Load the entire dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Calculate the number of samples for train, validation, and test
    total_samples = len(full_dataset)
    train_size = int(train_percentage * total_samples)
    test_size = total_samples - train_size 

    # Split the dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size])
    # Create DataLoader for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)

    return train_loader, test_loader


dataset_path ="/kaggle/input/fdata-adni-dataset/AugmentedAlzheimerDataset"

train_loader, test_loader = get_data_loaders(dataset_path , batch_size=64,batch_size_test=32)

classes = get_classes("/kaggle/input/fdata-adni-dataset/AugmentedAlzheimerDataset")
print(classes, len(classes))

dataloaders = {
    "train": train_loader,
    "test": test_loader,
}

dataset_sizes = {
    "train": len(train_loader.dataset),
    "test": len(test_loader.dataset),
}

print(len(train_loader), len(test_loader))

print(len(train_loader.dataset),
     len(test_loader.dataset))


# now, for the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

# Initialize ResNet-50
resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, len(classes ))  # Assuming your dataset has len(class_names) classes
resnet50 = resnet50.to(device)

# Define the criterion, optimizer, and learning rate scheduler

# criterion = nn.CrossEntropyLoss(weight=torch.Tensor())
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


import time
import copy
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_acc_history = []
    train_loss_history = []

    scaler = GradScaler()  # Initialize GradScaler for mixed precision

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in tqdm(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast():  # Mixed precision context
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # Scale gradients
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / total_train

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, train_acc_history, train_loss_history

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def test_model(model, criterion, dataloader, classes, device):
    model.eval()  # Set model to evaluation mode
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            correct_tensor = preds.eq(labels.data.view_as(preds))
            correct = correct_tensor.cpu().numpy()
            
            for i in range(len(labels)):
                label = labels.data[i]
                class_correct[label] += correct[i]
                class_total[label] += 1

    test_loss = running_loss / len(dataloader.dataset)
    overall_acc = 100 * sum(class_correct) / sum(class_total)

    print('Test Loss: {:.4f}'.format(test_loss))
    
    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print("Test Accuracy of {:>5s}: {:2.0f}% ({}/{})".format(class_name, acc, class_correct[i], class_total[i]))
        else:
            print("Test Accuracy of {:>5s}: NA".format(class_name))
            
    print("Overall Test Accuracy: {:.2f}%".format(overall_acc))
    
    # Plot the test accuracy for each class
    plt.figure(figsize=(8, 6))
    plt.bar(classes, [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(len(classes))])
    plt.xlabel('Class')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy for Each Class')
    plt.ylim(0, 100)
    plt.show()

    return test_loss, class_correct, class_total


# Train the model
resnet50, train_acc_history, train_loss_history = train_model(resnet50, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=15)

# Test the model
test_loss, test_acc_per_class, overall_acc = test_model(resnet50, criterion, test_loader, classes, device)


# Plotting training accuracy
plt.figure(figsize=(10, 5))
plt.plot( train_acc_history,label='Training Accuracy', color='blue')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plotting training loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Training Loss', color='red')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate predictions and true labels
def get_predictions_and_labels(model, dataloader, device):
    model.eval()
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return true_labels, predicted_labels

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

# Define the class labels
classes = ['AD', 'CN', 'EMCI', 'LMCI']

# Get true and predicted labels for training data
true_labels_train, predicted_labels_train = get_predictions_and_labels(resnet50, train_loader, device)

# Get true and predicted labels for testing data
true_labels_test, predicted_labels_test = get_predictions_and_labels(resnet50, test_loader, device)

# Print classification report for training data
print("Classification Report for Training Data")
print(classification_report(true_labels_train, predicted_labels_train, target_names=classes))

# Print classification report for testing data
print("Classification Report for Testing Data")
print(classification_report(true_labels_test, predicted_labels_test, target_names=classes))

# Calculate confusion matrix for training data
train_cm = confusion_matrix(true_labels_train, predicted_labels_train)

# Calculate confusion matrix for testing data
test_cm = confusion_matrix(true_labels_test, predicted_labels_test)

# Plot confusion matrix for training data
plot_confusion_matrix(train_cm, classes, title='Confusion Matrix for Training Data')
# Plot normalized confusion matrix for training data
plot_confusion_matrix(train_cm, classes, title='Normalized Confusion Matrix for Training Data', normalize=True)

# Plot confusion matrix for testing data
plot_confusion_matrix(test_cm, classes, title='Confusion Matrix for Testing Data')
# Plot normalized confusion matrix for testing data
plot_confusion_matrix(test_cm, classes, title='Normalized Confusion Matrix for Testing Data', normalize=True)





