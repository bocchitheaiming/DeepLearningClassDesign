import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import models, transforms

from utils.utils import FlowerDataset, set_up_data_loader, plot_confusion_matrix

# Flower class mapping
flower_class_dict = {
    0: "daisy",
    1: "dandelion",
    2: "roses", 
    3: "sunflowers",
    4: "tulips",
}

# Define data transforms
# Training transforms with data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms without augmentation
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets with appropriate transforms
train_flower_dataset = FlowerDataset(data_dir_path='./task_2/data', transform=train_transforms)
val_flower_dataset = FlowerDataset(data_dir_path='./task_2/data', transform=val_test_transforms)
test_flower_dataset = FlowerDataset(data_dir_path='./task_2/data', transform=val_test_transforms)

print(f"Total samples: {len(train_flower_dataset)}")
print(f"Classes: {train_flower_dataset.classes}")

# Split indices for train/val/test
total_size = len(train_flower_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Get the indices for splitting
indices = torch.randperm(total_size).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Create subset datasets
train_subset = Subset(train_flower_dataset, train_indices)
val_subset = Subset(val_flower_dataset, val_indices)  
test_subset = Subset(test_flower_dataset, test_indices)

# Create data loaders
train_dataset = DataLoader(train_subset, batch_size=32, shuffle=True)
valid_dataset = DataLoader(val_subset, batch_size=32, shuffle=False)
test_dataset = DataLoader(test_subset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")
print(f"Test samples: {len(test_subset)}")

# Train a resnet18 classification model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)

class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, epochs, optimizer, loss_fn):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = epochs    
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def train(self):
        # Train the model
        print("Start training: ")
        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")

            run.log(
                {
                    "loss": loss.item(),
                }
            )
        
        # save the model
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.model.state_dict(), f"./task_2/model/model_{time_stamp}.pth")

        # Validate the model
        print("Start validating: ")
        self.model.eval()
        total_samples_num = 0
        correct_pred_num = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct_pred_num += (pred == target).sum().item()
                total_samples_num += len(target)
        print(f"Validation accuracy: {correct_pred_num * 100 / total_samples_num}%")
    
    def evaluate(self):
        # Test the model
        print("Start testing: ")
        self.model.eval()
        pred_all = torch.tensor([], dtype=torch.long)
        target_all = torch.tensor([], dtype=torch.long)
        total_samples_num = 0
        correct_pred_num = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                output = self.model(data)
                pred = output.argmax(dim=1)
                pred_all = torch.cat((pred_all, pred))
                target_all = torch.cat((target_all, target))
                correct_pred_num += (pred == target).sum().item()
                total_samples_num += len(target)
        print(f"Testing accuracy: {correct_pred_num * 100 / total_samples_num}%")
        plot_confusion_matrix(target_all, pred_all, task_name="flower")

def load_and_test(model, model_path, real_test_dataset):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    pred_all = torch.tensor([], dtype=torch.long)
    target_all = torch.tensor([], dtype=torch.long)
    total_samples_num = 0
    correct_pred_num = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(real_test_dataset):
            output = model(data)
            pred = output.argmax(dim=1)
            pred_all = torch.cat((pred_all, pred))
            target_all = torch.cat((target_all, target))
            correct_pred_num += (pred == target).sum().item()
            total_samples_num += len(target)
    print(f"Testing accuracy: {correct_pred_num * 100 / total_samples_num}%")
    plot_confusion_matrix(target_all, pred_all, task_name="flower")

if __name__ == "__main__":
    # import swanlab
    # swanlab.login(api_key="JjBDaf9U5qSVTuKAHqFG4", save=True)
    # run = swanlab.init(
    #     project="DLClassDesign_task_2",
    #     config={
    #         "learning_rate": 0.0001,
    #         "epochs": 100,
    #     },
    # )

    # trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, epochs=5, optimizer=optim.Adam(model.parameters(), lr=0.0001), loss_fn=nn.CrossEntropyLoss())
    # trainer.train()
    # trainer.evaluate()

    saved_model_path = "./task_2/model/model_20250618_181343.pth"
    # real_test_dataset = set_up_data_loader(test_flower_dataset, batch_size=1, shuffle=False, split=None)
    # load_and_test(model, saved_model_path, real_test_dataset)
    real_test_data_dir = "./task_2/data/test"
    for file in os.listdir(real_test_data_dir):
        if file.endswith(".jpg"):
            from PIL import Image
            image_path = os.path.join(real_test_data_dir, file)
            image = Image.open(image_path)
            image = val_test_transforms(image)
            image = image.unsqueeze(0)
            output = model(image)
            pred = output.argmax(dim=1)
            print(f"Predicted class: {flower_class_dict[pred.item()]}")

