import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import models, transforms

from utils.utils import RemoteBlockDataset, RemoteBlockDatasetTest, set_up_data_loader, plot_confusion_matrix
from model_backbone.alexNet import AlexNet

# remote block class mapping
remote_block_class_dict = {
    0: "AnnualCrop", 
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "IndustrialBuildings",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake"
}

# Setup device for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Define data transforms
# Training transforms with data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms without augmentation
val_test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets with appropriate transforms
train_remote_block_dataset = RemoteBlockDataset(data_dir_path='./task_4/data/trainset', transform=train_transforms)
val_remote_block_dataset = RemoteBlockDataset(data_dir_path='./task_4/data/trainset', transform=val_test_transforms)
test_remote_block_dataset = RemoteBlockDataset(data_dir_path='./task_4/data/trainset', transform=val_test_transforms)

print(f"Total samples: {len(train_remote_block_dataset)}")
print(f"Classes: {train_remote_block_dataset.classes}")

# Split indices for train/val/test
total_size = len(train_remote_block_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Get the indices for splitting
indices = torch.randperm(total_size).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Create subset datasets
train_subset = Subset(train_remote_block_dataset, train_indices)
val_subset = Subset(val_remote_block_dataset, val_indices)  
test_subset = Subset(test_remote_block_dataset, test_indices)

# Create data loaders
train_dataset = DataLoader(train_subset, batch_size=32, shuffle=True)
valid_dataset = DataLoader(val_subset, batch_size=32, shuffle=False)
test_dataset = DataLoader(test_subset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")
print(f"Test samples: {len(test_subset)}")

# Train a resnet18 classification model
model = models.resnet18(pretrained=True)
# model = AlexNet(num_classes=10)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.to(device)
print(f"Model moved to {device}")

class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, epochs, optimizer, loss_fn, run = None, device = 'cpu'):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = epochs    
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.run = run    
        self.device = device
        
    def train(self):
        # Train the model
        print("Start training: ")
        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item():.4f}")
            if self.run is not None:
                self.run.log(
                    {
                        "loss": loss.item(),
                    }
                )
        
        # save the model
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.model.state_dict(), f"./task_4/model/model_{time_stamp}.pth")

        # Validate the model
        print("Start validating: ")
        self.model.eval()
        total_samples_num = 0
        correct_pred_num = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
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
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                pred = output.argmax(dim=1)
                pred_all = torch.cat((pred_all, pred.cpu()))
                target_all = torch.cat((target_all, target.cpu()))
                correct_pred_num += (pred == target).sum().item()
                total_samples_num += len(target)
        print(f"Testing accuracy: {correct_pred_num * 100 / total_samples_num}%")
        plot_confusion_matrix(target_all, pred_all, task_name="remote block")

def load_and_test(model, model_path, real_test_dataset, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    pred_all = torch.tensor([], dtype=torch.long)
    target_all = torch.tensor([], dtype=torch.long)
    total_samples_num = 0
    correct_pred_num = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(real_test_dataset):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            pred = output.argmax(dim=1)
            pred_all = torch.cat((pred_all, pred.cpu()))
            target_all = torch.cat((target_all, target.cpu()))
            correct_pred_num += (pred == target).sum().item()
            total_samples_num += len(target)
    print(f"Testing accuracy: {correct_pred_num * 100 / total_samples_num}%")
    plot_confusion_matrix(target_all, pred_all, task_name="remote block")

if __name__ == "__main__":
    import swanlab
    # swanlab.login(api_key="JjBDaf9U5qSVTuKAHqFG4", save=True)
    # run = swanlab.init(
    #     project="DLClassDesign_task_4",
    #     config={
    #         "learning_rate": 0.0001,
    #         "epochs": 10,
    #     },
    # )

    # trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, epochs=5, optimizer=optim.Adam(model.parameters(), lr=0.0001), loss_fn=nn.CrossEntropyLoss(), run = run, device=device)
    # # trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, epochs=5, optimizer=optim.Adam(model.parameters(), lr=0.0001), loss_fn=nn.CrossEntropyLoss(), run = None, device=device)
    # trainer.train()
    # trainer.evaluate()

    saved_model_path = "./task_4/model/model_20250619_192604.pth"
    real_test_dataset = RemoteBlockDatasetTest(data_dir_path='./task_4/data/test/testset', transform=val_test_transforms)
    print(len(real_test_dataset))
    real_test_dataset = set_up_data_loader(real_test_dataset, batch_size=1, shuffle=False, split=None)
    load_and_test(model, saved_model_path, real_test_dataset, device)

