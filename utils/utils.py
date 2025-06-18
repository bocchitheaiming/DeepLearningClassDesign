# Data utils
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pandas as pd


class IrisDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, header=None)
        self.X = self.data.iloc[:, :-1].values.astype('float32')
        self.y_str = self.data.iloc[:, -1]

        # Map class in string to ordinal integer
        self.classes = sorted(self.y_str.unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.y = self.y_str.map(self.class_to_idx).values.astype('int64')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
class FlowerDataset(Dataset):
    def __init__(self, data_dir_path: str, transform=None):
        import os
        from PIL import Image
        
        self.transform = transform

        self.classes = []
        for cls in os.listdir(data_dir_path):
            if not cls.endswith('.txt') and os.path.isdir(os.path.join(data_dir_path, cls)):
                self.classes.append(cls)
        self.classes.sort()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for cls in self.classes:
            cls_path = os.path.join(data_dir_path, cls)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default processing if no transform
            import numpy as np
            image = image.resize((224, 224))
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)
        
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)

# if __name__ == "__main__":
#     flowerDataset = FlowerDataset(data_dir_path='./task_2/data')
#     print(flowerDataset.X[0].shape)
    
def set_up_data_loader(dataset, batch_size, shuffle=True, split = None, vailid_set:bool = False):
    if split is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        if isinstance(split, list):
            if (len(split) != 2 and len(split) != 3):
                raise ValueError("split must be a list of two or three elements(for train, valid and test)")
            if (sum(split) != 1):
                raise ValueError("sum of split must be 1")
            if (len(split) == 2):
                train_size = int(len(dataset) * split[0])
                valid_size = len(dataset) - train_size
                train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
                return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
            else:
                train_size = int(len(dataset) * split[0])
                valid_size = int(len(dataset) * split[1])
                test_size = len(dataset) - train_size - valid_size
                train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
                return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
            
# Visualization utils
# Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
def plot_confusion_matrix(y_true, y_pred, title='Confusion matrix', cmap=plt.cm.Blues, task_name:str = "default"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False)
    plt.title(title)
    plt.savefig(f'./results/{task_name}_confusion_matrix.png')





