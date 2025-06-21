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

class CommentsDataset(Dataset):
    def __init__(self, data_path, text_encoder):
        self.data = pd.read_csv(data_path, sep='\t', header=None)
        self.X = self.data.iloc[1:, 2].values
        self.y = self.data.iloc[1:, 1].values.astype('int64')
        
        self.text_encoder = text_encoder

        self.max_text_length = 0
        for text in self.X:
            self.max_text_length = max(self.max_text_length, len(text))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.text_encoder(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)
    
    def get_max_text_length(self):
        return self.max_text_length

class TextEncoder:
    def __init__(self, data_path, max_length=100, min_freq=2):
        self.max_length = max_length # max length of the squence
        self.min_freq = min_freq # min frequency of the word

        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        # special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'

        import pandas as pd
        texts = pd.read_csv(data_path, sep='\t', header=None).iloc[1:, 2].values
        self.fit(texts)
        
    def _tokenize(self, text):
        # import re
        # text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        import jieba
        return list(jieba.cut(text))
    
    def fit(self, texts):
        # build vocabulary
        # count token
        vocab_count = {}
        for text in texts:
            tokens = self._tokenize(str(text))
            for token in tokens:
                vocab_count[token] = vocab_count.get(token, 0) + 1

        self.vocab = {word: count for word, count in vocab_count.items() 
                      if count >= self.min_freq}
        # add special tokens
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]
        
        # build word to index mapping
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        for i, token in enumerate(special_tokens):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
        
        for i, word in enumerate(sorted(self.vocab.keys())):
            idx = len(special_tokens) + i
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def encode(self, text):
        tokens = self._tokenize(str(text))
        indices = [self.word_to_idx.get(token, self.word_to_idx[self.UNK_TOKEN]) 
                  for token in tokens]
        # fix indices vector to the fixed length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([self.word_to_idx[self.PAD_TOKEN]] * (self.max_length - len(indices)))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_for_test(self, indices):
        # map indices:tensor to string
        indices = indices.tolist()
        return ''.join([self.idx_to_word[index] for index in indices])
    
    def get_vocab_size(self):
        return len(self.word_to_idx)
    
    def __call__(self, text):
        return self.encode(text)
    
if __name__ == "__main__":
    text_encoder = TextEncoder(data_path='./task_3/data/train.tsv')
    commentsDataset = CommentsDataset(data_path='./task_3/data/train.tsv', text_encoder=text_encoder)
    tmp = commentsDataset[0][0]
    print(text_encoder.decode_for_test(tmp))
    print(tmp)
    
# if __name__ == "__main__":
#     text_encoder = TextEncoder(data_path='./task_3/data/train.tsv')
#     print(text_encoder.decode_for_test(text_encoder.encode("要说不满意的话，那就是动力了，1.5自然吸气发动机对这款车有种小马拉大车的感觉。如今天气这么热，上路肯定得开空调，开了后动力明显感觉有些不给力不过空调制冷效果还是不错的。")))
#     print(text_encoder.get_vocab_size()) # 17368

class RemoteBlockDataset(Dataset):
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
            image = Image.new('RGB', (64, 64), (0, 0, 0))
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default processing if no transform
            import numpy as np
            image = image.resize((64, 64))
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)
        
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)
    
class RemoteBlockDatasetTest(Dataset):
    def __init__(self, data_dir_path: str, transform=None):
        import os
        from PIL import Image
        
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.str_to_idx = {
            "AnnualCrop": 0,
            "Forest": 1,
            "HerbaceousVegetation": 2,
            "Highway": 3,
            "IndustrialBuildings": 4,
            "Pasture": 5,
            "PermanentCrop": 6,
            "Residential": 7,
            "River": 8,
            "SeaLake": 9
        }

        for file in os.listdir(data_dir_path):
            if file.endswith(".jpg"):
                self.image_paths.append(os.path.join(data_dir_path, file))
                self.labels.append(self.str_to_idx[file.split('_')[0]])

        # remote_block_class_dict = {
        #     0: "AnnualCrop", 
        #     1: "Forest",
        #     2: "HerbaceousVegetation",
        #     3: "Highway",
        #     4: "IndustrialBuildings",
        #     5: "Pasture",
        #     6: "PermanentCrop",
        #     7: "Residential",
        #     8: "River",
        #     9: "SeaLake"
        # }


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
            image = Image.new('RGB', (64, 64), (0, 0, 0))
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default processing if no transform
            import numpy as np
            image = image.resize((64, 64))
            image = np.array(image, dtype=np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)
        
        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.long)
    
# if __name__ == "__main__":
#     remoteBlockDataset = RemoteBlockDataset(data_dir_path='./task_4/data/trainset')
#     print(remoteBlockDataset[0])
        
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





