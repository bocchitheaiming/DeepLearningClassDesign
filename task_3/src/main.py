import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import CommentsDataset, TextEncoder, set_up_data_loader, plot_confusion_matrix
from model_backbone.RNN import myRNN, myRNN_use_pytorch, myLSTM_use_pytorch, BiLSTMModel

# Setup device for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# device = torch.device('cpu')

text_encoder = TextEncoder(data_path='./task_3/data/train.tsv', max_length=150, min_freq=3)
commentsDataset = CommentsDataset(data_path='./task_3/data/train.tsv', text_encoder=text_encoder)

# Split the dataset into train, valid and test
train_dataset, valid_dataset, test_dataset = set_up_data_loader(commentsDataset, batch_size=128, split=[0.8, 0.1, 0.1])

# Train a RNN text classification model
# model = myRNN(vocab_size=text_encoder.get_vocab_size(), 
#               embedding_dim=128, 
#               hidden_size=128, 
#               output_size=2,
#               num_layers=1, 
#               dropout=0.3)

# model = myRNN_use_pytorch(vocab_size=text_encoder.get_vocab_size(), 
#                           embedding_dim=100, 
#                           hidden_size=128, 
#                           output_size=2,
#                           num_layers=2, 
#                           dropout=0.3)

# model = myLSTM_use_pytorch(vocab_size=text_encoder.get_vocab_size(), 
#                           embedding_dim=100, 
#                           hidden_size=128, 
#                           output_size=2,
#                           num_layers=2, 
#                           dropout=0.3)

model = BiLSTMModel(vocab_size=text_encoder.get_vocab_size(), 
                    embed_size=128, 
                    hidden_size=128, 
                    output_size=2)

model.init_weights()
# Move model to GPU if available
model = model.to(device)
print(f"Model moved to {device}")
class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, epochs, optimizer, loss_fn, device, run = None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = epochs    
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.run = run
    
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
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item()}")

            self.run.log(
                {
                    "loss": loss.item(),
                }
            )

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
        print(f"Vaildation accuracy: {correct_pred_num * 100 / total_samples_num}%")

        # save the model
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.model.state_dict(), f"./task_3/model/model_{time_stamp}.pth")
    
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
                # Move predictions back to CPU for concatenation
                pred_all = torch.cat((pred_all, pred.cpu()))
                target_all = torch.cat((target_all, target.cpu()))
                correct_pred_num += (pred == target).sum().item()
                total_samples_num += len(target)
        print(f"Testing accuracy: {correct_pred_num * 100 / total_samples_num}%")
        plot_confusion_matrix(target_all, pred_all, task_name="comments")

def load_and_test(model, model_path, real_test_dataset, device, text_encoder = text_encoder):
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
            for i in range(len(data)):
                print(text_encoder.decode_for_test(data[i]))
            output = model(data)
            pred = output.argmax(dim=1)
            print(pred.cpu())
            pred_all = torch.cat((pred_all, pred.cpu()))
            target_all = torch.cat((target_all, target.cpu()))
            total_samples_num += len(target)
            correct_pred_num += (pred == target).sum().item()
    print(f"Testing accuracy: {correct_pred_num * 100 / total_samples_num}%")
    plot_confusion_matrix(target_all, pred_all, task_name="comments")

if __name__ == "__main__":
    import swanlab
    # swanlab.login(api_key="JjBDaf9U5qSVTuKAHqFG4", save=True)
    # run = swanlab.init(
    #     # set project name
    #     project="DLClassDesign_task_3",
    #     # hyper parameter, no real meaning
    #     config={
    #         "learning_rate": 0.0003,
    #         "epochs": 10,
    #     },
    # )

    # trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, epochs=10, optimizer=optim.Adam(model.parameters(), lr=0.0003), loss_fn=nn.CrossEntropyLoss(), device=device, run=run)
    # trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, epochs=10, optimizer=optim.Adam(model.parameters(), lr=0.0003), loss_fn=nn.CrossEntropyLoss(), device=device, run=None)
    # trainer.train()
    # trainer.evaluate()

    real_test_dataset = CommentsDataset("./task_3/data/test/test.tsv", text_encoder=text_encoder)
    real_test_dataset = set_up_data_loader(real_test_dataset, batch_size=128, split=None)
    load_and_test(model, "./task_3/model/model_20250619_183109.pth", real_test_dataset, device, text_encoder = text_encoder)

    













