import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import IrisDataset, set_up_data_loader, plot_confusion_matrix

irisDataset = IrisDataset(data_path='./task_1/data/bezdekIris.data')
# Split the dataset into train, valid and test
train_dataset, valid_dataset, test_dataset = set_up_data_loader(irisDataset, batch_size=1, split=[0.8, 0.1, 0.1])

# Train a softmax classification model
model = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 3),
    nn.Softmax(dim=1)
)

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
                # if batch_idx % 10 == 0:
                #     # print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item()}")

            run.log(
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
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct_pred_num += (pred == target).sum().item()
                total_samples_num += len(target)
        print(f"Vaildation accuracy: {correct_pred_num * 100 / total_samples_num}%")

        # save the model
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(self.model.state_dict(), f"./task_1/model/model_{time_stamp}.pth")
    
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
        plot_confusion_matrix(target_all, pred_all)

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
            total_samples_num += len(target)
            correct_pred_num += (pred == target).sum().item()
    print(f"Testing accuracy: {correct_pred_num * 100 / total_samples_num}%")
    plot_confusion_matrix(target_all, pred_all, task_name="iris")

if __name__ == "__main__":
    # import swanlab
    # swanlab.login(api_key="JjBDaf9U5qSVTuKAHqFG4", save=True)
    # run = swanlab.init(
    #     # set project name
    #     project="DLClassDesign_task_1",
    #     # hyper parameter, no real meaning
    #     config={
    #         "learning_rate": 0.0001,
    #         "epochs": 100,
    #     },
    # )

    # trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, epochs=100, optimizer=optim.Adam(model.parameters(), lr=0.0001), loss_fn=nn.CrossEntropyLoss())
    # trainer.train()
    # trainer.evaluate()

    saved_model_path = "./task_1/model/model_20250618_192119.pth"
    test_real_dataset = IrisDataset(data_path='./task_1/data/test/test_1.data')
    test_real_dataset = set_up_data_loader(test_real_dataset, batch_size=1, shuffle=False, split=None)
    load_and_test(model, saved_model_path, test_real_dataset)













