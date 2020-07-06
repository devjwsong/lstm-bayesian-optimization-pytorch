from tqdm import tqdm
from custom_data import *
from lstm import *
from constant import *
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization

import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
import numpy as np


class Manager:
    def __init__(self):
        print("Loading dataset & vocab dict...")
        self.train_set, self.dev_set, self.test_set, self.word2idx = get_data()

        self.bayes_optimizer = BayesianOptimization(
            f=self.train,
            pbounds={
                'learning_rate': learning_rates,
                'batch_size': batch_sizes
            },
            random_state=777
        )

    def train(self, learning_rate, batch_size):
        batch_size = round(batch_size)
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(self.dev_set, batch_size=batch_size, shuffle=True)

        print("Loading model...")
        model = LSTM(len(self.word2idx)).to(device)
        criterion = nn.NLLLoss(reduction='mean')

        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        print("Initializing optimizer & loss function...")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_f1 = 0.0

        print("Train starts.")
        for epoch in range(1, epoch_num+1):
            model.train()

            total_train_losses = []
            total_train_preds = []
            total_train_targs = []

            for batch in tqdm(train_loader):
                x, y, lens = batch
                lens_sorted, idx = lens.sort(dim=0, descending=True)
                x_sorted = x[idx]
                y_sorted = y[idx]

                x, y, lens = x_sorted.to(device), y_sorted.to(device), lens_sorted.to(device)

                output = model(x, lens)  # (B, class_num)
                loss = criterion(output, y)  # ()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_losses.append(loss.item())
                total_train_preds += torch.argmax(output, dim=-1).tolist()
                total_train_targs += y.tolist()

            train_loss = np.mean(total_train_losses)
            train_f1 = f1_score(total_train_targs, total_train_preds, average='weighted')

            print(f"########## Epoch: {epoch} ##########")
            print(f"Train loss: {train_loss} || Train f1 score: {train_f1}")

            valid_loss, valid_f1 = self.validate(model, criterion, valid_loader)

            if valid_f1 > best_f1:
                print("***** Current best model saved. *****")
                torch.save(model.state_dict(), f"{ckpt_dir}/best_model_batch|{batch_size}_lr|{round(learning_rate, 4)}.pth")
                best_f1 = valid_f1

            print(f"Valid loss: {valid_loss} || Valid f1 score: {valid_f1} || Best f1 score: {best_f1}")

        return best_f1

    def validate(self, model, criterion,  valid_loader):
        model.eval()
        total_valid_losses = []
        total_valid_preds = []
        total_valid_targs = []

        for batch in tqdm(valid_loader):
            x, y, lens = batch
            lens_sorted, idx = lens.sort(dim=0, descending=True)
            x_sorted = x[idx]
            y_sorted = y[idx]

            x, y, lens = x_sorted.to(device), y_sorted.to(device), lens_sorted.to(device)

            output = model(x, lens)  # (B, class_num)
            loss = criterion(output, y)  # ()

            total_valid_losses.append(loss.item())
            total_valid_preds += torch.argmax(output, dim=-1).tolist()
            total_valid_targs += y.tolist()

        valid_loss = np.mean(total_valid_losses)
        valid_f1 = f1_score(total_valid_targs, total_valid_preds, average='weighted')

        return valid_loss, valid_f1

    def test(self, model_name, batch_size):
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=True)

        print("Loading model...")
        model = LSTM(len(self.word2idx))
        criterion = nn.NLLLoss(reduction='mean')

        model.load_state_dict(torch.load(f"{ckpt_dir}/{model_name}")).to(device)

        model.eval()
        total_test_losses = []
        total_test_preds = []
        total_test_targs = []

        for batch in tqdm(test_loader):
            x, y, lens = batch
            lens_sorted, idx = lens.sort(dim=0, descending=True)
            x_sorted = x[idx]
            y_sorted = y[idx]

            x, y, lens = x_sorted.to(device), y_sorted.to(device), lens_sorted.to(device)

            output = model(x, lens)  # (B, class_num)
            loss = criterion(output, y)  # ()

            total_test_losses.append(loss.item())
            total_test_preds += torch.argmax(output, dim=-1).tolist()
            total_test_targs += y.tolist()

        test_loss = np.mean(total_test_losses)
        test_f1 = f1_score(total_test_targs, total_test_preds, average='weighted')

        print("######## Test Results ########")
        print(f"Test loss: {test_loss} || Test f1 score: {test_f1}")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='train or test?')
    parser.add_argument('--model_name', type=str, help='name of model file if you want to test.')

    args = parser.parse_args()

    assert args.mode == 'train' or args.mode == 'test', "Please specify correct mode."

    manager = Manager()

    if args.mode == 'train':
        print("Training starts.")
        manager.bayes_optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei', xi=0.01)

        print("Best optimization option")
        print(manager.bayes_optimizer.max)
    elif args.mode == 'test':
        assert args.model_name is not None, "Please give the model name if you want to conduct test."

        print("Testing starts.")
        manager.test(args.model_name, batch_size=128)
