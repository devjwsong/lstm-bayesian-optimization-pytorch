from tqdm import tqdm
from custom_data import *
from lstm import *
from constant import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
import numpy as np


class Manager:
    def __init__(self, mode, model_name=None):
        print("Loading dataset & vocab dict...")
        train_set, dev_set, test_set, word2idx = get_data()
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        print("Loading model...")
        self.model = LSTM(len(word2idx))

        if mode == 'train':
            if not os.path.isdir(ckpt_dir):
                os.mkdir(ckpt_dir)

            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            print("Initializing optimizer & loss function...")
            self.optim = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.summary = SummaryWriter()
        elif mode == 'test':
            assert model_name is not None, "Please specify the model name if you want to test."

            self.model.load_state_dict(torch.load(f"{ckpt_dir}/{model_name}"))

        self.model = self.model.to(device)
        self.criterion = nn.NLLLoss(reduction='mean')

    def train(self):
        best_f1 = 0.0

        for epoch in range(1, epoch_num+1):
            self.model.train()

            total_train_losses = []
            total_train_preds = []
            total_train_targs = []

            for batch in tqdm(self.train_loader):
                x, y, lens = batch
                lens_sorted, idx = lens.sort(dim=0, descending=True)
                x_sorted = x[idx]
                y_sorted = y[idx]

                x, y, lens = x_sorted.to(device), y_sorted.to(device), lens_sorted.to(device)

                output = self.model(x, lens)  # (B, class_num)
                loss = self.criterion(output, y)  # ()

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_train_losses.append(loss.item())
                total_train_preds += torch.argmax(output, dim=-1).tolist()
                total_train_targs += y.tolist()

            train_loss = np.mean(total_train_losses)
            train_f1 = f1_score(total_train_targs, total_train_preds, average='weighted')

            print(f"########## Epoch: {epoch} ##########")
            print(f"Train loss: {train_loss} || Train f1 score: {train_f1}")

            valid_loss, valid_f1 = self.validate()

            if valid_f1 > best_f1:
                print("***** Current best model saved. *****")
                torch.save(self.model.state_dict(), f"{ckpt_dir}/best_model.pth")
                best_f1 = valid_f1

            print(f"Valid loss: {valid_loss} || Valid f1 score: {valid_f1} || Best f1 score: {best_f1}")

            self.summary.add_scalar('loss/train_loss', train_loss, epoch)
            self.summary.add_scalar('loss/validation_loss', valid_loss, epoch)
            self.summary.add_scalars('loss/loss_group', {'train': train_loss,
                                                   'validation': valid_loss}, epoch)

        self.summary.close()

    def validate(self):
        self.model.eval()
        total_valid_losses = []
        total_valid_preds = []
        total_valid_targs = []

        for batch in tqdm(self.valid_loader):
            x, y, lens = batch
            lens_sorted, idx = lens.sort(dim=0, descending=True)
            x_sorted = x[idx]
            y_sorted = y[idx]

            x, y, lens = x_sorted.to(device), y_sorted.to(device), lens_sorted.to(device)

            output = self.model(x, lens)  # (B, class_num)
            loss = self.criterion(output, y)  # ()

            total_valid_losses.append(loss.item())
            total_valid_preds += torch.argmax(output, dim=-1).tolist()
            total_valid_targs += y.tolist()

        valid_loss = np.mean(total_valid_losses)
        valid_f1 = f1_score(total_valid_targs, total_valid_preds, average='weighted')

        return valid_loss, valid_f1

    def test(self):
        self.model.eval()
        total_test_losses = []
        total_test_preds = []
        total_test_targs = []

        for batch in tqdm(self.test_loader):
            x, y, lens = batch
            lens_sorted, idx = lens.sort(dim=0, descending=True)
            x_sorted = x[idx]
            y_sorted = y[idx]

            x, y, lens = x_sorted.to(device), y_sorted.to(device), lens_sorted.to(device)

            output = self.model(x, lens)  # (B, class_num)
            loss = self.criterion(output, y)  # ()

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

    manager = Manager(args.mode, args.model_name)

    if args.mode == 'train':
        print("Training starts.")
        manager.train()
    elif args.mode == 'test':
        print("Testing starts.")
        manager.test()
