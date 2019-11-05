import torch
import torch.optim as optim
import torch.nn as nn
from lstm import LSTM
import os, argparse
from dataload import get_loader
import numpy as np
import sys
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
summary = SummaryWriter()


class Instructor:
    def __init__(self, args):
        self.args = args
        self.model = LSTM(self.args)

        if self.args.model_name is not "":
            self.model = self.model_load(self.model, self.args.model_name)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def make_dict(self):
        with open(os.path.join(self.args.data_path, self.args.vocab_name), 'r') as f:
            vocab_list = f.read().splitlines()

        vocab_to_int = {w: i+1 for i, w in enumerate(vocab_list)}
        int_to_vocab = {i+1: w for i, w in enumerate(vocab_list)}

        return vocab_to_int, int_to_vocab

    def train(self):
        print("Train starts.")
        recent_loss = sys.float_info.max
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_loader = get_loader("train.txt")
        dev_loader = get_loader("dev.txt")
        self.model.train()

        for e in range(1, self.args.epochs+1):
            epoch_train_loss = []
            epoch_valid_loss = []
            counter = 0

            for inputs, labels in tqdm(train_loader):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    print("CUDA is not available. This training operates with CPU.")
                counter += 1
                optimizer.zero_grad()
                pred = self.model(inputs)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()

                epoch_train_loss.append(loss.item())

                if counter % self.args.print_log == 0:
                    cur_loss = loss.item()
                    val_loss = self.validate(self.model, dev_loader, criterion)
                    epoch_valid_loss.append(val_loss)
                    self.model.train()

                    print(f"epoch: {e}, loss: {cur_loss}, val_loss: {val_loss}")

                    if cur_loss <= recent_loss:
                        now = datetime.datetime.today().strftime("%m%d_%H%M")
                        recent_loss = cur_loss
                        self.model_save(self.model, os.path.join(self.args.model_path, '{}_epoch{}.pth'.format(now, e)))

            summary.add_scalar('loss/train_loss', np.mean(epoch_train_loss), e)
            summary.add_scalar('loss/validation_loss', np.mean(epoch_valid_loss), e)
            summary.add_scalars('loss/loss_group', {'train': np.mean(epoch_train_loss),
                                                   'validation': np.mean(epoch_valid_loss)}, e)

        summary.close()

    def model_save(self, model, fname):
        with open(fname, 'wb') as f:
            torch.save(model.state_dict(), fname)

    def model_load(self, model, fname):
        return model.load_state_dict(torch.load(fname))

    def validate(self, dev_loader, criterion):
        print("Processing Validation...")
        val_losses = []
        self.model.eval()
        for dev_inputs, dev_labels in dev_loader:
            if torch.cuda.is_available():
                dev_inputs = dev_inputs.cuda()
                dev_labels = dev_labels.cuda()
            pred = self.model(dev_inputs)
            val_loss = criterion(pred, dev_labels)
            val_losses.append(val_loss.item())

        return np.mean(val_losses)

    def test(self):
        self.model.eval()
        test_loader = get_loader("test.txt")
        correct = 0
        total = len(test_loader)

        for inputs, labels in tqdm(test_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            pred = self.model(inputs)
            correct += (torch.argmax(pred, axis=1) == labels).sum().item()

        accuracy = correct / total * 100
        print(f"Accuracy: {accuracy:.2f}")


parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, help="train or test", type=bool)
parser.add_argument("--data_path", default="../data", help="path to data", type=str)
parser.add_argument("--model_path", default="../model", help="path to model", type=str)
parser.add_argument("--input_dim", default=195158, help="vocab size", type=int)
parser.add_argument("--emb_dim", default=400, help="embedding size", type=int)
parser.add_argument("--hid_dim", default=256, help="hidden size", type=int)
parser.add_argument("--output_dim", default=5, help="output size", type=int)
parser.add_argument("--num_layers", default=3, help="num of layers", type=int)
parser.add_argument("--drop_out", default=0.5, help="dropout value", type=float)
parser.add_argument("--vocab_name", default="vocab.txt", help="name of vocab file", type=str)
parser.add_argument("--batch_size", default=32, help="batch size", type=int)
parser.add_argument("--seq_len", default=600, help="length of input seq", type=int)
parser.add_argument("--learning_rate", default=0.0005, help="learning rate", type=float)
parser.add_argument("--epochs", default=10, help="num of epochs", type=int)
parser.add_argument("--print_log", default=1000, help="period to print log and validation", type=int)
parser.add_argument("--model_name", default="", help="model to load", type=str)

args = parser.parse_args()
instr = Instructor(args)

if args.test:
    instr.test()
else:
    instr.train()