import torch
import torch.optim as optim
import torch.nn as nn
from lstm import LSTM
import os
from dataload import get_loader
import numpy as np
import sys
import datetime
from tqdm import tqdm

DATA_PATH = "../data"
MODEL_PATH = "../model"

input_dim = 195158
emb_dim = 400
hid_dim = 256
output_dim = 5
num_layers = 3

vocab_name = "vocab.txt"
epochs = 10
print_log = 200

def make_dict():
    with open(os.path.join(DATA_PATH, vocab_name), 'r') as f:
        vocab_list = f.read().splitlines()

    vocab_to_int = {w: i+1 for i, w in enumerate(vocab_list)}
    int_to_vocab = {i+1: w for i, w in enumerate(vocab_list)}

    return vocab_to_int, int_to_vocab


def train(model):
    recent_loss = sys.float_info.max
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    train_loader = get_loader("train.txt")
    dev_loader = get_loader("dev.txt")
    model.train()

    for e in range(1, epochs+1):
        counter = 1
        for inputs, labels in tqdm(train_loader):
            counter += 1
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            if counter % print_log == 0:
                cur_loss = loss.item()
                val_loss = validate(model, dev_loader, criterion)
                model.train()

                print(f"epoch: {e}, loss: {cur_loss}, val_loss: {val_loss}")

                if cur_loss <= recent_loss:
                    now = datetime.datetime.today().strftime("%m%d_%H%M")
                    recent_loss = cur_loss
                    model_save(model, os.path.join(MODEL_PATH, '{}_epoch{}.pth'.format(now, e)))


def model_save(model, fname):
    with open(fname, 'wb') as f:
        torch.save(model.state_dict(), fname)


def model_load(model, fname):
    return model.load_state_dict(torch.load(fname))


def validate(model, dev_loader, criterion):
    val_losses = []
    model.eval()
    for dev_inputs, dev_labels in dev_loader:
        pred = model(dev_inputs)
        val_loss = criterion(pred, dev_labels)
        val_losses.append(val_loss.item())

    return np.mean(val_losses)


def test(model):
    model.eval()
    test_loader = get_loader("test.txt")
    correct = 0
    total = len(test_loader)

    for inputs, labels in test_loader:
        pred = model(inputs)
        correct += (torch.argmax(pred, axis=1) == labels).sum().item()

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}")

if __name__=="__main__":
    model = LSTM(input_dim, emb_dim, hid_dim, output_dim, num_layers)
    train(model)