from tqdm import tqdm
from constant import *
from torch.utils.data import Dataset

import torch
import matplotlib.pyplot as plt


def read_file(name):
    score2text = {}
    with open(f'{DATA_PATH}/{name}', 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        line = line.strip()
        text = line.split('\t')[-1]
        score = int(line.split('\t')[-3])-1

        if score not in score2text:
            score2text[score] = []

        score2text[score].append(text)

    return score2text


def read_vocab():
    word2idx = {'<pad>': 0, '<unk>': 1}
    with open(f'{DATA_PATH}/{vocab_name}', 'r') as f:
        lines = f.readlines()

    for line in lines:
        word = line.strip()
        word2idx[word] = len(word2idx)

    return word2idx


class CustomDataset(Dataset):
    def __init__(self, score2text, word2idx):
        scores = []
        texts = []
        lens = []
        for score, text_list in tqdm(score2text.items()):
            for text in text_list:
                scores.append(score)
                words = [word for word in text.split(' ')]
                words_idx = []
                for word in words:
                    if word in word2idx:
                        words_idx.append(word2idx[word])
                    else:
                        words_idx.append(word2idx['<unk>'])
                text_len = len(words_idx)

                if len(words_idx) > seq_len:
                    text_len = seq_len
                    words_idx = words_idx[:seq_len]
                else:
                    words_idx += ([word2idx['<pad>']] * (seq_len - len(words_idx)))

                texts.append(words_idx)
                lens.append(text_len)

        self.x = torch.LongTensor(texts)
        self.y = torch.LongTensor(scores)
        self.lens = torch.LongTensor(lens)

        assert self.x.shape[0] == self.y.shape[0], "The number of samples is not correct."
        assert self.x.shape == torch.Size([self.x.shape[0], seq_len]), "There is a sample with different length."

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.lens[idx]


def get_data():
    print("Making vocab dict...")
    word2idx = read_vocab()

    print("Reading data...")
    train_data = read_file(train_name)
    dev_data = read_file(dev_name)
    test_data = read_file(test_name)

    print("Making custom datasets...")
    train_set = CustomDataset(train_data, word2idx)
    dev_set = CustomDataset(dev_data, word2idx)
    test_set = CustomDataset(test_data, word2idx)

    return train_set, dev_set, test_set, word2idx


if __name__=='__main__':
    print("Reading data...")
    train_data = read_file(train_name)
    dev_data = read_file(dev_name)
    test_data = read_file(test_name)

    i = 0
    for score, text_list in train_data.items():
        for text in tqdm(text_list):
            words = [word for word in text.split(' ')]
            plt.scatter(i, len(words))
            i += 1

    for score, text_list in dev_data.items():
        for text in tqdm(text_list):
            words = [word for word in text.split(' ')]
            plt.scatter(i, len(words))
            i += 1

    for score, text_list in test_data.items():
        for text in tqdm(text_list):
            words = [word for word in text.split(' ')]
            plt.scatter(i, len(words))
            i += 1

    plt.show()
