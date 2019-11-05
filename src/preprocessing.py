import os
from tqdm import tqdm
import numpy as np

DATA_PATH = "../data/nsc_encoded"

train_name = "train.txt"
dev_name = "dev.txt"
test_name = "test.txt"

seq_len = 600

def read_file(name):
    review_list = []
    score_list= []
    with open(os.path.join(DATA_PATH, name), 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        review = line.split('\t')[0]
        score = int(line.split('\t')[1])-1

        review = [token for token in review.split()]

        review_list.append(review)
        score_list.append(score)

    return review_list, score_list


def pad_and_truncate(review_list):
    features = np.zeros((len(review_list), seq_len),  dtype=int)

    for i, review in enumerate(tqdm(review_list)):
        this_len = len(review)

        if this_len <= seq_len:
            zeroes = list(np.zeros(seq_len-this_len, dtype=int))
            new = review + zeroes

        else:
            new = review[0:seq_len]

        features[i,:] = np.array(new)

    return features


def get_data(name):
    review_list, score_list = read_file(name)
    review_list = pad_and_truncate(review_list)
    score_list = np.array(score_list)

    return review_list, score_list