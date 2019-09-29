import torch
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import get_data

batch_size = 32

def get_loader(name):
    review_list, score_list = get_data(name)
    dataset = TensorDataset(torch.from_numpy(review_list), torch.from_numpy(score_list))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataloader
