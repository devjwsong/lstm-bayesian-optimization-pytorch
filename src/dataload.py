import torch
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import get_data

def get_loader(name, args):
    review_list, score_list = get_data(name, args.seq_len)
    dataset = TensorDataset(torch.from_numpy(review_list), torch.from_numpy(score_list))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    return dataloader
