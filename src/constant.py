import torch

# Parameters for training and modeling
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rates = (0.0001, 0.001)
batch_sizes = (16, 128)
seq_len = 512
d_w = 256
d_h = 256
drop_out_rate = 0.5
layer_num = 3
bidirectional = True
class_num = 5
epoch_num = 10
ckpt_dir = '../saved_model'

# Parameters for Bayesian Optimization
init_points = 2
n_iter = 8

# Path for tensorboard
summary_path = '../runs'