import torch

# Path or parameters for data
DATA_PATH = '../data'
vocab_name = 'wordlist.txt'
train_name = 'train.txt'
dev_name = 'dev.txt'
test_name = 'test.txt'

# Parameters for training and modeling
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 0.0005
batch_size = 128
seq_len = 512
d_w = 256
d_h = 256
drop_out_rate = 0.5
layer_num = 3
bidirectional = True
class_num = 5
epoch_num = 15
ckpt_dir = '../saved_model'

# Path for tensorboard
summary_path = '../runs'