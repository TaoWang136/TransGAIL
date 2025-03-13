import torch
import sys
sys.path.append("C:/Users/14487/python-book/yielding_imitation/follow_code/Transformer-Encoder")
from transformer_encoder.encoder import TransformerEncoder

d_model = 512
n_heads = 8
batch_size = 1
max_len = 17
d_ff = 2048
dropout = 0.1
n_layers = 6


#def test_encoder():
enc = TransformerEncoder(d_model, d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
x = torch.randn(max_len, d_model)



if x.dim()==2:
    x=x.unsqueeze(0)
print('x',x.shape)

out = enc(x, None)
print('out',out.shape)
#assert x.size() == out.size()
