import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding
from torch.distributions import Categorical, MultivariateNormal

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor
import sys
sys.path.append("C:/Users/14487/python-book/follow_code/Transformer-Encoder-gae")
from transformer_encoder.encoder import TransformerEncoder

d_model =4
n_heads =2
batch_size = 1
max_len = 17
d_ff =32
dropout = 0.2
n_layers =3


number_car=8
feature=4


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, discrete=True):
        #print('状态的维度',state_dim)
        
        super(PolicyNetwork, self).__init__()

        #self.input_dim = state_dim + 4  # 拼接状态展平后的维度和行掩码维度
        # self.net = nn.Sequential(
            # nn.Linear(state_dim,16),
            # nn.Tanh(),
            # nn.Linear(16,8),
            # nn.Tanh(),
            # nn.Linear(8,4),
            # nn.Tanh(),
            # nn.Linear(4, action_dim)
        # )
        
        self.enco=nn.Sequential(nn.Linear(feature,d_model))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.enc = TransformerEncoder(d_model, d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)

        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states):

        if states.dim()==1:
            states=states.unsqueeze(0)
        #print('states',states.shape)
        states=states.reshape(states.shape[0],number_car,feature)###8代表八个车，2代表两个特征。
        #print('statesnew',states.shape,states)
        mask = (states.sum(dim=-1) != 0).float()
        #print('mask',mask)
        #mask=mask.unsqueeze(-1)#.repeat(1, 1,4)
        #print('mask',mask.shape,mask)
        states_encod=self.enco(states)
        #print('states_encod',states_encod.shape,states_encod)
        #('states_encod',states_encod.shape,states_encod)
        mean=self.enc(states_encod, mask)
        #print('mean',mean.shape)
        # mean = self.net(states)
        if mean.shape[0] == 1:
            mean = mean.squeeze(0)  # 移除第一个维度
        else:
            mean = mean  # 保持原样
        #print('mean2',mean.shape)
        std = torch.exp(self.log_std)
        cov_mtx = torch.eye(self.action_dim) * (std ** 2)
        distb = MultivariateNormal(mean, cov_mtx)

        return distb



class ValueNetwork(Module):
    def __init__(self, state_dim) -> None:
        super().__init__()

        self.net = Sequential(
            Linear(state_dim,32),
            Tanh(),
            Linear(32,16),
            Tanh(),
            Linear(16,8),
            Tanh(),
            Linear(8, 1))

    def forward(self, states):
        # states_flat = states.reshape(states.shape[0],-1)  # [12] 
        # if states_flat.dim()==1:
            # states_flat=states_flat.unsqueeze(0)
        # states_encod=self.enco(states_flat)
        # value=self.enc(states_encod, None)
        return self.net(states)


class Discriminator(Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if self.discrete:
            self.act_emb = Embedding(
                action_dim, state_dim
            )
            self.net_in_dim = 2 * state_dim
        else:  
            self.net_in_dim = state_dim+ action_dim
            #print('self.net_in_dim',self.net_in_dim)
        self.net = Sequential(
            Linear(self.net_in_dim,64),
            Tanh(),
            Linear(64,32),
            Tanh(),
            Linear(32,16),
            Tanh(),
            Linear(16,1),
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        if self.discrete:
            actions = self.act_emb(actions.long())        
        states_flat = states.reshape(states.shape[0],-1)
        sa = torch.cat([states_flat, actions], dim=-1).float() 
        #print('sa',sa.shape)
        return self.net(sa)
        
        
class Expert(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)

    def get_networks(self):
        return [self.pi]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()

        return action