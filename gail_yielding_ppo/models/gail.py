import numpy as np
import torch
from torch.nn import Module
from models.nets import PolicyNetwork, ValueNetwork, Discriminator
from utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch
from torch.cuda import FloatTensor
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import pickle
import sys
sys.path.append('C:/Users/14487/python-book/follow_code/gail_yielding_ppo')
#sys.path.append('C:/Users/14487/python-book/yielding_imitation/my_code/gail_yielding_ppo/models')
import random
#from experts import load_expert

#loaded_data_list = load_expert()
import os
import glob
import pandas as pd
from tqdm import tqdm
from path import train_path
#r"C:\Users\14487\python-book\ring_freewaydata\select_400\tra_one"
class GAIL(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None,
        seed=42
    ) -> None:
        super().__init__()
        self.set_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config
        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)
        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)
        
    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()
        state = FloatTensor(state)
        distb = self.pi(state)
        
        action = distb.sample().detach().cpu().numpy()
        return action

    def train(self, env, render=False):
        lr = self.train_config["lr"]

        num_epochs = self.train_config["num_epochs"]
        minibatch_size = self.train_config["minibatch_size"]
        gamma_ = self.train_config["gamma"]
        lambda_ = self.train_config["lambda"]
        eps = self.train_config["epsilon"]
        c1 = self.train_config["vf_coeff"]
        c2 = self.train_config["entropy_coeff"]
        normalize_advantage = self.train_config["normalize_advantage"]
        opt_pi = torch.optim.Adam(self.pi.parameters(), lr)
        opt_v = torch.optim.Adam(self.v.parameters(), lr)
        total_params = sum(p.numel() for p in self.pi.parameters() if p.requires_grad)
        # print('self.pi.parameters()',total_params)
        opt_d = torch.optim.Adam(self.d.parameters())
        exp_rwd_iter = []
        exp_obs = []
        exp_acts = []
        
        folder_path = train_path#r"C:\Users\14487\python-book\ring_freewaydata\select_400\tra_one"
        file_list = glob.glob(os.path.join(folder_path, "*.csv"))  
        #print('file_list',file_list)
        ##################RL
        #print('stop')
        rwd_iter_means = []
        for i in tqdm(range(1023), desc="Processing"):
            #print('file_list',file_list[i])
            expert=pd.read_csv(file_list[i%341])###%900
            #print('expert',file_list[i%341])
            # rel_v=expert['v'].values
            # Distance=expert['Distance'].values
            # angle=expert['angle_radians'].values
            #exp_obs = torch.tensor(np.array([group[['x_v','y_v', 'Distance', 'angle_radians']].to_numpy())
            #expert=expert[expert['x_car']!=0]
            exp_obs = torch.tensor(
            np.array([
                group[['x_v','y_v','Distance','angle_radians']].to_numpy()#'v_x_car','v_y_car','x_v','y_v','rel_x','rel_y','x_v','y_v','Distance','angle_radians'
                for _, group in expert.groupby("frame")
            ])
            )
            # exp_obs[:,0]=torch.exp(exp_obs[:,0])
            # exp_obs[:,1]=torch.exp(exp_obs[:,1])
            #exp_obs[exp_obs== 0] = float('nan')
            exp_obs=exp_obs[0:exp_obs.shape[0]-1]
            #print('exp_obs1',exp_obs.shape)
            exp_obs=exp_obs.reshape(exp_obs.shape[0],exp_obs.shape[1]*exp_obs.shape[2])
            #print('exp_obs1',exp_obs.shape)

            mask = (exp_obs != 0).float()
            #print('mask',mask.shape)
            row_sum = torch.sum(exp_obs.abs() * mask, dim=1, keepdim=True) + 1e-8

            exp_obs_normalized = (exp_obs / row_sum) * mask
            #print('exp_obs_normalized',exp_obs_normalized.shape)
            exp_obs=exp_obs_normalized
            #print('exp_obs2',exp_obs_normalized)
            exp_acts=torch.tensor(expert[expert['x_car']!=0].groupby("frame")[["a_x_car", "a_y_car"]].first().to_numpy())
            exp_acts=exp_acts[0:exp_acts.shape[0]-1]


            rwd_iter = []
            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            #print('len',len(expert['frame'].unique()))
            while steps < len(expert['frame'].unique())-1:
                #print('stepppppppppppppppppppppppppppppppppppppppppppppppppp',steps)
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []

                t = 0
                done = False

                ob = env.reset()
                #data_frame=env.unwrapped.save_time_step()
                
                while not done and steps <len(expert['frame'].unique()):
                    #print('ddd',done,steps,len(expert['frame'].unique()))
                    act = self.act(ob)
                    #print('act',act)
                    ep_obs.append(ob)
                    obs.append(ob)
                    ep_acts.append(act)
                    acts.append(act)
                    ob, rwd, done, info = env.step(act)
                    #data_frame=pd.concat([data_frame,env.unwrapped.save_time_step()])
                    ep_rwds.append(rwd)
                    ep_gms.append(gamma_ ** t)
                    ep_lmbs.append(lambda_ ** t)
                    t += 1
                    steps += 1
                if done:
                    rwd_iter.append(np.sum(ep_rwds))
                #data_frame.to_csv('C:/Users/14487/python-book/ring_freewaydata/data_test/data_%d.csv' % (i))
                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
             
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)
                
                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)).squeeze().detach()
                #print('ep_costs',ep_costs.shape)#([100])
                ep_disc_costs = ep_gms * ep_costs
                ep_disc_rets = FloatTensor([sum(ep_disc_costs[i:]) for i in range(t)])
                #print('ep_disc_rets',ep_disc_rets.shape)#[100])
                ep_rets = ep_disc_rets / ep_gms
                #print('ep_rets',ep_rets.shape)#[100]
                rets.append(ep_rets)
                

                self.v.eval()
                #print('ep_obs',ep_obs.shape)
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))).detach()
                ep_deltas = ep_costs.unsqueeze(-1)+gamma_ * next_vals- curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum() for j in range(t)])
                #print('ep_advs',ep_advs.shape)#[100]
                advs.append(ep_advs)
                #print('len(advs)',len(advs))
                gms.append(ep_gms)
                #print('gms',len(gms))

            rwd_iter_means.append(np.mean(rwd_iter))
            #print("Iterations: {},   Reward Mean: {}".format(i + 1, np.mean(rwd_iter)))
            obs = FloatTensor(np.array(obs))
            #print('obs',obs)#([300, 3])
            acts = FloatTensor(np.array(acts))
            #print('acts',acts)#([300, 1])
            rets = torch.cat(rets)
            #print('rets',rets.shape)#([300])
            advs = torch.cat(advs)
            #print('advs',advs.shape)#([300])
            gms = torch.cat(gms)
            #print('gms',gms.shape)#([300])

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()
            self.pi.eval()
            old_log_pi = self.pi(obs).log_prob(acts).detach()
            #print('old_log_pi',old_log_pi.shape)
            
        
            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            print('exp_scores',exp_obs.shape,exp_acts.shape,exp_obs.shape)
            nov_scores = self.d.get_logits(obs, acts)
            print('nov_scores',obs.shape,acts.shape,nov_scores.shape)
            
            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores))
            print('loss',loss)
            loss.backward()
            opt_d.step()
            


            self.pi.train()
            self.v.train()
            max_steps = num_epochs * (len(expert['frame'].unique()) // minibatch_size)
            #print('max_steps',max_steps)

            for _ in range(max_steps):
                minibatch_indices = np.random.choice(
                    range(steps), minibatch_size, False
                )
                mb_obs = obs[minibatch_indices]
                #print('mb_obs',mb_obs.shape)
                mb_acts = acts[minibatch_indices]
                mb_advs = advs[minibatch_indices]
                mb_rets = rets[minibatch_indices]

                mb_distb = self.pi(mb_obs)
                #print('mb_distb',mb_distb)
                mb_log_pi = mb_distb.log_prob(mb_acts)
                mb_old_log_pi = old_log_pi[minibatch_indices]

                r = torch.exp(mb_log_pi - mb_old_log_pi)
                L_clip = torch.minimum(
                    r * mb_advs, torch.clip(r, 1 - eps, 1 + eps) * mb_advs
                )
                #print('L_clip',L_clip.shape)
                L_vf = (self.v(mb_obs).squeeze() - mb_rets) ** 2
                #print('L_vf',L_vf.shape)
                S = mb_distb.entropy()
                #print('S',S.shape)
                opt_pi.zero_grad()
                opt_v.zero_grad()
                
                
                
                loss = (-1) * (L_clip - c1 * L_vf + c2 * S).mean()
                #print(loss)
                loss.backward()
                opt_pi.step()
                opt_v.step()

        return rwd_iter_means
