import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import pickle
import random
import sys
import os
import glob
import pandas as pd
pd.set_option('display.max_columns', None)  
# pd.set_option('display.max_rows', None)  
# pd.set_option('display.width', None)     
sys.path.append('C:/Users/14487/python-book/follow_code/gail_yielding_ppo_notrans')


from path import train_path
folder_path = train_path#r"C:\Users\14487\python-book\ring_freewaydata\select_400\tra_one"

file_list = glob.glob(os.path.join(folder_path, "*.csv"))  


from gym import data_process


class Safetyspacev1(gym.Env):

    def __init__(self):
        self.viewer = None
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=np.array([-38,0]),high=np.array([41,5]))
        self.seed()
        self.number=0
        self.current_id = None  
        self.cached_data = pd.DataFrame()  

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

        
    def step(self,a):
        #print('a',a)
        a[0] = np.clip(a[0],-6,6)
        a[1] = np.clip(a[1],-1.5,1.5)
        self.t=self.t+1

        frame=self.chosen_trajectory['frame'].unique()[self.t]
        #print('frame',frame)

        costs = 1
        #print('selv_x_car',self.time_step[self.time_step['v_x_car']!=0]['v_x_car'].values[0])
        #print('a',a)
        v_x_car=a[0]*0.4+self.time_step[self.time_step['x_car']!=0]['v_x_car'].values[0]#
        
        v_y_car=a[1]*0.4+self.time_step[self.time_step['x_car']!=0]['v_y_car'].values[0]
        #print('v_x_car',v_x_car)
        #print('_x_car',self.time_step[self.time_step['x_car']!=0]['x_car'].values[0])
        x_car=v_x_car*0.4+self.time_step[self.time_step['x_car']!=0]['x_car'].values[0]#
        y_car=v_y_car*0.4+self.time_step[self.time_step['x_car']!=0]['y_car'].values[0]#
        #print('x_car',x_car)
        
        self.time_step=self.chosen_trajectory[self.chosen_trajectory['frame']==frame]##
        #print('self.time_step',self.time_step)
        #self.time_step.to_csv('C:/Users/14487/python-book/e1.csv')
        #print('v_x_car',v_x_car)
        self.time_step['v_x_car'] = self.time_step['x_car'].apply(lambda x: v_x_car if x != 0 else x)#v_x_car
        self.time_step['v_y_car'] = self.time_step['x_car'].apply(lambda x: v_y_car if x != 0 else x)#v_y_car
        #print(self.time_step)

        self.time_step['x_car'] =self.time_step['x_car'].apply(lambda x: x_car if x != 0 else x)
        self.time_step['y_car'] =self.time_step['x_car'].apply(lambda x: y_car if x != 0 else x)
        
        self.time_step['a_x_car'] = a[0]#self.time_step['v_x_car'].apply(lambda x: v_x_car if x != 0 else x)
        self.time_step['a_y_car'] = a[1]#self.time_step['v_y_car'].apply(lambda x: v_y_car if x != 0 else x)
  

        self.time_step=data_process.stata_ca(self.time_step)

        self.state =self.time_step[['x_v','y_v','Distance','angle_radians']].values#'x_v','y_v','Distance','angle_radians'

      
        if self.t<len(self.chosen_trajectory['frame'].unique())-1:
            done=False
        else:
            done=True
        self.state=self.state.reshape(1,-1)

        mask = (self.state != 0).astype(np.float32)

        row_sum = np.sum(np.abs(self.state) * mask, axis=1, keepdims=True) + 1e-8

        self.state_normalized = (self.state / row_sum) * mask
 

        return self.state_normalized.squeeze(), -costs, done, {}###self.state.reshape(-1), -costs, done, {}

    def reset(self):
        self.t=0
        
        #print('self.chosen_trajectory',self.chosen_trajectory)
        self.chosen_trajectory =pd.read_csv(file_list[self.number%1],usecols=lambda col: col != 'Unnamed: 0')#random.choice(loaded_data_list)%300
        #print('envir',file_list[self.number%341])
        self.i_d=int(file_list[self.number%1].split('_')[-1].split('.')[0])#%300
        #print('self.chosen_trajectory',file_list[self.number])
        #print('self.chosen_trajectory',self.chosen_trajectory)
        frame=self.chosen_trajectory['frame'].unique()[self.t]
        #print(frame)
        self.time_step=self.chosen_trajectory[self.chosen_trajectory['frame']==frame]##

        self.state =self.time_step[['x_v','y_v','Distance','angle_radians']].values#'v_x_car','v_y_car','x_v','y_v','rel_x','rel_y','x_v','y_v','Distance','angle_radians'
        # print('ss',self.state.shape)

        #self.state =np.array([random.randint(10,35),random.randint(8,13)])
        self.number=self.number+1
        #print(self.number)
        #print('self.state',self.state)
        #break
        self.state=self.state.reshape(1,-1)
        #print('self.state',self.state.shape)
        mask = (self.state != 0).astype(np.float32)

        row_sum = np.sum(np.abs(self.state) * mask, axis=1, keepdims=True) + 1e-8

        self.state_normalized = (self.state / row_sum) * mask  
        

        # print('self.state_normalized',self.state_normalized.squeeze().shape)
        # print('self.state_normalized',self.state_normalized)
        return self.state_normalized.squeeze()#####self.state.reshape(-1)_normalized

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer: self.viewer.close()
    def _save_or_cache_data(self):

        if self.current_id == self.i_d:
            self.cached_data = pd.concat([self.cached_data, self.time_step], ignore_index=True)
        else:

            if not self.cached_data.empty:
                output_dir = 'C:/Users/14487/python-book/ring_freewaydata/data_test/'
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, f'data_{self.i_d}.csv')
                self.cached_data.to_csv(file_path, index=False)

            self.current_id = self.i_d
            self.cached_data = self.time_step.copy()
    def save_time_step(self):
        return self.time_step
    def i_d(self):
        return self.i_d