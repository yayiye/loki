# 代码用于离散环境的模型
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from datetime import datetime
import sys
import argparse

sys.path.append(os.getcwd())

MAIN_FOLDER = "data/train" + datetime.now().strftime("%Y%m%d-%H%M%S")

data_dir = './gcc_contrast/env_test/new-1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
# class PolicyNet(nn.Module):
#     def __init__(self, n_states, n_hiddens, n_actions):
#         super(PolicyNet, self).__init__()
#         self.fc1 = nn.Linear(n_states, n_hiddens)
#         self.fc2 = nn.Linear(n_hiddens, n_hiddens//2)
#         self.fc3 = nn.Linear(n_hiddens//2, n_actions)
#     def forward(self, x):
#         x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
#         x = F.relu(x)
#         x = self.fc2(x)  # [b, n_actions]
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
#         return x
 
# class ValueNet(nn.Module):
#     def __init__(self, n_states, n_hiddens):
#         super(ValueNet, self).__init__()
#         self.fc1 = nn.Linear(n_states, n_hiddens)
#         self.fc2 = nn.Linear(n_hiddens, n_hiddens//2)
#         self.fc3 = nn.Linear(n_hiddens//2, 1)
#     def forward(self, x):
#         x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
#         return x
 

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b, n_actions]
        x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
        return x

class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
        return x

class PPO:
    def __init__(self, n_states, n_actions, sw=None, gamma=0.95, lmbda=0.95, actor_lr=1e-4, critic_lr=1e-3, eps=0.2, n_hiddens=128):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)
 
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self._sw = sw
        self.epoch = 0
 
    # 动作选择
    def take_action(self, state):
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(state)
        # 创建以probs为标准的概率分布
        action_list = torch.distributions.Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_list.sample().item()
        return action
 
    # 训练
    def learn(self, transition_dict):

        self.epoch += 1

        #将数据集转换成单一的numpy数组
        states_np = np.array(transition_dict['states'], dtype=np.float32)
        # actions_np = np.array(transition_dict['actions'], dtype=np.float32)
        # rewards_np = np.array(transition_dict['rewards'], dtype=np.float32)
        next_states_np = np.array(transition_dict['next_states'], dtype=np.float32)
        # dones_np = np.array(transition_dict['dones'], dtype=np.float32)

        # 提取数据集
        states = torch.tensor(states_np, dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1,1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1,1)
        next_states = torch.tensor(next_states_np, dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1,1)
        
        
        # states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1,1)
        # rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1,1)
        # next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1,1)
 
        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1-dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value
 
        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []
 
        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage_np = np.array(advantage_list, dtype=np.float32)
        advantage = torch.tensor(advantage_np, dtype=torch.float).to(self.device)
 
        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
 
        # 一组数据训练 epochs 轮
        # for _ in range(self.epochs):
        # 每一轮更新一次策略网络预测的状态
        log_probs = torch.log(self.actor(states).gather(1, actions))
        # 新旧策略之间的比例
        ratio = torch.exp(log_probs - old_log_probs)
        # 近端策略优化裁剪目标函数公式的左侧项
        surr1 = ratio * advantage
        # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage

        # 策略网络的损失函数
        actor_loss = torch.mean(-torch.min(surr1, surr2))
        # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        # 梯度清0
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # 反向传播
        actor_loss.backward()
        critic_loss.backward()
        if self.epoch % 10 == 0:
            self._sw.add_scalar('loss/critic', critic_loss, self.epoch)
            self._sw.add_scalar('loss/actor', actor_loss, self.epoch)
        # 梯度更新
        self.actor_optimizer.step()
        self.critic_optimizer.step()


    def policy_state_dict(self):
        return self.actor.state_dict()

    def value_state_dict(self):
        return self.critic.state_dict()

    def load(self, path: str, map_location=None):
        if map_location == None:
            map_location = torch.device('cpu')
        self.actor.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict)

class PPOTrainer:
    
    def __init__(self, env, gamma, lmbda, actor_lr, critic_lr, eps, n_hiddens):
        self._obs_dim = env.get_obs_dim()
        self._action_dim = env.get_action_dim()
        self._sw = SummaryWriter(f'./{MAIN_FOLDER}/logs/trainer')
        self._agent = PPO(self._obs_dim, self._action_dim, self._sw, gamma, lmbda, actor_lr, critic_lr, eps, n_hiddens)
        self._env = env
        self._now_ep = 0
        self._step = 0
        self.main_folder = MAIN_FOLDER

    def train_one_episode(self):
        self._now_ep += 1
        self._env.reset()  # 一个reset就是读取一个视频
        done = False
        total_rew = 0
        count = 0
        states = None
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

        while not done:
            # TODO: repair the bug of the defination of the variable states
            if count == 30:
                states = self._env.get_obs()
                states = np.array(states).astype(np.float32)

            if count >= 30:
                actions = self._agent.take_action(states)
                next_states, rewards, done, info = self._env.step(actions, data_dir)
                next_states = np.array(states).astype(np.float32)
                
                print("rewards: ", rewards)
                print("actions: ", actions)

                transition_dict['states'].append(states)
                transition_dict['actions'].append(actions)
                transition_dict['next_states'].append(next_states)
                transition_dict['rewards'].append(rewards)
                transition_dict['dones'].append(done)

                self._step += 1
                total_rew += rewards
                states = next_states
            else:
                actions = 1
                self._env.step(actions, data_dir)

            if done:
                del states
            count += 1
        # 训练
        self._agent.learn(transition_dict)

        if self._now_ep % 200 == 0:
            self._sw.add_scalar(f'train_rew', total_rew, self._now_ep)
        return total_rew

    def test_one_episode(self):
        self._env.reset() 
        done = False
        total_rew = 0
        count = 0
        states = None
        while not done:
            # TODO: repair the bug of the defination of the variable states
            if count == 30:
                states = self._env.get_obs()
                states = np.array(states).astype(np.float32)
            if count >= 30:
                actions = self._agent.take_action(states)
                next_states, rewards, done, info = self._env.step(actions, data_dir)
                next_states = np.array(states).astype(np.float32)
                self._step += 1
                total_rew += rewards
                states = next_states
            else:
                actions = 1
                self._env.step(actions, data_dir)

            if done:
                del states
            count += 1

        self._sw.add_scalar(f'test_rew', total_rew, self._now_ep)
        return total_rew

    def save(self):
        path = f'./{MAIN_FOLDER}/models'
        if not os.path.exists(path):
            os.makedirs(path)
        save_pth = path + '/' + f'{self._now_ep}.pkl'
        torch.save(self._agent.policy_state_dict(), save_pth)