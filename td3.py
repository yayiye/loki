
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from datetime import datetime
import sys

sys.path.append(os.getcwd())


ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
# ACTOR_LR = 1e-5
# CRITIC_LR = 1e-4
TAU = 0.05
TARGET_STD = 0.5
DELAY = 2
GAMMA = 0.95 # 0.95
BATCH_SIZE = 128
START_UPDATE_SAMPLES = 200

MAIN_FOLDER = "data/train" + datetime.now().strftime("%Y%m%d-%H%M%S")

data_dir = './gcc_contrast/env_test/new-1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Actor(torch.nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc0 = torch.nn.Linear(obs_dim, 128)
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        a = torch.tanh(self.fc2(x))
        return a


class Critic(torch.nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc0 = torch.nn.Linear(obs_dim + action_dim, 256)
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        q = self.fc2(x)
        return q


class ReplayBuffer:

    def __init__(self, cap, state_dim, action_dim):
        self._states = np.zeros((cap, state_dim))
        self._actions = np.zeros((cap, action_dim))
        self._rewards = np.zeros((cap,))
        self._next_states = np.zeros((cap, state_dim))
        self._index = 0
        self._cap = cap
        self._is_full = False
        self._rnd = np.random.RandomState(19971023)

    def add(self, states, actions, rewards, next_states):
        self._states[self._index] = states
        self._actions[self._index] = actions
        self._rewards[self._index] = rewards
        self._next_states[self._index] = next_states

        self._index += 1
        if self._index == self._cap:
            self._is_full = True
            self._index = 0

    def sample(self, n):
        indices = self._rnd.randint(0, self._cap if self._is_full else self._index, (n,))
        s = self._states[indices]
        a = self._actions[indices]
        r = self._rewards[indices]
        s_ = self._next_states[indices]
        # s = torch.tensor(self._states[indices], dtype=torch.float32).to(device)
        # a = torch.tensor(self._actions[indices], dtype=torch.float32).to(device)
        # r = torch.tensor(self._rewards[indices], dtype=torch.float32).view(-1, 1).to(device)
        # s_ = torch.tensor(self._next_states[indices], dtype=torch.float32).to(device)
        return s, a, r, s_

    def n_samples(self):
        return self._cap if self._is_full else self._index


class TD3Agent:

    def __init__(self, obs_dim, act_dim, sw=None):
        self._actor = Actor(obs_dim, act_dim)
        self._critic = [Critic(obs_dim, act_dim) for _ in range(2)]
        self._target_actor = Actor(obs_dim, act_dim)
        self._target_critic = [Critic(obs_dim, act_dim) for _ in range(2)]

        self._target_actor.load_state_dict(self._actor.state_dict())
        for i in range(2):
            self._target_critic[i].load_state_dict(self._critic[i].state_dict())

        self._actor_opt = torch.optim.Adam(self._actor.parameters(), lr=ACTOR_LR)
        self._critic_opt = [
            torch.optim.Adam(self._critic[i].parameters(), lr=CRITIC_LR) for i in range(2)
        ]

        self._act_dim = act_dim
        self._obs_dim = obs_dim
        self._sw = sw
        self._step = 0


        '''### gpu'''
        self._actor.to(device)
        self._critic[0].to(device)
        self._critic[1].to(device)
        self._target_actor.to(device)
        self._target_critic[0].to(device)
        self._target_critic[1].to(device)

    def soft_upd(self):
        with torch.no_grad():
            for t, s in zip(self._target_actor.parameters(), self._actor.parameters()):
                t.copy_((1 - TAU) * t.data + TAU * s.data)
            for t, s in zip(self._target_critic[0].parameters(), self._critic[0].parameters()):
                t.copy_((1 - TAU) * t.data + TAU * s.data)
            for t, s in zip(self._target_critic[1].parameters(), self._critic[1].parameters()):
                t.copy_((1 - TAU) * t.data + TAU * s.data)

    def query_target_action(self, obs):
        o = torch.tensor(obs).to(device).float()
        with torch.no_grad():
            a = self._target_actor(o).to(device)
            a = a.detach().cpu().numpy()
        # TODO: 补全
        # target_noise = np.random.normal(0, TARGET_STD, a.shape)
        # return a + target_noise
        noise = np.random.normal(0, TARGET_STD)
        a[0] += noise
        a[0] = 0.004*a[0] + 0.012
        a[0] = a[0].clip(0.008, 0.016)
        return a

    def choose_action(self, obs):
        # o = torch.tensor(np.array(obs)).float()
        o = torch.tensor(obs).to(device).float()
        with torch.no_grad():
            a = self._actor(o).to(device)
            a = a.detach().cpu().numpy()
            # a[0] = np.clip(a[0], 0.0080, 0.0160)
            a[0] = 0.004*a[0] + 0.012
            a[0] = a[0].clip(0.008, 0.016)
            # '''gpu'''
            # a = a.detach().numpy()
        return a

    # TODO: please add the noise for the sake of exploration
    def choose_action_with_exploration(self, obs):
        a = self.choose_action(obs)
        noise = np.random.normal(0, TARGET_STD)
        a[0] += noise
        # a[0] = np.clip(a[0], 0.0080, 0.0160)
        a[0] = 0.004*a[0] + 0.012
        a[0] = a[0].clip(0.008, 0.016)
        return a

    def update(self, s, a, r, s_, a_):
        self._step += 1
        '''gpu上运行'''
        s_tensor = torch.tensor(s).float().to(device)
        a_tensor = torch.tensor(a).float().to(device)
        r_tensor = torch.tensor(r).float().view(-1, 1).to(device)
        next_s_tensor = torch.tensor(s_).float().to(device)
        next_a_tensor = torch.tensor(a_).float().to(device)

        if len(a_tensor.shape) == 1:
            a_tensor = a_tensor.view(-1, 1)
        if len(next_a_tensor.shape) == 1:
            next_a_tensor = next_a_tensor.view(-1, 1)

        self._actor_opt.zero_grad()
        self._critic_opt[0].zero_grad()
        self._critic_opt[1].zero_grad()

        # update critic
        next_sa_tensor = torch.cat([next_s_tensor, next_a_tensor], dim=1).to(device)
        with torch.no_grad():
            m = torch.min(self._target_critic[0](next_sa_tensor), self._target_critic[1](next_sa_tensor))
            target_q = r_tensor + GAMMA * m
        now_sa_tensor = torch.cat([s_tensor, a_tensor], dim=1).to(device)
        q_loss_log = [0, 0]
        for i in range(2):
            now_q = self._critic[i](now_sa_tensor)
            q_loss_fn = torch.nn.MSELoss()
            q_loss = q_loss_fn(now_q, target_q)
            self._critic_opt[i].zero_grad()
            q_loss.backward()
            self._critic_opt[i].step()
            # q_loss_log[i] = q_loss.detach().cpu().item()
            '''gpu'''
            q_loss_log[i] = q_loss.detach().item()

        # update actor
        a_loss_log = 0
        if self._step % DELAY == 0:
            new_a_tensor = self._actor(s_tensor).to(device)
            new_sa_tensor = torch.cat([s_tensor, new_a_tensor], dim=1).to(device)
            q = -self._critic[0](new_sa_tensor).mean()
            self._actor_opt.zero_grad()
            q.backward()
            self._actor_opt.step()
            # a_loss_log = q.detach().cpu().item()
            '''gpu'''
            a_loss_log = q.detach().item()
            self.soft_upd()

        if self._step % 500 == 0:
            self._sw.add_scalar('loss/critic_0', q_loss_log[0], self._step)
            self._sw.add_scalar('loss/critic_1', q_loss_log[1], self._step)
            self._sw.add_scalar('loss/actor', a_loss_log, self._step)

    def policy_state_dict(self):
        return self._actor.state_dict()

    def value_state_dict(self):
        return [self._critic[i].state_dict() for i in range(2)]

    def load(self, path: str, map_location=None):
        if map_location == None:
            map_location = torch.device('cpu')
        self._actor.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))

    def load_state_dict(self, state_dict):
        self._actor.load_state_dict(state_dict)


class TD3Trainer:

    def __init__(self, env):
        self._obs_dim = env.get_obs_dim()
        self._action_dim = env.get_action_dim()

        self._sw = SummaryWriter(f'./{MAIN_FOLDER}/logs/trainer')
        self._agent = TD3Agent(self._obs_dim, self._action_dim, self._sw)
        self._replay_buffer = ReplayBuffer(100000, self._obs_dim, self._action_dim)
        self._env = env
        self._now_ep = 0
        self._step = 0

    def train_one_episode(self):
        self._now_ep += 1
        # states = self._env.reset()  
        self._env.reset()  # 一个reset就是读取一个视频
        done = False
        total_rew = 0
        count = 0
        states = None
        while not done:
            # TODO: repair the bug of the defination of the variable states
            if count == 30:
                states = self._env.get_obs()
                # states = torch.tensor(self._env.get_obs()).to(device).float()
            if count >= 30:
                actions = self._agent.choose_action_with_exploration(states)
                next_states, rewards, done, info = self._env.step(actions, data_dir)
                # next_states = torch.tensor(next_states).to(device).float()
                self._step += 1

                self._replay_buffer.add(states, actions, rewards, next_states)

                if self._step % 20 == 0 and self._replay_buffer.n_samples() > START_UPDATE_SAMPLES:
                    for _ in range(20):
                        s, a, r, s_ = self._replay_buffer.sample(BATCH_SIZE)
                        a_ = self._agent.query_target_action(s_)
                        self._agent.update(s, a, r, s_, a_)

                total_rew += rewards
                states = next_states
            else:
                actions = [0.0087]
                self._env.step(actions, data_dir)

            if done:
                del states
            count += 1

        if self._now_ep % 200 == 0:
            self._sw.add_scalar(f'train_rew', total_rew, self._now_ep)
        return total_rew

    def test_one_episode(self):
        # states = self._env.reset()
        # done = False
        # total_rew = 0

        # while not done:
        #     # TODO: repair the bug of the defination of the variable states
        #     out_actions = self._agent.choose_action(states)
        #     actions = out_actions

        #     next_states, rewards, done, info = self._env.step(actions)

        #     total_rew += rewards
        #     states = next_states
        #     if done:
        #         del states

        # self._sw.add_scalar(f'test_rew', total_rew, self._now_ep)
        # return total_rew
        self._env.reset() 
        done = False
        total_rew = 0
        count = 0
        states = None
        while not done:
            # TODO: repair the bug of the defination of the variable states
            if count == 30:
                states = self._env.get_obs()
                # states = torch.tensor(self._env.get_obs()).to(device).float()
            if count >= 30:
                actions = self._agent.choose_action(states)
                next_states, rewards, done, info = self._env.step(actions, data_dir)
                # next_states = torch.tensor(next_states).to(device).float()
                self._step += 1
                total_rew += rewards
                states = next_states
            else:
                actions = [0.0087]
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
