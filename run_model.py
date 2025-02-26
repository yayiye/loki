from ppo import PPO
from gcc_net_env import Gccenv
import numpy as np
import time
import torch
import json

# model_path = "D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/data/train20250109-112111/models/51.pkl"
# model_path = 'D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/data/train20250116-170409-61-2/models/51.pkl'
model_path = 'D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/data/train20250116-161729-70/models/1.pkl'


data_dir = "D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/gcc_contrast/env_test/evaluate-1"
# data_dir = './gcc_contrast/env_test/evaluate-1'

action_list = [0.008, 0.0087, 0.0094, 0.0101, 0.0108, 0.0115, 0.0122, 0.0129, 0.0136, 0.0143]

def main():
    env = Gccenv()
    ppoagent = PPO(env.get_obs_dim(), env.get_action_dim())
    ppoagent.load(model_path)

    done = False
    total_rew = 0
    states = None
    count = 0

    while not done:
        if count == 30:
            states = env.get_obs()
            states = np.array(states).astype(np.float32)
        if count >= 30:
            actions = ppoagent.take_action(states)
            next_states, rewards, done, info = env.step(actions, data_dir)
            next_states = np.array(states).astype(np.float32)
            total_rew += rewards
            states = next_states
            with open("action.json", "a+") as f:
                f.write(json.dumps(action_list[actions]))
                f.write('\n')
        else:
            actions = 3
            env.step(actions, data_dir)

        if done:
            del states

        count += 1

if __name__ == '__main__':
    main()
