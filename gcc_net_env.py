import numpy as np
import time
import env_v2_plus as env
import gcc_plus
import random
import load_trace
import os

LOG_FILE = './gcc_contrast/env_test/'

COOKED_TRACE_FCC = './cooked_traces_test_four/FCC/'
COOKED_TRACE_GHSDPA = './cooked_traces_test_four/GHSDPA/'
COOKED_TRACE_MMGC = './cooked_traces_test_four/MMGC/'
COOKED_TRACE_OBOE = './cooked_traces_test_four/OBOE/'

DEFAULT_BITRATE = 300  # s kbps
MIN_DEFAULT_BITRATE = 300
MAX_DEFAULT_BITRATE = 4000

if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)

action_list = [0.008, 0.0087, 0.0094, 0.0101, 0.0108, 0.0115, 0.0122, 0.0129, 0.0136, 0.0143]


class Gccenv:
    def __init__(self):
        self.state_rate = None
        self.gamma = None
        self.tmp = None
        self.slope2 = None
        self.slope1 = None
        self.net_env = None
        self.trace_idx = -2
        self.package_group_list = None
        self.loss_list = None
        self.recv_bitrate_list = None
        self.jitter_list = None
        self.target_bitrate_list = None
        self.trace_queue = None  # 用于存储轨迹索引的队列

        self.reset()

    def prepare(self):

        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace('D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/cooked_traces_train/')
        # with open("test.log", "a", encoding="utf-8") as f:
        #     f.write(f"all_cooked_time: {len(all_cooked_time)}\n")

        # self.trace_idx = np.random.randint(len(all_cooked_time))
        self.trace_idx = self.trace_idx + 2

        self.net_env = env.Environment(all_cooked_time=all_cooked_time,
                                       all_cooked_bw=all_cooked_bw,
                                       all_file_names=all_file_names,
                                       train_tip=2,
                                       trace_idx=self.trace_idx,
                                       random_seed=10)
        

    def get_obs_dim(self):
        return 150

    def get_action_dim(self):
        return 10

    def reset(self):

        self.prepare()
        self.slope1 = 0.039
        self.slope2 = 0.0087
        self.tmp = 0
        self.net_env.sending_bitrate_kbps = DEFAULT_BITRATE
        self.gamma = 0
        self.state_rate = "increase"
        self.package_group_list = []
        self.loss_list = []
        self.recv_bitrate_list = []
        self.jitter_list = []
        self.target_bitrate_list = []

    def step(self, actions, data_dir):
        print(data_dir, flush=True)
        self.slope1 = 0.039
        self.slope2 = action_list[actions]
        done = False
        file_name, trace_idx, buffer, rcv_bitrate, \
            nack_count, end_of, rebuffer, send_bitrate, played_bitrate, \
            video_bitrate, rtt_list, loss_list, true_bandwidth, pack_group_delay, x_time = self.net_env.get_video_chunk()

        target_bitrate_old = self.net_env.sending_bitrate_kbps
        gamma_old = self.gamma



        self.package_group_list.append(np.max(pack_group_delay))
        self.loss_list.append(np.mean(loss_list))
        self.recv_bitrate_list.append(rcv_bitrate)
        self.target_bitrate_list.append(target_bitrate_old)
        if len(rtt_list) == 1:
            self.jitter_list.append(rtt_list[0])
        else:
            self.jitter_list.append(rtt_list[-1]-rtt_list[-2])


        '''当有30轮以上数据才开始交互'''
        if self.tmp >= 30:
            send_loss, send_delay, self.net_env.sending_bitrate_kbps, state_net, self.state_rate, self.gamma, \
            gradient_rtt, len_di, len_pack_group_delay = gcc_plus.gcc_plus_model(
                self.net_env.sending_bitrate_kbps, rcv_bitrate, pack_group_delay, loss_list, x_time, self.state_rate, self.gamma, self.slope1, self.slope2)
        '''设置上下限码率'''
        self.net_env.sending_bitrate_kbps = np.clip(self.net_env.sending_bitrate_kbps, MIN_DEFAULT_BITRATE, MAX_DEFAULT_BITRATE)
        '''前30轮不变发送码率'''
        if self.tmp < 30:
            self.net_env.sending_bitrate_kbps = DEFAULT_BITRATE

        # TODO: prepare the obs, rew
        obs = [None] * 5   # 创建一个长度为3的列表，每个元素初始化为None
        rew = 0
        flattened_obs = None
        if self.tmp >= 30:
            obs[0] = self.package_group_list[-30:]
            obs[1] = self.loss_list[-30:]
            obs[2] = [value/1000.0 for value in self.recv_bitrate_list[-30:]]
            obs[3] = [value/1000.0 for value in self.target_bitrate_list[-30:]]
            obs[4] = self.jitter_list[-30:]
            
            diff_list = [abs(self.target_bitrate_list[i+1] - self.target_bitrate_list[i])/1000.0 for i in range(-20, -1)]
            rew = obs[2][-1]
            rew = rew / 100.0
            flattened_obs = [element for sublist in obs for element in sublist]
        if end_of:
            done = True
        
        with open(data_dir, 'ab') as log_file:
            # print(file_name)
            if self.tmp == 0:
                log_file.write(("%s\t%20s\t%s\t%d\t" %
                                ("current_file_name:", file_name, "current_trace_id:", trace_idx) + '\n').encode())
                log_file.flush()
            if self.tmp % 30 == 0 and self.tmp != 0:
                log_file.write(("{:<s}\t" + ("{:<10s}\t" * 10) + ("{:<12s}\t" * 3) + '\n').format
                                ('tmp', 'target_b', 'send_b', 'send_loss', 'send_delay', \
                                'rcv_b', 'bandwidth', 'gamma', 'grad_delay', 'max_delay', \
                                'len_di', 'state_rate', 'state_net', 'lost_list').encode())
                log_file.flush()
            # with open("bitrate_check.log", "a", encoding="utf-8") as f:
            #     f.write(f"{self.net_env.sending_bitrate_kbps}     {send_bitrate}\n")
            if self.tmp >= 30:
                log_file.write(("{:<d}\t" + ("{:<10.2f}\t" * 6) + ("{:<10.6f}\t" * 3) + "{:<10d}\t" + (
                            "{:<12s}\t" * 2)+ ("{:<10.6f}\t" * 1) + '\n').format
                                (self.tmp, target_bitrate_old, send_bitrate, send_loss, send_delay, rcv_bitrate, \
                                true_bandwidth, gamma_old, gradient_rtt, np.max(pack_group_delay), len_di, \
                                self.state_rate, state_net, np.mean(loss_list)).encode())  # 这里记录的也是每一个时隙的决策值
                log_file.flush()


        self.tmp += 1
        return flattened_obs, rew, done, {}

    # # TODO: prepare get_rew
    # def get_rew(self, obs):
    #     reward = obs[2][-1] / 1000.0 * 5 - obs[1][-1] * 1 - obs[0][-1] / 1000.0 * 1
    #     return reward / 100.0

    def get_obs(self):
        obs = [None] * 5 
        obs[0] = self.package_group_list[-30:]
        obs[1] = self.loss_list[-30:]
        # obs[2] = self.recv_bitrate_list[-30:]           # 将每个元素除以1000
        obs[2] = [value/1000.0 for value in self.recv_bitrate_list[-30:]]
        # obs[3] = self.target_bitrate_list[-30:]
        obs[3] = [value/1000.0 for value in self.target_bitrate_list[-30:]]
        obs[4] = self.jitter_list[-30:]
        flattened_obs = [element for sublist in obs for element in sublist]
        return flattened_obs

# if __name__ == '__main__':
#     file_path = 'D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/cooked_traces_train/'
#     all_cooked_time = []
#     all_cooked_bw = []
#     all_file_names = []
#     all_cooked_time,all_cooked_bw,all_file_names = load_trace(file_path)
#     # print(len(all_file_names))