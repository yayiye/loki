import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import load_trace
from pylab import *

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
import env_v2_plus as env  # we add this file
import gcc_plus

COOKED_TRACE_FCC = './loki_for_upgrade_ppo/cooked_traces_test_four/FCC/'
COOKED_TRACE_GHSDPA = './loki_for_upgrade_ppo/cooked_traces_test_four/GHSDPA/'
COOKED_TRACE_MMGC = './loki_for_upgrade_ppo/cooked_traces_test_four/MMGC/'
COOKED_TRACE_OBOE = './loki_for_upgrade_ppo/cooked_traces_test_four/OBOE/'

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
LOG_FILE = './loki_for_upgrade_ppo/gcc_contrast/env_test/'
LOG_FILE_RTT = './loki_for_upgrade_ppo/gcc_contrast/env_rtt_list/'
LOG_FILE_LOSS = './loki_for_upgrade_ppo/gcc_contrast/env_loss_list/'
NN_MODEL = None
test_tip = 2  # 1:FCC;  2:GHSDPA;   3:MMGC;   4:OBOE
DEFAULT_BITRATE = 300  # s kbps
MIN_DEFAULT_BITRATE = 300
MAX_DEFAULT_BITRATE = 4000
width1_log = 10
width2_log = 12


def main():
    if not os.path.exists(LOG_FILE):
        os.makedirs(LOG_FILE)
    if not os.path.exists(LOG_FILE_RTT):
        os.makedirs(LOG_FILE_RTT)
    if not os.path.exists(LOG_FILE_LOSS):
        os.makedirs(LOG_FILE_LOSS)

    if test_tip == 1:  ##选择训练哪一个数据集
        COOKED_TRACE_FOLDER = COOKED_TRACE_FCC
    elif test_tip == 2:
        COOKED_TRACE_FOLDER = COOKED_TRACE_GHSDPA
    elif test_tip == 3:
        COOKED_TRACE_FOLDER = COOKED_TRACE_MMGC
    else:
        COOKED_TRACE_FOLDER = COOKED_TRACE_OBOE

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(COOKED_TRACE_FOLDER)
    # print(len(all_cooked_time),len(all_cooked_bw),len(all_file_names)) ##长度是test文件的数量


    buffer_log = []
    rcv_bit_log = []
    nack_count_log = []
    rebuffer_log = []
    send_bit_log = []
    played_bit_log = []
    video_bit_log = []
    rtt_list_log = []
    loss_list_log = []
    true_bandwidth_log = []
    target_bit_log = []
    gamma_log = []
    # target_bitrate = DEFAULT_BITRATE
    gamma = 0
    state_rate = "increase"
    epoch = 0
    trace_count = 0
    test_count = 5  ##
    tmp = 0
    trace_len = 0  ##计算每个trace的长度，方便计算均值

    if test_tip == 1:
        mark0 = "fcc_test.log"
        mark1 = "FCC"
    elif test_tip == 2:
        mark0 = "hsdpa_test.log"
        mark1 = "HSDPA"
    elif test_tip == 3:
        mark0 = "mmgc_test.log"
        mark1 = "MMGC"
    else:
        mark0 = "oboe_test.log"
        mark1 = "OBOE"

    with open(LOG_FILE + str(test_count) + '-' + mark0, 'wb') as log_file:  # 以二进制打开文件，只能写
        for i in range(test_count):
            net_env = env.Environment(all_cooked_time=all_cooked_time,
                                      all_cooked_bw=all_cooked_bw,
                                      all_file_names=all_file_names,
                                      train_tip=0,
                                      trace_idx=i,
                                      random_seed=10)
            net_env.sending_bitrate_kbps = DEFAULT_BITRATE
            while True:
                target_bit_log.append(net_env.sending_bitrate_kbps)  # 当前时刻的目标比特率，即决策比特率
                gamma_log.append(gamma)

                # GCC,使用环境net_env
                file_name, trace_idx, buffer, rcv_bitrate, \
                    nack_count, end_of, rebuffer, send_bitrate, played_bitrate, \
                    video_bitrate, rtt_list, loss_list, true_bandwidth, pack_group_delay, x_time = net_env.get_video_chunk()

                buffer_log.append(buffer)
                rcv_bit_log.append(rcv_bitrate)
                nack_count_log.append(nack_count)
                rebuffer_log.append(rebuffer)
                send_bit_log.append(send_bitrate)
                played_bit_log.append(played_bitrate)
                video_bit_log.append(video_bitrate)
                rtt_list_log.append(rtt_list)
                loss_list_log.append(loss_list)
                true_bandwidth_log.append(true_bandwidth)
                target_bitrate_old = net_env.sending_bitrate_kbps
                gamma_old = gamma
                '''当有30轮以上数据才开始交互'''
                if tmp >= 30:
                    send_loss, send_delay, net_env.sending_bitrate_kbps, state_net, state_rate, gamma, gradient_rtt, len_di, len_pack_group_delay = gcc_plus.gcc_plus_model(
                        net_env.sending_bitrate_kbps, rcv_bitrate, pack_group_delay, loss_list, x_time, state_rate, gamma)
                '''设置上下限码率'''
                net_env.sending_bitrate_kbps = max(MIN_DEFAULT_BITRATE, net_env.sending_bitrate_kbps)
                net_env.sending_bitrate_kbps = min(MAX_DEFAULT_BITRATE, net_env.sending_bitrate_kbps)
                '''前30轮不变发送码率'''
                if tmp < 30:
                    net_env.sending_bitrate_kbps = DEFAULT_BITRATE
                if tmp == 0:
                    log_file.write(("%s\t%20s\t%s\t%d\t" %
                                    ("current_file_name:", file_name, "current_trace_id:", trace_idx) + '\n').encode())
                if tmp % 30 == 0 and tmp != 0:
                    log_file.write(("{:<s}\t" + ("{:<10s}\t" * 10) + ("{:<12s}\t" * 3) + '\n').format
                                   ('tmp', 'target_b', 'send_b', 'send_loss', 'send_delay', \
                                    'rcv_b', 'bandwidth', 'gamma', 'grad_delay', 'max_delay', \
                                    'len_di', 'state_rate', 'state_net', 'lost_list').encode())
                    log_file.flush()
                if tmp >= 30:
                    log_file.write(("{:<d}\t" + ("{:<10.2f}\t" * 6) + ("{:<10.6f}\t" * 3) + "{:<10d}\t" + (
                                "{:<12s}\t" * 2)+ ("{:<10.6f}\t" * 1) + '\n').format
                                   (tmp, target_bitrate_old, send_bitrate, send_loss, send_delay, rcv_bitrate, \
                                    true_bandwidth, gamma_old, gradient_rtt, np.max(pack_group_delay), len_di, \
                                    state_rate, state_net, np.mean(loss_list)).encode())  # 这里记录的也是每一个时隙的决策值
                tmp = tmp + 1
                trace_len = trace_len + 1

                if end_of:  # env_v2里面get_vedio_chunk函数，表示换视频了
                    buffer_log = []
                    rcv_bit_log = []
                    nack_count_log = []
                    rebuffer_log = []
                    send_bit_log = []
                    played_bit_log = []
                    video_bit_log = []
                    rtt_list_log = []
                    loss_list_log = []
                    true_bandwidth_log = []
                    target_bit_log = []
                    gamma_log = []
                    net_env.sending_bitrate_kbps = DEFAULT_BITRATE
                    gamma = 0
                    state_rate = "increase"
                    trace_count = trace_count + 1  # 记录运行的trace的个数
                    trace_len = 0  # 记录一个trace的长度
                    epoch += 1
                    tmp = 0
                    if epoch % 10 == 0:
                        print("---------epoch %d--------" % epoch)
                    break


if __name__ == '__main__':
    main()
