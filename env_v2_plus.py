import numpy as np
import sys
import logging

RANDOM_SEED = 42
MIN_VIDEO_BITRATE = 300  # kbps
MAX_VIDEO_BITRATE = 4000  # kbps
BIT_RATE_INTERVAL = 50
BITRATE_LIST = range(MIN_VIDEO_BITRATE, MAX_VIDEO_BITRATE, BIT_RATE_INTERVAL)

VIDEO_SIZE_FILE = 'D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/frame_size_content/frame_size_'
FRAME_RATE = 30
GOP = 30
NFILE = 1
DEFAULT_BITRATE = 700
AUDIO_BIT_RATE = 128000  # original=96000 128000
MILLISECONDS_IN_SECOND = 1000.0  # 1s=1000ms
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
BITRATE_LEVELS = len(BITRATE_LIST)
LINK_RTT = 30  # millisec
PACKET_SIZE = 1500  # bytes ##以太网中每帧最多是1500字节，最少是46字节数据
DELAY_LIMIT_MS = 5000
NOISE_LOW = 0.99
NOISE_HIGH = 1.01
VIDEO_CHUNK_LEN = MILLISECONDS_IN_SECOND / FRAME_RATE  # millisec, every time add this amount to
FEEDBACK_DURATION = 1000.0  # in milisec
FRAME_DURATION = int(FRAME_RATE * FEEDBACK_DURATION / 1000.0)
PACKET_IN_NACK = 16.0
NACK_MAXTIMES = 10  # s这个最大次数是否和rtt有关

WEIGHT_FACTOR = 0.9  # 0.3
PACKET_HEAD_SIZE = 40  # original=8byte, but it should be 20(IP)+8(UDP)+12(RTP)=40byes
# A send report is approximately 65 bytes inc CNAME, which is used to calculate rtt
SR_SIZE = 65
NACK_MAXTIME_MS = 1000.0  # ms
FRAME_NUM_LIMIT = 18000
EXPEXTED_BUFFER = 1
PACER_MAX_DELAY = 400
TIME_OUT_PACKET = 1500  # ms
TIME_OUT_FRAME = 5000  # ms
# TIME_OUT_PACKET = 0  # ms
# TIME_OUT_FRAME = 0  # ms

# s 改了下面的GetStatus的case1.case2的self.frame_delay[temp_index]->self.frame_delay[self.index_var + temp_index]
# s 发送端：根据当前的输入码率计算当前帧发送的数据包数(有时会加上需要重传的包)，在以当前码率发送延迟过大时会加速
# s 接收端：根据发送端发送的数据包和根据带宽算出来的数据包数的最小包数算出接收的包数
logging.basicConfig(filename='example.log', filemode='a', encoding='UTF-8')
sys.stdout = open('D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/gcc_contrast/env_print_30_1s.log', 'w', encoding='UTF-8')
# sys.stdout = open('./loki_for_upgrade_ppo/gcc_contrast/env_print_30_1s.log', 'w', encoding='UTF-8')
'''D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/gcc_contrast/env_print_30_1s.log'''

logging.basicConfig(filename='test.log', filemode='a', encoding='UTF-8')

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, all_file_names, train_tip, trace_idx, random_seed=RANDOM_SEED):
        self.duration = FEEDBACK_DURATION / FRAME_RATE / MILLISECONDS_IN_SECOND  # s1/60, 读取带宽的步长
        assert len(all_cooked_time) == len(all_cooked_bw)
        '''这里trace_idx做了个修改'''
        np.random.seed(random_seed)  # s生成随机数种子
        self.all_cooked_time = all_cooked_time  # s 存储读取的所有文件的时间戳, 二维的数组
        self.all_cooked_bw = all_cooked_bw  # s 存储读取的所有文件的带宽信息，二维的数组
        self.all_file_names = all_file_names  # s 存储读取的所有文件的名字。一维的
        self.train_tip = train_tip  # s 区分训练或测试的标识（训练或测试时读取文件的顺序不太一样）
        self.video_chunk_counter = 0  # s 视频块计数器，这里用的时候变成了在一个trace上经历的视频帧数的计数器
        self.trace_idx = trace_idx  # s 每次读取的trace的id, 训练时随机变换，测试时按顺序变换。
        self.cooked_time = self.all_cooked_time[self.trace_idx]  # s id为self.trace_idx的文件的时间戳
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]  # s id为self.trace_idx的文件的带宽信息
        self.file_name = self.all_file_names[self.trace_idx]  # s id为self.trace_idx的文件的名字
        # randomize the start point of the trace
        # note: trace file starts with time 0
        # self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.mahimahi_ptr = 1  # s 当前读取的trace的（时间戳和带宽）的行数
        self.last_mahimahi_bw = 100  # s 当前读取的trace的当前时间戳的上一历史时刻的带宽
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]  # s 类似上条
        self.video_size = [[]] * NFILE  # in bytes
        self.send_frame = np.zeros(0)
        self.send_nack = np.zeros(0)  # s 发送端待发送的NACK消息，待确定
        self.frame_capture_time = np.zeros(0)  # s 发送端捕获每一帧的时间,步长为1/FRAME_RATE, 当前帧收到之后，这一帧的捕获时间会删除
        self.recv_frame = np.zeros(0)  # s 在AddFrame函数初始化的时候是当前帧不出意外收到的包数。每一帧传输结束之后，如果包都收到了，当前帧记录删除，否则记录当前帧没收到的包数
        self.recv_nack = np.zeros(0)
        self.nack_count = np.zeros(0)  # s 每一帧结束之后会清除, 应该是没发出去的包数
        self.preframedec = 0
        self.play_size = np.zeros(0)  # s 存入每一帧播放的总bit数，如果这一帧播放了，就会删除这一帧的信息
        self.temp_frame = np.zeros(0)  # s 在SimTrans开始初始化为self.recv_frame
        self.temp_nack = np.zeros(0)  # s 在SimTrans开始初始化为self.recv_nack
        self.src_frame = np.zeros(0)  # s 源标识符，但下面只在添加新帧时会压入当前帧的数据包数，无其余计算
        self.iframe = np.zeros(0)
        self.send_time_list = []  # s 记录每一帧各个数据包的发送时间，初始化为-1，中间会记录当前帧各个数据包的发送时间，然后接收到一个数据包，该数据包的记录就会被删除
        self.preRttMs = 50  # s 没用到
        self.curTimeMs = 0  # s 时钟,记录当前的时间，每次换视频时才会重新置零
        self.time = 0
        self.ifile = 0
        self.buffer_empty = 0.0
        self.buffer_size = 0.0
        self.time_temp = 0.0  # s 没变化，那就是每个视频的初始时间
        self.trans_time_ms = 0  # s 每一帧的传输时间，通过每一帧的数据包/1s内总的数据包数 * 1000得到
        self.packet_rtt = 10
        # s 上面这些是一整个视频里的参数
        for i in range(NFILE):  # i=0，NFILE=1
            self.video_size[i] = [[]] * BITRATE_LEVELS  ###BITRATE_LEVELS=74
            # self.vmaf[i] = [[]] * BITRATE_LEVELS
            for bitrate in range(BITRATE_LEVELS):
                self.video_size[i][bitrate] = []
                with open(VIDEO_SIZE_FILE + str(i) + '_' + str(BITRATE_LIST[bitrate])) as f:
                    ##按顺序打开frame_size_content文件夹下的一个个文件
                    iline = 0
                    for line in f:  # 在文件里面一行一行循环
                        if iline > FRAME_NUM_LIMIT:  # FRAME_NUM_LIMIT=18000>一个文件中数据的行数恒成立
                            break
                        self.video_size[i][bitrate].append(int(float(line.split()[0])))  ##取出文件中的第一列数据
                        iline += 1
        '''这里可以调整'''
        self.TOTAL_VIDEO_CHUNCK = 30 * FRAME_RATE  # s 在一条trace上经历的总的视频帧数
        '''下面的变量都是增加的'''
        self.sending_bitrate_kbps = DEFAULT_BITRATE  # 发送比特率
        self.end_of_session = False
        self.frame_num = 0
        '''这些状态值原来在get_video_chunk内部'''
        self.total_expect_packet = 0
        self.total_nack_sent_count = 0  # s 记录1s内发送的重传包的个数
        self.total_recv_packet = 0
        self.avg_received_bitrate_bps = 0.0
        self.avg_sending_bitrate_bps = 0.0  # s 1s时间内发送端实际发送的比特数(最后不一定全部接收)
        self.avg_played_video_size_bps = 0.0
        self.video_size_total = 0.0  # s 根据码率算出来的理论上应该播放的视频比特数，它不管丢包,重传这些
        self.buffer_empty = 0.0  # s 记录1s内卡顿的次数
        self.packet_rtt_list = []
        self.loss_fraction = 0.0  # 丢失部分
        self.avg_frame_delay = 10
        self.avg_rtt = 10
        self.frame_delay = np.zeros(len(self.send_frame))  # s 存储的每1s所有帧的延迟信息，该列表中间不会被清除
        self.index_var = 0
        self.last_send_packet = 0  # s 要记录上一帧发送的数据包数
        self.loss_list = []  # s 记录每一帧的丢包率，不考虑重传的包
        self.sum_packet = 0
        '''适应需要'''
        self.pack_group_delay = []
        self.x_time = []

    def get_video_chunk(self):  # s用sending_bitrate_kbps作为发送比特率，这里的发送比特率是与带宽匹配的，可理解为预测的带宽值。所以该比特率包括视频+音频信息。
        # with open("test.log", "a", encoding="utf-8") as f:
        #     f.write(f"trace_idx: {self.trace_idx}\n")
        #     f.write(f"self.file_name: {self.file_name}\n")
        #     f.write(f"sending_bitrate: {self.sending_bitrate_kbps}\n")

        # print("trace编号: ", self.trace_idx)
        # print("self.file_name:", self.file_name)
        # '''增加了这个，每一秒清空一次'''
        # print('实际发送码率！！！！！！！！！:', self.sending_bitrate_kbps)
        # with open("test.log", "a", encoding="utf-8") as f:
        #     f.write(f"file_name: {self.file_name}\n")
        if self.frame_num % FRAME_RATE == 0:
            self.total_expect_packet = 0
            self.total_nack_sent_count = 0  # s 记录1s内发送的重传包的个数
            self.total_recv_packet = 0
            self.avg_received_bitrate_bps = 0.0
            self.avg_received_bitrate_bps_1s = 0.0
            self.avg_sending_bitrate_bps = 0.0  # s 1s时间内发送端实际发送的比特数(最后不一定全部接收)
            self.avg_sending_bitrate_bps_1s = 0.0
            self.avg_played_video_size_bps = 0.0
            self.video_size_total = 0.0  # s 根据码率算出来的理论上应该播放的视频比特数，它不管丢包,重传这些
            self.buffer_empty = 0.0  # s 记录1s内卡顿的次数
            # self.packet_rtt_list = []
            self.loss_fraction = 0.0  # 丢失部分
            self.avg_frame_delay = 10
            self.avg_rtt = 10
            self.frame_delay = np.zeros(len(self.send_frame))  # s 存储的每1s所有帧的延迟信息，该列表中间不会被清除
            self.index_var = 0
            self.last_send_packet = 0  # s 要记录上一帧发送的数据包数
            # self.loss_list = []  # s 记录每一帧的丢包率，不考虑重传的包
            self.sum_packet = 0
            self.true_bandwidth_1s = 0
        # print("时间戳：", self.video_chunk_counter / FRAME_RATE)
        # s 2024.4.6
        # 计算考虑包头信息后的视频码率quality，每秒发送包数per_packets_second
        sending_bitrate_bps = self.sending_bitrate_kbps * 1000.0  # s 每1秒发送的总比特数。外面测试的GCC初始化为1000kbps,所以下面先以1000kbps算一下
        packets_per_second = np.ceil(float(sending_bitrate_bps / BITS_IN_BYTE) / PACKET_SIZE)  # s 预计每一秒发送的RTP包数，125
        overhead_bitrate_bps = packets_per_second * PACKET_HEAD_SIZE * BITS_IN_BYTE  # s 预计每一秒发送的包头的总比特数, 40000bit
        payload_bitrate_bps = sending_bitrate_bps - overhead_bitrate_bps  # s 每一秒预计的有效负荷（只有视频，音频接收带宽已经考虑了）的总比特数, 1460000bit
        video_bitrate_kbps = np.clip(payload_bitrate_bps / 1000.0, MIN_VIDEO_BITRATE, MAX_VIDEO_BITRATE)  # s 1460kbit
        # 存放视频信息的比特被限制在300kbps~4000kbps
        total_bandwidth_bit = 0  # s 根据带宽1s内可以接收的比特数，已经减去音频信息。（另外在发送端和接收端实际发送或预估的数据单位都是packet）
        # 开始循环模拟
        '''-------------------------开始循环模拟------------------------'''
        # print('\n')
        # print("**************************************************************************")
        # print("第几帧:", self.frame_num, "开始添加新的一帧")
        self.AddNewFrame(video_bitrate_kbps, self.frame_num % FRAME_RATE)  # s 973.12kbit

        # 发送端处理NACK消息
        assert self.frame_capture_time[-1] == self.curTimeMs
        # s 这里模拟的是定时处理NACK消息。>=的设计就是保证根据当前的网络状况(self.packet_rtt是变化的),发端能够收到收端发送的NACK消息。
        if self.curTimeMs - self.time_temp >= self.packet_rtt:
            # if self.playtime[-1] - self.time_temp >= self.temp_rtt and self.temp_rtt != 0:
            # print('rtt为：' + str(self.packet_rtt) + ',更新NACK，上一次更新时间为：' + str(self.time_temp) + '，当前时间为：' + str( self.curTimeMs))
            # print("发送端处理了一次NACK消息")
            # print("self.curTimeMs:", self.curTimeMs, "self.time_temp:", self.time_temp, "self.packet_rtt:", self.packet_rtt)
            self.HandleNack()
            self.time_temp = self.curTimeMs  # s 上一次处理NACK消息的时间
            # print("发送端处理完一次NACK消息")
            # print("self.time_temp:", self.time_temp)
        # 计算网络带宽大小bandwidth_recv_pkts（考虑音频数据）
        bandwidth_recv_pkts = self.CalBandwidth()  # s 接收端当前一帧时间内(1/FRAME_RATE s)平均的可接收数据包数, 已经减掉了音频信息
        # print("网络带宽大小bandwidth_recv_pkts:" + str(bandwidth_recv_pkts))
        '''注释掉'''
        # total_bandwidth_bit += bandwidth_recv_pkts * PACKET_SIZE * BITS_IN_BYTE  # s 每一帧累加一次，最后是1s内可接收的总比特数

        # 划定最大的数据范围
        # only retransmit a lost packet 10 times
        # s 是否就是丢了的包最多只重传10次
        # s 这里应该是在调整重传窗口大小，nack_count记录的是每一帧的NACK_count,当某一帧的NACK_count>10的时候，会变。
        # s 重传窗口的大小决定在添加重传包时，添加最近多少帧的重传包。下面JudgeNack()函数会用到
        # s near_retransmit_window != 0
        '''这里是不是要外部先定义一下才能放在内部调用????????'''
        near_retransmit_window = 0
        for m in range(len(self.nack_count)):
            # print("正在处理重传信息.......")
            # print("self.nack_count:", self.nack_count)
            if self.nack_count[-(m + 1)] > NACK_MAXTIMES:
                near_retransmit_window = m  # 最近重传窗口 #s 从self.nack_count后面遍历，遇到第一个NACK数大于10的帧，重传窗口就到这一帧
                # print("一次重传0--", "near_retransmit_window:", near_retransmit_window)
                assert (m != 0)  # s? self.nack_count[-1]>10不会出现
                break
            near_retransmit_window = len(self.nack_count)
            # print("一次重传1--", "near_retransmit_window:", near_retransmit_window)

        # print("near_retransmit_window:" + str(near_retransmit_window))

        # 根据评估码率判断是否发送Nack数据，本次可以发送的数据包数estimator_packets
        estimator_packets = np.ceil(packets_per_second * self.duration)  # s 按评估码率当前帧可以发送的数据包数，8.4->9
        # print("estimator_packets:", estimator_packets)
        # estimator_packets = bandwidth_recv_pkts
        # print("self.last_send_packet", self.last_send_packet)
        '''对应的就是这里的情况'''
        nack_flag, total_size = self.JudgeNack(estimator_packets, near_retransmit_window, self.last_send_packet)
        # print("判断完是否发送NACK了")
        # print("nack_flag:", nack_flag, "total_size:", total_size) #s nack_flag = TRUE仅表示当前帧可以和NACK消息一起发送，NACK消息可以为空

        # 根据数据量控制发送码率大小,发送码率决定了每次发送的大小
        # print("判断是否加速发送......")
        sending_bitrate_bps, send_packet = self.JudgePacing(packets_per_second, self.sending_bitrate_kbps, total_size)
        # print("判断是否加速完成")
        # print("sending_bitrate_bps:", sending_bitrate_bps, "send_packet:", send_packet)

        ##实际的发送码率，有重传帧时，实际发送码率会上调
        ## 参数是：每秒数据包(包括包头和视频信息)大小；网络传送比特率；
        ##返回值sending_bitrate_bps:如果规定的传输比特率不慢的话就不变，否则，以total_size为准增大比特率；send_packet为sending_bitrate_bps决定的
        # print("send_packet is",send_packet)
        # print("total_size is",total_size)
        true_send_packet = min(send_packet, total_size)
        # print("true_send_packet:", true_send_packet)
        ## total_size:根据最初应发的数据包数+重传数据包数；send_packet:根据实际发送码率算得的数据包数
        # print("true_send_packet is",true_send_packet)
        self.last_send_packet = true_send_packet

        # 计算得到发送数据量大小total_size，真实传输数据量大小real_received_pkt
        '''+=改='''
        self.avg_sending_bitrate_bps = true_send_packet * PACKET_SIZE * BITS_IN_BYTE  # s 要记录1s时间内实际发送的比特数
        self.avg_sending_bitrate_bps_1s += true_send_packet * PACKET_SIZE * BITS_IN_BYTE
        real_received_pkt = min(true_send_packet,
                                bandwidth_recv_pkts)  # s 当前这一帧接收端实际接收的数据包数，也就是实际上发送端和接收端都是以数据包为单位计算的。

        # loss_packet = LossPacket(real_received_pkt, 0.1)
        # print("丢包率为0.1，丢的包数：" + str(loss_packet))
        # real_received_pkt = real_received_pkt - loss_packet
        '''+=改='''
        self.avg_received_bitrate_bps = real_received_pkt * PACKET_SIZE * BITS_IN_BYTE  # s最终1秒内根据带宽可以接收的比特数
        self.avg_received_bitrate_bps_1s += real_received_pkt * PACKET_SIZE * BITS_IN_BYTE
        # print('bandwidth_recv_pkts:' + str(bandwidth_recv_pkts))
        # print('发送数据量大小true_send_packet:' + str(true_send_packet))
        # print('per_packets_second:' + str(per_packets_second))
        # print('真实传输数据量大小real_received_pkt:' + str(real_received_pkt))

        # 根据发送数据量大小total_size，真实传输数据量大小real_received_pkt，发送数据范围near_retransmit_window模拟接收与丢包
        # print("开始传输数据......")
        # s当前这一帧传输之后实际发送的数据包数，以及丢包率=实际接收的包数/实际发送的包数
        send_frame, loss = self.SimTrans(near_retransmit_window, total_size, real_received_pkt, nack_flag,
                                         true_send_packet)
        # print("模拟传输结束")
        # print("send_frame:", send_frame, "loss:", loss)

        # 标记发送时间
        self.AddSendTime(send_frame)

        # 计算状态值
        self.loss_list.append(loss)
        # s 如果当前帧有重传包的话，会影响这一帧的平均包延迟self.packet_rtt。重传的包的延迟会很大
        self.packet_rtt = self.GetStatus(real_received_pkt, bandwidth_recv_pkts, self.frame_num % FRAME_RATE, sending_bitrate_bps,
                                         send_packet)
        ##bandwidth_recv_pkts是根据cooked_trace文件计算出来的,看起来是重传的数据包的延迟时间
        self.packet_rtt_list.append(self.packet_rtt)
        self.avg_rtt = 15 / 16 * self.avg_rtt + 1 / 16 * self.packet_rtt
        '''-------------------------------------------------------------------------'''
        '''-----------------------------以上都是循环内的部分-----------------------------'''
        '''-------------------------------------------------------------------------'''
        # 数据平滑处理 PACKET_IN_NACK因为一个NACK请求包括16个包的请求信息
        avg_nack_sent_count = np.ceil(float(self.total_nack_sent_count) * np.random.uniform(0.8,
                                                                                            1.2) / PACKET_IN_NACK)
        # s 1s内发送NACK消息的次数，一次NACK消息包含16个没发送的数据包
        '''这里注释掉添加下面的这个'''
        # true_bandwidth = total_bandwidth_bit / 1000.0 / (FEEDBACK_DURATION / MILLISECONDS_IN_SECOND)
        true_bandwidth = bandwidth_recv_pkts * PACKET_SIZE * BITS_IN_BYTE * FRAME_RATE / 1000.0
        self.true_bandwidth_1s += bandwidth_recv_pkts * PACKET_SIZE * BITS_IN_BYTE / 1000.0
        assert self.total_recv_packet >= 0
        assert self.total_expect_packet >= 0
        # assert self.total_expect_packet - self.total_recv_packet >= 0

        # 计算丢包率 #s 不考虑重传包
        if self.total_expect_packet - self.total_recv_packet <= 0:
            packet_loss_rate = 0.0
        else:
            # s 1s内的总丢包率，不考虑重传包。如果这1s内带宽变化不大的话，每一帧丢包率的分母都是一样的，则packet_loss_rate=avg(self.loss_list)
            packet_loss_rate = float(self.total_expect_packet - self.total_recv_packet) / self.total_expect_packet

        self.avg_received_bitrate_bps *= np.random.uniform(NOISE_LOW, NOISE_HIGH)
        self.avg_sending_bitrate_bps *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # 计算帧延迟
        frame_delay = 0
        # print("np.count_nonzero(self.frame_delay):", np.count_nonzero(self.frame_delay))
        if np.count_nonzero(self.frame_delay) != 0:  ##该函数用于统计数组中非零元素的个数
            frame_delay = np.sum(self.frame_delay) / (np.count_nonzero(self.frame_delay))
            # s 最后的得到的是最后收到的那一帧的延迟
            ##如果要平均到每一帧，为什么不除以帧的总数  解答：有些没有完全接收到的帧没有算
        # frame_delay = Setmin(frame_delay, 10)  ###如果帧延迟<10，就设置为10
        self.avg_frame_delay = frame_delay
        '''改成+=1'''
        self.video_chunk_counter += 1

        '''增加的'''
        self.frame_num += 1

        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print("\n")
        # print("一次决策结束")
        # print("\n")
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        if self.video_chunk_counter + 1 >= self.TOTAL_VIDEO_CHUNCK:
            with open("test.log", "a", encoding="utf-8") as f:
                f.write(f"换视频了\n")
                
            # print("换视频了")
            # print(self.video_chunk_counter + FRAME_DURATION)
            self.end_of_session = True
            self.video_chunk_counter = 0
            # pick a random trace file ##相当于一个文件是一个视频
            if self.train_tip == 1:
                self.trace_idx = np.random.randint(len(self.all_cooked_time))  ##训练时随机
                with open("test.log", "a", encoding="utf-8") as f:
                    f.write(f"trace_idx = {self.trace_idx}\n")
            else:
                # self.trace_idx = self.trace_idx + 1  ##测试时再用
                with open("test.log", "a", encoding="utf-8") as f:
                    f.write(f"trace_idx = {self.trace_idx}\n")
                if self.trace_idx == len(self.all_cooked_time):
                    self.trace_idx = 0
            self.file_name = self.all_file_names[self.trace_idx]
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            # print(self.cooked_time) ##测试了一下白盒黑盒是不是用的同一个trace
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]
            # randomize the start point of the video
            # note: trace file starts with time 0
            if self.train_tip == 1:
                self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))  ##训练时随机
            else:
                self.mahimahi_ptr = 1  ##测试时从文件开头读取
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
            self.last_mahimahi_bw = 100
            self.send_frame = np.zeros(0)
            self.send_nack = np.zeros(0)
            self.frame_capture_time = np.zeros(0)
            self.recv_frame = np.zeros(0)
            self.recv_nack = np.zeros(0)
            self.nack_count = np.zeros(0)
            self.preframedec = 0
            self.play_size = np.zeros(0)
            self.temp_frame = np.zeros(0)
            self.temp_nack = np.zeros(0)
            self.src_frame = np.zeros(0)
            self.iframe = np.zeros(0)
            self.send_time_list = []  # s 记录的一个trace的每一帧再到每一个数据包的发送时间
            self.preRttMs = 50
            self.curTimeMs = 0
            self.time = 0
            self.ifile = np.random.randint(0, NFILE)
            self.buffer_size = 0.0
            self.time_temp = 0.0
            self.trans_time_ms = 0
            self.packet_rtt = 10

        '''修改'''
        if self.video_chunk_counter >= 30:
            if self.video_chunk_counter % FRAME_RATE == 0:
                # with open("test.log", "a", encoding="utf-8") as f:
                #     f.write(f"file_name: {self.file_name}\n")
                print(f"返回结果,{self.video_chunk_counter/FRAME_RATE}帧")
                print("file_name: ", self.file_name)
                print("trace_idx: ", self.trace_idx)
                print("rcv_bitrate: ", self.avg_received_bitrate_bps_1s / 1000.0)
                print("nack_count: ", avg_nack_sent_count)
                print("end_of: ", self.end_of_session)
                print("rebuffer: ", float(self.buffer_empty / FRAME_DURATION))
                print("send_bitrate: ", self.avg_sending_bitrate_bps / 1000.0)
                print("played_bitrate:", self.avg_played_video_size_bps / 1000.0)
                print("video_bitrate: ", self.video_size_total)
                print("rtt_list: ", self.packet_rtt_list[-30:])
                print("loss_list: ", self.loss_list[-30:])
                print("true_bandwidth: ", self.true_bandwidth_1s)
                print("pack_group_delay: ", self.pack_group_delay[-30:])
                print("x_time: ", self.x_time[-30:])
                print("\n")
            return self.file_name, \
                self.trace_idx, \
                self.buffer_size, \
                self.avg_received_bitrate_bps * FRAME_RATE / 1000.0, \
                avg_nack_sent_count, \
                self.end_of_session, \
                float(self.buffer_empty / FRAME_DURATION), \
                self.avg_sending_bitrate_bps * FRAME_RATE / 1000.0, \
                self.avg_played_video_size_bps / 1000.0, \
                self.video_size_total, \
                self.packet_rtt_list[-30:], \
                self.loss_list[-30:], \
                true_bandwidth, \
                self.pack_group_delay[-30:], \
                self.x_time[-30:]  # s 2024.4.6
        else:
            return self.file_name, \
                self.trace_idx, \
                self.buffer_size, \
                self.avg_received_bitrate_bps * FRAME_RATE / 1000.0, \
                avg_nack_sent_count, \
                self.end_of_session, \
                float(self.buffer_empty / FRAME_DURATION), \
                self.avg_sending_bitrate_bps * FRAME_RATE / 1000.0, \
                self.avg_played_video_size_bps / 1000.0, \
                self.video_size_total, \
                self.packet_rtt_list, \
                self.loss_list, \
                true_bandwidth, \
                self.pack_group_delay, \
                self.x_time  # s 2024.4.6

    ####rtt:ms; 比特率均为：kbps; 帧延迟：ms
    def recvpacket(self):
        self.test = 1
        pass

    def VideoByte2Packet(self, nbyte):  # s 用于计算数据包数，传入参数为有效负载的字节数，输出为去掉包头以后计算的数据包数
        npacket = int(np.ceil(float(nbyte) / (PACKET_SIZE - PACKET_HEAD_SIZE)))  # s nbyte/1460byte
        return npacket

    #  添加新帧的数据
    def AddNewFrame(self, quality, iframe):  # s (973.12kbit, 0)
        # current_frame_data_size = self.video_size[self.ifile][quality][
        #                               self.video_chunk_counter + iframe] * np.random.uniform(NOISE_LOW, NOISE_HIGH)
        ##np.random.uniform(0.85, 1.15)随机返回0.85与1.15之间的一个值      ###一共10帧，计算每一帧传输多少视频信息，单位是byte
        current_frame_data_size = quality * 1000.0 / FRAME_DURATION / BITS_IN_BYTE * np.random.uniform(0.85,
                                                                                                       1.15)  # s 当前帧(第iframe帧)的数据大小，18250byte*(0.85~1.15)
        # print("current_frame_data_size:", current_frame_data_size)

        current_frame_packet = self.VideoByte2Packet(current_frame_data_size)  # s 当前帧(第iframe帧)需要发几个数据包，12.5->13
        if (self.video_chunk_counter + iframe) % GOP == 0:  # s 判断当前帧是否为i帧，GOP = 30
            iframe_flag = 1
        else:
            iframe_flag = 0
        self.send_frame = np.append(self.send_frame, current_frame_packet)  # s 记录每一帧发送的数据包数，【9，】,发送出去的帧的信息会被清除掉
        # print("添新帧-self.send_frame:", self.send_frame )
        self.send_nack = np.append(self.send_nack, 0)
        # print("添新帧-self.send_nack:", self.send_nack )
        self.frame_capture_time = np.append(self.frame_capture_time, self.curTimeMs)  # s 播放之后会被清空, 当前时间戳以1/FRAME_RATE递增
        # print("添新帧-self.frame_capture_time:", self.frame_capture_time )
        self.frame_delay = np.append(self.frame_delay, 0)
        # print("添新帧-self.frame_delay:", self.frame_delay)
        self.iframe = np.append(self.iframe, iframe_flag)
        # print("添新帧-self.iframe:", self.iframe )
        self.recv_frame = np.append(self.recv_frame, current_frame_packet)
        # print("添新帧-self.recv_frame:", self.recv_frame )
        self.recv_nack = np.append(self.recv_nack, 0)
        # print("添新帧-self.recv_nack:", self.recv_nack )
        self.nack_count = np.append(self.nack_count, 0)
        # print("添新帧-self.nack_count:", self.nack_count )
        self.play_size = np.append(self.play_size,
                                   current_frame_packet * PACKET_SIZE * BITS_IN_BYTE)  # s 当前这一帧播放的总bit数，包括填充的
        # print("添新帧-self.play_size(bit):", self.play_size )
        # self.video_size_total += current_frame_data_size * 8
        self.video_size_total += current_frame_data_size * BITS_IN_BYTE / 1000.0  # s这里用了记录的是实际发送的视频总kbit数，看来上面不足一个数据包时是填充一些东西。
        # print("添新帧-self.video_size_total(bit,实际的(不包含凑整包的填充数据)):", self.video_size_total )
        self.src_frame = np.append(self.src_frame, current_frame_packet)
        # print("添新帧-self.src_frame:", self.src_frame )
        self.send_time_list.append(np.full_like(np.zeros(current_frame_packet), -1))  # s 初始化-1，现在还没有真正发送
        # print("添新帧-self.send_time_list:", self.send_time_list )
        # print('------------------------------------------')
        # print('self.send_frame:' + str(self.send_frame))
        # print("self.send_nack:" + str(self.send_nack))
        # print("self.playtime:" + str(self.playtime))
        # print("self.recvtime:" + str(self.recvtime))
        # print("self.iframe:" + str(self.iframe))
        # print("self.recv_frame:" + str(self.recv_frame))
        # print("self.recv_nack:" + str(self.recv_nack))
        # print("self.nack_count:" + str(self.nack_count))
        # print('------------------------------------------')

    # 发送端处理NACK消息(添加重传包)
    def HandleNack(self):
        assert (len(self.recv_frame) == len(self.frame_capture_time))
        # pkt_loss = 0
        # print("即将处理NACK消息..........")
        # print("self.recv_frame: ", self.recv_frame, "self.frame_capture_time: ", self.frame_capture_time, "self.send_frame:", self.send_frame)
        # print("self.send_nack:", self.send_nack, "self.recv_nack:", self.recv_nack, "self.nack_count:", self.nack_count)
        for i in range(len(self.recv_frame)):
            # s 此时self.recv_frame=【没完全接收到的帧的数据包数, ..., 当前帧不出意外应该接收的数据包数】
            if self.recv_frame[i] != self.send_frame[i]:
                # s 此时self.send_frame = 【没完全接收到的帧:0, ... , 当前帧预计发送出去数据包数】
                self.recv_nack[i] += self.recv_frame[i] - self.send_frame[i]
                self.recv_frame[i] = self.send_frame[i]
            else:
                pass
            if self.send_nack[i] != self.recv_nack[i]:
                self.send_nack[i] = self.recv_nack[i]
                self.nack_count[i] = self.nack_count[i] + 1
        # print("NACK消息处理完成..........")
        # print("self.recv_frame: ", self.recv_frame, "self.frame_capture_time: ", self.frame_capture_time, "self.send_frame:", self.send_frame)
        # print("self.send_nack:", self.send_nack, "self.recv_nack:", self.recv_nack, "self.nack_count:", self.nack_count)
        assert (len(self.recv_frame) == len(self.recv_nack))

    # 计算可用带宽
    def CalBandwidth(self):  ##并不是每次循环用到这个函数返回的带宽值都会改变，结果的一行不是trace文件60行数据计算得到的；
        time1 = 0  # s 当前位置上一个带宽采集点经历的时间
        time2 = self.duration  # s 1/10，当前位置下一个带宽采集点经历的时间
        print(len(self.cooked_bw))
        if self.mahimahi_ptr + 1 >= len(self.cooked_bw):  # s 当前的这个trace读取到了最后一行，就重新读取这个文件的第一行
            print(self.trace_idx)
            self.mahimahi_ptr = 1  # s 理解了这了为什么不初始化为0, 这样的话当需要重新返回第一行读取时，带宽的索引不会出界
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
            self.time = 0.0
        if self.cooked_time[self.mahimahi_ptr] + self.time >= self.cooked_time[
            self.mahimahi_ptr + 1]:  # s 一帧，即(1/FRAME_RATE)s采集一次当前带宽的值
            time2 = self.cooked_time[self.mahimahi_ptr] + self.time - self.cooked_time[self.mahimahi_ptr + 1]
            time1 = self.duration - time2
            self.mahimahi_ptr = self.mahimahi_ptr + 1
            self.time = time2
        self.time += self.duration
        # print(self.cooked_bw[self.mahimahi_ptr])
        if self.cooked_bw[self.mahimahi_ptr] < 0.01:
            self.cooked_bw[self.mahimahi_ptr] = 0.01

        bandwidth_bps = (time2 * self.cooked_bw[
            self.mahimahi_ptr] + time1 * self.last_mahimahi_bw) * B_IN_MB * np.random.uniform(NOISE_LOW, NOISE_HIGH) / (
                                time1 + time2)  # s 当前(1/FRAME_RATE)s的平均带宽*(0.99~1.01)
        self.last_mahimahi_bw = self.cooked_bw[self.mahimahi_ptr]

        bandwidth_recv_bytes = self.duration * (
                    bandwidth_bps - AUDIO_BIT_RATE) / BITS_IN_BYTE  # s 音频和视频是分开传输的，因此接收带宽这里要减去音频大小
        if bandwidth_recv_bytes < 0:
            bandwidth_recv_bytes = 0.1
        bandwidth_recv_pkts = np.ceil(float(bandwidth_recv_bytes) / PACKET_SIZE)  # s 以数据包为单位，这里竟然也向上取整了？s

        return bandwidth_recv_pkts

    # 判断是否发送NACK True发送 False不发送, 计算真实发送数据大小
    def JudgeNack(self, estimator_packets, near_retransmit_window, last_send_packet):
        # print("判断是否发送NACK.......")
        if last_send_packet <= estimator_packets:
            # print('nack+frame可以全部传输')
            # print("self.send_frame: ",self.send_frame, "self.send_nack:", self.send_nack)
            nack_flag = True  # s 当前帧发送了NACK消息
            total_size = np.sum(self.send_frame) + np.sum(self.send_nack[-near_retransmit_window:])
            # print("nack_flag:", nack_flag, "frame和需要重传的包一起")
            # print("total_size:", total_size)
            # print("send_frame is",np.sum(self.send_frame))
            # print("\033[0;31;47msend_nack is\033[0m",np.sum(self.send_nack[-near_retransmit_window:]))
            ####NACK的数值太大了，send_frame的数目太大了
        else:
            # print('nack+frame不能全部传输, 只传frame')
            # print("self.send_frame: ",self.send_frame, "self.send_nack:", self.send_nack)
            nack_flag = False  # s 当前帧没有发送NACK消息
            total_size = np.sum(self.send_frame)
            # print("nack_flag:", nack_flag)
            # print("total_size:", total_size)
        # nack_flag = True
        return nack_flag, total_size

    # 确定是否加速发送
    def JudgePacing(self, per_packets_second, sending_bitrate_kbps, total_size):
        ###127,1520,2~10
        # s 这里是传输延时吗？s
        self.trans_time_ms = total_size / per_packets_second * MILLISECONDS_IN_SECOND
        # print("self.trans_time_ms:", self.trans_time_ms)
        # if self.trans_time_ms >= MILLISECONDS_IN_SECOND / FRAME_RATE:
        if self.trans_time_ms >= 0.5 * PACER_MAX_DELAY:
            sending_bitrate_bps = total_size * PACKET_SIZE * BITS_IN_BYTE / (
                        0.5 * PACER_MAX_DELAY) * MILLISECONDS_IN_SECOND
            ### 2~20*1500*8/200*1000
            # print('发送延迟过大:' + str(self.trans_time_ms) + '，提高发送码率为:' + str(sending_bitrate_bps / 1000.0) + 'kbps')
        else:
            sending_bitrate_bps = sending_bitrate_kbps * 1000.0
            # print('发送延迟正常:' + str(self.trans_time_ms) + ', 发送码率为:' + str(sending_bitrate_bps / 1000.0) + 'kbps')
        send_packet = np.ceil(float(sending_bitrate_bps / BITS_IN_BYTE) / PACKET_SIZE * self.duration)
        return sending_bitrate_bps, send_packet

    # 标记发送时间
    def AddSendTime(self, true_send_packet):
        times = true_send_packet
        nframe = len(self.send_time_list)  # s 现在一个视频内累计传输的帧数
        # print("开始记录发送时间")
        # print("self.send_time_list:", self.send_time_list)
        # s 这一层循环的是现在时刻，前面记录的每一视频帧的数据
        for frame in range(nframe):
            # s 这一层循环的是每一帧记录的每一个数据包的信息
            for temp in range(len(self.send_time_list[frame])):
                # # print(len(self.send_time_list[frame]))
                if self.send_time_list[frame][temp] == -1:
                    self.send_time_list[frame][temp] = self.curTimeMs
                    times -= 1  # s 只有发送的数据包才会记录它的发送时间
                if times == 0:
                    break
            if times == 0:
                break
        # print("发送时间记录完毕")
        # print("self.send_time_list:", self.send_time_list)

    # 模拟传输
    def SimTrans(self, near_retransmit_window, total_size, real_received_pkt, nack_flag, true_send_packet):
        self.temp_frame = self.recv_frame.copy()
        self.temp_nack = self.recv_nack.copy()
        # print("self.temp_frame:", self.temp_frame, "self.temp_nack:", self.temp_nack)
        # s 下面在判断是否重传所有需要重传的数据包，待定
        if len(self.send_frame) <= near_retransmit_window:
            # print('全部考虑')
            # recv conditions of new frame packets and retransmitted frame packets
            if nack_flag:
                # print('nack+frame可以全部传输')
                sum_nack = np.sum(self.send_nack)  # s 需要重传的数据包的个数
                # print("self.send_nack:", self.send_nack)
            else:
                # print('nack+frame不能全部传输,只传frame')
                sum_nack = 0

            # recv_nack = min(sum_nack, real_received_pkt)
            # s? 下面用比例来算应该就是当再次出现丢包时，接收的包中重传包和新数据包的一个分配问题
            recv_nack = np.ceil(sum_nack / total_size * real_received_pkt)
            # print("recv_nack:", recv_nack)
            recv_frame = real_received_pkt - recv_nack  # s 实际当前帧的数据包数(不含重传包)
            # print("recv_frame:", recv_frame)
            send_nack = np.ceil(sum_nack / total_size * true_send_packet)
            # print("send_nack:", send_nack)
            send_frame = true_send_packet - send_nack  # s 当前帧发送的数据包数(不含重传包)
            # print("send_frame:", send_frame)
            # print("self.nack_count:", self.nack_count)
            self.send_nack, self.recv_nack, self.nack_count = recvpacket_real_nack(self.send_nack, self.recv_nack,
                                                                                   send_nack, recv_nack,
                                                                                   self.nack_count)
            self.send_frame, self.recv_frame = recvpacket_real(self.send_frame, self.recv_frame,
                                                               send_frame, recv_frame)
            self.total_nack_sent_count += sum_nack
            # s 处理完后，self.send_frame: [0:（上一帧需要重传的情况下：上一帧重传完了，都发出去了；否则，没有这一项），当前帧没发的数据包个数]
            # s 处理完后，self.recv_frame: [0/x:（上一帧需要重传的情况下：上一帧重传完了，都接收了；x是这一帧还没有传过来的数据包数；否则，没有这一项），当前帧没收到的数据包个数]
            # print("一些NACK消息处理完后")
            # print("self.send_frame:", self.send_frame, "self.recv_frame:", self.recv_frame)
            # print("self.send_nack:", self.send_nack, "self.recv_nack", self.recv_nack)
            # print("self.nack_count:", self.nack_count)
        else:
            # print('部分考虑：' + str(near_retransmit_window))
            if nack_flag:
                # print('nack+frame可以全部传输')
                sum_nack = np.sum(self.send_nack[-near_retransmit_window:])
            else:
                # print('nack+frame不能全部传输, 只传frame')
                sum_nack = 0
            # recv_nack = min(sum_nack, real_received_pkt)
            recv_nack = np.ceil(sum_nack / total_size * real_received_pkt)
            # print("recv_nack:", recv_nack)
            recv_frame = real_received_pkt - recv_nack
            # print("recv_frame:", recv_frame)
            send_nack = np.ceil(sum_nack / total_size * true_send_packet)
            # print("send_nack:", send_nack)
            send_frame = true_send_packet - send_nack
            # print("send_frame:", send_frame)
            self.send_nack[-near_retransmit_window:], self.recv_nack[
                                                      -near_retransmit_window:], self.nack_count = recvpacket_real_nack(
                self.send_nack[-near_retransmit_window:],
                self.recv_nack[-near_retransmit_window:], send_nack,
                recv_nack, self.nack_count)
            self.send_frame, self.recv_frame = recvpacket_real(self.send_frame, self.recv_frame,
                                                               send_frame, recv_frame)
            self.total_nack_sent_count += sum_nack

            # print("发送端和接收端一些信息更新后")
            # print("self.send_frame:", self.send_frame, "self.recv_frame:", self.recv_frame)
            # print("self.send_nack:", self.send_nack, "self.recv_nack", self.recv_nack)

        # 统计丢包信息的方案
        # 1、考虑所有包
        # self.total_expect_packet += total_size
        # self.total_recv_packet += real_received_pkt
        # 2、不考虑重传包
        self.total_expect_packet += true_send_packet - send_nack
        self.total_recv_packet += real_received_pkt - recv_nack
        assert self.total_expect_packet >= self.total_recv_packet
        # 3、不开启RTX方式
        # self.total_expect_packet += total_size - sum_nack
        # self.total_recv_packet += real_received_pkt
        # s 这里的判断条件有点奇怪，要确认一下send_nack记录的东西
        if true_send_packet == send_nack:
            loss = 0
        else:
            loss = 1 - (real_received_pkt - recv_nack) / (true_send_packet - send_nack)

        return send_frame, loss

    # 获取状态值
    def GetStatus(self, real_received_pkt, bandwidth_recv_pkts, iframe, sending_bitrate_bps, send_packet):
        # 更新接收时间
        # print("开始运行GetStatus.............")
        return_packet_rtt = 0
        return_packet_num = 0  # s 这一帧实际接收的数据包个数，包含重传包
        assert (len(self.send_time_list)) == len(self.recv_frame)
        # print("self.send_time_list:", self.send_time_list, "self.recv_frame:", self.recv_frame)
        # print("self.temp_frame:", self.temp_frame, "self.temp_nack:", self.temp_nack)

        try:
            for i in range(len(self.recv_frame)):
                assert (len(self.send_time_list[i])) == (self.temp_frame[i] + self.temp_nack[i])
        except(AssertionError):
            print(len(self.send_time_list), self.temp_frame, self.temp_nack)
        # s? 根据下面的计算：
        # s self.recv_frame, self.send_time_list, self.recv_nack是对应的，对应的为目前需要重传的帧和新的帧的数据
        # s self.recv_nack应该是对应每一帧收到的NACK的包数
        for temp_index in range(len(self.recv_frame)):
            # print("计算包延迟和帧延迟-case1-计算已经收到的几个包的延迟")
            # print("temp_index:", temp_index)
            assert (len(self.recv_frame) == len(self.recv_nack))
            # assert (len(self.recv_frame) == len(self.recvtime))
            # s self.temp_frame是模拟传输起始复制的self.recv_frame，它俩不等，说明有的包没接收到
            if self.recv_frame[temp_index] != self.temp_frame[temp_index]:
                return_packet_num += self.temp_frame[temp_index] - self.recv_frame[temp_index]
                for temp in range(int(self.temp_frame[temp_index] - self.recv_frame[temp_index])):
                    # print("temp:", temp)
                    # s 这个额外延迟的计算原理是什么？s  是传输延迟吗？s
                    extra_delay = np.random.uniform(0, self.duration * MILLISECONDS_IN_SECOND * min(1,
                                                                                                    send_packet / bandwidth_recv_pkts))
                    # print("extra_delay:", extra_delay)
                    # print("self.curTimeMs: ", self.curTimeMs, "self.send_time_list:", self.send_time_list)
                    # s 不出意外每一帧的几个数据包是同时发送的，self.send_time_list[temp_index][0]是当前计算的这一帧第一个数据包的发送时间
                    # s 理想情况交互时延迟是直接乘以2计算的，但实际情况，往，返两程应该有可能会不一样
                    return_packet_rtt += (extra_delay + self.curTimeMs - self.send_time_list[temp_index][0]) * 2
                    # print("return_packet_rtt:", return_packet_rtt)
                    # s 如果这样每次发送完就删掉该帧的第一个发送时间记录的话，应该就是默认不会乱序
                    self.send_time_list[temp_index] = np.delete(self.send_time_list[temp_index], 0, 0)
                    # print("一个包延迟计算完成--", "self.send_time_list:", self.send_time_list)
                    if len(self.send_time_list[temp_index]) == 0:
                        # print("计算帧延迟")
                        # s 帧延迟这里的赋值的位置有点怪?s, 这样的话最终得到的是收到的最后一帧的帧延迟

                        self.frame_delay[self.index_var + temp_index] = extra_delay + self.curTimeMs - \
                                                                        self.frame_capture_time[temp_index]
                        # s 2024.4.6
                        self.pack_group_delay.append(self.frame_delay[self.index_var + temp_index])
                        self.x_time.append(
                            self.frame_delay[self.index_var + temp_index] + self.frame_capture_time[temp_index])
                        # s 2024.4.6
                        # print("self.frame_delay:", self.frame_delay)

        for temp_index in range(len(self.recv_nack)):
            # print("计算包延迟和帧延迟-case2-收到的重传包的包延迟")
            # print("temp_index:", temp_index)
            assert (len(self.recv_frame) == len(self.recv_nack))
            # assert (len(self.recv_frame) == len(self.recvtime))
            # print("self.recv_nack", self.recv_nack, "  self.temp_nack", self.temp_nack)
            # s 这里是每丢16个包才会发送一个NACK,所以<16的时候丢的不算
            if self.recv_nack[temp_index] != self.temp_nack[temp_index]:
                return_packet_num += self.temp_nack[temp_index] - self.recv_nack[temp_index]
                for temp in range(int(self.temp_nack[temp_index] - self.recv_nack[temp_index])):
                    extra_delay = np.random.uniform(0, self.duration * MILLISECONDS_IN_SECOND * min(1,
                                                                                                    send_packet / bandwidth_recv_pkts))
                    # print("extra_delay:", extra_delay)
                    return_packet_rtt += (extra_delay + self.curTimeMs - self.send_time_list[temp_index][0]) * 2
                    # print("return_packet_rtt:", return_packet_rtt,"  self.curTimeMs:", self.curTimeMs, "  self.send_time_list:", self.send_time_list )
                    self.send_time_list[temp_index] = np.delete(self.send_time_list[temp_index], 0, 0)
                    # print("一个包延迟计算完成--", "self.send_time_list:", self.send_time_list)
                    if len(self.send_time_list[temp_index]) == 0:
                        # print("计算帧延迟")

                        self.frame_delay[self.index_var + temp_index] = extra_delay + self.curTimeMs - \
                                                                        self.frame_capture_time[temp_index]
                        # s 2024.4.6
                        self.pack_group_delay.append(self.frame_delay[self.index_var + temp_index])
                        self.x_time.append(
                            self.frame_delay[self.index_var + temp_index] + self.frame_capture_time[temp_index])
                        # s 2024.4.6
                        # print("self.frame_delay:", self.frame_delay)

        # DELAY_LIMIT_MS决定了多长时间内接收到的才算可以播放
        while len(self.recv_nack) > 0 and len(self.recv_frame) > 0:
            # s 下面这个判断条件索引为0的这一帧的数据包都收到了
            if self.recv_nack[0] == 0 and self.recv_frame[0] == 0:
                # print('多长时间内接收到的才算可以播放')
                if self.frame_delay[self.index_var] < DELAY_LIMIT_MS:
                    # print("self.frame_delay[self.index_var] < DELAY_LIMIT_MS:", self.index_var)
                    if self.iframe[0]:
                        # print('I帧收到了')
                        self.preframedec = 1
                        self.buffer_size += 1.0
                    elif self.preframedec:
                        # print('之前有I帧的P帧收到了')
                        self.buffer_size += 1.0
                    self.avg_played_video_size_bps += self.play_size[0]
                else:
                    # print('超时帧收到了')
                    self.preframedec = 0
                self.send_nack = np.delete(self.send_nack, 0, 0)
                self.send_frame = np.delete(self.send_frame, 0, 0)
                self.iframe = np.delete(self.iframe, 0, 0)
                self.frame_capture_time = np.delete(self.frame_capture_time, 0, 0)
                # self.recvtime = np.delete(self.recvtime, 0, 0)
                self.recv_frame = np.delete(self.recv_frame, 0, 0)
                self.recv_nack = np.delete(self.recv_nack, 0, 0)
                self.nack_count = np.delete(self.nack_count, 0, 0)
                self.play_size = np.delete(self.play_size, 0, 0)
                del self.send_time_list[0]
                self.index_var += 1
                # print("self.index_var:", self.index_var)
            else:
                # print("当前帧的数据包没完全收到")
                break
        # print('self.preframedec:' + str(self.preframedec))

        # new adding, retransmitting buffer maintains 1000ms
        # 超过1000ms的就不再发送了
        # s 下面这个1s决策一次的时候不会出现
        while len(self.send_nack) > 0 and len(self.send_frame) > 0:
            # print("计算包延迟和帧延迟-case3")
            # print("self.send_nack", self.send_nack, "self.send_frame", self.send_frame)
            # s 按照前面的初始化，我感觉下面的if条件不会出现?s
            if self.frame_capture_time[-1] - self.frame_capture_time[0] > NACK_MAXTIME_MS:
                # print('超过1000ms的就不再发送了')
                # print("self.frame_capture_time:", self.frame_capture_time)
                self.preframedec = 0
                if self.recv_frame[0] != 0 or self.recv_nack[0] != 0:
                    return_packet_num += self.recv_frame[0] + self.recv_nack[0]
                    for temp in range(int(self.recv_frame[0] + self.recv_nack[0])):
                        extra_delay = np.random.uniform(0, self.duration * MILLISECONDS_IN_SECOND * min(1,
                                                                                                        send_packet / bandwidth_recv_pkts))
                        return_packet_rtt += (extra_delay + self.curTimeMs - self.send_time_list[0][
                            0] + TIME_OUT_PACKET) * 2
                        self.frame_delay[self.index_var + 0] = extra_delay + self.curTimeMs - self.frame_capture_time[
                            0] + TIME_OUT_FRAME
                        # s 2024.4.6
                        self.pack_group_delay.append(self.frame_delay[self.index_var + 0])
                        self.x_time.append(
                            self.frame_delay[self.index_var + 0] - TIME_OUT_FRAME + self.frame_capture_time[0])
                        # s 2024.4.6
                        self.send_time_list[0] = np.delete(self.send_time_list[0], 0, 0)
                self.send_frame = np.delete(self.send_frame, 0, 0)
                self.send_nack = np.delete(self.send_nack, 0, 0)
                # self.recvtime = np.delete(self.recvtime, 0, 0)
                self.frame_capture_time = np.delete(self.frame_capture_time, 0, 0)
                self.iframe = np.delete(self.iframe, 0, 0)
                self.recv_frame = np.delete(self.recv_frame, 0, 0)
                self.recv_nack = np.delete(self.recv_nack, 0, 0)
                self.nack_count = np.delete(self.nack_count, 0, 0)
                self.play_size = np.delete(self.play_size, 0, 0)
                del self.send_time_list[0]
                self.index_var += 1
            else:
                # print("case3-break")
                break

        # print(self.video_chunk_counter)
        # print(i)
        if (self.video_chunk_counter + iframe >= EXPEXTED_BUFFER):
            if self.buffer_size > 6:
                # print('-------------3倍速播放-------------')
                self.buffer_size -= 3.0
            elif self.buffer_size > 3:
                # print('-------------2倍速播放-------------')
                self.buffer_size -= 2.0
            elif self.buffer_size > 0:
                # print('--------------正常播放-------------')
                self.buffer_size -= 1.0
            else:
                # print('---------------卡顿---------------')
                self.buffer_empty += 1.0

        self.curTimeMs += self.duration * MILLISECONDS_IN_SECOND
        # print('self.buffer_size:' + str(self.buffer_size))
        # print('self.buffer_empty:' + str(self.buffer_empty))
        # print('self.buffer_empty:' + str(self.buffer_empty))
        # self.iframe = (self.iframe + 1) % GOP

        # print('\n')
        # print("return_packet_rtt:", return_packet_rtt, "return_packet_num", return_packet_num)
        if return_packet_num != 0:
            return_rtt = return_packet_rtt / return_packet_num
        else:
            return_rtt = 0

        # assert self.index_var <= 30
        return return_rtt


def recvpacket(buffer_list, npacket):
    if np.sum(buffer_list) == 0:
        return buffer_list

    if npacket < 0:
        npacket = 0

    if npacket >= np.sum(buffer_list):
        new_buffer_list = np.zeros(len(buffer_list))
        return new_buffer_list

    buffer_list_cumsum = np.cumsum(buffer_list)  # s cumsum:计算列表的依次累加和。
    end_index = (buffer_list_cumsum >= npacket).argmax()
    new_buffer_list = np.zeros(len(buffer_list))
    new_buffer_list[end_index] = buffer_list_cumsum[end_index] - npacket

    index = end_index + 1
    while index < len(buffer_list):
        new_buffer_list[index] = buffer_list[index]
        index += 1

    return new_buffer_list  # s 感觉记录的是重传之后还剩下的数据包


def recvpacket_real(target_list, true_list, tar_npacket, true_npacket):
    if np.sum(true_list) == 0:
        return target_list, true_list
    # 需要根据target_list来确定传输的是哪几帧的数据包
    new_target = recvpacket(target_list, tar_npacket)
    temp = recvpacket(target_list, true_npacket)

    new_true = np.zeros(len(true_list))

    for i in range(len(true_list)):
        new_true[i] = true_list[i] - (target_list[i] - temp[i])

    return new_target, new_true


# s (发送端NACK列表, 接收端NACK列表, 没发出去的数据包数, 没接收到的数据包数, line556)
# s (发送端NACK列表, 接收端NACK列表, 没发出去的数据包数, 没接收到的数据包数, line556)
def recvpacket_real_nack(target_list, true_list, tar_npacket, true_npacket, nack_count):
    if np.sum(true_list) == 0:
        return target_list, true_list, nack_count
    # 需要根据target_list来确定传输的是哪几帧的数据包
    new_target = recvpacket(target_list, tar_npacket)
    temp = recvpacket(target_list, true_npacket)

    new_true = np.zeros(len(true_list))

    for i in range(len(true_list)):
        if target_list[i] - temp[i] != 0:
            new_true[i] = true_list[i] - (target_list[i] - temp[i])
            # nack_count[i] = nack_count[i] + 1
        else:
            new_true[i] = true_list[i]

    return new_target, new_true, nack_count


def Setmin(temp, min):
    if temp < min:
        temp = min
    return temp


def LossPacket(total_packet, rate):
    loss_packet = 0
    for i in range(int(total_packet)):
        random_num = np.random.randint(0, 10)
        if random_num < rate * 10:
            loss_packet += 1
        else:
            pass
    return loss_packet

if __name__ == '__main__':
    # print('测试')
    import load_trace

    np.random.seed(42)
    # TRAIN_TRACES = './traces/bandwidth_4/'  # 这个是什么？
    TRAIN_TRACES = 'D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/cooked_traces_train/'
    # TRAIN_TRACES = './traces/bandwidth_2/'
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
    # print('all_cooked_time: ' + str(all_cooked_time[0][0]))
    # print('all_cooked_bw: ' + str(all_cooked_bw[0][0]))
    net_env = Environment(all_cooked_time=all_cooked_time,
                          all_cooked_bw=all_cooked_bw,
                          all_file_names=all_file_names,
                          train_tip=0,
                          trace_idx=np.random.randint(len(all_cooked_time)),
                          random_seed=0)
    
    stalling_rate_list = []
    packet_loss_rate_list = []
    frame_delay_list = []
    avg_rtt_list = []
    received_bit_rate_list = []
    nack_sent_count_list = []
    buffer_size_list = []
    sending_bit_rate_list = []
    avg_played_video_size_list = []
    video_size_total_list = []

    for i in range(50):
        rtt, frame_delay, wait_time, buffer_size, packet_loss_rate, \
            received_bit_rate, nack_sent_count, end_of_video, stalling_rate, avg_sending_bitrate, avg_played_video_size_bps, video_size_total, rtt_list = \
            net_env.get_video_chunk(4500)
        stalling_rate_list.append(stalling_rate)
        packet_loss_rate_list.append(packet_loss_rate)
        frame_delay_list.append(frame_delay)
        avg_rtt_list.append(rtt)
        received_bit_rate_list.append(received_bit_rate)
        nack_sent_count_list.append(nack_sent_count)
        buffer_size_list.append(buffer_size)
        sending_bit_rate_list.append(avg_sending_bitrate)
        avg_played_video_size_list.append(avg_played_video_size_bps)
        video_size_total_list.append(video_size_total)
        # print('==========================================')

    # print('--------------------------------------')
    # print('stalling_rate:' + str(stalling_rate_list))
    # print('packet_loss_rate:' + str(packet_loss_rate_list))
    # print('frame_delay:' + str(frame_delay_list))
    # print('avg_rtt:' + str(avg_rtt_list))
    # print('avg_sending_bitrate:' + str(sending_bit_rate_list))
    # print('received_bit_rate:' + str(received_bit_rate_list))
    # print('nack_sent_count:' + str(nack_sent_count_list))
    # print('buffer_size:' + str(buffer_size_list))
    # print('avg_played_video_size:' + str(avg_played_video_size_list))
    # print('video_size_total:' + str(video_size_total_list))
    # print('\n')

    # print('--------------------------------------')
    # print('stalling_rate:' + str(np.mean(stalling_rate_list)))
    # print('packet_loss_rate:' + str(np.mean(packet_loss_rate_list)))
    # print('frame_delay:' + str(np.mean(frame_delay_list)))
    # print('rtt:' + str(np.mean(avg_rtt_list)))
    # print('avg_sending_bitrate:' + str(np.mean(sending_bit_rate_list)))
    # print('received_bit_rate:' + str(np.mean(received_bit_rate_list)))
    # print('nack_sent_count:' + str(np.mean(nack_sent_count_list)))
    # print('buffer_size:' + str(np.mean(buffer_size_list)))
    # print('avg_played_video_size:' + str(np.mean(avg_played_video_size_list)))
    # print('video_size_total:' + str(np.mean(video_size_total_list)))
    # print('\n')

    # print('--------------------------------------')
    # print(np.mean(stalling_rate_list))
    # print(np.mean(packet_loss_rate_list))
    # print(np.mean(frame_delay_list))
    # print(np.mean(avg_rtt_list))
    # print(np.mean(sending_bit_rate_list))
    # print(np.mean(received_bit_rate_list))
    # print(np.mean(nack_sent_count_list))
    # print(np.mean(buffer_size_list))
    # print(np.mean(avg_played_video_size_list))
    # print(np.mean(video_size_total_list))
    # print('\n')
#
#
# print(delay)
# print(wait_time)
# print(buffer_size)
# print(packet_loss_rate)
# print(received_bit_rate)
# print(nack_sent_count)
# print(end_of_video)
# print(rebuf_size)
