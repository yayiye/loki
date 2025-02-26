1.初始化:
增加self.sending_bitrate_bps = 700
全部换上self.sending_bitrate_bps
frame_num改成self.frame_num

2.get_video_chunk函数内部: 
get_video_chunk改成1帧return一次
函数内添加的量全部移动到外部定义
删掉total_bandwidth_bit += bandwidth_recv_pkts * PACKET_SIZE * BITS_IN_BYTE  # s 每一帧累加一次，最后是1s内可接收的总比特数
self.avg_sending_bitrate_bps += true_send_packet * PACKET_SIZE * BITS_IN_BYTE  # +=改成=
self.avg_received_bitrate_bps += real_received_pkt * PACKET_SIZE * BITS_IN_BYTE 同理
self.AddNewFrame和self.GetStatus函数都frame_num%30
self.video_chunk_counter改成+=1



3.不要漏
self.end_of_session后面不要忘记设置一个True???
self.frame_num在视频结束后要清零


4.观察点
self.buffer_size在换视频的时候才清零


5.问题
self.frame_delay的处理，如果每一秒清空的话，新的一秒留下一些0，这样凑不齐30个数据






神经网络训练
1.使用td3连续动作空间，只计算两个slope的大小，设置好上下限
