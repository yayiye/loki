import os
import logging
import numpy as np
import multiprocess as mp
import tensorflow as tf
__all__ = [tf]
import env_v2_plus as env #we add this file   ##要改env
import load_trace
import sys
import our_ppo as ppo
import gcc_plus
import libconcerto
import datetime
# from env_v2 import VIDEO_CHUNK_TIME
from torch.utils.tensorboard import SummaryWriter

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

Alpha=5.0  ##5节中的reward
Beta=0.5

Alpha_a = 5 ##线性reward
Alpha_b = 1
Alpha_c = 1
Alpha_d = 3

##黑盒GCC参数
LR_RATE = 1.2*1e-3  ##黑盒GCC的学习率
S_INFO_gcc=153
S_LEN_gcc=3

##PPO参数
S_INFO=15*4+15*4+15*4+1#loss_list,rtt_list,jitter_list,吞吐量
S_LEN = 6  # take how many chunks in the past
ACTOR_LR_RATE = 0.0005  
CRITIC_LR_RATE = 0.0015  
NUM_AGENTS = 4
TRAIN_SEQ_LEN = 600  # take as a train batch
TRAIN_EPOCH = 50000   
MODEL_SAVE_INTERVAL = 20
RANDOM_SEED = 42
RAND_RANGE = 10000

REWARD_INTERVAL=100 ##更新loss中的熵权重
decay_interval=10
low_bondary=0.1
entropy_weight_decay=0.95
INITIAL_ENTROPY_WEIGHT=1



M_IN_K = 1000.0
MILLISECONDS_IN_SECOND = 1000.0
FEEDBACK_DURATION = 1000.0  #in milisec

SUMMARY_DIR='./tensor_lokix/fusion/'
MODEL_DIR = './lokix/fusion_mmodels'
TRAIN_TRACES = './cooked_traces_train/'
LOG_FILE ='./lokix/fusion_log/'
checkpoint_path_gcc = './gcc_results/4.0models/'  ##
NN_MODEL = None

FACTOR_LIST=[0.85,0.89,0.96,0.98,0.99,1.01,1.02,1.03,1.04,1.05] ##黑盒GCC的
ACTION_BIT=[0.7,0.83,0.98,1.13,1.27,1.42,1.56,1.71,1.87,2.0]
A_DIM=10
DEFAULT_ACTION=2
EPS=0.2

  ###env_v2的比特率单位均为kbps
def stable_sigmoid(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig   ##可以改为1/2或1/3x，把函数拉宽，降低斜率

def black_gcc(actor, convey_bit, s_batch_gcc, bit_rate, convey_bit_old,loss_list,rtt_list,send_bitrate,rcv_bitrate):

    state_gcc = np.array(s_batch_gcc[-1], copy=True)  # 将list类型换成array类型，复制下数组 153*3,state改变s_batch不会随着改变
    state_gcc = np.roll(state_gcc, -1, axis=1)  # 按列左移，将状态计入历史
    rtt_jitter = []
    rtt_jitter.append(0)
    for i in range(len(rtt_list) - 1):
        rtt_jitter.append(rtt_list[i + 1] - rtt_list[i])  # rtt列表前后相邻值的差
    while len(rtt_jitter) > 60:
        rtt_jitter.pop(0)  # 取剩下的后面60个值
    # this should be S_INFO number of terms
    state_gcc[0:60, -1] = loss_list  # 第二项是-1，取得是最后一列
    state_gcc[60:120, -1] = rtt_jitter
    x = np.zeros(15)
    x[0] = send_bitrate / 1000.0
    state_gcc[120:135, -1] = x
    y = np.zeros(15)
    y[0] = rcv_bitrate / 1000.0
    state_gcc[135:150, -1] = y
    # 记录当前选择的动作
    state_gcc[150, -1] = convey_bit_old / 1000.0  ##历史时刻1

    # w(s) 加权的交叉熵惩罚 GCC预测值与网络环境预测值
    if convey_bit <= bit_rate:  # w(s)  黑盒决策<专家决策
        state_gcc[151, -1] = (bit_rate - convey_bit) * (bit_rate - convey_bit) / 1000000.0
    else:
        state_gcc[151, -1] = (bit_rate - convey_bit) * (bit_rate - convey_bit) / 1000000.0 + 8

    # Smoothing bitrate switch，历史时刻取3，平滑率比特函数,以及φ(t,k)整体没问题,用的都是黑盒GCC的
    state_gcc[152, -1] = 0.5 * abs(convey_bit / 1000.0 - 0.5 * state_gcc[150, 2] - 0.25 * state_gcc[150, 1] - 0.125 * state_gcc[150, 0])

    action_prob, action = actor.predict(
        np.reshape(state_gcc, (-1, S_INFO_gcc, S_LEN_gcc)))  ##预测下一个时刻的action

    convey_bit_old = convey_bit  ##历史时刻1的

    bit_rate = gcc.gcc_model(convey_bit, loss_list, rtt_list)

    action_vec = np.zeros(A_DIM, dtype=float)
    action_vec[action] = 1  ##记录黑盒的

    s_batch_gcc.append(state_gcc)
    del rtt_jitter[:]

    return bit_rate,  convey_bit_old, action_prob

def fusion_function(fg,fh):
    """
    if fg.argmax()<=4: ##论文中是4
        ffg=np.exp(20*fg)
    else:
        ffg=stable_sigmoid(fg)
    ffh=np.multiply(ffg,fh)
    action = ffh.argmax()
    bitrate=ACTION_BIT[ffh.argmax()]*1000
    return bitrate,ffh,action
    """
    ##新的融合方式
    # print("fh:", fh)
    a = fh[0,0:5]
    b = fh[0,5:]
    # print("a:",a,"b:",b)
    if fg.argmax()<=4:
        action = a.argmax()
    else:
        action = b.argmax() + 5     

    bitrate=ACTION_BIT[action]*1000   
    return bitrate,action  
     

def compute_reward(throughput, loss, delay, bitrate, last_bitrate):

    reward=Alpha*(throughput-loss)/(0.09*delay+9)-Beta*abs(bitrate-last_bitrate) 
    
    ## throughput：Mbps; delay:s; bitrate:Mbps
    return reward 

# def compute_reward(throughput, rebuffer, delay, bitrate, last_bitrate):

#     # reward=Alpha*(throughput-loss)/delay-Beta*abs(bitrate-last_bitrate) 
#     # new_reward
#     reward = Alpha_a*throughput - Alpha_b*rebuffer - Alpha_c*delay - Alpha_d*abs(bitrate-last_bitrate)
#     ##Loki 5式子(2)
#     ## throughput：Mbps; delay:s; bitrate:Mbps
#     return reward 

def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log2(x[i])
    return H

def entropy_w_decay(epoch, reward_sum, actor, best_reward, best_interval): ##新加的，要放在中心代理部分，actor会多一个参数
    # if epoch % REWARD_INTERVAL != 0:
    #     interval_reward = interval_reward + reward_sum  ##这部分放在外面

    cur_interval = epoch / REWARD_INTERVAL
    cur_entropy_weight = actor.beta.eval()  ##eval将字符串转换成有效的表达式
    print("Last entropy weight:" + str(format(cur_entropy_weight, '.2f'))) ##格式化输出字符串
    print("Current interval:" + str(format(cur_interval, '.0f')))
    reward_sum /= REWARD_INTERVAL
    # first interval or update
    if epoch == REWARD_INTERVAL or reward_sum > best_reward * 1.01:
        best_reward = reward_sum
        best_interval = cur_interval
    elif cur_interval - decay_interval >= best_interval and cur_entropy_weight > low_bondary:
        new_entropy_weight = max(cur_entropy_weight * entropy_weight_decay, low_bondary)
        actor.update_beta(new_entropy_weight)
        best_interval = cur_interval
        print("Update entropy weight:" + str(format(new_entropy_weight, '.2f')))
    else:
        print("Best interval:" + str(format(best_interval, '.0f')))
    reward_sum = 0

    return reward_sum, best_reward, best_interval




def build_summaries():   ##tensorboard图
    td_loss = tf.compat.v1.Variable(0.)
    tf.compat.v1.summary.scalar("adv_A", td_loss)
    eps_total_reward = tf.compat.v1.Variable(0.)  ##刚开始因为这个没有用compat.v1
    tf.compat.v1.summary.scalar("Reward", eps_total_reward)
    # eps_min_reward = tf.compat.v1.Variable(0.)  ##刚开始因为这个没有用compat.v1
    # tf.compat.v1.summary.scalar("Min_Reward", eps_min_reward)
    entropy = tf.compat.v1.Variable(0.)
    tf.compat.v1.summary.scalar("Entropy", entropy)

    summary_vars = [td_loss, eps_total_reward, entropy]
    summary_ops = tf.compat.v1.summary.merge_all()

    return summary_ops, summary_vars

def central_agent(net_params_queues, exp_queues,hyper_params_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + "/_central",
                        filemode='a',
                        level=logging.INFO)  ##用于输出运行日志

    with tf.compat.v1.Session(config=config) as sess:

        summary_ops, summary_vars = build_summaries()

        actor = ppo.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE,entropy_weight=INITIAL_ENTROPY_WEIGHT)
        critic = ppo.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_lokiplus = os.path.join(SUMMARY_DIR, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        writer = tf.compat.v1.summary.FileWriter(summary_lokiplus, sess.graph)  # training monitor
        sess.run(tf.compat.v1.global_variables_initializer())
        # print([v.name for v in tf.all_variables()])
        # writer = tf.compat.v1.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.compat.v1.train.Saver(max_to_keep=5)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        epoch = 0
        #s 这里用于继续之前训练的模型接着训练
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")
            parse = NN_MODEL[10:-5].split('_')
            epoch = int(parse[-1])  

        reward_sum = 0  ##新加的
        best_reward = 0
        best_interval = -1
        # max_reward, max_epoch = -10000., 0
        # tick_gap = 0
        # while True:  # assemble experiences from agents, compute the gradients
        actor_net_params = actor.get_network_params()
        critic_net_params = critic.get_network_params()
        actor.update_beta(INITIAL_ENTROPY_WEIGHT)
        for i in range(NUM_AGENTS):
            net_params_queues[i].put([actor_net_params, critic_net_params])
            hyper_params_queues[i].put(INITIAL_ENTROPY_WEIGHT)

        for epoch in range(epoch, TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            # print(epoch)
            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_v = 0.0
            total_adv_A = 0.0
            total_entropy = 0.0
            total_agents = 0.0
            min_reward = 0.0
            actor_gradient_batch = []
            critic_gradient_batch = []
            nreward = TRAIN_SEQ_LEN


            for i in range(NUM_AGENTS):
                s_batch, a_batch, p_batch, r_batch, terminal, info = exp_queues[i].get()
                  ##td_batch是adv(A)
                actor_gradient, critic_gradient, td_batch, v_batch = \
                    ppo.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0), ##以维度0来堆叠
                        a_batch=np.vstack(a_batch),  ##按垂直方向堆叠
                        p_batch=np.vstack(p_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)
                # r = actor.rr.eval()
                # print(r)

                # 网络权重层数
                nreward = min(nreward,len(actor_gradient))
                # print(nreward,'nreward')

                actor_gradient_batch.append(actor_gradient)
                # print(i)
                # print(actor_gradient)
                critic_gradient_batch.append(critic_gradient)
                if(np.mean(r_batch) < min_reward):
                    min_reward = np.mean(r_batch)
                total_reward += np.mean(r_batch)  ##4个agent的reward和
                total_adv_A += np.mean(td_batch)  ##4个agent的adv(A)和
                total_v += np.mean(v_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.mean(info['entropy'])
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            #mean_actor_gradient_batch = np.zeros(nreward)
            #mean_critic_gradient_batch
            
            actor_gradient_batch = np.divide(actor_gradient_batch , NUM_AGENTS)
            critic_gradient_batch = np.divide(critic_gradient_batch , NUM_AGENTS)
            # for j in range(1,nreward):
            for j in range(nreward):
                for i in range(1, NUM_AGENTS):
                    actor_gradient_batch[0][j] = np.add(actor_gradient_batch[0][j], actor_gradient_batch[i][j])
                    critic_gradient_batch[0][j] = np.add(critic_gradient_batch[0][j], critic_gradient_batch[i][j])

            mean_actor_gradient_batch = actor_gradient_batch[0]   ##这里是均值为何刚开始没有除以4,因为前面divide那里已经除过了
            mean_critic_gradient_batch = critic_gradient_batch[0]
            actor.apply_gradients(mean_actor_gradient_batch)
            critic.apply_gradients(mean_critic_gradient_batch)

            # log training information
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_adv_A = total_adv_A / total_agents
            avg_v=total_v / total_agents
            avg_entropy = total_entropy / total_agents   ##求4个agent的均值
            # print(avg_entropy, 'shang1')

            ##新加的,entropy_decay

            # if epoch % REWARD_INTERVAL != 0:
            #     reward_sum += avg_reward 
            # else:
            #     reward_sum,best_reward,best_interval =entropy_w_decay(epoch, reward_sum, actor, best_reward, best_interval)


            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            # entropy_weight = agents['crf'].actor.beta
            entropy_weight = actor.beta.eval()
            # print(entropy_weight)
            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                hyper_params_queues[i].put(entropy_weight)


            logging.info("Epoch:%06d\tadv_A:%6.5f\tAvg_reward:%8.2f\tAvg_v:%8.2f\tAvg_entropy:%7.6f\tentropy_weight:%7.6f"%\
                         (epoch,avg_adv_A,avg_reward,avg_v,avg_entropy,entropy_weight))
              ##min_reward是4个agent中最小的平均reward值

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_adv_A,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy

            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            if epoch % MODEL_SAVE_INTERVAL == 0:
                print ("---------epoch %d--------" % epoch)
                # if(epoch % 10000 == 0):
                #     maxreward = 0
                # # Save the neural net parameters to disk.
                save_path = saver.save(sess, MODEL_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                
                logging.info("a:%6d\tb:%6d\tactor_lr:%6.5f\tcritic_lr:%6.5f\tinitiall_entropy:%4.2f\tmin_entropy:%4.2f"%\
                         (Alpha,Beta,ACTOR_LR_RATE,CRITIC_LR_RATE,INITIAL_ENTROPY_WEIGHT,low_bondary))






def agent(agent_id, all_cooked_time, all_cooked_bw, all_file_names, net_params_queue, exp_queue, hyper_params_queue):
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                          all_cooked_bw=all_cooked_bw,
                          all_file_names=all_file_names,
                          train_tip=1,
                          random_seed=agent_id)  ##删掉了文件名这个参数
    ##黑盒GCC提取模型
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess_gcc = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    sess_gcc.run(tf.compat.v1.global_variables_initializer())
    actor_gcc = libconcerto.Network(sess_gcc, S_INFO=S_INFO_gcc, S_LEN=S_LEN_gcc, A_DIM=A_DIM, LR_RATE=LR_RATE)  #
    module_file_gcc = tf.compat.v1.train.latest_checkpoint(checkpoint_path_gcc)
    actor_gcc.load(module_file_gcc)  ##这里可以用是因为actor有load函数来加载模型
    tf.compat.v1.reset_default_graph()

    ##创建PPO会话窗口

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.compat.v1.global_variables_initializer())
    with open(LOG_FILE + "agent_" + str(agent_id), 'wb') as log_file:
        actor = ppo.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE,entropy_weight=INITIAL_ENTROPY_WEIGHT)
        critic = ppo.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)
        entropy_weight = hyper_params_queue.get()
        actor.update_beta(entropy_weight) ##熵权重更新函数

        action = DEFAULT_ACTION
        convey_bit=ACTION_BIT[action] * 1000.0 #  初始值保持和黑盒GCC一致
        convey_bit_old=convey_bit
        bit_rate = convey_bit ##黑盒GCC里面的白盒GCC

        action_vec = np.zeros(A_DIM)
        action_vec[action] = 1

        s_batch_gcc = [np.zeros((S_INFO_gcc, S_LEN_gcc,), dtype=float)]  # 1*153*3
        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        p_batch = [np.full(A_DIM, 1 / A_DIM)]
        r_batch = []
        entropy_record = []
        epoch=0

        time_stamp = 0
        # rand_times = 1
        while True:
            if epoch==TRAIN_EPOCH:
                break
            file_name, trace_idx, avg_rtt, frame_delay, buffer, avg_loss, rcv_bitrate, \
            nack_count, end_of, rebuffer, send_bitrate, played_bitrate, \
            video_bitrate, rtt_list, loss_list, true_bandwidth = net_env.get_video_chunk(convey_bit)

            ##played_bitrate：kbps
            time_stamp += 1 # in frame

            reward = compute_reward(rcv_bitrate/1000, np.average(loss_list), np.average(rtt_list), convey_bit/1000, convey_bit_old/1000)
            ##单位和缩放, 这里的convey_bit_old用的是上一时刻GCC记下的
            r_batch.append(reward)

            # last_crf = crf
            # last_fps = fps
            rtt_jitter = []
            rtt_jitter.append(0)
            for i in range(len(rtt_list) - 1):
                rtt_jitter.append(rtt_list[i + 1] - rtt_list[i])  # rtt列表前后相邻值的差
            while len(rtt_jitter) > 60:
                rtt_jitter.pop(0)  # 取剩下的后面60个值
            ##黑盒GCC做出动作
            bit_rate, convey_bit_old, action_prob_gcc = \
                black_gcc(actor_gcc, convey_bit, s_batch_gcc, bit_rate, convey_bit_old, loss_list, rtt_list,
                          send_bitrate, rcv_bitrate)

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)
            # dequeue history record
            state = np.roll(state, -1, axis=1)

            state[0:60, -1] = loss_list
            state[60:120, -1] = rtt_jitter
            state[120:180,-1] = rtt_list
            state[180,-1] = rcv_bitrate  ##论文中的接收吞吐量；吞吐率，用cooked_trace计算出来的带宽不是吞吐率，而且这个带宽实际是不能得到的

            action_prob_ppo, action_ppo = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            ##fusion_bit, ffh, action = fusion_function(action_prob_gcc, action_prob_ppo)
            ##新的融合方式
            fusion_bit, action = fusion_function(action_prob_gcc, action_prob_ppo)      

            log_file.write(("%07d\t%5.1f\t%5.1f\t%5.1f\t%5.1f\t%5d\t%5d\t%5.1f\t%5.1f\t%5.1f\t%5.1f\t%5.1f\t%5.1f\t"
                            % (time_stamp, convey_bit, send_bitrate, rcv_bitrate,played_bitrate,
                                action, action_ppo, true_bandwidth, np.average(loss_list), rebuffer,
                               np.average(rtt_list), frame_delay,
                               reward)).encode())  ##当前时刻的传送比特率
            log_file.flush()

            # convey_bit_old = convey_bit  ##黑盒GCC函数里面赋过值了
            convey_bit = fusion_bit  ##每一秒的决策值
            # log_file.write((" ".join(str(format(i, '.3f')) for i in ffh[0]) + '****').encode())
            # log_file.flush()
            log_file.write((" ".join(str(format(i, '.3f')) for i in action_prob_gcc[0]) + '****').encode())
            log_file.flush()
            log_file.write((" ".join(str(format(i, '.3f')) for i in action_prob_ppo[0]) + '\n').encode())
            log_file.flush()
            entropy_record.append(compute_entropy(action_prob_ppo[0]))
            del rtt_jitter[:]

            # report experience to the coordinator
            if end_of:
                exp_queue.put([s_batch[1:],  # ignore the first frame since we don't have the control over it
                               a_batch[1:],
                               p_batch[1:],
                               r_batch[1:],
                               end_of,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)
                entropy_weight = hyper_params_queue.get()
                actor.update_beta(entropy_weight)  ##熵权重更新函数

                # _ = actor.set_entropy_weight()
                # print(_)

                del s_batch[:]
                del a_batch[:]
                del p_batch[:]
                del r_batch[:]
                del entropy_record[:]
                del s_batch_gcc[:]  ##为了换视频后state复制的时候清除掉之前的视频信息
                epoch=epoch+1

                # log_file.write(("time stamp\tcrf\tfps\tvmaf\tgreed\tiframe\tSI\tTI\tbuffer\tstall\trcv_br\tdelay\tavg_dly\tloss\treward\taction prob\n").encode())

            # store the state and action into batches
            if end_of:
                action = DEFAULT_ACTION
                convey_bit=ACTION_BIT[action] * 1000.0 #Mbps->kbps 保持和黑盒GCC一致
                convey_bit_old=convey_bit
                bit_rate = convey_bit

                action_vec = np.zeros(A_DIM)
                action_vec[action]=1

                s_batch_gcc.append(np.zeros((S_INFO_gcc, S_LEN_gcc)))
                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                p_batch.append(np.full(A_DIM, 1 / A_DIM)) ##返回一个以1/A_DIM为填充值的1*A_DIM的数组,
                
                log_file.write(('\n').encode())  ##当前时刻的传送比特率
                log_file.flush()
                log_file.write(("%s\t"
                                % ("t_stamp/convey_bit/send_bit/rev_bit/played_bit/action/ppo_action/bandwidth/loss"
                                   "/rebuffer/avg_rtt/frame_delay/reward/action_prob_gcc/action_prob_ppo"
                                   )+'\n').encode())
                log_file.flush()
                log_file.write(("%8s\t%5d\t%8s\t%s\t%8s\t%d\t"
                                % ("epoch: ",epoch+1,"file_name:",file_name,"trace_idx:", trace_idx) + '\n').encode())
                log_file.flush()
            else:
                ##黑盒GCC的s_batch在函数里面已经添加过了
                s_batch.append(state)
                p_batch.append(action_prob_ppo[0])
                action_vec = np.zeros(A_DIM)
                action_vec[action_ppo] = 1
                #action_vec[action] = 1  ##以最终的动作尝试训练
                a_batch.append(action_vec)

def main():

    np.random.seed(RANDOM_SEED)

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    if not os.path.exists(LOG_FILE):
        os.makedirs(LOG_FILE)  
     # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    hyper_params_queues = [] ##新加的，熵权重

    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))
        hyper_params_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues,hyper_params_queues))
    coordinator.start()

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)  
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw, all_file_names,
                                       net_params_queues[i],
                                       exp_queues[i],
                                       hyper_params_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start() ##主进程和子进行都会正常开启运行，当只使用直到所有进程运行结束后，代码运行结束。
    os.system('chmod -R 777 ' + SUMMARY_DIR)  ##应该是设置SUMMARY_DIR文件可读，可写，可执行操作
    # wait unit training is done
    coordinator.join() ##join()方法可以在当前位置阻塞主进程，等执行join()的进程结束后再继续执行主进程的代码逻辑
    


if __name__ == '__main__':
    main()
