import numpy as np

alpha = 0.9
# slope1 = 0.039
# slope2 = 0.0087
increase_slope = 1.35 #1.08
decrease_slope = 0.85
threshold_gain = 120  # 4 * 20

action_list = [0.008, 0.0087, 0.0094, 0.0101, 0.0108, 0.0115, 0.0122, 0.0129, 0.0136, 0.0143]
'''
frame_capture_time[]表示这个包组第一个包的发送时间，frame_delay+frame_capture_time 对应xi，frame_delay之间的差值对应di，注意把后面连续的0去掉
这里可以考虑打印的参数:ki,frame_delay除去0的长度，xi，。。。
'''

def y_modify(y_ini):
    
    acc = []
    y = []
    for i in range(len(y_ini)):
        if i == 0:
            acc.append(y_ini[0])
        else:
            acc.append(y_ini[i] + acc[i - 1])

    y.append(acc[0])
    for i in range(len(acc)-1):
        y.append(alpha*y[i]+(1-alpha)*acc[i+1])

    return y    

def loss_based(loss_list, send_rate):
    
    loss = np.mean(loss_list)
    # with open("test.log", "a", encoding="utf-8") as f:
    #     f.write(f"loss: {loss}\n") 
    if loss < 0.02:
        loss_send = 1.05 * send_rate
    elif loss < 0.1:
        loss_send = send_rate
    else:
        loss_send = send_rate * (1 - 0.5 * loss)

    return loss_send    

def network_judge(gradient_delay, gamma_ada, state_rate, slope1, slope2):

    if np.abs(gradient_delay) <= gamma_ada:
        gamma_new = gamma_ada + slope1*(np.abs(gradient_delay) - gamma_ada) * 1.0/30 # 1.0/30是什么意思
        state_net = "kBwNormal"
    else:
        gamma_new = gamma_ada + slope2*(np.abs(gradient_delay) - gamma_ada) * 1.0/30
        if gradient_delay > gamma_ada:
            state_net = "kBwOverusing"
        elif gradient_delay < -gamma_ada:
            state_net = "kBwUnderusing"

    if state_net == "kBwNormal":
        if state_rate == "decrease":
            state_rate_new = "hold"
        elif state_rate == "hold":
            state_rate_new = "increase"
        else:
            state_rate_new = "increase"
    elif state_net == "kBwOverusing":
        state_rate_new = "decrease"       
    else:
        state_rate_new = "hold"
        
    return state_rate_new, state_net, gamma_new    

def delay_based(gamma, send_rate, rcv_rate, pack_group_delay, x_time, state_rate, slope1, slope2):
    delay_time = []
    x = []
    y = []
    # s pack_group_delay = 0待定
    # s 计算di, 下面的计算默认了 len(pack_group_delay)>=2
    for i in range(len(pack_group_delay)-1):
        delay_time.append(pack_group_delay[i + 1] - pack_group_delay[i])

    # # 针对空列表的情况
    # if delay_time == []:
    #     return increase_slope*send_rate, "increase", gamma, 0, len(delay_time)

    x = x_time
    for i in range(len(x_time)):
        x[i] = x_time[i] - x_time[0]
    y = y_modify(delay_time)

    length = min(len(x), len(y))
    x = x[:length]
    y = y[:length]
    
    # s 拟合斜率
    coefficients = np.polyfit(x, y, 1)
    # s 得到拟合的斜率和截距
    gradient_delay = coefficients[0]
    # with open("test.log", "a", encoding="utf-8") as f:
    #     f.write(f"gradient: {gradient_delay}\n")
    gradient_delay_modify = gradient_delay * threshold_gain
    
    state_rate_new, state_net, gamma_new = network_judge(gradient_delay_modify, gamma, state_rate, slope1, slope2)
    if state_rate_new == "increase":
        sending_bitrate = increase_slope * send_rate
    elif state_rate_new == "hold":
        sending_bitrate = send_rate
    else:
        sending_bitrate = decrease_slope * rcv_rate
    
    return sending_bitrate, \
        state_rate_new, state_net, gamma_new, gradient_delay_modify, \
        len(delay_time), len(pack_group_delay)


def gcc_plus_model(send_rate, rcv_rate, pack_group_delay, loss_list, x_time, state_rate, gamma, slope1=0.039,
                   slope2=0.0087):
    # s delay_based

    # slope2 = action_list[actions]
    send_delay, state_rate_new, state_net, gamma_new, gradient_delay, len_di, len_pgd \
        = delay_based(gamma, send_rate, rcv_rate, pack_group_delay, x_time, state_rate, slope1, slope2)
    
    # s loss_based
    send_loss = loss_based(loss_list, send_rate)

    send_final = np.minimum(send_delay, send_loss)

    # with open("test.log", "a", encoding="utf-8") as f:
    #     f.write(f"send_delay: {send_delay}\nsend_loss: {send_loss}\nsend_final: {send_final}\n\n")

    return send_loss, send_delay, send_final, \
        state_rate_new, state_net, gamma_new, gradient_delay, \
        len_di, len_pgd
