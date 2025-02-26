import matplotlib.pyplot as plt
from matplotlib import rcParams

# 配置中文字体，防止乱码
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（SimHei）显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 定义函数读取文件并提取数据
def read_log_file(file_path):
    tmp = []
    target_b = []
    send_b = []
    bandwidth = []

    # 打开文件并逐行读取
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过表头和无关数据行
            if line.startswith("tmp") or line.strip() == "":
                continue

            # 按列解析数据
            data = line.split()
            if len(data) >= 8:  # 确保数据行有效
                tmp.append(int(data[0]))  # 时间步
                target_b.append(float(data[1]))  # 目标比特率
                send_b.append(float(data[2]))  # 发送码率
                bandwidth.append(float(data[6]))  # 带宽

    return tmp, target_b, send_b, bandwidth


# 主函数
def main():
    # 设置日志文件路径
    file_path = 'D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/gcc_contrast/env_test/evaluate-1'

    # 读取数据
    tmp, target_b, send_b, bandwidth = read_log_file(file_path)

    # 绘制图像
    plt.figure(figsize=(12, 6))
    plt.plot(tmp, target_b, label="目标码率 (target_b)", linestyle="--", marker="o")
    plt.plot(tmp, send_b, label="发送码率 (send_b)", linestyle="-", marker="s")
    plt.plot(tmp, bandwidth, label="带宽 (bandwidth)", linestyle=":", marker="^")

    # 添加标题和标签
    plt.title("目标码率 vs 发送码率 vs 带宽", fontsize=16)
    plt.xlabel("时间步 (tmp)", fontsize=14)
    plt.ylabel("比特率 (kbps)", fontsize=14)
    plt.legend(fontsize=12)

    # 显示网格
    plt.grid(alpha=0.5)

    # 显示图像
    plt.tight_layout()
    plt.show()


# 执行主函数
if __name__ == "__main__":
    main()