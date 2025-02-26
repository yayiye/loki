import json
import matplotlib.pyplot as plt

# 读取action.json文件
def load_action_data(file_path):
    with open(file_path, 'r') as f:
        # 读取所有数据
        data = f.readlines()
    
    # 解析JSON格式的数据
    action_data = [json.loads(line.strip()) for line in data]
    return action_data

# 绘制折线图
def plot_action_data(action_data):

    plt.figure(figsize=(10, 6))
    plt.plot(action_data, marker='.', linestyle='', color='b', label='Action Data')
    plt.title('Action Data Line Plot')
    plt.xlabel('Index')
    plt.ylabel('Action Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 设置action.json的路径
    file_path = 'D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/action.json'
    
    # 读取数据
    action_data = load_action_data(file_path)
    
    # 绘制折线图
    plot_action_data(action_data)
