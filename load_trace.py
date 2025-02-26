import os
import numpy as np

def load_trace(cooked_trace_folder):  # 一列带宽，一列时间
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:  # 这里修改了！！！
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print(file_path)
        with open(file_path, 'rb') as f:     # 二进制只读方式打开文件
            for line in f:
                parse = line.split()   # 拆分字符串
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)
    # for i in range(len(all_file_names)):
    #         with open("file_name.log", "a", encoding="utf-8") as f:
    #                     f.write(f"file_name : {all_file_names[i]}\n")
    # 把文件都读取到了下面三个列表中
    return all_cooked_time, all_cooked_bw, all_file_names

# if __name__ == '__main__':
#     file_path = 'D:/vscode/code/python/python_object/loki_for_upgrade/loki_for_upgrade_ppo/cooked_traces_train/'
#     all_cooked_time = []
#     all_cooked_bw = []
#     all_file_names = []
#     all_cooked_time,all_cooked_bw,all_file_names = load_trace(file_path)
#     print(len(all_cooked_time))