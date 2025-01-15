import numpy as np
def trans(c, k):
    return np.power(10, k*(c-1)) - np.power(10, -k)

# 假设你的数据存储在一个名为"data.txt"的文件中
# id = 'AQM_3'
# file_name = "/home/xiatao/vbnet/GraphGPS-main/Predict/{}/{}/mae/logging.log".format(id,id)
file_name="/home/xiatao/vbnet/GraphGPS-main/Predict/AQM_3/AQM_3/mae/logging.log"
# 初始化一个空列表来存储数据
data_list = []

# 初始化一个标志变量，用于控制读取数据的开始和结束
start_reading = False

# 读取文件并处理每一行
with open(file_name, 'r') as file:
    for line in file:
        # 检查是否到达开始读取的标记行
        if line.strip() == "Num parameters: 626405":
            start_reading = True
            continue  # 跳过这一行，继续读取下一行
        # 检查是否到达结束读取的标记行
        if line.strip() == "Done predicting!":
            break  # 到达结束标记行，停止读取
        # 如果已经到达开始标记行，并且还没有到达结束标记行，则处理当前行
        if start_reading:
            # 去除行尾的换行符并分割标识串和数值
            parts = line.strip().split(' : ')
            if len(parts) == 2:
                identifier = parts[0]  # 标识串
                value = float(parts[1])  # 对应的数值
                data_list.append((identifier, value))

# 根据数值对数据进行排序（从大到小）
data_list = sorted(data_list, key=lambda x: x[1], reverse=True)

# trans_data_list = [(identifier, trans(value, k=4.0)) for identifier, value in data_list]
# sum_of_values = sum(value for _, value in trans_data_list)
# normalized_data_list = [(identifier, value / sum_of_values) for identifier, value in trans_data_list]

# sum_of_values = sum(value for _, value in data_list)
# data_list = [(identifier, value / sum_of_values) for identifier, value in data_list]
max_of_values = max(value for _, value in data_list)


# 将排序后的数据写入新的文件
output_file_name = "/home/xiatao/vbnet/GraphGPS-main/Predict/AQM_3/AQM_3/mae/sort_logging.log"
# output_file_name = "/home/xiatao/vbnet/GraphGPS-main/Predict/{}/{}/mae/logging_sort.log".format(id,id)

with open(output_file_name, 'w') as output_file:
    for identifier, value in data_list:
        # 按照指定格式写入标识串和数值
        # if x <=0.98:
        #     x+=value
        #     output_file.write(f"{identifier} : {value}\n")
        output_file.write(f"{identifier} : {value / max_of_values}\n")
