import os
import re
import subprocess

def read_xyz_coordinates(file_path, point_number):
    """
    读取指定 Point 的分子坐标。

    参数:
        file_path (str): 文件路径
        point_number (int): 需要提取的 Point 编号

    返回:
        list: 分子坐标列表，每个原子为 (元素, x, y, z)
              如果未找到指定 Point，则返回空列表。
    """
    coordinates = []
    point_found = False
    target_point = f"Point    {point_number}"
    target_point_2 = f"Point   {point_number}"
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # 检查是否是目标 Point
            if line == target_point or line == target_point_2:
                point_found = True
                coordinates = []  # 初始化新 Point 的坐标
                continue
            
            # 如果已经找到目标 Point，读取坐标数据
            if point_found:
                if line.startswith("Point"):  # 下一个 Point 开始，停止读取
                    break
                parts = line.split()
                if len(parts) == 4:  # 分子坐标行
                    element = parts[0]  # 元素符号
                    x, y, z = map(float, parts[1:])  # 坐标

                    coordinates.append((element, x, y, z))
    return coordinates

def remove_blank_lines(data):
    # 按行分割，过滤掉空行后重新拼接
    return "\n".join(line for line in data.splitlines() if line.strip())

def update_molecule(file_path, molecule_content):
    """
    替换 Python 文件中 `molecule` 变量三引号内的内容。

    Args:
        file_path (str): Python 文件路径。
        molecule_content (list): 列表，表示需要插入到 `molecule` 中的内容。
    """
    # 将列表内容转换为字符串，每行一个分子坐标
    molecule_content = [str(x) for x in molecule_content]
    result = []
    for line in molecule_content:
        # 去掉括号和单引号，将逗号替换为空格
        formatted_line = line.replace("(", "").replace(")", "").replace("'", "").replace(",", "")
        result.append(formatted_line.strip())

    new_molecule = "\n".join(result) # 去掉空行
    
    # 打开并读取文件内容
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 替换 `molecule` 的三引号内容
    # print(new_molecule)
    new_molecule = remove_blank_lines(new_molecule)
    # print(new_molecule)
    updated_content = re.sub(
        r'(molecule\s*=\s*""")([\s\S]*?)(\s*""")',
        rf'\1\n{new_molecule}\n\3',
        content
    )
    updated_content = "\n".join(line for line in updated_content.splitlines() if line.strip())
    
    # 将更新后的内容写回文件
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_content)

    # print(f"Updated `molecule` in {file_path}.")

def run_irc_points(total_points):

    for point in range(1, total_points + 1):
        print(point)
        molecule_content=read_xyz_coordinates("/home/xiatao/vbnet/GraphGPS-main/plt/IRC.xyz", point)
        print(molecule_content)
        update_molecule("/home/xiatao/vbnet/GraphGPS-main/graphgps/loader/dataset/predict_vb.py", molecule_content)
        command = f"python main.py --cfg /home/xiatao/vbnet/GraphGPS-main/configs/IRC/{point}.yaml"
        subprocess.run(command, shell=True)  # 调用子进程运行命令


# 调用函数
if __name__ == "__main__":

    total_points = 21
    run_irc_points(total_points)
