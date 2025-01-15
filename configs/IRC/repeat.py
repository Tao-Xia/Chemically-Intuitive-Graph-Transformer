# 原始 YAML 文件路径
template_path = "/home/xiatao/vbnet/GraphGPS-main/configs/IRC/1.yaml"
# 输出路径
output_dir = "./"

# 替换数字范围
for i in range(2, 22):  # 从 2 到 21
    # 读取模板文件
    with open(template_path, "r") as f:
        content = f.read()
    
    # 替换路径中的 1 为当前的数字 i
    new_content = content.replace("/IRC/1", f"/IRC/{i}")
    
    # 写入新文件
    output_path = f"{output_dir}/{i}.yaml"
    with open(output_path, "w") as f:
        f.write(new_content)

    # print(f"Generated {output_path}")
