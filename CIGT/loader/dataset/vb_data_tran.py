import os
import numpy as np
from rdkit import Chem
import re
import scipy.sparse as sp

global __ATOM_LIST__
__ATOM_LIST__ = \
    ['h',  'he',
     'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne',
     'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',
     'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
     'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
     'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe',
     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt',
     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
     'fr', 'ra', 'ac', 'th', 'pa', 'u',  'np', 'pu']

def int_atom(atom):
    """
    convert str atom to integer atom
    """
    global __ATOM_LIST__
    #print(atom)
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1

def replace_zeros(lst):
    # 找到列表中的最小非零元素值
    min_non_zero = min(filter(lambda x: x != 0, lst))
    
    # 将零值替换为最小非零元素值的一半
    for i in range(len(lst)):
        if lst[i] == 0:
            lst[i] = min_non_zero / 2
    
    return lst

def read_data_from_xmo(filename):
    """
    read data from xmo file
    atom_symbols : 元素符号
    Z:原子序数
    N:每一个价键结构的原子总数
    atom_charges:原子电荷
    R:原子坐标
    W:价键结构权重
    nc_orb:非活性轨道
    ac_orb:活性轨道
    nae:活性电子
    norb：分子总轨道数目
    nab：一个轨道用多少基函数展开，orbitals in primitive basis function
    """

    atomic_symbols = []
    
    atom_charges = []
      
    W = []    # vb structure weight
    lowdin_W = []
    inverse_W = []
    renormalized_W = []
    nc_orb = [] # non-active orbital
    ac_orb = [] # active orbital
    vbs = []
    with open(filename, "r") as file:

        file = file.read().strip().splitlines()
        VBS_index1 = file.index("              ******  WEIGHTS OF STRUCTURES ****** ")
        VBS_index2 = file.index("         Lowdin Weights")
        
        for line_number, line in enumerate(file[VBS_index1 + 2 : VBS_index2 - 1]):
            line = line.split()
            
            weight, orbital_index= line[1], line[3:]
            vbs.append(orbital_index)
            W.append(float(weight)) # return W of all vb structures 价键结构权重
            nc_orb_index = [s for s in orbital_index if ':' in s ] # 非活轨道索引
            ac_orb_index = [s for s in orbital_index if ':' not in s] # 活性轨道索引
            
            ac_orb.append(ac_orb_index) 
            nc_orb.append(nc_orb_index)# return nc_orb, eg:['1:8','9', '10']

        Lowdin_Weights_index1 = file.index("         Lowdin Weights")
        Lowdin_Weights_index2 = file.index("         Inverse Weights")

        for line_number, line in enumerate(file[Lowdin_Weights_index1 + 2 :Lowdin_Weights_index2 -1]):
            line = line.split()
            weight, orbital_index= line[1], line[3:]
            lowdin_W.append(float(weight)) # return W of all vb structures 价键结构权重
            
    
        inverse_weight_index1 = file.index("         Inverse Weights")
        inverse_weight_index2 = file.index("         Renormalized Weights")

        for line_number, line in enumerate(file[inverse_weight_index1 + 2 : inverse_weight_index2 - 1]):
            line = line.split()
            weight, orbital_index= line[1], line[3:]
            inverse_W.append(float(weight)) # return W of all vb structures 价键结构权重
            
        renormalized_weight_index1 = file.index("         Renormalized Weights")
        renormalized_weight_index2 = file.index("                 ******  OPTIMIZED ORBITALS  ******")

        for line_number, line in enumerate(file[renormalized_weight_index1 + 2 : renormalized_weight_index2 - 3]):
            line = line.split()
            weight, orbital_index= line[1], line[3:]
            renormalized_W.append(float(weight)) # return W of all vb structures 价键结构权重

    index_w = list(enumerate(W))
    sort_w = sorted(index_w, key=lambda x: x[1], reverse=True)
    vb2class = [0] * len(W)
    trans_w = [0] * len(W)
    # print(sort_w)
    sum_class = 0
    cout = 0
    for i, x in sort_w :
        sum_class+=x
        if sum_class < 0.98:
           cout += 1   
           vb2class[i] = 1
    # count_1 = 0
    # sum_weight = 0
    # for i, x in sort_w:
    #     sum_weight += x
    #     if sum_weight < 0.7:
    #         count_1 += 1
    #         trans_w[i] = x
    # print(cout)
    # lowdin_W = np.array(lowdin_W) / np.max(lowdin_W)
    # lowdin_W = ((lowdin_W - np.mean(lowdin_W)) / np.std(lowdin_W))
    # print("mean ", np.mean(lowdin_W))
    # print("sum ",sum(lowdin_W))
    # print("转换前中位数 ", np.median(np.sort(lowdin_W)))
    lowdin_W = np.array(lowdin_W) / np.max(lowdin_W)
    # print("转换前非零最小值 ", np.min(lowdin_W[non_zero_index]) )
    # print("sum ",sum(lowdin_W))
    # print("mean ",np.mean(lowdin_W))
    # print("转换后中位数 ", np.median(lowdin_W))
    # print(len(lowdin_W))

    # print("中位数 ", np.median(lowdin_W))
    # non_zero_index = np.nonzero(lowdin_W)
    # print("非零最小值 ", np.min(lowdin_W[non_zero_index]) )

    # print("最小非零：", lowdin_W)
    # lowdin_W = np.array(lowdin_W) / np.max(lowdin_W)  
    # log_lowdin_W = np.log10(np.array(lowdin_W) + 1)

    # # print(log_lowdin_W)
    # linear_lowdin_w = 5 * np.array(lowdin_W) + 1
    # lowdin_W = 10 * np.array(lowdin_W) + 1
    # 将存在的0值替换为最小值的一半
    # W = replace_zeros(W)
    # W = np.log(np.abs(W))
    # lowdin_W = replace_zeros(lowdin_W)
    # inverse_W = replace_zeros(inverse_W)
    # renormalized_W = replace_zeros(renormalized_W)

    # log函数预处理
    # lowdin_W = np.array(lowdin_W)
    # lowdin_W = np.log10(lowdin_W + 1e-5)


    ac_orb = np.array(ac_orb).astype(float).astype(int) # 将活性轨道索引值转为 numpy int 
    
    atomic_symbols = [x.replace('','') for x in atomic_symbols]
    # Z = [int_atom(atom) for atom in atomic_symbols] # tansform atom str to atom number eg.C -> 6

     # 每一个价键结构的原子数目 
    Z = []
    # 打开文本文件并读取内容
    with open(filename, 'r') as file:
        content = file.read()

    # 找到 $geo 的索引（不区分大小写）
    geo_index = content.lower().find("$geo")
    if geo_index == -1:
        print("未找到 $geo")
    else:
        # 找到 $end 在 $geo 之后的索引
        end_index = content.lower().find("$end", geo_index)
        if end_index == -1:
            print("未找到 $end")
        else:
            # 提取 $geo 到 $end 之间的内容
            str_R = content[geo_index+4:end_index+ len("$end")-4].strip()
            # print("找到的 $geo 至 $end 之间的内容:")
    lines = str_R.strip().splitlines()

    # 初始化存储浮点数坐标的列表
    R = []

    # 处理每行数据，将坐标数据解析为浮点数并存储
    for line in lines:
        parts = line.split()
        Z.append(int_atom(parts[0]))
        # coords = [float(part) for part in parts[-3:]]  #
        R.append(parts[-3:])
    R = np.array(R, dtype=float)
    Z = np.array(Z)
    N = np.array(len(Z))
    with open(filename, "r") as file:
        text = file.read()

    # 使用正则表达式匹配 nae
    match_nae = re.search(r'(?i)nae=(\d+)', text)
    nae = match_nae.group(1)
    nae = int(nae)
    # match_nab = re.search(r'nab=(\d+)', text)
    # nab = match_nab.group(1)

    #使用正则表达式匹配 nao
    match_nao = re.search(r'(?i)nao=(\d+)', text)
    nao = match_nao.group(1)
    nao = int(nao)
    # 计算一个分子的总轨道数目为多少，只包含前三周期元素
    nel = sum(Z)
    if (nel - nao) % 2 == 0:
        num_double_orbital = (nel - nao) // 2
    else:
        num_double_orbital = (nel - nao) // 2 + 1
    norb = num_double_orbital + nao

    # 计算一个原子用多少基函数展开
    dict_Z_b = [1,6,7,8,9,15,16,17] # 包含 H C N O F P S Cl
    dict_orb_B = [5,15,15,15,15,19,19,19]
    dict_Z_B = dict(zip(dict_Z_b, dict_orb_B))
    Z_B = np.vectorize(lambda x: dict_Z_B[x])
    nab = sum(Z_B(Z))

    return vb2class, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs

# vb2class, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs = read_data_from_xmo('/home/xiatao/vbnet/data/VBSCF/12105_VBSCF.xmo')

# print(lowdin_W)12105
# print(vbs[0])
# print(sum(inverse_W))

# print(sum(renormalized_W))

# def read_vbslist_from_xmo(filename):
#     """
#     read VBS weight and VB structure list from xmvb xmo file
#     eg 1:8 9 10 -> [1:8, 9, 10]
    
#     """
#     W = []    
#     nc_orb = []

#     with open(filename, "r") as file:

#         file = file.read().strip().splitlines()
#         VBS_index1 = file.index("              ******  W OF STRUCTURES ****** ")
#         VBS_index2 = file.index("         Lowdin W")
        
#         for line_number, line in enumerate(file[VBS_index1 + 2 : VBS_index2 - 1]):
#             line = line.split()
#             weight, orbital_index = line[1], line[3:]
#             W.append(float(weight)) # return W of all vb structures 
#             nc_orb.append(orbital_index)# return nc_orb, eg:['1:8','9', '10']
#     return W, nc_orb
from collections import defaultdict

# 初始化全局变量
prev_value = None
label_counts = defaultdict(int)
def rename_bf(value, label):
    global prev_value, label_counts  # 使用全局变量 prev_value 和 label_counts

    if prev_value is None or value != prev_value:
        label_counts.clear()  # 如果 value 和前一个值不一样，清空字典重新计数
    prev_value = value

    if label not in label_counts:
        label_counts[label] = 1
    else:
        label_counts[label] += 1

    count = label_counts[label]

    if label in ['S']:
        return f'{count}{label}'
    elif label in ['X', 'Y', 'Z']:
        return f'{count+1}P{label}'
    elif label in ['XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ']:
        return f'{count+2}D{label}'
    else:
        return label
    

def AC(filename, cov_factor):
    """
    compute AC matrix of a xyz corrdinates, from a xmo file

    purpose: give steleton connection

    根据共价半径进行判断骨架结构

    """
    vb2class, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs = read_data_from_xmo(filename)
    num_atoms = np.shape(R)[0]
    #任意两个原子之间的距离矩阵
    dist_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            # Calculate Euclidean distance between atoms i and j
            dist = np.linalg.norm(R[i] - R[j]) # 计算两个原子之间的距离
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    pt = Chem.GetPeriodicTable()
    ac = np.zeros((num_atoms, num_atoms))
    # rel_dist_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        R_i = pt.GetRcovalent(int(Z[i])) * cov_factor # 原子i的共价半径
        for j in range(i+1 , num_atoms):
            R_j = pt.GetRcovalent(int(Z[j])) * cov_factor # 原子j的共价半径
            if dist_matrix[i, j] <= (R_i +R_j) :
                ac[i, j] = 1
                ac[j, i] = 1
            # rel_dist_matrix[i, j] = (R_i + R_j) / dist_matrix[i, j]
            # rel_dist_matrix[j, i] = (R_i + R_j) / dist_matrix[j, i]

    del dist_matrix
    return ac
 

# print(AC('/home/xiatao/vbnet/data/VBSCF/335_VBSCF.xmo',cov_factor=1.2))

def get_oipbf(filename, vbscf=True, localize=True):
    """
    oipbf : ORBITALS IN PRIMITIVE BASIS FUNCTIONS

    nab : 一个轨道用多少个基函数展开

    num_orb : 一个分子的总轨道数目
    """
    vb2class, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs = read_data_from_xmo(filename)

    nel = np.sum(Z)
    if vbscf:
        num_orb = nao + (nel - nae) // 2 
    if localize:
        num_orb = nel // 2 
    oipbf=[] # ORBITALS IN PRIMITIVE BASIS FUNCTIONS
    oipbf1 = []
    oipbf2 = []
    orbital_energy = []
    with open(filename, "r") as file:

        file = file.read().strip().splitlines()
        index0 = file.index("       ******  ORBITALS IN PRIMITIVE BASIS FUNCTIONS ******")
        index1 = index0 + 6     

        # orbital_energy_index = file.index(" ORBITAL ENERGY")
        # for line in (file[orbital_energy_index + 4 : orbital_energy_index + 4 + num_orb]):
        #     line = line.split()
        #     orbital_energy.append(float(line[1]))

        for line_number, line in enumerate(file[index1 : index1 + nab]):
            line = line.split()
            oipbf1.append(line)
        
        if num_orb % 5 == 0:
            m = (num_orb // 5) - 1
        else:
            m = (num_orb // 5) 

        index2 = index1 + nab + 4

        for i in range(m):
            index1 = index2
            index_x = index1 + nab 
            index3 = index_x + 4
            index2 = index3
            for line_number, line in enumerate(file[index1 : index_x]):
                line = line.split()
                oipbf2.append(line[4:])
    
        # 字符串型转化为浮点数
        oipbf1 =  [[float(val) if val.replace('.', '', 1).isdigit() else val for val in sublist] for sublist in oipbf1]
        oipbf2 = [[float(val) for val in sublist] for sublist in oipbf2]
        
        #oipbf2转化
        c = []
        for i in range(nab):
            q = [i + k * nab for k in range(m)] # 拼接索引
            x = sum([oipbf2[index] for index in q], [])
            c.append(x)
                             

        for i in range(len(oipbf1)):
            temp = oipbf1[i] + c[i]
            oipbf.append(temp)
    
    return  oipbf, Z
     

def orb_atom(filename, vbscf:True, localize:False):
    """
    获取活性轨道分别定域在哪一个原子上，
    nae 活性电子数
    nab 一个轨道需要多少基函数展开
    norb 一个分子的总轨道数目
    aca:活性轨道所在的原子编号
    aco：活性轨道编号
    """
    vb2class, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs = read_data_from_xmo(filename)
    if vbscf:
        oipbf, Z = get_oipbf(filename, vbscf=True, localize=False)
    if localize:
        oipbf, Z = get_oipbf(filename, vbscf=False, localize=True)


    oipbf = np.array(oipbf)
    AS = oipbf[:, 1] # 原子符号所在列，提取出成单独一列
    BS = oipbf[:, 3] # 基函数所在列，提取出单独成一列
    atom_index = oipbf[:, 2].astype(float).astype(int) # 原子索引所在列，提取出单独为一列,并转化为整数
    oipbf = np.delete(oipbf, (0, 1, 2, 3), 1).astype(float) # 删除已经提取为单独一列的量,以及第一列基函数序号，即只保留轨道系数
    # col_index = 0 # 字符串转化为float
     
    m = np.max(np.abs(oipbf), 0) # 输出每一列的最大值（绝对值）
    row_index = np.argmax(np.abs(oipbf), axis=0) # 输出每一列最大值所在的行索引
    oa = atom_index[row_index] # 轨道定域在哪一个原子，返回原子编号

    aca = oa[-(int(nao)):] # 取最后的nao个活性轨道定域所在原子的编号
    aco = np.arange(len(oa) - nao + 1, len(oa) + 1)
    
    
    return  aca, aco

# print(orb_atom('/home/xiatao/vbnet/data/C7H7NO.xmo'))
   


def edge_index(filename, cov_factor):
    """
    compute adjacency matrix of a vb structure

    ac_orb_atom:活性轨道对应的原子编号
    
    """       
    vb2class, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs = read_data_from_xmo(filename)
    aca, aco = orb_atom(filename, vbscf=True, localize=False)  
    ac = AC(filename, cov_factor)  # 
    
    orb_atom_dict = dict(zip(aco, aca)) 
    vfunc = np.vectorize(lambda x: orb_atom_dict[x]) # 函数向量化

    ac_orb_atom = vfunc(ac_orb) # 活性轨道编号转化为原子编号
    
    #创建空列表存储邻接矩阵
    edge_index = []
    for i in range(ac_orb_atom.shape[0]):
        ac1 = np.copy(ac)

        if (ac_orb_atom.shape[1] % 2) == 0: #如果是偶数个活性电子
            for j in range(0, ac_orb_atom.shape[1] - 1, 2):
                    a = ac_orb_atom[i][j]
                    b = ac_orb_atom[i][j+1]
                    if a != b: # 某一轨道占据了两个电子，即某个原子存在一个负电荷
                        ac1[a-1,b-1] = 1
                        ac1[b-1,a-1] = 1
                    # np.fill_diagonal(ac1, 0)
        else: # 如果是奇数个活性电子
            for j in range(0, ac_orb_atom.shape[1] - 1, 2):
                    a = ac_orb_atom[i][j]
                    b = ac_orb_atom[i][j+1]              
                    if a != b:   
                        ac1[a-1,b-1] = 1
                        ac1[b-1,a-1] = 1
                    # np.fill_diagonal(ac1, 0)
        edge_index.append(ac1)# 将新计算的部分拼接到之前的结果后面
    edge_index = np.array(edge_index)

    adjacency=[]
    for x in edge_index:
        edge_index_temp = sp.coo_matrix(x)
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
        adjacency.append(indices)

    return adjacency

# print(edge_index('/home/xiatao/vbnet/data/VBSCF/241_VBSCF.xmo',  cov_factor=1.2))


def LMO_coffe(boys_filename, vbscf_filename):
    """
    读取定域化的轨道做初始猜测构建节点特征和边特征
    """
    vb2class_single_file, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W_single_file, inverse_W, renormalized_W,vbs = read_data_from_xmo(vbscf_filename)

    oipbf_local, Z = get_oipbf(boys_filename, vbscf=False, localize=True)
    oipbf_local = np.array(oipbf_local, )
    oipbf_vbscf, Z = get_oipbf(vbscf_filename, vbscf=True, localize=False)
    basis_func = oipbf_local[:, [2,3]] # 保留原子索引及基函数符号
    
    basis_sym = [[value, rename_bf(value, label)] for value, label in basis_func] # 基函数转换，s->1s
    basis_sym = np.array(basis_sym)
    
    atom_sym = oipbf_local[:,1] # 存放原子符号
    aca, aco = orb_atom(vbscf_filename, vbscf=True, localize=False) # 活性原子编号，活性轨道编号   
    orb_atom_dict = dict(zip(aco, aca-1)) # 活性轨道编号对应的原子编号
    vfunc = np.vectorize(lambda x: orb_atom_dict[x]) # 函数向量化
    ac_orb_atom = vfunc(ac_orb) # 活性轨道编号对应的原子编号

    # 对VBSCF计算输出文件的轨道系数进行判断活性轨道的信息
    oipbf_coffe = np.delete(oipbf_vbscf, (0, 1, 2, 3), 1).astype(float)# 删除已经提取为单独一列的量,以及第一列基函数序号，即只保留轨道系数
    row_index = np.argmax(np.abs(oipbf_coffe), axis=0) # 输出每一列最大值所在的行索引
    bt = row_index[-nao:]
    active_atom_sym = atom_sym[bt] # 活性轨道所在的原子（符号）
    ac_atom_func = [] # 存储每个活性轨道所对应的原子轨道类型
    for index in bt:
        ac_atom_func.append(basis_sym[index][1]) # 
    
    ac_orb_basis_dict= dict(zip(aco,ac_atom_func)) # 活性轨道编号对应的活性轨道类型
    ac_atom_sym_orb_dict = dict(zip(aco, active_atom_sym)) # 活性轨道编号对那个的原子符号字典
    
    ac_atom_basis_dict = dict(zip(aca-1,ac_atom_func)) # 活性原子编号对应的活性轨道类型


    oipbf_local = np.delete(oipbf_local, (0, 1, 3), 1).astype(float) # 删除已经提取为单独一列的量,以及第一列基函数序号，即只保留轨道系数



    dict_Z_b = [1,6,7,8,9,15,16,17] # 包含 H C N O F P S Cl
    dict_orb_B = [5,15,15,15,15,19,19,19]
    dict_Z_B = dict(zip(dict_Z_b, dict_orb_B))
    Z_B = np.vectorize(lambda x: dict_Z_B[x])
    num_basis = Z_B(Z)
    abs_local_oipbf = np.abs(oipbf_local)
    # oipbf = oipbf[:, np.argsort(orbital_energy)]
    s = []
    n = 0
    m = 0
    # 计算每个原子的基函数对各个轨道的贡献

    # print(np.sum(oipbf**2, axis=0))

    for x in num_basis:
        m += x
        s.append(np.sum(abs_local_oipbf[n:m, 1:], axis=0))
        n += x
    
    local_dict = {}
    for i, x_1 in enumerate(s): # 对每个原子
        
        for j, x_2 in enumerate(x_1):  # 一个原子对所有轨道的贡献值
            
            if x_2 >= 0.5: # 如果贡献值大于等于0.5
                if j+1 not in local_dict:
                    local_dict[j+1] = []
                local_dict[j+1].append(i+1) # key : orbital , value : atom

                # print('atom:{} orbital:{}'.format(i+1,j+1), x_2)
    
    # orbital_guess = []
    # # bond_orbital = []
    # single = 0
    # double = 0
    # for i,x in enumerate(local_dict.values()):
    #     if len(x) == 1 : # 如果这个轨道只定域在一个原子上
    #         # single += 1
    #         # bond_orbital.append(tuple(x*2))
    #         # orbital_guess.append(list(local_dict.keys())[i])
    #         print(list(local_dict.keys())[i], x)
    #     else:
    #         # double += 1
    #         # bond_orbital.append(tuple(x))
    #         # orbital_guess.append(list(local_dict.keys())[i])
    #         print(list(local_dict.keys())[i], x)
    # for i,x in enumerate(local_dict.values()):
    #     if len(x) > 1 : # 如果是成键轨道，定域在2个或多个原子间
    #         double += 1
    #         bond_orbital.append(tuple(x))
    #         orbital_guess.append(list(local_dict.keys())[i])
    #         print(list(local_dict.keys())[i], x)
    # 自定义活性轨道系数
    pz_2 = np.zeros(19)
    pz_2[[5,9]] = 0.66, 0.44 
    pz_3 = np.zeros(19)
    pz_3[[5,9]] = 0.44, 0.66
    experience_active_orb ={'2PZ':pz_2} 
    
    active_orb_dict = {}
    active_orb_dict[2,0,0,1] = pz_2
    
    # 存放原子轨道对应量子数的字典
    quam_nun_dict = {'1S':[1,0,0,0], '2S':[2,0,0,0], '3S':[3,0,0,0], '4S':[4,0,0,0], '2PX':[2,1,0,0], '2PY':[2,0,1,0],'2PZ':[2,0,0,1],'3PX':[3,1,0,0],'3PY':[3,0,1,0],'3PZ':[3,0,0,1],'4PX':[4,1,0,0],'4PY':[4,0,1,0],'4PZ':[4,0,0,1],'3DXX':[3,2,0,0],'3DXY':[3,1,1,0],'3DXZ':[3,1,0,1],'3DYZ':[3,0,1,1],'3DYY':[3,0,2,0],'3DZZ':[3,0,0,2]}
    node_feat_s = np.zeros((len(Z), 19)) # 初始化节点特征向量
    # Cl原子在ccd基组下展开所用的基函数类型顺序，所对应的量子数
    max_basis = [[1,0,0,0],[2,0,0,0],[3,0,0,0],[4,0,0,0],[2,1,0,0],[2,0,1,0],[2,0,0,1],[3,1,0,0],[3,0,1,0],[3,0,0,1],[4,1,0,0],[4,0,1,0],[4,0,0,1],[3,2,0,0],[3,1,1,0],[3,1,0,1],[3,0,2,0],[3,0,1,1],[3,0,0,2]]
    
    matrix1_tuples = [tuple(row) for row in max_basis]
    index_map = {t: i for i, t in enumerate(matrix1_tuples)}

    # 原子序号对应相加的索引字典
    atom_basis_add = {}
    atom_basis_add[6] = [0, 1, 2, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18]

    edge_feat = []
    node_feat = []
    adjacency = edge_index(vbscf_filename,  cov_factor=1.2)
    # print(adjacency[0])
    for i, x in enumerate(local_dict.values()):

        orbital_num = list(local_dict.keys())[i]
        atom_num = list(local_dict.values())[i]

        if len(x) == 1:           
            orbital_coff = oipbf_local[oipbf_local[:, 0].astype(float) == atom_num, orbital_num] # 提取原子atom_num所对应的第orbital_num个轨道的轨道系数
            basis_ = basis_sym[basis_sym[:,0].astype(float) == atom_num, 1] # 提取原子atom_num的基函数符号
            
            # 将基函数符号转换为量子数
            quan_num = [quam_nun_dict.get(x, 'Unknow') for x in basis_]
            
            # 获取每一组基函数在基准CL原子基函数中的向量位置,得到索引列表
            add_indices = [index_map.get(tuple(row), []) for row in quan_num]
            
            orbital_coff = np.where(np.abs(orbital_coff) < 0.1, 0, orbital_coff)
            orbital_coff = orbital_coff / np.sum(np.abs(orbital_coff)) # 归一化
            # print(orbital_coff)
            node_feat_s[atom_num[0] - 1, add_indices] += (orbital_coff)# 非活性部分

    for i in range(len(adjacency)): # 
        node_feat.append(node_feat_s) # 非活性部分一样的

    for x in adjacency:  

        edge_orbital = np.zeros((len(x[0]), 19))

        for i, atom_pair in enumerate(local_dict.values()):
            orbital_num = list(local_dict.keys())[i]
            atom_num = list(local_dict.values())[i]
            # print(atom_num)
            if len(atom_num) == 2:
                # 获取源节点和目标节点的轨道系数
                orbital_coff_0 = oipbf_local[oipbf_local[:, 0].astype(float) == atom_num[0], orbital_num] 
                orbital_coff_1 = oipbf_local[oipbf_local[:, 0].astype(float) == atom_num[1], orbital_num]

                orbital_coff_0 = np.where(np.abs(orbital_coff_0) < 0.1, 0, orbital_coff_0)
                orbital_coff_1 = np.where(np.abs(orbital_coff_1) < 0.1, 0, orbital_coff_1)

                if np.sum(np.abs(orbital_coff_0)) == 0:
                    orbital_coff_0 = orbital_coff_0
                else:
                    orbital_coff_0 = orbital_coff_0 / np.sum(np.abs(orbital_coff_0))
                if np.sum(np.abs(orbital_coff_1)) == 0:
                    orbital_coff_1 = orbital_coff_1
                else:
                    orbital_coff_1 = orbital_coff_1 / np.sum(np.abs(orbital_coff_1))

                atom_basis_sym_0 = basis_sym[basis_sym[:,0].astype(float) == atom_num[0], 1] # 提取原子atom_num[0]的基函数符号
                atom_basis_sym_1 = basis_sym[basis_sym[:,0].astype(float) == atom_num[1], 1] # # 提取原子atom_num[1]的基函数符号
                
                # print(trans_basis)
                # 将基函数符号转换为量子数
                quan_num_0 = [quam_nun_dict.get(basis, 'Unknow') for basis in atom_basis_sym_0]
                quan_num_1 = [quam_nun_dict.get(basis, 'Unknow') for basis in atom_basis_sym_1]
                
                # 获取每组基函数在基准CL基函数中的向量位置
                add_indices_0 = [index_map.get(tuple(row), []) for row in quan_num_0]
                add_indices_1 = [index_map.get(tuple(row), []) for row in quan_num_1]
                
                mask_source = (x[0] == (atom_num[0] - 1)) & (x[1] == (atom_num[1] - 1))
                mask_target = (x[1] == (atom_num[0] - 1)) & (x[0] == (atom_num[1] - 1))
                # 获取边特征矩阵中的行索引
                indice_1 = np.nonzero(mask_source)[0][0] # 源节点的行索引
                indice_2 = np.nonzero(mask_target)[0][0] # 目标节点行索引
                
                node_feat[i][atom_num[0]-1, add_indices_0] +=orbital_coff_0
                node_feat[i][atom_num[1]-1, add_indices_1] +=orbital_coff_1

                edge_orbital_1 = np.copy(edge_orbital)
                edge_orbital_1[indice_1, add_indices_0] += orbital_coff_0
                edge_orbital_1[indice_1, add_indices_1] += orbital_coff_1
                edge_orbital_1[indice_2, add_indices_0] += orbital_coff_0
                edge_orbital_1[indice_2, add_indices_1] += orbital_coff_1
        edge_feat.append(edge_orbital_1)
        
    
    for i in range(ac_orb.shape[0]):
        if (ac_orb.shape[1] % 2) == 0: # 如果是偶数个活性电子
            for j in range(0, ac_orb.shape[1] - 1, 2):
                if ac_orb[i][j] == ac_orb[i][j+1]: # ac_orb_atom[i][j]表示活性轨道序号
                    # 获取该活性轨道对应的原子轨道类型
                    ac_orb_sym_0 = ac_orb_basis_dict[ac_orb[i][j]] 
                    ac_orb_sym_1 = ac_orb_basis_dict[ac_orb[i][j+1]] 

                    # 获取该活性轨道定域所在原子的原子符号
                    # ac_atom_sym_0 = ac_atom_sym_orb_dict[ac_orb[i][j]]
                    # ac_atom_sym_1 = ac_atom_sym_orb_dict[ac_orb[i][j+1]]
                    
                    # 将原子符号转为原子序号
                    # ac_atom_0 = int_atom(ac_atom_sym_0)
                    # ac_atom_1 = int_atom(ac_atom_sym_1)
                    
                    # quan_num_0 = ac_atom_0 + quam_nun_dict.get(ac_orb_sym_0, 'Unknow') 
                    # quan_num_1 = ac_atom_1 + quam_nun_dict.get(ac_orb_sym_1, 'Unknow') 
                    

                    # add_indices_0 = atom_basis_add.get(ac_atom_0)
                    # add_indices_1 = atom_basis_add.get(ac_atom_1)
                    
                    # add_indices_0 = [index_map.get(tuple(quan_num_0), []) ]
                    # add_indices_1 = [index_map.get(tuple(quan_num_1), []) ]
                    if ac_orb_sym_0 != '2PZ':
                        print(vbscf_filename)
                    if ac_orb_sym_0 != '2PZ':
                        print(vbscf_filename)
                    node_feat[i][orb_atom_dict[ac_orb[i][j]]] += experience_active_orb[ac_orb_sym_0]
                    node_feat[i][orb_atom_dict[ac_orb[i][j+1]]] += experience_active_orb[ac_orb_sym_1]
                else: # 如果不相同，是成键情况
                    # 获取该活性轨道对应的原子轨道类型
                    ac_orb_sym_0 = ac_orb_basis_dict[ac_orb[i][j]] 
                    ac_orb_sym_1 = ac_orb_basis_dict[ac_orb[i][j+1]]
                    
                    ac_atom_index_0 = orb_atom_dict[ac_orb[i][j]]
                    ac_atom_index_1 = orb_atom_dict[ac_orb[i][j+1]]

                    mask_source = (adjacency[i][0] == ac_atom_index_0) & (adjacency[i][1] == ac_atom_index_1)
                    mask_target = (adjacency[i][1] == ac_atom_index_0) & (adjacency[i][0] == ac_atom_index_1)
                    indice_0 = np.nonzero(mask_source)[0][0]
                    indice_1 = np.nonzero(mask_target)[0][0]
                    if ac_orb_sym_0 != '2PZ':
                        print(vbscf_filename)
                    if ac_orb_sym_0 != '2PZ':
                        print(vbscf_filename)
                    
                    node_feat[i][ac_atom_index_0] += experience_active_orb[ac_orb_sym_0]
                    node_feat[i][ac_atom_index_1] += experience_active_orb[ac_orb_sym_1]
                    edge_feat[i][indice_0] += experience_active_orb[ac_orb_sym_0]
                    edge_feat[i][indice_1] += experience_active_orb[ac_orb_sym_1]
        else: # 如果是奇数个活性电子
            # 获取最后一个活性轨道，其为单占活性轨道
            ac_orb_sym_last = ac_orb_basis_dict[ac_orb[i][-1]]
            node_feat[i][orb_atom_dict[ac_orb[i][j]]] += experience_active_orb[ac_orb_sym_last]
            
            for j in range(0, ac_orb.shape[1] - 2, 2):
                if ac_orb_atom[i][j] == ac_orb_atom[i][j+1]:
                    # 获取该活性轨道对应的原子轨道类型
                    ac_orb_sym_0 = ac_orb_basis_dict[ac_orb[i][j]] 
                    ac_orb_sym_1 = ac_orb_basis_dict[ac_orb[i][j+1]] 

                    node_feat[i][orb_atom_dict[ac_orb[i][j]]] += experience_active_orb[ac_orb_sym_0]
                    node_feat[i][orb_atom_dict[ac_orb[i][j+1]]] += experience_active_orb[ac_orb_sym_1]
                else:
                    ac_orb_sym_0 = ac_orb_basis_dict[ac_orb[i][j]] 
                    ac_orb_sym_1 = ac_orb_basis_dict[ac_orb[i][j+1]]
                    
                    ac_atom_index_0 = orb_atom_dict[ac_orb[i][j]]
                    ac_atom_index_1 = orb_atom_dict[ac_orb[i][j+1]]

                    mask_source = (adjacency[i][0] == ac_atom_index_0) & (adjacency[i][1] == ac_atom_index_1)
                    mask_target = (adjacency[i][1] == ac_atom_index_0) & (adjacency[i][0] == ac_atom_index_1)
                    indice_1 = np.nonzero(mask_source)[0][0]
                    indice_2 = np.nonzero(mask_target)[0][0]
                    if ac_orb_sym_0 != '2PZ':
                        print(vbscf_filename)
                    if ac_orb_sym_0 != '2PZ':
                        print(vbscf_filename)
                    
                    node_feat[i][ac_atom_index_0] += experience_active_orb[ac_orb_sym_0]
                    node_feat[i][ac_atom_index_1] += experience_active_orb[ac_orb_sym_1]
                    edge_feat[i][indice_1] += experience_active_orb[ac_orb_sym_0]
                    edge_feat[i][indice_2] += experience_active_orb[ac_orb_sym_1]
                    
    # print(ac_orb_atom)
    # print(len(edge_feat))
    # print(len(node_feat))
    # print(edge_feat[0])
    
    return node_feat, edge_feat

# LMO_coffe('/home/xiatao/vbnet/data/LOCAL/335_boys.xmo', '/home/xiatao/vbnet/data/VBSCF/335_VBSCF.xmo')

def BD(filename, cov_factor):
    """
    compute bond order matrix of a vb structure

    ac_orb_atom:活性轨道对应的原子编号
    
    """
    vb2class, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs = read_data_from_xmo(filename)  
    aca, aco = orb_atom(filename,vbscf=True, localize=False) # 活性原子编号，活性轨道编号

    num_atoms = np.shape(R)[0]
    dist_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            # Calculate Euclidean distance between atoms i and j
            dist = np.linalg.norm(R[i] - R[j]) # 计算两个原子之间的距离
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    pt = Chem.GetPeriodicTable()
    ac = np.zeros((num_atoms, num_atoms))
    rel_dist_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        R_i = pt.GetRcovalent(int(Z[i])) * cov_factor # 原子i的共价半径
        for j in range(i+1 , num_atoms):
            R_j = pt.GetRcovalent(int(Z[j])) * cov_factor # 原子j的共价半径
            if dist_matrix[i, j] <= (R_i +R_j) :
                ac[i, j] = 1
                ac[j, i] = 1
                rel_dist_matrix[i, j] = (R_i + R_j) / dist_matrix[i, j]
                rel_dist_matrix[j, i] = (R_i + R_j) / dist_matrix[j, i]

    nsae = np.zeros((np.shape(ac)))  # number of shared active electron
    
    orb_atom_dict = dict(zip(aco, aca)) 
    vfunc = np.vectorize(lambda x: orb_atom_dict[x]) # 函数向量化

    ac_orb_atom = vfunc(ac_orb) # 活性轨道编号转化为原子编号
    # 根据分子骨架邻接矩阵和活性轨道部分计算键级矩阵
    # 创建空列表存储依据活性空间修改后的ac1，用于存放不同价键结构的键级矩阵
    NSE = []  # number of shared electron
    NSAE = [] # number of shared active electron
    RBL = []
    for i in range(ac_orb_atom.shape[0]):
        bd1 = np.copy(ac) # 每次的初始键级矩阵都是初始骨架邻接矩阵
        nsae_1 = np.copy(nsae) # number of shared active electron
        rbl = np.copy(rel_dist_matrix)
        if (ac_orb_atom.shape[1] % 2) == 0: # 如果是偶数个活性电子
            for j in range(0, ac_orb_atom.shape[1] - 1, 2):
                    a = ac_orb_atom[i][j]
                    b = ac_orb_atom[i][j+1]
                    if a != b :
                        bd1[a-1][b-1] += 1
                        bd1[b-1][a-1] += 1
                        nsae_1[a-1][b-1] += 1
                        nsae_1[b-1][a-1] += 1
                        dist = np.linalg.norm(R[a-1] - R[b-1])
                        r_cov = (pt.GetRcovalent(int(Z[a-1])) + pt.GetRcovalent(int(Z[b-1])))* cov_factor
                        rbl[a-1, b-1] = r_cov / dist
                    # np.fill_diagonal(bd1, 0)
        else: # 如果是奇数个活性电子
            for j in range(0, ac_orb_atom.shape[1] - 2, 2):
                    a = ac_orb_atom[i][j]
                    b = ac_orb_atom[i][j+1]
                    c = ac_orb_atom[i][-1]
                    
                    if a != b :
                        bd1[a-1][b-1] += 1
                        bd1[b-1][a-1] += 1
                        nsae_1[a-1][b-1] += 1
                        nsae_1[b-1][a-1] += 1
                        dist = np.linalg.norm(R[a-1] - R[b-1])
                        r_cov = (pt.GetRcovalent(int(Z[a-1])) + pt.GetRcovalent(int(Z[b-1])))* cov_factor
                        rbl[a-1, b-1] =  r_cov / dist
                    # np.fill_diagonal(bd1, 0)
        # edge_index_temp = sp.coo_matrix(bd1)
        # nse = 2 * edge_index_temp.data  # nse : number of shared electron

        NSAE.append(nsae_1)   # number of shared active electron 
        NSE.append(bd1) # number of shared electron
        RBL.append(rbl)
    NSAE = np.array(NSAE) # number of shared nonactive electron
    NSE = np.array(NSE) # number of shared active electron
    RBL = np.array(RBL)
    return NSE, NSAE, RBL

# BD('/home/xiatao/vbnet/data/before/C6H6_6e_6o.xmo',cov_factor=1.2)

def check_cov(arr, nao, nae):
    # 找出一个共价结构
    if nae > nao:
        pair = nae - nao
    else:
        pair = 0

    for vbs in arr:
        for i in range(len(vbs) - pair):
            sub_arr = vbs[i+ 2 * pair:]
            if len(sub_arr) % 2 == 0: # 如果是偶数个电子
                for j in range(0, len(sub_arr), 2):
                    if sub_arr[j] != sub_arr[j+1]:
                        return vbs
            else:
                print("Don't exist convlent structure")


def charge(filename):
    """
    compute charge  of every atom for a vb structure , base on active space

    return : Charge [num_atoms]
    """  
    vb2class, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs = read_data_from_xmo(filename)      
    aca, aco = orb_atom(filename, vbscf=True, localize=False) # 活性原子编号，活性轨道编号
    
    orb_atom_dict = dict(zip(aco, aca)) 
    vfunc = np.vectorize(lambda x: orb_atom_dict[x]) # 函数向量化
    
    n_ac_orb = np.zeros(len(Z)).astype(int)
    for x in list(orb_atom_dict.values()):
        n_ac_orb[x - 1] += 1
    
    ac_orb_atom = vfunc(ac_orb) # 活性轨道编号转化为原子编号
    # 计算每个原子的电荷
    #初始化长度为原子数，数值维0的数组，电荷初始化信息
    
    c = np.zeros((len(Z)))
    ae = np.zeros(len(Z)) # 初始化每个原子的活性电子数目
    found_valid = False
    # 计算每个原子的活性电子数目
    #根据共价结构判断每个原子的活性电子数目
    
    if nae <= nao: #对于活性电子数不超过活性轨道数的情况，即共价结构每个轨道都是单占                     
        for vbs in ac_orb_atom:
            if (ac_orb_atom.shape[1] % 2) == 0: # 如果是偶数个电子
                if found_valid:
                    break
                
                for i in range(0, ac_orb_atom.shape[1] - 1, 2):
                    if vbs[i] == vbs[i + 1]:
                        break
                    else:
                        ae[vbs[i] - 1] += 1
                        ae[vbs[i+1] - 1] += 1
                else:
                    found_valid = True
            else: # 如果是奇数个电子
                if found_valid:
                    break
                
                for i in range(0, ac_orb_atom.shape[1] - 2, 2):
                    if vbs[i] == vbs[i + 1]:
                        break
                    else:
                        ae[vbs[i] - 1] += 1
                        ae[vbs[i+1] - 1] += 1
                        ae[vbs[-1] - 1] += 1
                else:
                    found_valid = True
    else:  # 如果活性电子数目大于活性轨道数目
        cov = check_cov(ac_orb_atom, nae, nao)
        for item in cov:
            ae[item - 1] += 1

    Charge = [] # 每个原子的电荷     
    N_AE = [] # 每个原子的活性电子数目
    N_AB = [] # 每个原子的活性轨道数目
    N_E = [] # 每个原子的电子数目
    
    x = Z - ae # 惰性电子 = 每个原子的初始电子数 - 每个原子的活性电子数 ，下面根据活性空间计算每个价键结构中每个原子的总电子数目
    
    for vbs in ac_orb_atom:
        c1 = np.copy(c) # 初始化电荷矩阵
        c0 = np.copy(c) #
        z1 = np.copy(Z) # 
        x1 = np.copy(x) # 初始化电子数目矩阵
        # charge = Z - 惰性电子矩阵加上活性空间部分的电子分布情况
       
        for j in range(ac_orb_atom.shape[1]):
            x1[vbs[j] - 1] += 1
            c0[vbs[j] - 1] += 1
        
         # 初态 - 末态
        c1 = z1 - x1 # 电荷
        c2 = x1 # 每个原子的电子数目
        ae_0 = c0 # 每个原子的活性电子数目

        N_AB.append(n_ac_orb) # 每个原子的活性轨道数目
        Charge.append(c1) # 每个原子的形式电荷
        N_AE.append(ae_0) # 每个原子的活性电子数目
        N_E.append(c2) # 每个原子的电子数目

    Charge = np.array(Charge).astype(int) #转为numpy 三维数组
    N_AE = np.array(N_AE).astype(int)
    N_AB = np.array(N_AB).astype(int)
    N_E  =np.array(N_E).astype(int)

    return Charge, N_AE, N_AB, N_E

# print(charge('/home/xiatao/vbnet/data/before/C6H6_6e_6o.xmo'))

label_counts = {}
prev_value = None  # 用于存储前一个 value 的值


 
def get_vec(crd):
    """ 
    Get the vector of the sequential coordinate.
    """
    # (B, A, D)
    crd_ = np.roll(crd, -1, axis=-2)
    vec = crd_ - crd
    # (B, A-1, D)
    return vec[:, :-1, :]

def get_dis(crd):
    """ Get the distance of the sequential coordinate.
    """
    # (B, A-1, D)
    vec = get_vec(crd)
    # (B, A-1, 1)
    dis = np.linalg.norm(vec, axis=-1, keepdims=True)
    return dis, vec
 
def get_angle(crd):
    """ Get the bond angle of the sequential coordinate.
    """
    EPSILON = 1e-08
    # (B, A-1, 1), (B, A-1, D)
    dis, vec = get_dis(crd)
    vec_ = np.roll(vec, -1, axis=-2)
    dis_ = np.roll(dis, -1, axis=-2)
    # (B, A-1, 1)
    angle = np.einsum('ijk,ijk->ij', vec, vec_)[..., None] / (dis * dis_ + EPSILON)
    # (B, A-2, 1), (B, A-1, 1), (B, A-1, D)
    return np.arccos(angle[:, :-1, :]), dis, vec

def get_dihedral(crd):
    """ Get the dihedrals of the sequential coordinate.
    """
    EPSILON = 1e-08
    # (B, A-2, 1), (B, A-1, 1), (B, A-1, D)
    angle, dis, vec_0 = get_angle(crd)
    # (B, A-1, D)
    vec_1 = np.roll(vec_0, -1, axis=-2)
    vec_2 = np.roll(vec_1, -1, axis=-2)
    vec_01 = np.cross(vec_0, vec_1)
    vec_12 = np.cross(vec_1, vec_2)
    vec_01 /= np.linalg.norm(vec_01, axis=-1, keepdims=True) + EPSILON
    vec_12 /= np.linalg.norm(vec_12, axis=-1, keepdims=True) + EPSILON
    # (B, A-1, 1)
    dihedral = np.einsum('ijk,ijk->ij', vec_01, vec_12)[..., None]
    # (B, A-3, 1), (B, A-2, 1), (B, A-1, 1)
    return np.arccos(dihedral[:, :-2, :]), angle, dis
 
def get_inner_crd(R):
    """ 
    Concat the distance, angles and dihedrals to get the inner coordinate.
    """
    # (B, A-3, 1), (B, A-2, 1), (B, A-1, 1)
    
    R = R.reshape(1,R.shape[0],3)
    dihedral, angle, dis = get_dihedral(R)
    # (B, A, 1)
    dihedral_ = np.pad(dihedral, ((0, 0), (3, 0), (0, 0)), mode='constant', constant_values=0)
    angle_ = np.pad(angle, ((0, 0), (2, 0), (0, 0)), mode='constant', constant_values=0)
    dis_ = np.pad(dis, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
    # (B, A, 3)
    inner_crd = np.concatenate((dis_, angle_, dihedral_), axis=-1)
    inner_crd = inner_crd.reshape(R.shape[1],3) # transform to 2-dimesions array
    return inner_crd


def active_space(filename, ac_orb, nao, R):
    """
    计算每一个价键结构的活性空间矩阵,可以是n,i,j,k+原子索引，或者是n,i,j,k+内坐标
    """
    # oipbf, Z = np.array(get_oipbf(filename, vbscf=True, localize=False)) # 读取轨道系数
    oipbf, Z = get_oipbf(filename, vbscf=True, localize=False)# 读取轨道系数
    oipbf = np.array(oipbf)
    # vb2class, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs = read_data_from_xmo(filename)

    # BS = oipbf[:, 3] # 基函数所在列，提取出单独成一列
    atom_basis = np.array(oipbf[:, 2:4]) # 原子索引和基函数合并放在一起
    atom_index = oipbf[:, 2].astype(float).astype(int) # 原子索引所在列，提取出单独为一列,并转化为整数
    oipbf = np.delete(oipbf, (0, 1, 2, 3), 1).astype(float) # 删除已经提取为单独一列的量,以及第一列基函数序号，即只保留轨道系数
    # col_index = 0 # 字符串转化为float

    row_index = np.argmax(np.abs(oipbf), axis=0) # 输出每一列最大值所在的行索引
    atom_basis = [[value, rename_bf(value, label)] for value, label in atom_basis] # 基函数转换，s->1s

    bt = row_index[-nao:] # 活性轨道的行索引
    at = atom_index[row_index][-nao:] # 活性轨道对应的原子索引

    aca, aco = orb_atom(filename) # 活性原子编号，活性轨道编号   
    orb_atom_dict = dict(zip(aco, aca))
    vfunc = np.vectorize(lambda x: orb_atom_dict[x]) # 函数向量化
    ac_orb_atom = vfunc(ac_orb) # 活性轨道编号对应的原子编号

    # bf = atom_basis[row_index][-nao:] # 活性轨道对应的基函数名称
    basis_function = [] # 存储每个活性轨道所对应的原子轨道类型
    for index in bt:
        basis_function.append(atom_basis[index][1]) # 
    
    orb_basis_dict= dict(zip(aco,basis_function)) # 活性轨道对应的基函数符号字典
    
    # 将每一个价键结构的活性轨道编号转化为其对应的原子轨道类型
    mapped_values = [] 

    for row in ac_orb:
        mapped_row = [orb_basis_dict.get(value, 'Unknown') for value in row]
        mapped_values.append(mapped_row)
        

    quam_nun_dict = {'1S':[1,0,0,0], '2S':[2,0,0,0], '3S':[3,0,0,0], '4S':[4,0,0,0], '2PX':[2,1,0,0], '2PY':[2,0,1,0],'2PZ':[2,0,0,1],'3PX':[3,1,0,0],'3PY':[3,0,1,0],'3PZ':[3,0,0,1],'4PX':[4,1,0,0],'4PY':[4,0,1,0],'4PZ':[4,0,0,1],'3DXX':[3,2,0,0],'3DXY':[3,1,1,0],'3DXZ':[3,1,0,1],'3DYZ':[3,0,1,1],'3DYY':[3,0,2,0],'3DZZ':[3,0,0,2]}

    # 将每一个价键结构的基函数符号转为量子数
    quan_num = []
    for row in mapped_values:
        quan_num_row = [quam_nun_dict.get(basis, 'Unknow') for basis in row]
        quan_num.append(quan_num_row)

    # z_matrix = []
    # inner_crd = get_inner_crd(R)# 获取活性轨道的内坐标
    # for i in range(len(ac_orb_atom)):
    #     m = inner_crd[ac_orb_atom[i]]
    #     z_matrix.append(m)

    active_space_matrix = []
    for i in range(len(quan_num)):
        # # 两种活性矩阵，第一种是以为n,i,j,k+活性原子索引
        active_space_m = np.concatenate((quan_num[i], ac_orb_atom[i].reshape(-1,1)), axis=1)
        # padding_zero = np.zeros((R.shape[0] - nae,5))
        # active_space_m = np.vstack((active_space_m, padding_zero))
        active_space_matrix.append(active_space_m) 
    
    #     #   第二种：n,i,j,k+内坐标
    #     x = np.concatenate((quan_num[i], z_matrix[i]), axis=1)
    #     active_space_matrix.append(x)
    
    return  active_space_matrix

# active_space('/home/xiatao/vbnet/241_VBSCF.xmo',ac_orb, nao,R)


def active_conv(filename):
    vb2class, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs = read_data_from_xmo(filename)
    aca, aco = orb_atom(filename) # 活性原子编号，活性轨道编号

    orb_atom_dict = dict(zip(aco, aca)) 
    vfunc = np.vectorize(lambda x: orb_atom_dict[x]) # 函数向量化

    ac_orb_atom = vfunc(ac_orb) # 活性轨道编号转化为原子编号
    ac_orb_atom = ac_orb_atom - 1
    ac_orb = ac_orb - np.min(ac_orb)
    R_active = []
    for i in range(len(ac_orb_atom)):
        R_ = R[ac_orb_atom[i]]
        R_active.append(R_)

    # 计算活性电子的edge_index
    active_edge_index = []
    e = np.zeros((nae, nae))
    
    for i in range(len(ac_orb)):
        e1 = np.copy(e)
        if (ac_orb.shape[1] % 2 == 0): # 偶数个活性电子
            for j in range(0, ac_orb.shape[1] - 1, 2):
                a = ac_orb[i][j]
                b = ac_orb[i][j+1]
                e1[a, b] = 1
                e1[b, a] = 1
        else: #如果奇数个电子
            for j in range(0, ac_orb.shape[1] - 1, 2):
                a = ac_orb[i][j]
                b = ac_orb[i][j+1]
                e1[a, b] = 1
                e1[b, a] = 1

        active_edge_index.append(e1)
    active_edge_index = np.array(active_edge_index)

    active_adjacency=[]
    for x in active_edge_index:
        edge_index_temp = sp.coo_matrix(x)
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
        active_adjacency.append(indices)

    # 计算活性电子的bond_order
    # active_bd = []

    # for i in range(len(ac_orb)):
    #     bd1 = np.copy(e)
    #     if (ac_orb.shape[1] % 2) == 0:
    #         for j in range(0, ac_orb.shape[1] - 1, 2):
    #             a = ac_orb[i][j]
    #             b = ac_orb[i][j+1]
    #             if a==b:
    #                 bd1[a][b] = 0
    #             else:
    #                 bd1[a][b] += 1
    #                 bd1[b][a] += 1
    #     else:
    #         for j in range(0, ac_orb.shape[1] - 2, 2):
    #             a = ac_orb[i][j]
    #             b = ac_orb[i][j+1]

    #             if a == b:
    #                 bd1[a][b] = 0
    #             else:
    #                 bd1[a][b] += 1
    #                 bd1[b][a] += 1
    #     active_bd.append(bd1)
    # active_bd = np.array(active_bd)

    return active_adjacency, R_active
# print(active_conv('/home/xiatao/vbnet/data/C9H10_8e_8o.xmo'))

def graph_generation(local_folder_path, vbscf_folder_path):
    # 存放在两个文件夹
    # 遍历连个文件夹
    local_file_names = os.listdir(local_folder_path)
    vbscf_file_names = os.listdir(vbscf_folder_path)

    node_feat = []
    edge_feat = []
    vb_2=[]
    lowdin_weight = []
    adjacency = []
    coor = []
    for i in range(len(local_file_names)):
        numbers = re.findall(r'\d+', vbscf_file_names[i])
        cid = ''.join(numbers)
        print(cid)
        node_feat_single_file, edge_feat_single_file = LMO_coffe(os.path.join(local_folder_path, str(cid)+'_boys.xmo'), os.path.join(vbscf_folder_path, str(cid)+'_VBSCF.xmo'))
        vb2class_single_file, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W_single_file, inverse_W, renormalized_W,vbs = read_data_from_xmo(os.path.join(vbscf_folder_path, str(cid)+'_VBSCF.xmo'))
        edge_index_single_file = edge_index(os.path.join(vbscf_folder_path, str(cid)+'_VBSCF.xmo'), cov_factor=1.2)
        
        # 分子坐标
        for i in range(len(lowdin_W_single_file)):
            coor.append(R)
        # 2分类标签
        for x in vb2class_single_file:
            vb_2.append(x)
        # lowdin_weight标签
        for x in lowdin_W_single_file:
            lowdin_weight.append(x)
        # node feature
        for x in node_feat_single_file:
            node_feat.append(x)
        # edge feature
        for x in edge_feat_single_file:
            edge_feat.append(x)
        # adjacency martrix
        for x in edge_index_single_file:
            adjacency.append(x)
    
    return coor, vb_2, lowdin_weight, node_feat, edge_feat, adjacency


# graph_generation('/home/xiatao/vbnet/data/LOCAL', '/home/xiatao/vbnet/data/VBSCF')

def process_files_in_folder(vbscf_folder_path):
    # 获取文件夹中的所有文件名
    
    # 遍历每个文件名
    pos=[]
    lowdin_weight = []
    atom_number = [] # 原子序号
    N_atom = [] # 原子数目
    adjacency = [] # 邻接矩阵
    bd = []#键级
    edge_feat = []
    Nae = []
    atom_charge = []
    log_lowdin_weight = []
    linear_lowdin_weight = []
    vbclass2 = []
    AE = []
    node_feat = []
    vbscf_file_names = os.listdir(vbscf_folder_path)
    for i in range(len(vbscf_file_names)):
        numbers = re.findall(r'\d+', vbscf_file_names[i])
        cid = ''.join(numbers)
        print(cid)
        vb2c, atomic_symbols, Z, N, atom_charges, R, W, nc_orb, ac_orb, nae, norb, nab, nao, lowdin_W, vbs = read_data_from_xmo(os.path.join(vbscf_folder_path, str(cid)+'_VBSCF.xmo'))
        NSE, NSAE, RBL = BD(os.path.join(vbscf_folder_path, str(cid)+'_VBSCF.xmo'), cov_factor=1.2) # 键级
        Charge, N_AE, N_AB, N_E = charge(os.path.join(vbscf_folder_path, str(cid)+'_VBSCF.xmo')) # 电荷
        natom = np.tile(N, (len(nc_orb),1))
        N = np.tile(N, (len(nc_orb)-1 ,1)) # 其中每个元素表示每一个价键结构的原子数目
        Z = np.tile(Z, (len(nc_orb),1)) # 原子序号
        W = np.array(W) #结构权重


        for i in range(len(vb2c)):
            vbclass2.append(vb2c[i])
            # N_atom.append(natom[i])
            lowdin_weight.append(lowdin_W[i])
            # log_lowdin_weight.append(log_lowdin_W[i])
            # linear_lowdin_weight.append(linear_lowdin_w[i])
            # atom_number.append(Z[i])
            # atom_charge.append(C[i])

            # nf = np.concatenate((Z[i].reshape(-1,1), Charge[i].reshape(-1,1), N_AE[i].reshape(-1,1), N_AB[i].reshape(-1, 1), N_E[i].reshape(-1,1)), axis=1)

            # Z, NAE, NNE, NAO
            nf = np.concatenate((Z[i].reshape(-1,1), N_AE[i].reshape(-1,1), (N_E[i] - N_AE[i]).reshape(-1,1) ,N_AB[i].reshape(-1, 1)), axis=1)

            node_feat.append(nf)

            pos.append(np.concatenate((Z[i].reshape(-1, 1), R), axis=1))
            
            edge_index_NSE = sp.coo_matrix(NSE[i])
            nse = 2 * edge_index_NSE.data  # number of shared electrons
            indices = np.vstack((edge_index_NSE.row, edge_index_NSE.col))  # 我们真正需要的coo形式
            nsae = 2* (NSAE[i])[indices[0], indices[1]] # get active electron information
            rbl = (RBL[i])[indices[0], indices[1]]
            nsne = nse - nsae # number of nonactive electron
            # concate(nsne, nsae, rbl)
            bond_feat = np.concatenate((nsne.reshape(-1,1), nsae.reshape(-1, 1), rbl.reshape(-1,1)) ,axis=1) # 
            edge_feat.append(bond_feat)
            
            adjacency.append(indices)
    
    # np.savez('train_data.npz',vbclass2=vbclass2, node_feat=node_feat, pos=pos, edge_feat=edge_feat, adjacency=adjacency, lowdin_weight=lowdin_weight)
    

    return  vbclass2, node_feat, pos, edge_feat, adjacency, lowdin_weight


# vbclass2, node_feat, pos, edge_feat, adjacency, lowdin_weight = process_files_in_folder('/home/xiatao/vbnet/data/test')
# print(edge_feat[50])
# print(edge_feat[0])
# np.save('linear_lowdin_weight.npy', linear_lowdin_weight)

# 从文件中加载矩阵
# loaded_matrix = np.load('linear_lowdin_weight.npy')
# loaded_matrix = (loaded_matrix  - 1) / 5
# print("initial mean", np.mean(loaded_matrix))
# print("initial:",np.std(loaded_matrix))
# loaded_matrix = 10 * loaded_matrix + 1
# print(np.max(loaded_matrix))
# print(np.min(loaded_matrix))
# print(np.mean(loaded_matrix))
# print("linear",np.std(loaded_matrix))
# vbscf_file_names = os.listdir('/home/xiatao/vbnet/data/VBSCF')
# print(len(vbscf_file_names))
# 计算加载的矩阵的平均值
# for i in range(40):
#     print(loaded_matrix[i] - np.mean(loaded_matrix))


