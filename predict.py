import torch
from tqdm import tqdm
from rdkit import Chem
import ctypes
from vbinfo_ctypes import struct_VbInfo
import os.path as osp
import scipy.sparse as sp
import numpy as np
from sklearn.utils import shuffle
from torch_scatter import scatter
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

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

def AC(Z, R, cov_factor):
    """
    compute AC matrix of a xyz corrdinates, from a xmo file

    purpose: give steleton connection

    根据共价半径进行判断骨架结构

    """
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
    for i in range(num_atoms):
        R_i = pt.GetRcovalent(Z[i]) * cov_factor # 原子i的共价半径
        for j in range(i+1 , num_atoms):
            R_j = pt.GetRcovalent(Z[j]) * cov_factor # 原子j的共价半径
            if  dist_matrix[i, j] <= (R_i +R_j) :
                ac[i, j] = 1
                ac[j, i] = 1
    return ac 


def cal_edge_index(aca, aco, ac_orb, ac):
    """
    compute adjacency matrix of a vb structure
    aca:活性原子编号
    aco:活性轨道编号
    ac_orb: 价键结构序列中的活性部分
    ac:由AC函数计算出的骨架连接
    """     
    orb_atom_dict = dict(zip(aco, aca)) 
    # print(orb_atom_dict)
    vfunc = np.vectorize(lambda x: orb_atom_dict[x]) # 函数向量化

    ac_orb_atom = vfunc(ac_orb) # 活性轨道编号转化为原子编号
    
    #创建空列表存储邻接矩阵
    adjacency = []
    for i in range(ac_orb_atom.shape[0]):
        ac1 = np.copy(ac)

        if (ac_orb_atom.shape[1] % 2) == 0: #如果是偶数个活性电子
            for j in range(0, ac_orb_atom.shape[1] - 1, 2):
                    a = ac_orb_atom[i][j]
                    b = ac_orb_atom[i][j+1]
                    if a != b: # 某一轨道占据了两个电子，即某个   
                        ac1[a-1,b-1] = 1
                        ac1[b-1,a-1] = 1
        else: # 如果是奇数个活性电子
            for j in range(0, ac_orb_atom.shape[1] - 1, 2):
                    a = ac_orb_atom[i][j]
                    b = ac_orb_atom[i][j+1]              
                    if a != b:
                    #     break    
                    # else:    
                        ac1[a-1,b-1] = 1
                        ac1[b-1,a-1] = 1
        adjacency.append(ac1)# 将新计算的部分拼接到之前的结果后面
    adjacency = np.array(adjacency)
    # print(adjacency[-1])
    return adjacency, ac_orb_atom


def BD(ac, ac_orb_atom, Z, R, cov_factor=1.2):
    """
    compute bond order matrix of a vb structure
    ac : 初始邻接矩阵
    ac_orb_atom : 活性轨道对应的原子编号
    """  
    num_atoms = np.shape(ac)[0]
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
    
    # orb_atom_dict = dict(zip(aco, aca)) 
    # vfunc = np.vectorize(lambda x: orb_atom_dict[x]) # 函数向量化

    # ac_orb_atom = vfunc(ac_orb) # 活性轨道编号转化为原子编号
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

def check_cov(self, arr, nao, nae):
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


def cal_charge(Z, nae, nao, ac_orb_atom, aca, aco):
    """
    compute charge  of every atom for a vb structure , base on active space
    return : Charge, N_AE, N_AB, N_E
    """      
    #    aca, aco = orb_atom(filename, vbscf=True, localize=False) # 活性原子编号，活性轨道编号
    
    orb_atom_dict = dict(zip(aco, aca)) 
    # vfunc = np.vectorize(lambda x: orb_atom_dict[x]) # 函数向量化
    
    n_ac_orb = np.zeros(len(Z)).astype(int)
    for x in list(orb_atom_dict.values()):
        n_ac_orb[x - 1] += 1
    
    # ac_orb_atom = vfunc(ac_orb) # 活性轨道编号转化为原子编号
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

    # charge = [] # 每个原子的电荷     
    N_AE = [] # 每个原子的活性电子数目
    N_AB = [] # 每个原子的活性轨道数目
    N_E = [] # 每个原子的电子数目
    
    x = Z - ae # 惰性电子 = 每个原子的初始电子数 - 每个原子的活性电子数 ，下面根据活性空间计算每个价键结构中每个原子的总电子数目
    
    for vbs in ac_orb_atom:
        # c1 = np.copy(c) # 初始化电荷矩阵
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
        # charge.append(c1) # 每个原子的形式电荷
        N_AE.append(ae_0) # 每个原子的活性电子数目
        N_E.append(c2) # 每个原子的电子数目

    # charge = np.array(charge).astype(int) #转为numpy 三维数组
    N_AE = np.array(N_AE).astype(int)
    N_AB = np.array(N_AB).astype(int)
    N_E  =np.array(N_E).astype(int)

    return N_AE, N_AB, N_E


label_counts = {}
prev_value = None  # 用于存储前一个 value 的值

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

def active_space(R, aco, ac_orb, ac_orb_atom, active_orbital):
    """
    计算每一个价键结构的活性空间矩阵,可以是n,i,j,k+原子索引，或者是n,i,j,k+内坐标
    """

    orb_basis_dict= dict(zip(aco, active_orbital)) # 活性轨道对应的基函数符号字典
    
    # 将每一个价键结构的活性轨道编号转化为其对应的原子轨道类型
    mapped_values = [] 

    for row in ac_orb:
        mapped_row = [orb_basis_dict.get(value, 'Unknown') for value in row]
        mapped_values.append(mapped_row)
        

    quam_nun_dict = {'1S':[1,0,0,0], '2S':[2,0,0,0], '3S':[3,0,0,0], '4S':[4,0,0,0], '2PX':[2,1,0,0], '2PY':[2,0,1,0],'2PZ':[2,0,0,1],'3DXX':[3,2,0,0],'3DXY':[3,1,1,0],'3DXZ':[3,1,0,1],'3DYZ':[3,0,1,1],'3DYY':[3,0,2,0],'3DZZ':[3,0,0,2]}

    # 将每一个价键结构的基函数符号转为量子数
    quan_num = []
    for row in mapped_values:
        quan_num_row = [quam_nun_dict.get(basis, 'Unknow') for basis in row]
        quan_num.append(quan_num_row)
    
    z_matrix = []
    inner_crd = get_inner_crd(R)# 获取活性轨道的内坐标
    for i in range(len(ac_orb_atom)):
        m = inner_crd[ac_orb_atom[i]]
        z_matrix.append(m)

    active_space_matrix = []
    for i in range(len(quan_num)):
    #     # 两种活性矩阵，第一种是以为n,i,j,k+活性原子索引
        # active_space_m = np.concatenate((quan_num[i], ac_orb_atom[i].reshape(-1,1)), axis=1)
        # # padding_zero = np.zeros((R.shape[0] - nae,5))
        # # active_space_m = np.vstack((active_space_m, padding_zero))
        # active_space_matrix.append(active_space_m) 
        
        # 第二种：n,i,j,k+内坐标
        x = np.concatenate((quan_num[i],z_matrix[i], ac_orb_atom[i].reshape(-1,1)), axis=1)
        active_space_matrix.append(x)
    
    return active_space_matrix

def genstr(nao, nae , numl , nel):
        """
        genereate valence bond structure series number 
        """
        vblib = ctypes.CDLL("/home/xiatao/xmvb/install-dir/lib/libxeda.exe.so")

        # 设置genstr函数的参数类型和返回类型
        vblib.genstr.argtypes = [ctypes.POINTER(struct_VbInfo)]  # genstr需要一个指向VbInfo结构体的指针
        vblib.genstr.restype = ctypes.c_int  # 假设genstr返回一个整数
        
        # 创建VbInfo结构体的实例
        vb_str = struct_VbInfo()
        vb_str.nao = nao
        vb_str.nae = nae
        vb_str.nmul = numl
        vb_str.nel = nel
        vb_str.strclass = b"FULL"

        # 调用genstr函数
        result = vblib.genstr(ctypes.byref(vb_str))

        vb_structure=[]
        # print(f"vb_str.nor={vb_str.nor}")
        # print("valence bond structure:")
        for i in range(0, vb_str.nstr):
            for j in range(0, vb_str.nel):
                vb_structure.append(vb_str.ntstr[i * vb_str.nel + j])
            #     print(f"{vb_str.ntstr[i*vb_str.nel+j]}",end=" ")
            # print("")
        
        vb_structure = [vb_structure[i:i+nel] for i in range(0,len(vb_structure),nel)] # 
       
        ac_orb = [sub_list[-nae:] for sub_list in vb_structure] # 只取活性部分
        
        return  ac_orb


def read(data):
    """
    read atom type and 3D corrdinates
    """
    lines = data.strip().split('\n') 
    Z = []
    R = []
    # 逐行解析数据
    for line in lines:
        parts = line.split()  # 使用空格分割每行
        atom_num = int_atom(parts[0])  # 第一个元素是原子类型
        coordinates = list(map(float, parts[-3:]))  # 将剩余部分转换为浮点数列表
        R.append(coordinates)
        Z.append(atom_num)
    nel = sum(Z)
    return Z, np.array(R), nel

def feature_matrix(DATA, active_orbital, nae, nao, numl):

    Z, R, nel = read(DATA)
    
    aco = np.arange(int((nel-nae)/2 + 1), int((nel-nae)/2 + 1 + int(nao))).astype(int)

    aca = active_orbital

    # active_orbital_num = list(active_orbital_dict.keys()) # 活性轨道对应的原子编号
    
    ac_orb = genstr(nao, nae , numl , nel) # 价键结构活性空间序列

    ac = AC(Z, R, cov_factor=1.2)
    # print(ac)
    adjacency, ac_orb_atom = cal_edge_index(aca, aco, ac_orb, ac)

    # number of shared electron 
    # number of active shared electron
    # relative bond length 
    NSE, NSAE, RBL = BD(ac, ac_orb_atom, Z, R) # for edge_feat

    # charge, number of active electron, number of active orbital, number of electron
    N_AE, N_AB, N_E = cal_charge(Z, nae, nao, ac_orb_atom, aca, aco) 
    # active_space_matrix = active_space(R, aco, ac_orb, ac_orb_atom, active_orbital)

    # num_atom = len(R)
    # N_sp = np.tile(num_atom, (len(ac_orb)-1, 1))
    # split = np.cumsum(N_sp)
    # N = np.tile(num_atom, (len(ac_orb), 1))

    Z = np.tile(Z, (len(ac_orb), 1))
    # R = np.tile(R, (len(ac_orb), 1))
    # R = np.split(R, split)
    # print(len(R))
    # nel = np.tile(nel, (len(ac_orb), 1) )
    # nae = np.tile(nae, (len(ac_orb), 1))


    node_feat=[]
    edge_feat=[]
    edge_index=[]
    for i in range(len(adjacency)):
        # node feature
        # 原子序数，活性电子数，非活性电子数，活性轨道数
        nf = np.concatenate((Z[i].reshape(-1,1), N_AE[i].reshape(-1,1), (N_E[i] - N_AE[i]).reshape(-1,1) ,N_AB[i].reshape(-1, 1)), axis=1)
        node_feat.append(nf)

        # edge feature
        edge_index_NSE = sp.coo_matrix(NSE[i])
        nse = 2 * edge_index_NSE.data  # number of shared electrons
        indices = np.vstack((edge_index_NSE.row, edge_index_NSE.col))  # 我们真正需要的coo形式
        nsae = 2* (NSAE[i])[indices[0], indices[1]] # get active electron information
        rbl = (RBL[i])[indices[0], indices[1]]
        nsne = nse - nsae # number of nonactive electron
        # concate(nsne, nsae, rbl)
        bond_feat = np.concatenate((nsne.reshape(-1,1), nsae.reshape(-1, 1), rbl.reshape(-1,1)) ,axis=1) # 
        edge_feat.append(bond_feat)
        
        edge_index.append(indices)
    return node_feat, edge_feat, edge_index, ac_orb

class vb_test(InMemoryDataset):
    r"""   
    convert vb structure to graph
    """
    def __init__(self, root = '/home/xiatao/vbnet/GraphGPS-main/Predict/' , transform = None, pre_transform = None, pre_filter = None):
        
        self.folder = osp.join(root)
        # self.train_folder = train_floder
        super(vb_test, self).__init__(self.folder, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'train_data.npz'

    @property
    def processed_file_names(self):
        return 'train_data.pt'
    # def download(self):
    #     download_url(self.url, self.raw_dir)

    def process(self):
        node_feat, edge_feat, adjacency, ac_orb = feature_matrix(molecule, active_orbital, nae, nao, numl)
             
        print('convert VB structure to grpah ...')
        data_list = []

        index = torch.arange(0, len(node_feat), step=1)
        
        for i in tqdm(range(len(node_feat))): # len(N):价键结构的总数
            # get feature matrix 
            
            node_feat_i = torch.tensor(node_feat[i], 
                                       dtype=torch.float)
            
            edge_feat_i = torch.tensor(edge_feat[i], 
                                       dtype=torch.float)
            
            adjacency_i = torch.tensor(adjacency[i], 
                                       dtype=torch.long)
            
            ac_orb_i = torch.tensor(ac_orb[i], 
                                    dtype=torch.int)
            
            orb_index_i = index[i].clone().detach()
            # max_lenght_i = max_length[i].clone().detach().requires_grad_(True)
            # log_lowdin_weight_i = torch.tensor(log_lowdin_weight[i], dtype=torch.float)
            # linear_lowdin_weight_i = torch.tensor(linear_lowdin_weight[i], dtype=torch.float)
            # edge_num_i = torch.tensor(edge_num[i],dtype=torch.long)

            data = Data(x=node_feat_i, edge_attr=edge_feat_i , edge_index=adjacency_i, orb = ac_orb_i, y = orb_index_i)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])

# if __name__ == '__main__':

    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     #构型（支持xyz、gamess格式）
#     molecule = """
# H 0.0 0.0 0.0
# H 0.0 0.0 1.0
#     """
    
# #     # active space 
#     nao = 2
#     nae = 2
#     numl = 1
#     # ac_orb = genstr(nao=18, nae=20, numl=1, nel=268)
# # #     # active orbital
# # #     # 通过创建字典的方式来表示活性轨道及其所属原子。key表示原子序号，value表示活性轨道。
# # #     # 例如 1 : ['2PZ'] 表示1号原子的活性轨道是2PZ
#     active_orbital = [1, 2]
                  
#     # Z, R, nel = read(molecule)
#     # ac_orb = genstr(nao=nao, nae=nae, numl=numl, nel=nel)
 
#     # must refer the task name
#     dataset = vb_test(root = '/home/xiatao/vbnet/GraphGPS-main/Predict/' ,transform=None, pre_transform=None, pre_filter=None)

    # if torch.cuda.is_available():
    #         model = model.cuda()
#     model.load_state_dict(torch.load('/home/xiatao/vbnet/model/crossentrop/model_0.98.pt'))
#     model.eval()

    # preds = torch.Tensor([])
    # for step, batch_data in enumerate(tqdm(predict_data)):
    #     batch_data = batch_data
    #     out = model(batch_data)
    #     prob, pred = torch.max(out, dim=1)
    #     preds = torch.cat([preds, pred.detach_()], dim=0)
    
    # a = (nel-nae)//2 # number of non-active orbital
    # nae = np.tile(nae, (len(preds), 1))
    # vb_str = dataset.vb_str
    # vb_str = torch.split(vb_str, tuple(nae.reshape(-1)))

    # if_predict_weight = False

    # if if_predict_weight:
    #     preds = list(enumerate(preds.squeeze(-1)))
    #     preds = sorted(preds, key=lambda x: x[1], reverse=True)
    #     for i, w in preds:
    #         if a !=0: # if exits non-active orbital
    #             output_str = ' '.join([str(num.item()) for num in vb_str[i]])
    #             print('1:{}'.format(a) + ' ' + output_str + ' ' + '  {}'.format(w) )
                
    #         else:
    #             print(output_str + ' ' + w)
    
    # else:
    # # classier
    #     indice = torch.nonzero(preds).squeeze()
    #     important_vbstr = [vb_str[i] for i in indice]
    #     print("number of important valence bond structure:", len(important_vbstr))
    #     for i in range(len(important_vbstr)):
    #         output_str = ' '.join([str(num.item()) for num in important_vbstr[i]])
    #         if a !=0:
    #             print('1:{}'.format(a) + ' ' + output_str )
    #         else:
    #             print(output_str)
    
      



