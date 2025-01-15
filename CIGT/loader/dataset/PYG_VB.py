import os.path as osp
import numpy as np
from tqdm import tqdm
import math
import torch
from sklearn.utils import shuffle
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from CIGT.loader.dataset import vb_data_tran
# import vb_data_tran

# from torch_geometric.utils import degree
# from torch_geometric.loader import DataLoader

class VBData(InMemoryDataset):
    r"""   
    convert vb structure to graph
    """
    def __init__(self, root = '/home/xiatao/vbnet/GraphGPS-main/graphgps/', task='regression', transform = None, pre_transform = None, pre_filter = None):
        self.task = task
        if task == 'regression':
            self.folder = osp.join(root, task)
        elif task == 'classification_binary':
            self.folder = osp.join(root, task)
        else:
            self.folder = osp.join(root, task)
        # self.train_folder = train_floder
        super(VBData, self).__init__(self.folder, transform, pre_transform, pre_filter)
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
        """prosess valence bond structure to graph"""

        # pos, vb_2, lowdin_weight, node_feat, edge_feat, adjacency = data_tran.graph_generation('/home/xiatao/vbnet/data/LOCAL', '/home/xiatao/vbnet/data/VBSCF')
        vbclass2, node_feat, pos, edge_feat, adjacency, lowdin_weight = vb_data_tran.process_files_in_folder('/home/xiatao/vbnet/data/VBSCF')
        # vb_train_data = np.load('train_data.npz')
        # vbclass2 = vb_train_data['vbclass2']
        # node_feat = vb_train_data['node_feat']
        # pos = vb_train_data['pos']
        # edge_feat = vb_train_data['edge_feat']
        # adjacency = vb_train_data['adjacency']
        # lowdin_weight = vb_train_data['lowdin_weight']
        # # edge_num = []
        # for x in adjacency:
        #     edge_num_j = x.shape[1]
        #     edge_num.append(edge_num_j)

        # max_length = []
        # for i in range(len(pos)):
        #     if np.all(adjacency[i] == 0) or adjacency[i].ndim == 0:
        #         ri = pos[i].reshape(-1, 1, 3)  # 将 R 扩展为 (num_atoms, 1, 3)
        #         rj = pos[i].reshape(1, -1, 3)  # 将 R 扩展为 (1, num_atoms, 3)
        #         distances = np.linalg.norm(ri - rj, axis=2)  # 计算欧几里得距离
        #         max_l = torch.tensor(np.max(distances))                  
        #     else:
        #         m, n = torch.from_numpy(adjacency[i])
        #         dist = (torch.from_numpy(pos[i])[m] - torch.from_numpy(pos[i])[n]).norm(dim=-1)
        #         max_l = torch.max(dist)
        #     max_length.append(max_l)
        
        print('convert VB structure to grpah ...')

        data_list = []

        for i in tqdm(range(len(pos))): # len(N):价键结构的总数
            # get feature matrix 

            pos_i = torch.tensor(pos[i], dtype=torch.float)
            
            vbclass2_i = torch.tensor(vbclass2[i], dtype=torch.int)

            # lowdin_weight_i = torch.tensor((torch.log10(lowdin_weight[i]+ torch.tensor(1e-4)) + torch.tensor(4.0)) / torch.tensor(4.0), dtype=torch.float)
            lowdin_weight_i = torch.tensor(lowdin_weight[i], dtype=torch.float)
            node_feat_i = torch.tensor(node_feat[i], dtype=torch.float)
            edge_feat_i = torch.tensor(edge_feat[i], dtype=torch.float)
            adjacency_i = torch.tensor(adjacency[i], dtype=torch.long)
            # max_lenght_i = max_length[i].clone().detach().requires_grad_(True)
            # log_lowdin_weight_i = torch.tensor(log_lowdin_weight[i], dtype=torch.float)
            # linear_lowdin_weight_i = torch.tensor(linear_lowdin_weight[i], dtype=torch.float)
            # edge_num_i = torch.tensor(edge_num[i],dtype=torch.long)
            if self.task == 'classification_binary':
                data = Data(pos=pos_i, vbclass2 = vbclass2_i, y = vbclass2_i, x=node_feat_i, edge_attr=edge_feat_i , edge_index=adjacency_i)
            elif self.task == 'regression':
                data = Data(pos=pos_i, vbclass2 = vbclass2_i, y = lowdin_weight_i, x=node_feat_i, edge_attr=edge_feat_i , edge_index=adjacency_i)
            else:
                data = Data(pos=pos_i, vbclass2 = vbclass2_i, y = lowdin_weight_i, x=node_feat_i, edge_attr=edge_feat_i , edge_index=adjacency_i)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size],dtype=torch.long), torch.tensor(ids[train_size:train_size + valid_size],dtype=torch.long), torch.tensor(ids[train_size + valid_size:],dtype=torch.long)
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

if __name__ == '__main__':
    dataset = VBData(root = '/home/xiatao/vbnet/GraphGPS-main/', task='regression', transform = None, pre_transform = None, pre_filter = None)
    
    print(torch.mean(dataset.y))
    # dataset.split_idxs = dataset.get_idx_split(data_size=len(dataset.y), train_size=math.floor(0.9*len(dataset.y)), valid_size=math.floor(0.05*len(dataset.y)), seed=42)
    # n = 0
    # for i in range(len(dataset)):
    #     n+=dataset[i].num_nodes
    # # 假设你已经有了 edge_index 和 x（节点特征）
    # print(n/(len(dataset)))

    # 计算每个节点的度数


