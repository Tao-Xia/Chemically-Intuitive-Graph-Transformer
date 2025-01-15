import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder

class Gaussian_Emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=2, num_gaussians=20):
        super(Gaussian_Emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians) # 创建一个等差数列,start开始，stop结束，包含nun_gaussians个数
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2 # 高斯分布函数的系数，-1/(2*(\sigma)^2)
        self.register_buffer('offset', offset) 
        
        #将 offset 注册为缓冲区，以确保在模型中保持不变。这是因为 offset 张量不属于模型的可学习参数，而只是用于计算高斯函数的中心位置，因此它不会被优化器更新。将其注册为缓冲区可以确保它始终处于模型的状态中，并且在保存和加载模型时也能正确地处理它。

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1) # [n_edges, n_gaussian]
        return torch.exp(self.coeff * torch.pow(dist, 2))
    

def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, ):
    """
    calculate sinc radial basis function:
    
    sin(n *pi*d/d_cut)/d
    """
    n = torch.arange(edge_size, device=edge_dist.device) + 1
    return torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi) / edge_dist.unsqueeze(-1)
      


@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        if cfg.dataset.name in ['MNIST', 'CIFAR10']:
            self.in_dim = 1
        elif cfg.dataset.name in ['VB']:
            self.in_dim = 2
        else:
            raise ValueError("Input edge feature dim is required to be hardset "
                             "or refactored to use a cfg option.")
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)
        self.Gaussian_Emb = Gaussian_Emb(0, 2, 20)

    def forward(self, batch):
        batch.edge_attr, batch.geo = batch.edge_attr[: , :2] , batch.edge_attr[: , -1]
        batch.geo = self.Gaussian_Emb(batch.geo)
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch
