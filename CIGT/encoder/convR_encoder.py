# import networkx as nx
# import torch
# import torch.nn.functional as F
# from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.register import register_edge_encoder
# from torch_geometric.utils import to_dense_adj
# from rdkit import Chem
# # Permutes from (batch, node, node, head) to (batch, head, node, node)
# BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)

# # Inserts a leading 0 row and a leading 0 column with F.pad
# INSERT_GRAPH_TOKEN = (1, 0, 1, 0)


# def convR_pre_processing(data, conv_factor=1.2):
#     """Implementation of Graphormer pre-processing. Computes in- and out-degrees
#     for node encodings, as well as spatial types (via shortest-path lengths) and
#     prepares edge encodings along shortest paths. The function adds the following
#     properties to the data object:

#     - spatial_types
#     - graph_index: An edge_index type tensor that contains all possible directed edges 
#                   (see more below)
#     - shortest_path_types: Populates edge attributes along all shortest paths between two nodes

#     Similar to the adjacency matrix, any matrix can be batched in PyG by decomposing it
#     into a 1D tensor of values and a 2D tensor of indices. Once batched, the graph-specific
#     matrix can be recovered (while appropriately padded) via ``to_dense_adj``. We use this 
#     concept to decompose the spatial type matrix and the shortest path edge type tensor
#     via the ``graph_index`` tensor.

#     Args:
#         data: A PyG data object holding a single graph
#         distance: The distance up to which types are calculated

#     Returns:
#         The augmented data object.
#     """

#     N = data.num_nodes
#     graph_index = torch.empty(2, N ** 2, dtype=torch.long)

#     # full connected graph_index
#     for i in range(N):
#         for j in range(N):
#             graph_index[0, i * N + j] = i
#             graph_index[1, i * N + j] = j
    
#     pos = data.pos  # (num_nodes, 3) tensor
#     atomic_numbers = data.x[:, 0].long()  # Atomic numbers (num_nodes,) tensor
    
#     # Step 2: Get node pairs from the edge_index
#     row, col = graph_index[0], graph_index[1]
    
#     # Step 3: Compute Euclidean distances between connected nodes
#     edge_distances = torch.norm(pos[row] - pos[col], p=2, dim=1)  # (num_edges,) tensor
    
#     # Step 4: Calculate covalent radii for each node based on atomic numbers
#     pt = Chem.GetPeriodicTable()
#     radii_row = torch.tensor([pt.GetRcovalent(int(a)) for a in atomic_numbers[row]], device=data.pos.device)
#     radii_col = torch.tensor([pt.GetRcovalent(int(a)) for a in atomic_numbers[col]], device=data.pos.device)
    
#     # Step 5: Sum of covalent radii for each edge (radii_row + radii_col)
#     radii_sum = (radii_row + radii_col) * conv_factor  # (num_edges,) tensor
    
#     # Step 6: Compute final edge features by dividing radii_sum by edge_distances
#     # Ensure we avoid division by zero
#     edge_distances = edge_distances.clamp(min=1e-6)  # Avoid division by zero
#     spatial_types= radii_sum / edge_distances  # (num_edges,) tensor

#     data.spatial_types = spatial_types.unsqueeze(-1)
#     data.graph_index = graph_index

#     return data


# class convR_BiasEncoder(torch.nn.Module):
#     def __init__(self, num_heads: int, use_graph_token: bool = False):
#         """Implementation of the bias encoder of Graphormer.
#         This encoder is based on the implementation at:
#         https://github.com/microsoft/Graphormer/tree/v1.0
#         Note that this refers to v1 of Graphormer.

#         Args:
#             num_heads: The number of heads of the Graphormer model
#         """
#         super().__init__()
#         self.num_heads = num_heads

#         # Takes into account disconnected nodes
#         self.spatial_encoder = torch.nn.Linear(1, num_heads)
        
#         # self.edge_dis_encoder = torch.nn.Embedding(
#         #     num_spatial_types * num_heads * num_heads, 1)
#         # self.edge_encoder = torch.nn.Embedding(num_edge_types, num_heads)

#         self.use_graph_token = use_graph_token
#         if self.use_graph_token:
#             self.graph_token = torch.nn.Parameter(torch.zeros(1, num_heads, 1))
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.spatial_encoder.weight.data.normal_(std=0.02)
#         # self.edge_encoder.weight.data.normal_(std=0.02)
#         # self.edge_dis_encoder.weight.data.normal_(std=0.02)
#         if self.use_graph_token:
#             self.graph_token.data.normal_(std=0.02)

#     def forward(self, data):
#         """Computes the bias matrix that can be induced into multi-head attention
#         via the attention mask.

#         Adds the tensor ``attn_bias`` to the data object, optionally accounting
#         for the graph token.
#         """
#         # To convert 2D matrices to dense-batch mode, one needs to decompose
#         # them into index and value. One example is the adjacency matrix
#         # but this generalizes actually to any 2D matrix
#         spatial_types: torch.Tensor = self.spatial_encoder(data.spatial_types)
#         spatial_encodings = to_dense_adj(data.graph_index,
#                                          data.batch,
#                                          spatial_types)
#         bias = spatial_encodings.permute(BATCH_HEAD_NODE_NODE) # to suit attention matrix dimsion
        
#         if hasattr(data, "shortest_path_types"):
#             edge_types: torch.Tensor = self.edge_encoder(
#                 data.shortest_path_types)
#             edge_encodings = to_dense_adj(data.graph_index,
#                                           data.batch,
#                                           edge_types)

#             spatial_distances = to_dense_adj(data.graph_index,
#                                              data.batch,
#                                              data.spatial_types)
#             spatial_distances = spatial_distances.float().clamp(min=1.0).unsqueeze(1)

#             B, N, _, max_dist, H = edge_encodings.shape

#             edge_encodings = edge_encodings.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
#             edge_encodings = torch.bmm(edge_encodings, self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads))
#             edge_encodings = edge_encodings.reshape(max_dist, B, N, N, self.num_heads).permute(1, 2, 3, 0, 4)
#             edge_encodings = edge_encodings.sum(-2).permute(BATCH_HEAD_NODE_NODE) / spatial_distances
#             bias += edge_encodings

#         if self.use_graph_token:
#             bias = F.pad(bias, INSERT_GRAPH_TOKEN)
#             bias[:, :, 1:, 0] = self.graph_token
#             bias[:, :, 0, :] = self.graph_token

#         B, H, N, _ = bias.shape
#         data.attn_bias = bias.reshape(B * H, N, N)
#         return data


# def add_graph_token(data, token):
#     """Helper function to augment a batch of PyG graphs
#     with a graph token each. Note that the token is
#     automatically replicated to fit the batch.

#     Args:
#         data: A PyG data object holding a single graph
#         token: A tensor containing the graph token values

#     Returns:
#         The augmented data object.
#     """
#     B = len(data.batch.unique())
#     tokens = torch.repeat_interleave(token, B, 0)
#     data.x = torch.cat([tokens, data.x], 0)
#     data.batch = torch.cat(
#         [torch.arange(0, B, device=data.x.device, dtype=torch.long), data.batch]
#     )
#     data.batch, sort_idx = torch.sort(data.batch)
#     data.x = data.x[sort_idx]
#     return data


# class NodeEncoder(torch.nn.Module):
#     def __init__(self, embed_dim, num_in_degree, num_out_degree,
#                  input_dropout=0.0, use_graph_token: bool = True):
#         """Implementation of the node encoder of Graphormer.
#         This encoder is based on the implementation at:
#         https://github.com/microsoft/Graphormer/tree/v1.0
#         Note that this refers to v1 of Graphormer.

#         Args:
#             embed_dim: The number of hidden dimensions of the model
#             num_in_degree: Maximum size of in-degree to encode
#             num_out_degree: Maximum size of out-degree to encode
#             input_dropout: Dropout applied to the input features
#             use_graph_token: If True, adds the graph token to the incoming batch.
#         """
#         super().__init__()
#         self.in_degree_encoder = torch.nn.Embedding(num_in_degree, embed_dim)
#         self.out_degree_encoder = torch.nn.Embedding(num_out_degree, embed_dim)

#         self.use_graph_token = use_graph_token
#         if self.use_graph_token:
#             self.graph_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
#         self.input_dropout = torch.nn.Dropout(input_dropout)
#         self.reset_parameters()

#     def forward(self, data):
#         in_degree_encoding = self.in_degree_encoder(data.in_degrees)
#         out_degree_encoding = self.out_degree_encoder(data.out_degrees)

#         if data.x.size(1) > 0:
#             data.x = data.x + in_degree_encoding + out_degree_encoding
#         else:
#             data.x = in_degree_encoding + out_degree_encoding

#         if self.use_graph_token:
#             data = add_graph_token(data, self.graph_token)
#         data.x = self.input_dropout(data.x)
#         return data

#     def reset_parameters(self):
#         self.in_degree_encoder.weight.data.normal_(std=0.02)
#         self.out_degree_encoder.weight.data.normal_(std=0.02)
#         if self.use_graph_token:
#             self.graph_token.data.normal_(std=0.02)


# @register_edge_encoder("convR")
# class convREncoder(torch.nn.Sequential):
#     def __init__(self, *args, **kwargs):
#         encoders = [
#             convR_BiasEncoder(cfg.gt.num_heads,)
#         ]

#         super().__init__(*encoders)
