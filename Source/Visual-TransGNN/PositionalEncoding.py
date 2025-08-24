import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.preprocessing import MinMaxScaler

class PositionalEncoding(nn.Module):
    def __init__(self, latent_dim, num_users, num_items, device):
        super(PositionalEncoding, self).__init__()
        self.latent_dim = latent_dim
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.device = device
        
        # MLP for each encoding type
        self.spe_mlp = nn.Sequential(
            nn.Linear(1, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, latent_dim // 4)
        )
        
        self.de_mlp = nn.Sequential(
            nn.Linear(1, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, latent_dim // 4)
        )
        
        self.pre_mlp = nn.Sequential(
            nn.Linear(1, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, latent_dim // 4)
        )
        
        # Combination MLP
        self.comb_mlp = nn.Sequential(
            nn.Linear(latent_dim + (latent_dim // 4) * 3, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def compute_shortest_path(self, adj):
        # Convert sparse adj to dense for shortest path calculation
        adj_dense = adj.to_dense().cpu().numpy()
        dist_matrix = shortest_path(adj_dense, directed=False, unweighted=True)
        dist_matrix = torch.from_numpy(dist_matrix).float().to(self.device)
        return dist_matrix

    def compute_pagerank(self, adj, damping=0.85, max_iter=100):
        # Convert sparse adj to dense for pagerank calculation
        adj_dense = adj.to_dense().cpu()
        num_nodes = adj_dense.shape[0]
        
        # Initialize
        pr = torch.ones(num_nodes).to(self.device) / num_nodes
        out_degree = adj_dense.sum(dim=1).float().clamp(min=1)
        
        for _ in range(max_iter):
            # Power iteration
            new_pr = (1 - damping) / num_nodes + damping * (
                adj_dense.transpose(0, 1) @ (pr.cpu() / out_degree)
            ).to(self.device)
            
            # Check convergence
            if torch.norm(new_pr - pr, 1) < 1e-6:
                break
            pr = new_pr
            
        return pr

    def forward(self, x, adj, node_indices=None):
        # Convert sparse adj to dense if needed
        adj_dense = adj.to_dense() if adj.is_sparse else adj
        
        # Determine if processing users or items
        is_user = x.size(0) == self.num_users  # self.num_users cần được khai báo trong __init__
        
        if node_indices is None:
            node_indices = torch.arange(x.size(0)).to(self.device)
        
        # Degree encoding
        degrees = adj_dense.sum(dim=1).float().unsqueeze(1)[node_indices]
        de = self.de_mlp(degrees)
        
        # PageRank encoding
        pagerank = self.compute_pagerank(adj).unsqueeze(1)[node_indices]
        pre = self.pre_mlp(pagerank)
        
        # Shortest path encoding
        if not hasattr(self, 'dist_matrix'):
            self.dist_matrix = self.compute_shortest_path(adj)
        
        # For all nodes
        if is_user:
            # User nodes - use distance to themselves (0)
            spe_self = self.spe_mlp(torch.zeros(x.size(0), 1).to(self.device))
        else:
            # Item nodes - compute actual distances
            dists = self.dist_matrix[node_indices, node_indices].unsqueeze(1)
            spe_self = self.spe_mlp(dists)
        
        # Combine all encodings
        combined = torch.cat([x, spe_self, de, pre], dim=1)
        return self.comb_mlp(combined)