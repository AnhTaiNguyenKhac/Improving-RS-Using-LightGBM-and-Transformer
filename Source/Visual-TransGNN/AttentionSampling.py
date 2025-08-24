import torch
import torch.nn as nn
import torch.nn.functional as F

# class AttentionSampling(nn.Module):
#     def __init__(self, alpha, top_k):
#         super().__init__()
#         self.alpha = alpha
#         self.top_k = top_k

#     def forward(self, embeds, adj):
#         # Step 1: Normalize
#         X = F.normalize(embeds, p=2, dim=1)

#         # Step 2: Semantic similarity
#         S_semantic = torch.mm(X, X.t())

#         # Step 3: Structural similarity
#         A_hat = adj.to_dense() + torch.eye(adj.size(0), device=adj.device)
#         S_structural = torch.mm(A_hat, S_semantic)

#         # Step 4: Combine
#         S = S_semantic + self.alpha * S_structural
#         S.fill_diagonal_(-float('inf'))

#         # Step 5: Top-k
#         topk_values, topk_indices = torch.topk(S, self.top_k, dim=1)

#         # Step 6: Build sparse adjacency matrix from top-k
#         row_idx = torch.arange(S.size(0), device=embeds.device).unsqueeze(1).expand(-1, self.top_k)
#         col_idx = topk_indices
#         indices = torch.stack([row_idx.reshape(-1), col_idx.reshape(-1)], dim=0)
#         values = torch.ones(indices.size(1), device=embeds.device)  # or use topk_values.flatten()

#         adj_topk = torch.sparse_coo_tensor(
#             indices,
#             values,
#             size=(S.size(0), S.size(0))
#         )

#         return adj_topk

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSampling(nn.Module):
    def __init__(self, alpha, top_k):
        super().__init__()
        self.alpha = alpha
        self.top_k = top_k

    def forward(self, embeds, adj):
        # Step 1: Normalize embeddings
        X = F.normalize(embeds, p=2, dim=1)

        # Step 2: Calculate semantic similarity
        S_semantic = torch.mm(X, X.t())

        # Step 3: Calculate structural similarity
        A_hat = adj.to_dense() + torch.eye(adj.size(0), device=adj.device)
        S_structural = torch.mm(A_hat, S_semantic)

        # Step 4: Combine similarities
        S = S_semantic + self.alpha * S_structural
        S.fill_diagonal_(-float('inf'))  # Mask self-similarity

        # Step 5: Get top-k most similar nodes
        topk_values, topk_indices = torch.topk(S, self.top_k, dim=1)
        
        # Step 6: Get unique sampled nodes (remove duplicates)
        sampled_nodes = torch.unique(topk_indices)
        
        # Step 7: Build sparse adjacency matrix for top-k connections
        row_idx = torch.arange(S.size(0), device=embeds.device).unsqueeze(1).expand(-1, self.top_k)
        col_idx = topk_indices
        
        # Create indices for sparse matrix
        indices = torch.stack([
            row_idx.reshape(-1),
            col_idx.reshape(-1)
        ], dim=0)
        
        # Use similarity scores as edge weights (optional)
        values = topk_values.flatten()  # or torch.ones(indices.size(1))
        
        adj_topk = torch.sparse_coo_tensor(
            indices,
            values,
            size=(S.size(0), S.size(0)),
            device=embeds.device
        ).coalesce()

        return sampled_nodes, adj_topk


