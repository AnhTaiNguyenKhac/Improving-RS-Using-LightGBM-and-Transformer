import torch
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import pairPredict 
from Transformer import Encoder_Layer, TransformerEncoderLayer
from AttentionSampling import AttentionSampling
from PositionalEncoding import PositionalEncoding
import numpy as np

def load_similarity_matrix(sim_matrix_path):
    try:
        sim_matrix = np.load(sim_matrix_path)
        print(f"Loaded similarity matrix from {sim_matrix_path}")
        sim_matrix = (sim_matrix + sim_matrix.T) / 2  # Make symmetric
        
        # Check for NaN or inf values
        if np.any(np.isnan(sim_matrix)) or np.any(np.isinf(sim_matrix)):
            print("Warning: Similarity matrix contains NaN or inf values")
            sim_matrix = np.nan_to_num(sim_matrix)
            
        return torch.tensor(sim_matrix, dtype=torch.float32)
    except Exception as e:
        print(f"Failed to load similarity matrix: {e}")
        return None

class TransGNN(nn.Module):
    def __init__(self):
        super(TransGNN, self).__init__()
        self.args = args
        self.user_embeding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.user, args.latdim)))
        self.item_embeding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.item, args.latdim)))
        self.user_transformer_encoder = TransformerEncoderLayer(d_model=args.latdim, num_heads=args.num_head, dropout=args.dropout)
        self.item_transformer_encoder = TransformerEncoderLayer(d_model=args.latdim, num_heads=args.num_head, dropout=args.dropout)
        self.raw_embedding_proj = nn.Linear(1280, self.args.latdim)
        self.attention_sampler = AttentionSampling(alpha=0.5, top_k=10)
        self.raw_node_embeddings = torch.tensor(np.load('Data/vibrent/nodeEmbeddings.npy'), dtype=torch.float32).cuda()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.visual_sim = load_similarity_matrix("Data/vibrent/similarity_matrix.npy").cuda()
        self.visual_sim_normalized = None
        self.attention_fc = nn.Linear(2 * args.latdim, 1)
        self.num_users = args.user
        self.num_items = args.item
        # Thêm layer gate cho PE
        self.pe_gate = nn.Linear(args.latdim, args.latdim)
        
        # Khởi tạo với giá trị nhỏ
        nn.init.uniform_(self.pe_gate.weight, -0.01, 0.01)
        nn.init.constant_(self.pe_gate.bias, 0.5)
        self.pos_encoder = PositionalEncoding(
            latent_dim=args.latdim,
            num_users=args.user,
            num_items=args.item,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

    def user_transformer_layer(self, embeds, mask=None):
        assert len(embeds.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embeds.shape)
        if len(embeds.shape) == 2:
            embeds = embeds.unsqueeze(dim=0)
            embeds = self.user_transformer_encoder(embeds, mask)
            embeds = embeds.squeeze()
        else:
            embeds = self.user_transformer_encoder(embeds, mask)
        return embeds

    def item_transformer_layer(self, embeds, mask=None):
        assert len(embeds.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embeds.shape)
        if len(embeds.shape) == 2:
            embeds = embeds.unsqueeze(dim=0)
            embeds = self.item_transformer_encoder(embeds, mask)
            embeds = embeds.squeeze()
        else:
            embeds = self.item_transformer_encoder(embeds, mask)
        return embeds

    # def build_sparse_adj_from_topk(self, topk_indices, num_nodes, device):
    #     row_indices = []
    #     col_indices = []
    #     for i in range(num_nodes):
    #         neighbors = topk_indices[i]
    #         if isinstance(neighbors, torch.Tensor):
    #             neighbors = neighbors.tolist()
    #         row_indices.extend([i] * len(neighbors))
    #         col_indices.extend(neighbors)
    #     indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    #     values = torch.ones(len(row_indices), dtype=torch.float, device=device)
    #     adj_topk = torch.sparse.FloatTensor(indices, values, torch.Size([num_nodes, num_nodes]))
    #     return adj_topk

    # def sample_adj(self, adj):
    #     embeds = self.raw_node_embeddings  # <- đây là Tensor, không phải list
    #     topk_indices = self.attention_sampler(embeds, adj)
    #     adj_topk = self.build_sparse_adj_from_topk(topk_indices, num_nodes=embeds.size(0), device=embeds.device)
    #     return adj_topk



    # def gnn_message_passing(self, adj, embeds):
    #     user_embeds = embeds[:args.user]
    #     item_embeds = embeds[args.user:]

    #     # Chuẩn hóa ma trận tương đồng ảnh nếu chưa có
    #     if self.visual_sim_normalized is None:
    #         eps = 1e-7
    #         degree = torch.sum(self.visual_sim, dim=1)
    #         degree_inv_sqrt = torch.pow(degree + eps, -0.5)
    #         degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    #         self.visual_sim_normalized = degree_inv_sqrt.view(-1, 1) * self.visual_sim * degree_inv_sqrt.view(1, -1)

    #     # Graph message passing (user-item)
    #     graph_embeds = torch.spmm(adj, embeds)

    #     # Visual similarity propagation (item-item)
    #     sim_prop = torch.mm(self.visual_sim_normalized, item_embeds) + item_embeds

    #     # Attention combination
    #     combined = torch.cat([graph_embeds[args.user:], sim_prop], dim=1)
    #     attention = torch.sigmoid(self.attention_fc(combined))
    #     new_item_embeds = attention * graph_embeds[args.user:] + (1 - attention) * sim_prop

    #     new_embeds = torch.cat([graph_embeds[:args.user], new_item_embeds], 0)
    #     return new_embeds
    
    def gnn_message_passing(self, adj, embeds):
        # Initialize visual similarity matrix if not exists
        if self.visual_sim is not None and self.visual_sim_normalized is None:
            try:
                eps = 1e-7
                degree = torch.sum(self.visual_sim, dim=1)
                degree_inv_sqrt = torch.pow(degree + eps, -0.5)
                degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
                self.visual_sim_normalized = degree_inv_sqrt.view(-1, 1) * self.visual_sim * degree_inv_sqrt.view(1, -1)
            except Exception as e:
                print(f"Failed to normalize visual similarity matrix: {e}")
                self.visual_sim_normalized = None
        
        # Basic graph propagation
        graph_embeds = torch.spmm(adj, embeds)
        
        # Only apply visual propagation if normalized matrix exists
        if self.visual_sim_normalized is not None:
            item_embeds = embeds[self.args.user:]
            sim_prop = torch.mm(self.visual_sim_normalized, item_embeds)
            graph_embeds[self.args.user:] += self.alpha * sim_prop
        
        return graph_embeds

    def forward(self, adj):
        # 1. Kết hợp raw features và learned embeddings
        raw_embed = self.raw_embedding_proj(self.raw_node_embeddings)
        learned_embed = torch.cat([self.user_embeding, self.item_embeding], dim=0)
        combined_embed = self.alpha * raw_embed + (1 - self.alpha) * learned_embed
        embeds_list = [combined_embed]

        # 2. GNN với Attention Sampling và PE động
        for block_idx in range(self.args.block_num):
            # 2.1. Attention Sampling - Lấy subgraph quan trọng
            with torch.no_grad():  # Giảm memory footprint
                try:
                    # Lấy top-k nodes và adjacency tương ứng
                    sampled_nodes, adj_topk = self.attention_sampler(embeds_list[-1], adj)
                except RuntimeError as e:
                    print(f"Block {block_idx}: Sampling failed, using full graph: {e}")
                    sampled_nodes = torch.arange(combined_embed.size(0), device=combined_embed.device)
                    adj_topk = adj

            # 2.2. Message Passing trên subgraph
            current_embeds = self.gnn_message_passing(adj_topk, embeds_list[-1])

            # 2.3. Positional Encoding CHO SAMPLED NODES
            sampled_pe = self.pos_encoder(
                current_embeds[sampled_nodes], 
                adj_topk,  # Chỉ xét adjacency của subgraph
                sampled_nodes
            )
            
            # 2.4. Gated Residual PE
            pe_gate = torch.sigmoid(self.pe_gate(current_embeds[sampled_nodes]))
            current_embeds[sampled_nodes] = current_embeds[sampled_nodes] + pe_gate * sampled_pe

            # 2.5. Transformer Processing
            user_embeds = self.user_transformer_layer(
                current_embeds[:self.args.user]
            )
            item_embeds = self.item_transformer_layer(
                current_embeds[self.args.user:]
            )
            current_embeds = torch.cat([user_embeds, item_embeds], dim=0)
            
            embeds_list.append(current_embeds)

        # 3. Tổng hợp có trọng số (layer càng sâu càng ít quan trọng)
        weights = torch.linspace(1.0, 0.5, len(embeds_list), device=current_embeds.device)
        final_embeds = torch.sum(torch.stack([w*e for w,e in zip(weights, embeds_list)]), dim=0)
        
        return final_embeds, final_embeds[:self.args.user], final_embeds[self.args.user:]


    def pickEdges(self, adj):
        idx = adj._indices()
        rows, cols = idx[0, :], idx[1, :]
        mask = torch.logical_and(rows <= args.user, cols > args.user)
        rows, cols = rows[mask], cols[mask]
        edgeSampNum = int(args.edgeSampRate * rows.shape[0])
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        edgeids = torch.randint(rows.shape[0], [edgeSampNum])
        pckUsrs, pckItms = rows[edgeids], cols[edgeids] - args.user
        return pckUsrs, pckItms

    def pickRandomEdges(self, adj):
        edgeNum = adj._indices().shape[1]
        edgeSampNum = int(args.edgeSampRate * edgeNum)
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        rows = torch.randint(args.user, [edgeSampNum])
        cols = torch.randint(args.item, [edgeSampNum])
        return rows, cols

    def bprLoss(self, user_embeding, item_embeding, ancs, poss, negs):
        ancEmbeds = user_embeding[ancs]
        posEmbeds = item_embeding[poss]
        negEmbeds = item_embeding[negs]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - ((scoreDiff).sigmoid() + 1e-6).log().mean()
        return bprLoss

    def calcLosses(self, ancs, poss, negs, adj):
        embeds, user_embeds, item_embeds = self.forward(adj)
        user_embeding, item_embeding = embeds[:args.user], embeds[args.user:]
        bprLoss = self.bprLoss(user_embeding, item_embeding, ancs, poss, negs) + self.bprLoss(user_embeds, item_embeds, ancs, poss, negs)
        return bprLoss

    def predict(self, adj):
        embeds, user_embeds, item_embeds = self.forward(adj)
        return user_embeds, item_embeds
