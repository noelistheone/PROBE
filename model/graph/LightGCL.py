import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE

# LightGCL (Cai et al., ICLR 2023): LightGCN main view + a low-rank SVD-reconstructed
# augmented view; contrastive between the two. A strong recent GCL SOTA baseline, added to
# establish the true bar (does AdaG's sparse win already beat the newest SOTA?).


class LightGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCL, self).__init__(conf, training_set, test_set)
        c = self.config['LightGCL']
        self.n_layers = int(c['n_layer'])
        self.cl_rate = float(c['lambda'])
        self.temp = float(c['tau'])
        self.q = int(c['q']) if 'q' in c else 5
        self.model = LightGCL_Encoder(self.data, self.emb_size, self.n_layers, self.q).cuda()

    def train(self):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                (u_g, i_g), (u_s, i_s) = model(both=True)
                user_emb, pos_item_emb, neg_item_emb = u_g[user_idx], i_g[pos_idx], i_g[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                uu = torch.unique(torch.tensor(user_idx, device=u_g.device))
                ii = torch.unique(torch.tensor(pos_idx, device=u_g.device))
                cl = InfoNCE(u_s[uu], u_g[uu], self.temp) + InfoNCE(i_s[ii], i_g[ii], self.temp)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + self.cl_rate * cl
                optimizer.zero_grad(); batch_loss.backward(); optimizer.step()
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        return torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1)).cpu().numpy()


class LightGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, q):
        super().__init__()
        self.data = data; self.emb_size = emb_size; self.n_layers = n_layers
        init = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(init(torch.empty(data.user_num, emb_size))),
            'item_emb': nn.Parameter(init(torch.empty(data.item_num, emb_size))),
        })
        self.norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).cuda()
        # truncated SVD of the normalized adjacency for the augmented view
        u, s, vt = svds(data.norm_adj.astype(np.float32), k=q)
        self.svd_u = torch.tensor((u * s).astype(np.float32)).cuda()   # (n, q)
        self.svd_vt = torch.tensor(vt.astype(np.float32)).cuda()        # (q, n)

    def forward(self, both=False):
        ego = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        e = ego; g_layers = []; s_layers = []
        for _ in range(self.n_layers):
            e_g = torch.sparse.mm(self.norm_adj, e)                     # main (graph) view
            e_s = self.svd_u @ (self.svd_vt @ e)                         # low-rank SVD view
            g_layers.append(e_g); s_layers.append(e_s)
            e = e_g
        G = torch.stack(g_layers, 1).mean(1)
        u_g, i_g = torch.split(G, [self.data.user_num, self.data.item_num])
        if not both:
            return u_g, i_g
        S = torch.stack(s_layers, 1).mean(1)
        u_s, i_s = torch.split(S, [self.data.user_num, self.data.item_num])
        return (u_g, i_g), (u_s, i_s)
