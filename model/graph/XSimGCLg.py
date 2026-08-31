import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from model.graph.XSimGCL import XSimGCL_Encoder

# XSimGCLg = XSimGCL (strongest GCL backbone here) + explicit geometric regularization
# (DirectAU-style alignment + uniformity), to test whether the geometric-reg principle
# genuinely improves the STRONG backbone and GENERALIZES across datasets.


class XSimGCLg(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCLg, self).__init__(conf, training_set, test_set)
        c = self.config['XSimGCLg']
        self.cl_rate = float(c['lambda'])
        self.eps = float(c['eps'])
        self.temp = float(c['tau'])
        self.n_layers = int(c['n_layer'])
        self.layer_cl = int(c['l_star'])
        self.geom_w = float(c['geom_w']) if 'geom_w' in c else 0.0
        # DENSITY-ADAPTIVE geometric reg: weight the geom terms per node by (1+deg)^(-beta),
        # normalized to mean 1, so COLD/TAIL (low-degree) nodes get strong regularization and
        # well-trained DENSE (high-degree) nodes get light — beta=0 recovers uniform geom reg.
        self.geom_beta = float(c['geom_beta']) if 'geom_beta' in c else 0.0
        # gate: 'power' = (1+deg)^(-beta) (smooth); 'sigmoid' = sharp threshold that ZEROES geom
        # reg for well-trained dense nodes (>= tau-quantile log-degree) so they stay pure XSimGCL.
        self.geom_gate = c['geom_gate'] if 'geom_gate' in c else 'power'
        self.geom_a = float(c['geom_a']) if 'geom_a' in c else 5.0        # sigmoid sharpness
        self.geom_tau = float(c['geom_tau']) if 'geom_tau' in c else 0.5  # degree quantile threshold
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl, self.temp)
        adaptive = self.geom_w > 0 and (self.geom_beta > 0 or self.geom_gate == 'sigmoid')
        if adaptive:
            import numpy as np
            ud = np.asarray(self.data.interaction_mat.sum(axis=1)).ravel()
            idg = np.asarray(self.data.interaction_mat.sum(axis=0)).ravel()
            # Control: keep the weight DISTRIBUTION identical but destroy its correspondence to
            # degree, by permuting the degree vector. If the gate works because it tracks how well
            # each node is determined by data, this control should not work; if it works merely
            # because the weights are heterogeneous, it should.
            if 'geom_perm' in c and str(c['geom_perm']).lower() == 'true':
                rng = np.random.RandomState(int(c['geom_perm_seed']) if 'geom_perm_seed' in c else 12345)
                ud, idg = rng.permutation(ud), rng.permutation(idg)
            if self.geom_gate == 'sigmoid':
                ldu, ldi = np.log1p(ud), np.log1p(idg)
                tu = np.quantile(ldu, self.geom_tau); ti = np.quantile(ldi, self.geom_tau)
                uw = 1.0 / (1.0 + np.exp(self.geom_a * (ldu - tu)))       # ~1 cold, ~0 dense
                iw = 1.0 / (1.0 + np.exp(self.geom_a * (ldi - ti)))
            else:
                uw = (1.0 + ud) ** (-self.geom_beta)
                iw = (1.0 + idg) ** (-self.geom_beta)
            uw = uw / uw.mean(); iw = iw / iw.mean()
            self.user_w = torch.tensor(uw, dtype=torch.float32).cuda()
            self.item_w = torch.tensor(iw, dtype=torch.float32).cuda()
        else:
            self.user_w = self.item_w = None

    @staticmethod
    def _alignment(x, y, w=None):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        a = (x - y).norm(p=2, dim=1).pow(2)
        return (w * a).sum() / w.sum() if w is not None else a.mean()

    @staticmethod
    def _uniformity(x, w=None):
        x = F.normalize(x, dim=-1)
        if w is None:
            return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()
        n = x.size(0)
        iu = torch.triu_indices(n, n, offset=1, device=x.device)
        wij = w[iu[0]] * w[iu[1]]                         # pairwise degree weights (tail-heavy)
        e = torch.pdist(x, p=2).pow(2).mul(-2).exp()
        return (wij * e).sum().div(wij.sum()).log()

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                if self.geom_w > 0:
                    if self.user_w is not None:   # density-adaptive weighting
                        uw = self.user_w[user_idx]; iw = self.item_w[pos_idx]
                        geom = self._alignment(user_emb, pos_item_emb, uw) + \
                            0.5 * (self._uniformity(user_emb, uw) + self._uniformity(pos_item_emb, iw))
                    else:                          # uniform geom reg (beta=0)
                        geom = self._alignment(user_emb, pos_item_emb) + \
                            0.5 * (self._uniformity(user_emb) + self._uniformity(pos_item_emb))
                    batch_loss = batch_loss + self.geom_w * geom
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        return torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1)).cpu().numpy()
