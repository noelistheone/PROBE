import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
from scipy.sparse.linalg import svds
import numpy as np
from model.graph.XSimGCL import XSimGCL_Encoder
from model.graph.SimGCL import SimGCL_Encoder
import os


# PT4Rec_Enhanced: Prompt-based Recommendation with Performance Enhancements
class PT4Rec_Enhanced(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(PT4Rec_Enhanced, self).__init__(conf, training_set, test_set)

        args = OptionConf(self.config['PT4Rec_Enhanced'])
        self.n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        prompt_size = int(args['-prompt_size'])
        self.user_prompt_num = int(args['-user_prompt_num'])
        self.pretrain_model = args['-pretrain_model']
        
        # New hyperparameters with backward compatibility
        self.dual_attention = args.contain('-dual_attention') and args['-dual_attention'].lower() == 'true'
        self.use_popularity = args.contain('-use_popularity') and args['-use_popularity'].lower() == 'true'
        self.pop_gamma = float(args['-pop_gamma']) if args.contain('-pop_gamma') else 0.1
        self.pop_topk = int(args['-pop_topk']) if args.contain('-pop_topk') else 100
        self.fusion_type = args['-fusion_type'] if args.contain('-fusion_type') else 'multiply'
        self.warmup_epochs = int(args['-warmup_epochs']) if args.contain('-warmup_epochs') else 0
        self.stage2_cl_weight = float(args['-stage2_cl_weight']) if args.contain('-stage2_cl_weight') else 0.0
        self.prompt_dropout = float(args['-prompt_dropout']) if args.contain('-prompt_dropout') else 0.0
        self.align_weight = float(args['-align_weight']) if args.contain('-align_weight') else 0.0
        self.uniform_weight = float(args['-uniform_weight']) if args.contain('-uniform_weight') else 0.0
        self.neg_mixup = args.contain('-neg_mixup') and args['-neg_mixup'].lower() == 'true'
        self.n_negs = int(args['-n_negs']) if args.contain('-n_negs') else 1

        # Freeze the pre-trained GCL encoder during prompt-tuning (faithful to the
        # "prompt-tuning of a frozen encoder" premise). Default true; set false to
        # jointly fine-tune the encoder (the original, non-faithful behaviour).
        self.freeze_encoder = (not args.contain('-freeze_encoder')) or args['-freeze_encoder'].lower() == 'true'
        # Popularity top-K: dataset-adaptive floor(0.4% * M) unless -pop_adaptive false.
        self.pop_adaptive = (not args.contain('-pop_adaptive')) or args['-pop_adaptive'].lower() == 'true'

        # Geometry-preserving regularizer (the paper's "L2-SP + Uniformity" module, implemented
        # as a DEGREE-ADAPTIVE alignment+uniformity term: cold/tail nodes get strong geometric
        # regularization, well-trained dense nodes get light). geom_w=0 disables (legacy behaviour).
        self.geom_w = float(args['-geom_w']) if args.contain('-geom_w') else 0.0
        self.geom_beta = float(args['-geom_beta']) if args.contain('-geom_beta') else 1.0
        
        # Encoder parameters from config (fix hardcoded values)
        self.xsimgcl_eps = float(args['-xsimgcl_eps']) if args.contain('-xsimgcl_eps') else 0.2
        self.xsimgcl_layer_cl = int(args['-xsimgcl_layer_cl']) if args.contain('-xsimgcl_layer_cl') else 1
        self.simgcl_eps = float(args['-simgcl_eps']) if args.contain('-simgcl_eps') else 0.1
        self.simgcl_n_layers = int(args['-simgcl_n_layers']) if args.contain('-simgcl_n_layers') else 3
        
        if self.config.contain('num.max.preepoch'):
            self.maxPreEpoch = int(self.config['num.max.preepoch'])
        else:
            self.maxPreEpoch = 0

        # Derive pretrained model path from config (dataset name + pretrain model + epochs).
        # IMPORTANT: the cache key MUST include the split (valid.ratio + split.seed) so a
        # checkpoint trained on a different train/val partition (or the full, un-split data)
        # is never silently loaded — that would leak validation edges into the frozen encoder.
        import os
        dataset_name = os.path.basename(os.path.dirname(self.config['training.set']))
        vr = self.config['valid.ratio'] if self.config.contain('valid.ratio') else 0.1
        ss = self.config['split.seed'] if self.config.contain('split.seed') else 2024
        self.split_tag = f'vr{vr}_ss{ss}'
        # Include the run seed so each seed pre-trains its OWN encoder (the pre-training
        # sampler is seeded); otherwise all seeds would share one encoder and the reported
        # variance would understate total pipeline variance.
        run_seed = os.environ.get('RUN_SEED', 'na')
        if self.config.contain('pretrain_model_path'):
            # Allow explicit override in yaml
            self.pretrained_model_path = self.config['pretrain_model_path']
        else:
            self.pretrained_model_path = f'./pretrained_model/{self.pretrain_model}_{dataset_name}_pretrain_{self.maxPreEpoch}_{self.split_tag}_seed{run_seed}.pt'
        print(f'Pretrained model path: {self.pretrained_model_path}')

        self.user_prompt_H = True
        self.user_prompt_M = True
        self.user_prompt_R = True

        if self.pretrain_model == 'XSimGCL':
            self.model = XSimGCL_Encoder(self.data, self.emb_size, self.xsimgcl_eps, self.n_layers, self.xsimgcl_layer_cl, temp)
        elif self.pretrain_model == 'SimGCL':
            self.model = SimGCL_Encoder(self.data, self.emb_size, eps=self.simgcl_eps, n_layers=self.simgcl_n_layers)

        if self.user_prompt_num != 0:         
            self.user_prompt_generator = [Prompts_Generator(self.emb_size, prompt_size, self.prompt_dropout).cuda() for _ in range(self.user_prompt_num)]
            self.user_attention = Attention(prompt_size, self.user_prompt_num, self.dual_attention).cuda()
            
            # Popularity debiasing prompt generator
            if self.use_popularity:
                self.pop_prompt_generator = Prompts_Generator(self.emb_size, prompt_size, self.prompt_dropout).cuda()
            
            # MLP fusion module
            if self.fusion_type == 'mlp':
                self.fusion_mlp = Fusion_MLP(self.emb_size, prompt_size).cuda()

        self.interaction_mat = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda()
        self.user_matrix, self.Item_matrix = self._adjacency_matrix_factorization()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()
        
        # Compute top-K popular items for popularity debiasing
        if self.use_popularity:
            self.top_k_items = self._compute_top_k_items()
        if self.geom_w > 0:
            self._init_geom_weights()
    
    def _compute_top_k_items(self):
        """Compute top-K popular items based on TRAINING interaction frequency.
        K is dataset-adaptive (floor(0.4% * M)) unless overridden."""
        if self.pop_adaptive:
            self.pop_topk = max(1, int(0.004 * self.data.item_num))
        item_degrees = self.data.interaction_mat.sum(axis=0).A1
        k = min(self.pop_topk, len(item_degrees))
        top_k_items = np.argsort(item_degrees)[-k:]
        print(f'[model] popularity top-K = {k} (adaptive={self.pop_adaptive}, M={self.data.item_num})')
        return torch.tensor(top_k_items).cuda()
    
    def global_pop_records(self, item_emb):
        """Aggregate global top-K popular items embedding."""
        top_k_emb = item_emb[self.top_k_items]
        global_pop_emb = top_k_emb.mean(dim=0, keepdim=True)
        return global_pop_emb.expand(self.data.user_num, -1)
    
    def _init_geom_weights(self):
        """Per-node degree weights (1+deg)^(-beta), mean-normalized: cold/tail ~high, dense ~low."""
        ud = np.asarray(self.data.interaction_mat.sum(axis=1)).ravel()
        idg = np.asarray(self.data.interaction_mat.sum(axis=0)).ravel()
        uw = (1.0 + ud) ** (-self.geom_beta); uw = uw / uw.mean()
        iw = (1.0 + idg) ** (-self.geom_beta); iw = iw / iw.mean()
        self.geom_uw = torch.tensor(uw, dtype=torch.float32).cuda()
        self.geom_iw = torch.tensor(iw, dtype=torch.float32).cuda()

    @staticmethod
    def _align_w(x, y, w):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        a = (x - y).norm(p=2, dim=1).pow(2)
        return (w * a).sum() / w.sum()

    @staticmethod
    def _unif_w(x, w, t=2.0):
        x = F.normalize(x, dim=-1)
        n = x.size(0)
        iu = torch.triu_indices(n, n, offset=1, device=x.device)
        wij = w[iu[0]] * w[iu[1]]
        e = torch.pdist(x, p=2).pow(2).mul(-t).exp()
        return (wij * e).sum().div(wij.sum().clamp_min(1e-12)).clamp_min(1e-12).log()

    def alignment(self, x, y):
        """Alignment loss for representation learning."""
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()
    
    def uniformity(self, x, t=2):
        """Uniformity loss for representation learning."""
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def pre_train(self):
        """Unified pre-training method. Loads from self.pretrained_model_path if exists, else trains and saves."""
        print('############## Pre-Training Phase ##############')


        if os.path.exists(self.pretrained_model_path):
            self.model.load_state_dict(torch.load(self.pretrained_model_path))
            print(f'Loaded pretrained model from {self.pretrained_model_path}')
            return
        else:
            print(f'No pretrained model found at {self.pretrained_model_path}, start pre-training...')

        pre_trained_model = self.model.cuda()
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=self.lRate)

        for epoch in range(self.maxPreEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                if self.pretrain_model == 'XSimGCL':
                    rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = pre_trained_model(True)
                    cl_loss = pre_trained_model.cal_cl_loss([user_idx, pos_idx], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb)
                else:  # SimGCL
                    cl_loss = pre_trained_model.cal_cl_loss([user_idx, pos_idx])
                optimizer.zero_grad()
                cl_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('pre-training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item())

        os.makedirs(os.path.dirname(self.pretrained_model_path), exist_ok=True)
        torch.save(pre_trained_model.state_dict(), self.pretrained_model_path)
        print(f'Saved pretrained model to {self.pretrained_model_path}')

    def _csr_to_pytorch_dense(self, csr):
        array = csr.toarray()
        dense = torch.Tensor(array)
        return dense.cuda()

    def user_historical_records(self, item_emb):    # 用户的历史记录 根据所有选择过该用户的商品的embedding
        user_profiles = torch.mm(self.interaction_mat, item_emb)
        return user_profiles

    def _adjacency_matrix_factorization(self):
        """Real low-rank decomposition of the (training) user-item adjacency via
        truncated SVD: R ~= U diag(s) V^T. Returns sqrt(s)-scaled user/item factors
        used as the "M" (adjacency-decomposition) prompt source. Cached per dataset.

        NOTE: the original code constructed a torchnmf NMF but NEVER called fit(), so
        the M view was a fixed RANDOM matrix. This computes an actual factorization."""
        print('######### Adjacency Matrix Factorization (truncated SVD) #############')
        dataset_name = os.path.basename(os.path.dirname(self.config['training.set']))
        cache = f'./pretrained_model/{dataset_name}_svd{self.emb_size}_{getattr(self, "split_tag", "vr0.1_ss2024")}.npz'
        if os.path.exists(cache):
            d = np.load(cache)
            user_f, item_f = d['user_f'], d['item_f']
            print(f'Loaded cached SVD factors from {cache}')
        else:
            R = self.data.interaction_mat.tocsr().astype(np.float32)  # sparse (N x M)
            k = min(self.emb_size, min(R.shape) - 1)
            U, s, Vt = svds(R, k=k)               # ascending singular values
            order = np.argsort(-s)                 # sort descending
            U, s, Vt = U[:, order], s[order], Vt[order]
            sqrt_s = np.sqrt(np.maximum(s, 0.0))
            user_f = (U * sqrt_s).astype(np.float32)        # N x k
            item_f = (Vt.T * sqrt_s).astype(np.float32)     # M x k
            if k < self.emb_size:                  # pad if rank < embedding size
                user_f = np.pad(user_f, ((0, 0), (0, self.emb_size - k)))
                item_f = np.pad(item_f, ((0, 0), (0, self.emb_size - k)))
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, user_f=user_f, item_f=item_f)
            print(f'Computed & cached SVD factors -> {cache}  (rank={k})')
        user_profiles = torch.tensor(user_f, dtype=torch.float32).cuda()
        item_profiles = torch.tensor(item_f, dtype=torch.float32).cuda()
        return user_profiles, item_profiles

    def _high_order_relations(self, item_emb, user_emb):  # 高阶关系
        # small dataset
        # emb = torch.cat((user_emb, item_emb), 0)
        # inputs = torch.sparse.mm(self.ui_high_order, emb)
        # inputs = inputs[:self.data.user_num, :]
        # return inputs

        # big dataset Ciao
        ego_embeddings = torch.cat((user_emb, item_emb), 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_profiles, item_profiles = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_profiles, item_profiles

    def _set_prompt_mode(self, train=True):
        """Toggle train/eval on the prompt sub-modules so prompt dropout is OFF at
        inference (the original code left dropout active during evaluation)."""
        if self.user_prompt_num == 0:
            return
        mods = list(self.user_prompt_generator) + [self.user_attention]
        if self.use_popularity:
            mods.append(self.pop_prompt_generator)
        if self.fusion_type == 'mlp':
            mods.append(self.fusion_mlp)
        for m in mods:
            m.train() if train else m.eval()

    def train(self):
        self.pre_train()

        model = self.model.cuda()

        # Snapshot the frozen pre-trained user embedding as the L2-SP anchor e_bar_u
        # (Eq. 5 / Theorem 1: the fine-tuned representation is regularized toward THIS).
        with torch.no_grad():
            pu, _ = model()
            self.pretrained_user_emb = pu.detach()

        # Freeze the GCL encoder during prompt-tuning (faithful "frozen encoder" premise).
        if self.freeze_encoder:
            for p in model.parameters():
                p.requires_grad = False

        # Collect trainable parameters: encoder only if NOT frozen.
        params = [] if self.freeze_encoder else list(model.parameters())
        if self.user_prompt_num != 0:
            params += list(self.user_attention.parameters())
            for gen in self.user_prompt_generator:
                params += list(gen.parameters())
            if self.use_popularity:
                params += list(self.pop_prompt_generator.parameters())
            if self.fusion_type == 'mlp':
                params += list(self.fusion_mlp.parameters())

        optimizer = torch.optim.Adam(params, lr=self.lRate)
        print(f'[model] freeze_encoder={self.freeze_encoder}, trainable tensors={len(params)}')

        # When the encoder is frozen, the prompt INPUTS (base user/item emb, the H/M/R
        # profile matrices, and the popularity input) are constant across all batches and
        # epochs. Precompute them ONCE to avoid re-running the expensive multi-hop graph
        # propagations every batch (audit L3). Only the trainable prompt generators/
        # attention run per batch.
        self._frozen_inputs = None
        if self.freeze_encoder:
            with torch.no_grad():
                fu, fi = model()
                self._frozen_inputs = {
                    'user_emb': fu.detach(), 'item_emb': fi.detach(),
                    'H': self.user_historical_records(fi).detach(),
                    'R': self._high_order_relations(fi, fu)[0].detach(),
                    'pop': self.global_pop_records(fi).detach() if self.use_popularity else None,
                }

        metrics = []
        print('############## Downstream Training Phase ##############')
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size, self.n_negs)):
                if self.freeze_encoder:
                    prompted_user_emb, prompted_item_emb = self.generate_prompts_cached()
                else:
                    user_emb, item_emb = model()
                    prompted_user_emb, prompted_item_emb = self.generate_prompts(user_emb, item_emb)

                user_idx, pos_idx, neg_idx = batch
                rec_user_emb = prompted_user_emb
                rec_item_emb = prompted_item_emb

                user_emb_batch = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[pos_idx]
                neg_item_emb = rec_item_emb[neg_idx]
                
                # Handle multiple negatives: reshape from [batch_size * n_negs, emb_size] to [batch_size, n_negs, emb_size]
                if self.n_negs > 1:
                    batch_size = len(user_idx)
                    neg_item_emb = neg_item_emb.view(batch_size, self.n_negs, -1)
                    
                    # Select hardest negative
                    neg_scores = torch.matmul(user_emb_batch.unsqueeze(1), neg_item_emb.transpose(1, 2)).squeeze(1)
                    hard_neg_idx = torch.argmax(neg_scores, dim=1)
                    neg_item_emb = neg_item_emb[torch.arange(batch_size), hard_neg_idx]
                    
                    # Apply mixup if enabled
                    if self.neg_mixup:
                        lam = torch.rand(batch_size, 1).cuda()
                        neg_item_emb = lam * pos_item_emb + (1 - lam) * neg_item_emb
                
                rec_loss = bpr_loss(user_emb_batch, pos_item_emb, neg_item_emb)
                reg_loss = l2_reg_loss(self.reg, user_emb_batch, pos_item_emb)
                
                # Warm-up strategy: only basic loss during warm-up
                if epoch < self.warmup_epochs:
                    batch_loss = rec_loss + reg_loss
                else:
                    batch_loss = rec_loss + reg_loss
                    
                    # Stage-2 contrastive loss only makes sense if the encoder is
                    # still trainable (frozen encoder -> constant CL, no gradient).
                    if self.stage2_cl_weight > 0 and not self.freeze_encoder and hasattr(model, 'cal_cl_loss'):
                        if self.pretrain_model == 'XSimGCL':
                            rec_user_emb_raw, rec_item_emb_raw, cl_user_emb, cl_item_emb = model(True)
                            cl_loss = model.cal_cl_loss([user_idx, pos_idx], rec_user_emb_raw, cl_user_emb, rec_item_emb_raw, cl_item_emb)
                        else:
                            cl_loss = model.cal_cl_loss([user_idx, pos_idx])
                        batch_loss += self.stage2_cl_weight * cl_loss

                    # L2-SP (Eq. 5): anchor the fine-tuned user representation to its
                    # frozen pre-trained embedding e_bar_u. (Previously this term was
                    # DirectAU user<->item alignment, which is NOT what Eq.5/Theorem 1
                    # analyze.) align_weight == mu_sp.
                    if self.align_weight > 0:
                        anchor = self.pretrained_user_emb[user_idx]
                        l2sp_loss = (user_emb_batch - anchor).pow(2).sum(dim=1).mean()
                        align_loss = l2sp_loss
                        batch_loss += self.align_weight * l2sp_loss

                    if self.uniform_weight > 0:
                        uniform_loss = (self.uniformity(user_emb_batch) + self.uniformity(pos_item_emb)) / 2
                        batch_loss += self.uniform_weight * uniform_loss

                    # Geometry-preserving module, degree-adaptive (the paper's Sec 3.2.2 intent).
                    if self.geom_w > 0:
                        uw = self.geom_uw[user_idx]; iw = self.geom_iw[pos_idx]
                        geom = self._align_w(user_emb_batch, pos_item_emb, uw) + \
                            0.5 * (self._unif_w(user_emb_batch, uw) + self._unif_w(pos_item_emb, iw))
                        batch_loss += self.geom_w * geom
                
                # Backward and optimize
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if n % 100 == 0:
                    loss_info = f'training: epoch {epoch + 1}, batch {n}, rec_loss: {rec_loss.item():.4f}'
                    if epoch >= self.warmup_epochs:
                        if self.stage2_cl_weight > 0 and 'cl_loss' in locals():
                            loss_info += f', cl_loss: {cl_loss.item():.4f}'
                        if self.align_weight > 0:
                            loss_info += f', align_loss: {align_loss.item():.4f}'
                        if self.uniform_weight > 0:
                            loss_info += f', uniform_loss: {uniform_loss.item():.4f}'
                    print(loss_info)
                    
            self._set_prompt_mode(train=False)   # dropout OFF for inference
            with torch.no_grad():
                if self.freeze_encoder:
                    self.user_emb, self.item_emb = self.generate_prompts_cached()
                else:
                    user_emb, self.item_emb = self.model()
                    self.user_emb, _ = self.generate_prompts(user_emb, self.item_emb)
            self._set_prompt_mode(train=True)
            if epoch % 5 == 0:
                metric = self.fast_evaluation(epoch)
                metrics.append(metric)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        self._set_prompt_mode(train=False)   # dropout OFF for inference
        with torch.no_grad():
            if self.freeze_encoder and getattr(self, '_frozen_inputs', None) is not None:
                self.best_user_emb, self.best_item_emb = self.generate_prompts_cached()
            else:
                user_emb, item_emb = self.model.forward()
                self.best_user_emb, self.best_item_emb = self.generate_prompts(user_emb, item_emb)
        self._set_prompt_mode(train=True)

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
    def generate_prompts(self, user_emb, item_emb):
        user_prompts = []
        u = 0
        if self.user_prompt_num != 0:
            if self.user_prompt_H:
                user_prompts.append(self.user_prompt_generator[u](self.user_historical_records(item_emb)))
                u += 1
            if self.user_prompt_M:
                user_prompts.append(self.user_prompt_generator[u](self.user_matrix))
                u += 1
            if self.user_prompt_R:
                user_prompts.append(self.user_prompt_generator[u](self._high_order_relations(item_emb, user_emb)[0]))
            
            # Generate combined prompt from attention
            user_prompt = torch.stack(user_prompts, dim=1)
            prompt = self.user_attention(user_prompt, user_emb)
            
            # Apply fusion strategy
            if self.fusion_type == 'mlp':
                prompted_user_emb = self.fusion_mlp(user_emb, prompt)
            else:  # multiply (default, backward compatible)
                prompted_user_emb = prompt * user_emb
            
            # Apply popularity debiasing if enabled
            if self.use_popularity:
                pop_input = self.global_pop_records(item_emb)
                pop_prompt = self.pop_prompt_generator(pop_input)
                # Residual unlearning: subtract popularity bias
                prompted_user_emb = prompted_user_emb - self.pop_gamma * (user_emb * pop_prompt)
        else:
            prompted_user_emb = user_emb

        return prompted_user_emb, item_emb

    def generate_prompts_cached(self):
        """Fast prompt path used when the encoder is frozen: the H/M/R profile inputs,
        the base user/item embeddings, and the popularity input are precomputed once in
        self._frozen_inputs; only the trainable prompt generators / attention / pop
        generator run here. Numerically identical to generate_prompts() with a frozen
        encoder, but avoids re-running the multi-hop graph propagation every batch."""
        fi = self._frozen_inputs
        user_emb, item_emb = fi['user_emb'], fi['item_emb']
        if self.user_prompt_num == 0:
            return user_emb, item_emb
        prompts, u = [], 0
        if self.user_prompt_H:
            prompts.append(self.user_prompt_generator[u](fi['H'])); u += 1
        if self.user_prompt_M:
            prompts.append(self.user_prompt_generator[u](self.user_matrix)); u += 1
        if self.user_prompt_R:
            prompts.append(self.user_prompt_generator[u](fi['R']))
        user_prompt = torch.stack(prompts, dim=1)
        prompt = self.user_attention(user_prompt, user_emb)
        if self.fusion_type == 'mlp':
            prompted_user_emb = self.fusion_mlp(user_emb, prompt)
        else:
            prompted_user_emb = prompt * user_emb
        if self.use_popularity:
            pop_prompt = self.pop_prompt_generator(fi['pop'])
            prompted_user_emb = prompted_user_emb - self.pop_gamma * (user_emb * pop_prompt)
        return prompted_user_emb, item_emb


class Attention(nn.Module):
    """Attention mechanism with optional dual (adversarial) attention."""
    def __init__(self, prompt_size, prompt_num, dual_attention=False):
        super(Attention, self).__init__()
        initializer = nn.init.xavier_uniform_
        self.prompt_num = prompt_num
        self.dual_attention = dual_attention
        
        if dual_attention:
            # Dual attention: positive and negative attention matrices
            self.A_pos = nn.Parameter(initializer(torch.empty(prompt_size, prompt_num)))
            self.A_neg = nn.Parameter(initializer(torch.empty(prompt_size, prompt_num)))
        else:
            # Standard single attention matrix (backward compatible)
            self.A = nn.Parameter(initializer(torch.empty(prompt_size, prompt_num)))

    def forward(self, prompt_base, embeddings):
        if self.dual_attention:
            # Positive attention
            score_pos = torch.matmul(embeddings, self.A_pos)
            alpha_pos = F.softmax(score_pos, dim=1)
            
            # Negative attention (adversarial)
            score_neg = torch.matmul(embeddings, self.A_neg)
            alpha_neg = F.softmax(score_neg, dim=1)
            
            # Combined attention: alpha in range (-1, 1)
            alpha = alpha_pos - alpha_neg
            alpha = alpha.unsqueeze(1)
            prompt = torch.bmm(alpha, prompt_base)
            prompt = prompt.squeeze(1)
        else:
            # Standard attention (backward compatible)
            alpha = torch.matmul(embeddings, self.A)
            alpha = F.softmax(alpha, dim=1)
            alpha = alpha.unsqueeze(1)
            prompt = torch.bmm(alpha, prompt_base)
            prompt = prompt.squeeze(1)
        return prompt

    
class Prompts_Generator(nn.Module):
    """Prompt generator with optional dropout regularization."""
    def __init__(self, emb_size, prompt_size, dropout=0.0):
        super(Prompts_Generator, self).__init__()
        self.W = nn.Linear(emb_size, prompt_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, inputs):
        prompts = self.W(inputs)
        prompts = self.activation(prompts)
        if self.dropout is not None:
            prompts = self.dropout(prompts)
        return prompts


class Fusion_MLP(nn.Module):
    """MLP-based fusion of user embeddings and prompts."""
    def __init__(self, emb_size, prompt_size):
        super(Fusion_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_size + prompt_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size),
            nn.Tanh()
        )
    
    def forward(self, user_emb, prompt):
        combined = torch.cat([user_emb, prompt], dim=1)
        return self.mlp(combined)
    