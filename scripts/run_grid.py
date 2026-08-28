#!/usr/bin/env python
"""Multi-seed experiment grid driver for the OURS resubmission.

Generates a config per (model, dataset), then runs it under the fixed harness
(train/val/test split, val-based selection, frozen encoder, real L2-SP, batched
GPU eval) for each seed. Every run auto-persists a JSON to results/wm/ via
GraphRecommender._persist_json. Resumable: skips runs whose JSON already exists.

Usage:
  python scripts/run_grid.py --models OURS_XSim,PTbase_XSim,DirectAU \
      --datasets douban-book --seeds 2024,2025,2026
  python scripts/run_grid.py --all --datasets douban-book,yelp2018,ml-1M --seeds 2024,2025,2026
"""
import argparse, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

DATASETS = {  # name -> (train, test, has_social_trust)
    'douban-book': ('./dataset/douban-book/train.txt', './dataset/douban-book/test.txt', True),
    'yelp2018':    ('./dataset/yelp2018/train.txt',    './dataset/yelp2018/test.txt',    False),
    'ml-1M':       ('./dataset/ml-1M/train.txt',       './dataset/ml-1M/test.txt',       False),
    'amazon-kindle': ('./dataset/amazon-kindle/train.txt', './dataset/amazon-kindle/test.txt', False),
    'iFashion':    ('./dataset/iFashion/train.txt',    './dataset/iFashion/test.txt',    False),
    'ml-1M-temporal': ('./dataset/ml-1M-temporal/train.txt', './dataset/ml-1M-temporal/test.txt', False),
}

# OURS / PT4Rec-baseline dash-strings (both freeze the encoder = faithful prompt-tuning).
_OURS = ('-n_layer 2 -temp 0.2 -prompt_size 64 -user_prompt_num 3 -pretrain_model {bk} '
         '-simgcl_eps 0.1 -simgcl_n_layers 3 -xsimgcl_eps 0.2 -xsimgcl_layer_cl 1 '
         '-dual_attention true -use_popularity true -pop_gamma 0.1 -pop_adaptive true '
         '-fusion_type multiply -warmup_epochs 10 -stage2_cl_weight 0.0 -prompt_dropout 0.1 '
         '-align_weight 0.02 -uniform_weight 0.01 -n_negs 4 -neg_mixup true -freeze_encoder true')
_PTBASE = ('-n_layer 2 -temp 0.2 -prompt_size 64 -user_prompt_num 3 -pretrain_model {bk} '
           '-simgcl_eps 0.1 -simgcl_n_layers 3 -xsimgcl_eps 0.2 -xsimgcl_layer_cl 1 '
           '-dual_attention false -use_popularity false -fusion_type multiply -warmup_epochs 10 '
           '-stage2_cl_weight 0.0 -prompt_dropout 0.1 -align_weight 0.0 -uniform_weight 0.0 '
           '-n_negs 1 -neg_mixup false -freeze_encoder true')

# tag -> (model_name, param_block, max_epoch, needs_social, preepoch)
MODELS = {
    'MF':        ('MF',       {}, 200, False, 0),
    'LightGCN':  ('LightGCN', {'n_layer': 2}, 200, False, 0),
    'SGL':       ('SGL',      {'n_layer': 2, 'lambda': 0.1, 'drop_rate': 0.1, 'aug_type': 1, 'temp': 0.2}, 200, False, 0),
    'NCL':       ('NCL',      {'n_layer': 3, 'ssl_reg': '1e-6', 'proto_reg': '1e-7', 'tau': 0.05, 'hyper_layers': 1, 'alpha': 1, 'num_clusters': 2000}, 200, False, 0),
    'SimGCL':    ('SimGCL',   {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2}, 100, False, 0),
    'XSimGCL':   ('XSimGCL',  {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15}, 100, False, 0),
    'DirectAU':  ('DirectAU', {'gamma': 2, 'n_layers': 3}, 100, False, 0),
    'BUIR':      ('BUIR',     {'n_layer': 2, 'tau': 0.995, 'drop_rate': 0.2}, 100, False, 0),
    'SelfCF':    ('SelfCF',   {'n_layer': 2, 'tau': 0.05}, 100, False, 0),
    'SSL4Rec':   ('SSL4Rec',  {'tau': 0.07, 'alpha': 0.1, 'drop': 0.1}, 100, False, 0),
    'CPTPP':     ('CPTPP',    '-n_layer 2 -lambda 0.1 -droprate 0.1 -augtype 1 -temp 0.2 -inputs_type 2 -prompt_size 256', 100, False, 0),
    'MHCN':      ('MHCN',     {'n_layer': 2, 'ss_rate': 0.01}, 200, True, 0),
    'SEPT':      ('SEPT',     {'n_layer': 2, 'ss_rate': 0.005, 'drop_rate': 0.3, 'ins_cnt': 10}, 200, True, 0),
    'PTbase_Sim':  ('PT4Rec_Enhanced', _PTBASE.format(bk='SimGCL'),  100, False, 20),
    'PTbase_XSim': ('PT4Rec_Enhanced', _PTBASE.format(bk='XSimGCL'), 100, False, 20),
    'OURS_Sim':    ('PT4Rec_Enhanced', _OURS.format(bk='SimGCL'),    100, False, 20),
    'OURS_XSim':   ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL'),   100, False, 20),
    # --- fairness diagnostics: better-pretrained frozen encoder, and unfrozen encoder ---
    'PTbase_XSim_p50':  ('PT4Rec_Enhanced', _PTBASE.format(bk='XSimGCL'), 100, False, 50),
    'OURS_XSim_p50':    ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL'),   100, False, 50),
    'PTbase_XSim_p100': ('PT4Rec_Enhanced', _PTBASE.format(bk='XSimGCL'), 100, False, 100),
    'OURS_XSim_p100':   ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL'),   100, False, 100),
    'PTbase_XSim_nofz': ('PT4Rec_Enhanced', _PTBASE.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false'), 100, False, 20),
    'OURS_XSim_nofz':   ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false'), 100, False, 20),
    # --- NEW principled method: signed (bipolar) layer aggregation on a SimGCL backbone ---
    'SGCL_L2':      ('SGCL', {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'signed': 'true'}, 100, False, 0),
    'SGCL_L3':      ('SGCL', {'n_layer': 3, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'signed': 'true'}, 100, False, 0),
    'SGCL_L4':      ('SGCL', {'n_layer': 4, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'signed': 'true'}, 100, False, 0),
    'SGCL_L3_uns':  ('SGCL', {'n_layer': 3, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'signed': 'false'}, 100, False, 0),
    'SGCL_gl':      ('SGCL', {'n_layer': 3, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'signed': 'true', 'router': 'global'}, 100, False, 0),
    'SGCL_geom':    ('SGCL', {'n_layer': 3, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'signed': 'false', 'router': 'global', 'geom_w': 0.5}, 100, False, 0),
    'SGCL_gl_geom': ('SGCL', {'n_layer': 3, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'signed': 'true', 'router': 'global', 'geom_w': 0.5}, 100, False, 0),
    # clean SimGCL mean-aggregation backbone + one principle at a time (isolate geom / pop)
    'Smean':        ('SGCL', {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'router': 'mean'}, 100, False, 0),
    'Smean_geom':   ('SGCL', {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'router': 'mean', 'geom_w': 0.5}, 100, False, 0),
    'Smean_pop':    ('SGCL', {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'router': 'mean', 'pop_w': 0.1}, 100, False, 0),
    'Smean_pop05':  ('SGCL', {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'router': 'mean', 'pop_w': 0.5}, 100, False, 0),
    # geom_w sweep for the promising geometric-regularization method (GRGCL)
    'GRGCL_w01':    ('SGCL', {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'router': 'mean', 'geom_w': 0.1}, 100, False, 0),
    'GRGCL_w02':    ('SGCL', {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'router': 'mean', 'geom_w': 0.2}, 100, False, 0),
    'GRGCL_w05':    ('SGCL', {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'router': 'mean', 'geom_w': 0.5}, 100, False, 0),
    'GRGCL_w10':    ('SGCL', {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'router': 'mean', 'geom_w': 1.0}, 100, False, 0),
    'GRGCL_w20':    ('SGCL', {'n_layer': 2, 'lambda': 0.5, 'eps': 0.1, 'tau': 0.2, 'router': 'mean', 'geom_w': 2.0}, 100, False, 0),
    # DECISIVE: geometric-reg on the STRONGEST backbone (XSimGCL) — does it help + generalize?
    'XSimGCLg_w00': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 0.0}, 100, False, 0),
    'XSimGCLg_w05': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 0.5}, 100, False, 0),
    'XSimGCLg_w10': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0}, 100, False, 0),
    'XSimGCLg_w20': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 2.0}, 100, False, 0),
    # DENSITY-ADAPTIVE geom reg (geom_beta>0: tail-focused) — does it beat XSimGCL on BOTH regimes?
    'AdaG_b05':   ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0, 'geom_beta': 0.5}, 100, False, 0),
    'AdaG_b10':   ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0, 'geom_beta': 1.0}, 100, False, 0),
    'AdaG_b20':   ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0, 'geom_beta': 2.0}, 100, False, 0),
    'AdaG_b10w2': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 2.0, 'geom_beta': 1.0}, 100, False, 0),
    'AdaG_b20w2': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 2.0, 'geom_beta': 2.0}, 100, False, 0),
    # SSU (Spectral-Subspace Uniformity) — innovation-panel top pick. Beat AdaG on dense while holding sparse?
    'SSU_11':  ('SSU', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'lambda_h': 1.0, 'lambda_l': 1.0, 'rho': 0.9, 't': 2.0}, 100, False, 0),
    'SSU_h2':  ('SSU', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'lambda_h': 2.0, 'lambda_l': 1.0, 'rho': 0.9, 't': 2.0}, 100, False, 0),
    'SSU_l0':  ('SSU', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'lambda_h': 1.0, 'lambda_l': 0.0, 'rho': 0.9, 't': 2.0}, 100, False, 0),
    'SSU_r95': ('SSU', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'lambda_h': 1.0, 'lambda_l': 1.0, 'rho': 0.95, 't': 2.0}, 100, False, 0),
    'OURSgeom_b05': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false') + ' -geom_w 1.0 -geom_beta 0.5', 100, False, 20),
    'OURSgeom_b10': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false') + ' -geom_w 1.0 -geom_beta 1.0', 100, False, 20),
    'OURSgeom_w2':  ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'OURSg_w3':     ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false') + ' -geom_w 3.0 -geom_beta 0.5', 100, False, 20),
    'OURSg_w5':     ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false') + ' -geom_w 5.0 -geom_beta 0.5', 100, False, 20),
    'OURSg_nodual': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-dual_attention true','-dual_attention false').replace('-use_popularity true','-use_popularity false') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'OURSg_nod_w5': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-dual_attention true','-dual_attention false').replace('-use_popularity true','-use_popularity false') + ' -geom_w 5.0 -geom_beta 0.5', 100, False, 20),
    'OURSnm_w2': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-neg_mixup true','-neg_mixup false').replace('-n_negs 4','-n_negs 1') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'OURSnm_w5': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-neg_mixup true','-neg_mixup false').replace('-n_negs 4','-n_negs 1') + ' -geom_w 5.0 -geom_beta 0.5', 100, False, 20),
    'OURSnm_w0': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-neg_mixup true','-neg_mixup false').replace('-n_negs 4','-n_negs 1') + ' -geom_w 0.0', 100, False, 20),
    'SW_sp0p0': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-align_weight 0.02','-align_weight 0.0') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_sp0p002': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-align_weight 0.02','-align_weight 0.002') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_sp0p02': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-align_weight 0.02','-align_weight 0.02') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_sp0p2': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-align_weight 0.02','-align_weight 0.2') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_sp1p0': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-align_weight 0.02','-align_weight 1.0') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_u0p0': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-uniform_weight 0.01','-uniform_weight 0.0') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_u0p001': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-uniform_weight 0.01','-uniform_weight 0.001') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_u0p01': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-uniform_weight 0.01','-uniform_weight 0.01') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_u0p1': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-uniform_weight 0.01','-uniform_weight 0.1') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_u1p0': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-uniform_weight 0.01','-uniform_weight 1.0') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_wu0': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-warmup_epochs 10','-warmup_epochs 0') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_wu5': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-warmup_epochs 10','-warmup_epochs 5') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_wu10': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-warmup_epochs 10','-warmup_epochs 10') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_wu20': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-warmup_epochs 10','-warmup_epochs 20') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_wu30': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-warmup_epochs 10','-warmup_epochs 30') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_g0p1': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-pop_gamma 0.1','-pop_gamma 0.1') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_g0p3': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-pop_gamma 0.1','-pop_gamma 0.3') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SW_g0p5': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true', '-freeze_encoder false').replace('-pop_gamma 0.1','-pop_gamma 0.5') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'AB_full':   ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true','-freeze_encoder false') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'AB_nodual': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true','-freeze_encoder false') .replace('-dual_attention true','-dual_attention false') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    # capacity-matched popularity control: the head and its parameters stay, gamma=0 so it
    # cannot subtract anything -- separates 'the subtraction is inert' from 'the capacity is inert'.
    # dose-matched adaptive arms: same mu_g as the uniform column, so the contrast isolates beta
    'AdaG_b05w2': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 2.0, 'geom_beta': 0.5}, 100, False, 0),
    'AdaG_b10w2': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 2.0, 'geom_beta': 1.0}, 100, False, 0),
    # frozen encoder WITH the geometric term, so the frozen/joint ratio changes one thing only
    'OURSgeom_w2_fz': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'AB_popg0': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true','-freeze_encoder false').replace('-pop_gamma 0.1','-pop_gamma 0.0') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'AB_nopop':  ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true','-freeze_encoder false') .replace('-use_popularity true','-use_popularity false') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'AB_nogeom': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true','-freeze_encoder false'), 100, False, 20),
    'AB_nohn':   ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true','-freeze_encoder false') .replace('-n_negs 4','-n_negs 1').replace('-neg_mixup true','-neg_mixup false') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'AB_geomonly': ('PT4Rec_Enhanced', _OURS.format(bk='XSimGCL').replace('-freeze_encoder true','-freeze_encoder false') .replace('-dual_attention true','-dual_attention false').replace('-use_popularity true','-use_popularity false').replace('-n_negs 4','-n_negs 1').replace('-neg_mixup true','-neg_mixup false') + ' -geom_w 2.0 -geom_beta 0.5', 100, False, 20),
    'SRC':      ('SRC', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'gamma': 0.1, 'beta': 1.0}, 100, False, 0),
    'LightGCL': ('LightGCL', {'n_layer': 2, 'lambda': 0.2, 'tau': 0.2, 'q': 5}, 100, False, 0),
    # innovation-panel-2 candidates: PCNS (negative selection) + OT-CF (objective)
    'PCNS':     ('PCNS', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'n_cand': 32, 'pcns_lambda': 1.0}, 100, False, 0),
    'PCNS_dns': ('PCNS', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'n_cand': 32, 'pcns_lambda': 0.0}, 100, False, 0),
    'PCNS_l2':  ('PCNS', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'n_cand': 32, 'pcns_lambda': 2.0}, 100, False, 0),
    'OTCF_r05': ('OTCF', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'rho': 0.5, 'sinkhorn_eps': 0.05, 'sinkhorn_iters': 8}, 100, False, 0),
    'OTCF_r10': ('OTCF', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'rho': 1.0, 'sinkhorn_eps': 0.05, 'sinkhorn_iters': 8}, 100, False, 0),
    # sharp-gate AdaG: zero geom reg for dense nodes (pure XSimGCL) -> aim >= XSimGCL on BOTH regimes
    'AdaS_t5':   ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0, 'geom_gate': 'sigmoid', 'geom_tau': 0.5, 'geom_a': 5.0}, 100, False, 0),
    'AdaS_t3':   ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0, 'geom_gate': 'sigmoid', 'geom_tau': 0.3, 'geom_a': 5.0}, 100, False, 0),
    'AdaS_t7':   ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0, 'geom_gate': 'sigmoid', 'geom_tau': 0.7, 'geom_a': 5.0}, 100, False, 0),
    'AdaS_t5w2': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 2.0, 'geom_gate': 'sigmoid', 'geom_tau': 0.5, 'geom_a': 5.0}, 100, False, 0),
    'AdaS_t6':  ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0, 'geom_gate': 'sigmoid', 'geom_tau': 0.6, 'geom_a': 5.0}, 100, False, 0),
    'AdaS_t8':  ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0, 'geom_gate': 'sigmoid', 'geom_tau': 0.8, 'geom_a': 5.0}, 100, False, 0),
    'AdaS_t9':  ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0, 'geom_gate': 'sigmoid', 'geom_tau': 0.9, 'geom_a': 5.0}, 100, False, 0),
    'AdaS_t9w05': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 0.5, 'geom_gate': 'sigmoid', 'geom_tau': 0.9, 'geom_a': 5.0}, 100, False, 0),
    'AdaS_t99': ('XSimGCLg', {'n_layer': 2, 'l_star': 1, 'lambda': 0.2, 'eps': 0.2, 'tau': 0.15, 'geom_w': 1.0, 'geom_gate': 'sigmoid', 'geom_tau': 0.99, 'geom_a': 5.0}, 100, False, 0),
}

ALL_ORDER = ['MF','LightGCN','SGL','NCL','SSL4Rec','DirectAU','BUIR','SEPT','SelfCF','MHCN','CPTPP',
             'PTbase_Sim','OURS_Sim','PTbase_XSim','OURS_XSim','SimGCL','XSimGCL']


def build_yaml(tag, dataset):
    base = tag[:-4] if tag.endswith('_VR0') else (tag[:-8] if tag.endswith('_SELTEST') else tag)
    name, params, max_epoch, needs_social, preepoch = MODELS[base]
    train, test, has_social = DATASETS[dataset]
    lines = [f'training.set: {train}', f'test.set: {test}']
    if needs_social:
        if not has_social:
            return None  # skip social model on non-social dataset
        lines.append(f'social.data: ./dataset/{dataset}/trust.txt')
    vr = '0.0' if tag.endswith('_VR0') else '0.1'
    seltest = tag.endswith('_SELTEST')
    lines += ['model:', f'  name: {name}', '  type: graph',
              'item.ranking.topN: [10,20]', 'embedding.size: 64',
              f'max.epoch: {max_epoch}', 'batch.size: 2048',
              'learning.rate: 0.001', 'reg.lambda: 0.0001',
              f'valid.ratio: {vr}', 'split.seed: 2024', 'eval.every: 5']
    if seltest: lines.append('select.on: test')
    if preepoch > 0:
        lines.append(f'num.max.preepoch: {preepoch}')
    if isinstance(params, str):
        lines.append(f'{name}: {params}')
    elif params:
        lines.append(f'{name}:')
        for k, v in params.items():
            lines.append(f'  {k}: {v}')
    lines.append('output: ./results/')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='')
    ap.add_argument('--datasets', default='douban-book')
    ap.add_argument('--seeds', default='2024,2025,2026')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    os.chdir(REPO)
    os.makedirs('conf/grid', exist_ok=True)
    os.makedirs('results/wm', exist_ok=True)
    models = ALL_ORDER if args.all else [m.strip() for m in args.models.split(',') if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    jobs = []
    for ds in datasets:
        for tag in models:
            y = build_yaml(tag, ds)
            if y is None:
                print(f'skip {tag} on {ds} (needs social)'); continue
            cfg = f'conf/grid/{tag}__{ds}.yaml'
            with open(cfg, 'w') as f:
                f.write(y)
            for seed in seeds:
                out_json = f'results/wm/{tag}__{ds}__seed{seed}.json'
                jobs.append((tag, ds, seed, cfg, out_json))

    print(f'{len(jobs)} runs queued')
    for i, (tag, ds, seed, cfg, out_json) in enumerate(jobs):
        if os.path.exists(out_json):
            print(f'[{i+1}/{len(jobs)}] SKIP (exists) {tag} {ds} seed{seed}')
            continue
        print(f'[{i+1}/{len(jobs)}] RUN {tag} {ds} seed{seed}', flush=True)
        if args.dry:
            continue
        env = dict(os.environ, RUN_SEED=str(seed), RUN_TAG=tag)
        t0 = time.time()
        r = subprocess.run([PY, 'main.py', '--config', cfg], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f'   -> exit {r.returncode} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
