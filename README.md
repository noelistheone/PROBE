# Degree-Adaptive Regularization for Graph Contrastive Recommenders

Code and per-seed result records for the paper *"Geometry Is What Survives: Degree-Adaptive
Regularization for Graph Contrastive Recommenders under a Leakage-Controlled Protocol"*.

The repository contains everything needed to re-run the experiments and to regenerate every
table and figure in the paper directly from the released records. It is anonymous: no author,
institution, or account information appears anywhere in the code or in the records.

## What is here

```
main.py, SELFRec.py        entry point and dispatcher
base/, data/, util/        harness: split, graph construction, sampling, evaluation
model/graph/               the models compared in the paper
scripts/                   experiment driver and the table / figure generators
conf/grid/                 one config per (tag, dataset) actually run
results/wm/                per-seed JSON records from the current harness
results/wm_noisefloor/     18 repeats of one configuration (the measured noise floor)
results/prev_sampler/      records produced under the earlier, slower negative sampler
                           (the cells marked with a dagger in the transfer table)
```

The method itself is in `model/graph/PT4Rec_Enhanced.py`: `_init_geom_weights` computes the
degree weights `w_n = (1+d_n)^-beta`, normalized to unit mean, and `_align_w` / `_unif_w` apply
them to the alignment and uniformity terms. The regularizer has no learnable parameters.

## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is required (the encoders call `.cuda()` directly). All results in the paper were
produced on a single RTX 4090.

## Data

The datasets are public but are not redistributed here. Place each one under
`dataset/<name>/{train.txt,test.txt}`, one interaction per line as `user item rating`
(whitespace separated); `rating` is read but not used. The four datasets are Amazon-Kindle,
Yelp2018, Douban-Book and MovieLens-1M (ratings binarized at >= 4).

The validation split is *not* a file: it is carved out of `train.txt` inside the loader
(`data/ui_graph.py`), per user, with a fixed `split.seed`, before the graph, the popularity
counts and the SVD view are built. The chronological split used for the ML-1M robustness check
is built by

```bash
python scripts/make_temporal_split.py dataset/ml-1M/ratings_ts.txt dataset/ml-1M-temporal
```

which holds out each user's most recent 20% of interactions (`ratings_ts.txt` is the raw
MovieLens file reformatted as `user item rating timestamp`).

## Running experiments

A single run:

```bash
python main.py --config ./conf/grid/OURSgeom_w2__douban-book.yaml
```

The grid driver generates the config, runs every seed and writes one JSON record per run into
`results/wm/`. It skips runs whose record already exists, so it is resumable.

```bash
python scripts/run_grid.py --models OURSgeom_w2,PTbase_XSim_nofz,XSimGCLg_w00 \
    --datasets douban-book --seeds 2024,2025,2026
python scripts/run_grid.py --all --datasets douban-book,yelp2018,ml-1M --seeds 2024,2025,2026
```

Two protocol switches are exposed as ordinary config keys, which is what makes the
decomposition in the paper measurable:

| key | effect |
|---|---|
| `valid.ratio: 0.1` | fraction of each user's training interactions held out for validation; `0` disables the split entirely (the `_VR0` arms) |
| `select.on: test` | keeps the validation split carved out but selects the reported epoch on the test set (the `_SELTEST` arms) |
| `-freeze_encoder true/false` | whether the pre-trained encoder receives gradients during prompt-tuning |

## Reproducing the numbers

```bash
python scripts/report_paper_numbers.py     # every number in the paper, with its source tag
python scripts/significance.py             # paired tests over seeds, ours vs the best baseline
python scripts/per_user_significance.py    # Wilcoxon signed-rank over users, from the ranking dumps
python scripts/make_table.py  > tab_main.tex
python scripts/make_transfer.py > tab_transfer.tex
python scripts/make_protocol.py > tab_protocol.tex
python scripts/make_ablation.py > tab_ablation.tex
python scripts/make_figs.py                # writes figures/
```

All tables use the three canonical seeds 2024, 2025 and 2026.

### Which tag is which row

| paper | tag |
|---|---|
| XSimGCL (backbone) | `XSimGCLg_w00` |
| PT4Rec | `PTbase_XSim_nofz` |
| Ours (geometric module off) | `OURS_XSim_nofz` |
| Ours | `OURSgeom_w2` |
| leave-one-out ablation | `AB_full`, `AB_nogeom`, `AB_nodual`, `AB_nopop`, `AB_nohn`, `AB_geomonly` |
| selection on test / no validation split | tags suffixed `_SELTEST` / `_VR0` |
| encoder genuinely frozen | `PTbase_XSim`, `OURS_XSim` (in `results/prev_sampler/`) |
| regularizer on the encoder alone | `XSimGCLg_w00` (off), `XSimGCLg_w20` (uniform), `AdaG_b05`, `AdaG_b10` |
| dose-response | `XSimGCLg_w00`, `XSimGCLg_w10`, `XSimGCLg_w20` |
| noise floor | `results/wm_noisefloor/` (Douban); on ML-1M, `AB_full` and `OURSgeom_w2` are byte-identical configurations, giving six runs of one config |
| capacity-matched popularity control | `AB_popg0` |
| chronological split | any tag on dataset `ml-1M-temporal` |

The PT4Rec baseline is run through the same code path with every added module disabled
(`-dual_attention false -use_popularity false -align_weight 0.0 -uniform_weight 0.0 -n_negs 1
-neg_mixup false`), so the comparison isolates the modules rather than two implementations.

Each record stores the metrics, the epoch chosen on validation, the validation scores at that
epoch, the exact hyperparameter string, and (for later runs) the singular spectrum and effective
rank of the learned embeddings.

## Acknowledgement

The harness is built on the SELFRec library for self-supervised recommendation.

```
@article{yu2023self,
  title={Self-supervised learning for recommender systems: A survey},
  author={Yu, Junliang and Yin, Hongzhi and Xia, Xin and Chen, Tong and Li, Jundong and Huang, Zi},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  publisher={IEEE}
}
```
