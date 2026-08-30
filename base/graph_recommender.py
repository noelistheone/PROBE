from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []
        self.topN = [int(num) for num in self.ranking]
        self.max_N = max(self.topN)

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # print dataset statistics
        print(f'Training Set Size: (user number: {self.data.training_size()[0]}, '
              f'item number: {self.data.training_size()[1]}, '
              f'interaction number: {self.data.training_size()[2]})')
        print(f'Validation Set Size: (user number: {self.data.valid_size()[0]}, '
              f'item number: {self.data.valid_size()[1]}, '
              f'interaction number: {self.data.valid_size()[2]})')
        print(f'Test Set Size: (user number: {self.data.test_size()[0]}, '
              f'item number: {self.data.test_size()[1]}, '
              f'interaction number: {self.data.test_size()[2]})')
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def _generate_reclist(self, eval_users, filter_dicts):
        """Rank max_N items for each eval user, masking out any item that appears
        in one of filter_dicts (e.g. training items, and validation items at test
        time). This is the single leakage-controlled ranking routine used for both
        validation-time selection and final test-time evaluation.

        Uses a fast batched GPU path when the model exposes full user_emb/item_emb
        tensors (all graph models here do); falls back to the per-user predict()
        loop otherwise."""
        try:
            import torch
            ue = getattr(self, 'user_emb', None)
            ie = getattr(self, 'item_emb', None)
            if torch.is_tensor(ue) and torch.is_tensor(ie):
                return self._batched_reclist(eval_users, filter_dicts, ue, ie)
        except Exception as e:
            print('batched ranking unavailable, falling back to per-user:', e)

        rec_list = {}
        user_count = len(eval_users)
        for i, user in enumerate(eval_users):
            candidates = self.predict(user)
            for fd in filter_dicts:
                for item in fd.get(user, {}):
                    iid = self.data.item.get(item)
                    if iid is not None:
                        candidates[iid] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 5000 == 0:
                print(f'\rRanking: {i}/{user_count}', end='', flush=True)
        print('')
        return rec_list

    def _batched_reclist(self, eval_users, filter_dicts, ue, ie, chunk=1024):
        """Batched full ranking on GPU: scores = U @ I^T, mask filtered items, top-N."""
        import torch
        rec_list = {}
        ie_t = ie.t().contiguous()
        with torch.no_grad():
            for start in range(0, len(eval_users), chunk):
                batch_users = eval_users[start:start + chunk]
                rows = torch.tensor([self.data.user[u] for u in batch_users],
                                    device=ue.device, dtype=torch.long)
                scores = ue[rows] @ ie_t                      # (b, M)
                for bi, user in enumerate(batch_users):       # mask train/valid items
                    for fd in filter_dicts:
                        d = fd.get(user)
                        if not d:
                            continue
                        cols = [self.data.item[it] for it in d if it in self.data.item]
                        if cols:
                            scores[bi, torch.tensor(cols, device=scores.device, dtype=torch.long)] = -1e8
                topv, topi = torch.topk(scores, self.max_N, dim=1)
                topv = topv.cpu().numpy(); topi = topi.cpu().numpy()
                for bi, user in enumerate(batch_users):
                    rec_list[user] = [(self.data.id2item[int(topi[bi, r])], float(topv[bi, r]))
                                      for r in range(self.max_N)]
                print(f'\rRanking(batched): {min(start + chunk, len(eval_users))}/{len(eval_users)}',
                      end='', flush=True)
        print('')
        return rec_list

    def test(self):
        # Final evaluation on the TEST set. Mask training AND validation items so
        # neither the trained-on nor the model-selection interactions can be re-ranked.
        return self._generate_reclist(list(self.data.test_set.keys()),
                                      [self.data.training_set_u, self.data.valid_set])

    def valid(self):
        # Model-selection ranking on the VALIDATION set. Mask training items only.
        return self._generate_reclist(list(self.data.valid_set.keys()),
                                      [self.data.training_set_u])

    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = user + ':' + ''.join(
                f" ({item[0]},{item[1]}){'*' if item[0] in self.data.test_set[user] else ''}"
                for item in rec_list[user]
            )
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_dir = self.output
        file_name = f"{self.config['model']['name']}@{current_time}-top-{self.max_N}items.txt"
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = f"{self.config['model']['name']}@{current_time}-performance.txt"
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print(f'The result of {self.model_name}:\n{"".join(self.result)}')
        self._persist_json(rec_list)

    def _persist_json(self, rec_list):
        """Persist a structured JSON with standard + beyond-accuracy metrics so
        every run is recorded to disk (not just the human-readable txt)."""
        import os, json
        from util.evaluation import Metric
        try:
            from util.extended_eval import extended_metrics
        except Exception:
            extended_metrics = None
        metrics = {}
        for n in self.topN:
            predicted = {u: rec_list[u][:n] for u in rec_list}
            hits = Metric.hits(self.data.test_set, predicted)
            metrics[f'HitRatio@{n}'] = Metric.hit_ratio(self.data.test_set, hits)
            metrics[f'Precision@{n}'] = Metric.precision(hits, n)
            metrics[f'Recall@{n}'] = Metric.recall(hits, self.data.test_set)
            metrics[f'NDCG@{n}'] = Metric.NDCG(self.data.test_set, predicted, n)
            if extended_metrics is not None:
                try:
                    metrics.update(extended_metrics(self.data.test_set, rec_list, self.data, n))
                except Exception as e:
                    print('extended_metrics failed:', e)
        try:
            dataset = os.path.basename(os.path.dirname(self.config['training.set']))
        except Exception:
            dataset = 'unknown'
        # embedding geometry diagnostics (no-collapse evidence: singular spectrum / effective rank)
        geo = {}
        try:
            import torch as _t
            ue, ie = getattr(self, 'user_emb', None), getattr(self, 'item_emb', None)
            if _t.is_tensor(ue) and _t.is_tensor(ie):
                for nm, E in (('user', ue), ('item', ie)):
                    X = _t.nn.functional.normalize(E.detach().float(), dim=-1)
                    sv = _t.linalg.svdvals(X - X.mean(0, keepdim=True))
                    p = (sv / sv.sum().clamp_min(1e-12)).clamp_min(1e-12)
                    geo[nm + '_eff_rank'] = float(_t.exp(-(p * p.log()).sum()))
                    geo[nm + '_sv_top20'] = [float(x) for x in sv[:20]]
                    geo[nm + '_cond'] = float(sv[0] / sv[min(63, len(sv)-1)].clamp_min(1e-12))
        except Exception as e:
            geo['error'] = str(e)

        record = {
            'geometry': geo,
            'curve': getattr(self, 'curve', None),
            'model': self.model_name,
            'dataset': dataset,
            'seed': int(os.environ.get('RUN_SEED', -1)),
            'best_val_epoch': self.bestPerformance[0] if getattr(self, 'bestPerformance', None) else None,
            # validation metrics at the selected epoch (for leak-free config selection in sweeps)
            'val_metrics': self.bestPerformance[1] if getattr(self, 'bestPerformance', None) else None,
            'config': str(self.config[self.model_name]) if self.config.contain(self.model_name) else '',
            'metrics': metrics,
        }
        tag = os.environ.get('RUN_TAG', self.model_name)
        out = os.path.join('results', 'wm')
        os.makedirs(out, exist_ok=True)
        fname = os.path.join(out, f"{tag}__{dataset}__seed{record['seed']}.json")
        with open(fname, 'w') as f:
            json.dump(record, f, indent=2)
        print('Persisted metrics ->', fname)

    def fast_evaluation(self, epoch):
        # Speed control: only actually evaluate every `eval.every` epochs (+ always the
        # last), to avoid a full-ranking eval on every epoch. Default 1 = every epoch.
        ev = int(self.config['eval.every']) if self.config.contain('eval.every') else 1
        if ev > 1 and (epoch % ev != 0) and (epoch != self.maxEpoch - 1):
            return None
        # Select the best epoch on the VALIDATION set (no test-set peeking). If no
        # validation split was configured (valid.ratio<=0), fall back to the test set.
        # 'select.on: test' keeps the validation split carved out (so the TRAINING data is
        # identical) but performs model selection on the test set, isolating the selection axis.
        sel = self.config['select.on'] if self.config.contain('select.on') else 'valid'
        use_valid = len(self.data.valid_set) > 0 and sel != 'test'

        # Diagnostic instrument (off by default): record BOTH the validation and the test score at
        # every evaluation epoch of a single run, so the selection effect can be measured exactly --
        # test-at-argmax-test minus test-at-argmax-val -- without the run-to-run noise that separate
        # val-selected and test-selected runs would carry. This is a measurement, not model selection:
        # nothing here influences which checkpoint the run keeps.
        if self.config.contain('log.curve') and str(self.config['log.curve']).lower() == 'true':
            if not hasattr(self, 'curve'):
                self.curve = []
            v_list = ranking_evaluation(self.data.valid_set, self.valid(), [self.max_N])
            t_list = ranking_evaluation(self.data.test_set, self.test(), [self.max_N])
            gv = lambda mm: {k: float(v) for m in mm[1:] for k, v in [m.strip().split(':')]}
            self.curve.append({'epoch': epoch + 1,
                               'valid_NDCG': gv(v_list).get('NDCG'),
                               'test_NDCG': gv(t_list).get('NDCG')})
        print('Evaluating the model (%s)...' % ('validation' if use_valid else 'test'))
        eval_set = self.data.valid_set if use_valid else self.data.test_set
        rec_list = self.valid() if use_valid else self.test()
        measure = ranking_evaluation(eval_set, rec_list, [self.max_N])

        performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}

        if self.bestPerformance:
            count = sum(1 if self.bestPerformance[1][k] > performance[k] else -1 for k in performance)
            if count < 0:
                self.bestPerformance = [epoch + 1, performance]
                self.save()
        else:
            self.bestPerformance = [epoch + 1, performance]
            self.save()

        print('-' * 80)
        print(f'Real-Time Ranking Performance (Top-{self.max_N} Item Recommendation)')
        measure_str = ', '.join([f'{k}: {v}' for k, v in performance.items()])
        print(f'*Current Performance*\nEpoch: {epoch + 1}, {measure_str}')
        bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
        print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
        print('-' * 80)
        return measure
