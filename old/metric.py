# -*- encoding: utf-8 -*-
# @Author: RenZihou

from collections import defaultdict

import numpy as np
import pandas as pd
import torch


def dcg(y, k=1000):
    """discounted cumulative gain"""
    return np.sum((np.power(2, y[:k]) - 1) / (np.log2(np.arange(2, min(y.size, k) + 2))))


def idcg(y, k=1000):
    """ideal DCG"""
    y = np.sort(y)[::-1]
    return dcg(y, k)


def eval_err(model, valid_set, k_list, device):
    """evaluate err on validation set at k_list"""
    model.eval()
    err = defaultdict(list)
    with torch.no_grad():
        for x, y in valid_set.generate_group_per_query():
            if x is None or x.shape[0] == 0:
                continue
            y_tensor = model.forward(torch.Tensor(x).to(device))
            scores = y_tensor.cpu().numpy().reshape(-1)
            labels = y.reshape(-1)
            result = pd.DataFrame({'score': scores, 'label': labels})\
                .sort_values('score', ascending=False)
            rel_rank = result['label'].values
            for k in k_list:
                p = 1
                e = 0
                for r in range(0, min(len(rel_rank), k)):
                    r_g = (2 ** rel_rank[r] - 1) / 2 ** 4
                    e += p * r_g / (r + 1)
                    p *= 1 - r_g
                err[k].append(e)
    return {k: np.mean(err[k]) for k in k_list}


def eval_ndcg(model, valid_set, k_list, device):
    """evaluate ndcg on validation set at k_list"""
    model.eval()
    ndcg = defaultdict(list)
    with torch.no_grad():
        for x, rel in valid_set.generate_group_per_query():
            if x is None or x.shape[0] == 0:
                continue
            y_tensor = model.forward(torch.Tensor(x).to(device))
            scores = y_tensor.cpu().numpy().reshape(-1)
            labels = rel.reshape(-1)
            # ipdb.set_trace()
            result = pd.DataFrame({'score': scores, 'label': labels})\
                .sort_values('score', ascending=False)
            rel_rank = result['label'].values
            for k in k_list:
                # if ndcg.maxDCG(rel_rank) == 0:
                #     continue
                # ndcg_k = ndcg.evaluate(rel_rank)
                # if not np.isnan(ndcg_k):
                #     session_ndcgs[k].append(ndcg_k)
                idcg_ = idcg(rel_rank, k)
                if idcg_:
                    ndcg[k].append(dcg(rel_rank, k) / idcg_)
    return {k: np.mean(ndcg[k]) for k in k_list}


def evaluate(model, valid_set, k_list, device):
    """evaluate NDCG and ERR on validation set @ k_list"""
    model.eval()
    ndcg = defaultdict(list)
    err = defaultdict(list)
    with torch.no_grad():
        for x, y in valid_set.generate_group_per_query():
            if x is None or x.shape[0] == 0:
                continue
            y_tensor = model(torch.tensor(x).to(device))
            scores = y_tensor.cpu().numpy().reshape(-1)
            labels = y.reshape(-1)
            # ipdb.set_trace()
            result = pd.DataFrame({'score': scores, 'label': labels})\
                .sort_values('score', ascending=False)
            rel_rank = result['label'].values
            for k in k_list:
                # evaluate ndcg
                idcg_ = idcg(rel_rank, k)
                if idcg_:
                    ndcg[k].append(dcg(rel_rank, k) / idcg_)
                # evaluate err
                p = 1
                e = 0
                for r in range(0, min(len(rel_rank), k)):
                    r_g = (2 ** rel_rank[r] - 1) / 2 ** 4
                    e += p * r_g / (r + 1)
                    p *= 1 - r_g
                err[k].append(e)
    return {k: np.mean(ndcg[k]) for k in k_list}, {k: np.mean(err[k]) for k in k_list}


if __name__ == '__main__':
    pass
