"""implements LambdaRank"""
# -*- encoding: utf-8 -*-
# @Author: RenZihou

import numpy as np
import torch

from metric import idcg


def train_lambdarank_one_epoch(model: 'DNN', train_set, device, args):
    """train one epoch"""
    model.train()
    model.zero_grad()
    count = 0
    grad_batch, y_pred_batch = [], []

    for x, rel in train_set.generate_group_per_query():
        if np.sum(rel) == 0:
            # negative session, cannot learn useful signal
            continue
        n = 1.0 / idcg(rel)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        y_pred = model(x_tensor)
        y_pred_batch.append(y_pred)
        # compute the rank order of each document
        # order the document using the relevance score, higher score's order rank's higher.
        rank_order = y_pred.reshape(-1).argsort(descending=True).argsort() + 1

        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp(args.sigma * (y_pred - y_pred.t()))
            y_tensor = torch.tensor(rel, dtype=torch.float32, device=device).view(-1, 1)
            rel_diff = y_tensor - y_tensor.t()
            pos_pairs = (rel_diff > 0).type(torch.float32)
            neg_pairs = (rel_diff < 0).type(torch.float32)
            sij = pos_pairs - neg_pairs
            gain_diff = torch.pow(2.0, y_tensor) - torch.pow(2.0, y_tensor.t())
            rank_order_tensor = rank_order.reshape(-1, 1)
            decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) \
                - 1.0 / torch.log2(rank_order_tensor.t() + 1.0)
            delta_ndcg = torch.abs(n * gain_diff * decay_diff)
            lambda_update = args.sigma * (0.5 * (1 - sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, 1, keepdim=True)
            grad_batch.append(lambda_update)

        count += 1
        if count % args.batch == 0:
            for grad, y_pred in zip(grad_batch, y_pred_batch):
                y_pred.backward(grad / args.batch)
            model.optimizer.step()
            model.zero_grad()
            grad_batch, y_pred_batch = [], []
    model.scheduler.step()


if __name__ == '__main__':
    pass
