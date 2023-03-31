"""implements PG Rank"""
# -*- encoding: utf-8 -*-
# @Author: RenZihou

import numpy as np
import torch

from metric import dcg, idcg


def log_pi(score, ranking, device):
    """log(pi_theta(r | q))"""
    def log_sum_exp(x, dim=0):
        """numerically stable log sum exp"""
        s, _ = torch.max(x, dim=dim, keepdim=True)
        outputs = s + (x - s).exp().sum(dim=dim, keepdim=True).log()
        return outputs.squeeze(dim)

    subtracts = torch.zeros_like(score, device=device)
    log_probs = torch.zeros_like(score, device=device)
    for i in range(score.size()[0]):
        pos_i = ranking[i]
        log_probs[i] = score[pos_i] - log_sum_exp(score - subtracts)
        subtracts[pos_i] = score[pos_i] + 1e6
    return torch.sum(log_probs)


def train_pgrank_one_epoch(model: 'DNN', train_set, device, args):
    """train PG Rank one epoch"""
    model.train()
    model.zero_grad()
    count = 0
    # loss_batch = []
    batch_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=True)

    for x, rel in train_set.generate_group_per_query():
        if np.sum(rel) == 0 or rel.size == 1:
            continue
        score = model(torch.tensor(x, dtype=torch.float32, device=device))
        score = 2 * torch.sigmoid(score) + 2
        probs = torch.softmax(score, dim=0)
        for i in range(args.sample_size):  # Monte Carlo sampling from eq.5 in paper
            ranking = np.random.choice(probs.shape[0], size=probs.shape[0], replace=False,
                                       p=probs.squeeze(1).cpu().detach().numpy())
            with torch.no_grad():
                ndcg = dcg(ranking, ranking.size) / idcg(ranking, ranking.size)  # no grad?
            loss = log_pi(score, ranking, device) * -ndcg
            # loss_batch.append(loss)
            batch_loss = batch_loss + loss

        count += 1
        if count & args.batch == 0:
            # for loss in loss_batch:
            #     loss.backward(retain_graph=True)
            batch_loss = batch_loss / args.batch
            # print(batch_loss)
            # print(torch.sum(model.linear1.weight))
            batch_loss.backward()
            model.optimizer.step()
            model.zero_grad()
            # loss_batch = []
            batch_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=True)
    model.scheduler.step()


if __name__ == '__main__':
    pass
