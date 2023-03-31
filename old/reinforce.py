"""implements Top-k Off-Policy Correction for a REINFORCE Model"""
# -*- encoding: utf-8 -*-
# @Author: RenZihou
import ipdb
import numpy as np
import torch

from metric import dcg, idcg


def train_reinforce_one_epoch(model: 'DNN', train_set, device, args):
    """train REINFORCE one epoch"""
    model.train()
    model.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()

    for x, rel in train_set.generate_group_per_query():
        if np.sum(rel) == 0:
            # negative session, cannot learn useful signal
            continue

        score = model(torch.tensor(x, dtype=torch.float32, device=device))
        score = torch.sigmoid(score)
        probs = torch.softmax(score, dim=0).squeeze(1)
        ranking = torch.multinomial(probs, probs.shape[0], replacement=True)  # action
        ranking_prob = probs[ranking.detach()]  # pi(d | q) in ranking order
        with torch.no_grad():  # no gradient for reward
            action_rel = rel[ranking.cpu().detach().numpy()]
            if np.sum(action_rel) == 0:
                continue
            reward = torch.zeros(probs.shape[0], dtype=torch.float32, device=device)
            for k in range(probs.shape[0]):
                reward[k] = dcg(action_rel, k + 1) / idcg(action_rel, k + 1)
            # reward = dcg(action_rel, probs.shape[0]) / idcg(action_rel, probs.shape[0])
            # alpha(d | q) in ranking order
            topk_grad = probs.shape[0] * (1 - ranking_prob) ** (probs.shape[0] - 1)
        loss = criterion(score.squeeze(1), torch.tensor(rel, dtype=torch.float32, device=device))
        loss = torch.sum(reward * topk_grad) * loss

        loss.backward()
        model.optimizer.step()
        model.zero_grad()

    model.scheduler.step()


if __name__ == '__main__':
    pass
