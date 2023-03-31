"""implements cross entropy"""
# -*- encoding: utf-8 -*-
# @Author: RenZihou

import ipdb
import numpy as np
import torch


def train_crossentropy_one_epoch(model: 'DNN', train_set, device, args):
    """train cross entropy one epoch"""
    model.train()
    model.zero_grad()
    count = 0
    loss = torch.nn.CrossEntropyLoss()
    batch_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=True)

    for x, y_true in train_set.generate_group_per_query():
        if np.sum(y_true) == 0:
            # negative session, cannot learn useful signal
            continue
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_true, dtype=torch.float32, device=device).reshape(-1, 1)
        y_pred = model(x_tensor)
        y_pred = torch.sigmoid(y_pred)
        # y_pred = torch.clamp(y_pred, 1e-5, 1)
        a_true = y_tensor / torch.sum(y_tensor)
        a_pred = y_pred / torch.sum(y_pred)
        lo = loss(a_pred.t(), a_true.t())
        # batch_loss = batch_loss + lo

        lo.backward()
        model.optimizer.step()
        model.zero_grad()

        # count += 1
        # if count % args.batch == 0:
        #     batch_loss.backward()
        #     optimizer.step()
        #     model.zero_grad()
        #     grad_batch, y_pred_batch = [], []
        #     batch_loss = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=True)

    model.scheduler.step()


if __name__ == '__main__':
    pass
