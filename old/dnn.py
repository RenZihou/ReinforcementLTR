"""
implements deep neural model (trained with fine-grained labels) as baseline
"""
# -*- encoding: utf-8 -*-
# @Author: RenZihou

import torch
import torch.nn as nn
from tqdm import trange

from metric import evaluate


class DNN(nn.Module):
    """deep neural network"""

    def __init__(self, features: int):
        super(DNN, self).__init__()
        self.linear1 = nn.Linear(features, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 1)
        self.activation = nn.ELU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)

    def forward(self, x):
        """forward"""
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)  # no activation after last layer
        return x

    def save(self, checkpoint):
        """save model"""
        torch.save(self.state_dict(), checkpoint + '.ckpt')

    def load(self, checkpoint):
        """load model"""
        self.load_state_dict(torch.load(checkpoint + '.ckpt'))


class ActorCritic:
    """Actor-Critic model with two DNNs"""
    def __init__(self, features: int):
        self.actor = DNN(features)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        # self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=200, gamma=0.5)
        self.critic = DNN(features + 1)  # with one action input
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        # self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=200, gamma=0.5)

        self.actor_target = DNN(features)
        self.critic_target = DNN(features + 1)

    def save(self, checkpoint):
        """save model"""
        self.actor.save(checkpoint + '_actor')
        self.critic.save(checkpoint + '_critic')

    def load(self, checkpoint):
        """load model"""
        self.actor.load(checkpoint + '_actor')
        self.critic.load(checkpoint + '_critic')

    def train(self):
        """enter train phase"""
        self.actor.train()
        self.critic.train()

    def eval(self):
        """enter evaluate phase"""
        self.actor.eval()
        self.critic.eval()

    def to(self, device):
        """move model to device"""
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        return self

    def __call__(self, *args, **kwargs):
        """forward"""
        return self.actor(*args, **kwargs)


def train(model, train_one_epoch, train_set, valid_set, device, args):
    """
    train DNN on any algorithm
    :param model: DNN model
    :param train_one_epoch: implemented algorithm for training one epoch
    :param train_set: training dataset
    :param valid_set: validation dataset
    :param device: device
    :param args: training arguments
    """
    best_ndcg = 0
    pbar = trange(args.epoch)
    for e in pbar:
        train_one_epoch(model, train_set, device, args)
        # print(torch.sum(model.linear1.weight))
        if (e + 1) % args.eval_epoch == 0:
            ndcg, err = evaluate(model, valid_set, [1, 3, 5, 10], device)
            pbar.set_postfix({'ndcg@10': ndcg[10], 'err@10': err[10]})
            if ndcg[10] > best_ndcg:
                best_ndcg = ndcg[10]
                model.save(args.checkpoint)


if __name__ == '__main__':
    pass
