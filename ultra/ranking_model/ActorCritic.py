# -*- encoding: utf-8 -*-
# @Author: RenZihou

import torch
import torch.nn as nn

from ultra.ranking_model import DNN, device
import ultra.utils


class ActorCritic:
    def __init__(self, hparams_str, feature_size):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """
        self.actor = DNN(hparams_str, feature_size)
        self.critic = DNN(hparams_str, feature_size + 1)

    def build(self, input_list, noisy_params=None, noise_rate=0.05, **kwargs):
        """if you want score, call model.actor.build"""
        input_size = input_list[0].shape[0]
        action = self.actor.build(input_list, noisy_params, noise_rate, **kwargs)
        action = torch.cat(action, dim=0)
        action = torch.tanh(action)
        input_list = torch.cat(input_list, dim=0).to(device=device)
        state = torch.cat((input_list, action), dim=1)
        state_list = torch.split(state, input_size, dim=0)
        value = self.critic.build(state_list, noisy_params, noise_rate, **kwargs)
        return value

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
        self.critic.to(device)
        return self

    def load_state_dict(self, state_dict):
        """load state dict"""
        self.actor.load_state_dict(state_dict)

    def state_dict(self):
        """get state dict"""
        return self.actor.state_dict()


if __name__ == '__main__':
    pass
