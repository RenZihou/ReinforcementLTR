"""implements Deep Deterministic Policy Gradient"""
# -*- encoding: utf-8 -*-
# @Author: RenZihou

import torch


def train_ddpg_one_epoch(model: 'ActorCritic', train_set, device, args):
    """train DDPG one epoch"""
    model.train()
    criterion = torch.nn.MSELoss()

    for x, rel in train_set.generate_group_per_query():
        x = torch.tensor(x, dtype=torch.float32, device=device)
        score = model.actor(x)  # x is state
        score = torch.tanh(score)
        # action = score  # TODO random action? warmup?

        # simulate reward and update network
        reward = 0

        # Critic update
        model.critic.zero_grad()
        q_batch = model.critic(torch.concat([x, score], dim=1))
        q_targ = model.critic_target(torch.concat([x, model.actor_target(x)], dim=1))
        value_loss = criterion(q_batch, reward + args.gamma * q_targ)
        value_loss.backward()
        model.critic_optimizer.step()

        # Actor update
        model.actor.zero_grad()
        policy_loss = -model.critic(torch.concat([x, model.actor(x)], dim=1))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        model.actor_optimizer.step()

        # soft update target
        for param, target_param in zip(model.critic.parameters(), model.critic_target.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - args.tau) + param.data * args.tau
            )
        for param, target_param in zip(model.actor.parameters(), model.actor_target.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - args.tau) + param.data * args.tau
            )


if __name__ == '__main__':
    pass
