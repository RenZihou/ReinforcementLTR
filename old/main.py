# -*- encoding: utf-8 -*-
# @Author: RenZihou

from argparse import ArgumentParser
import pathlib
import time

import torch

from dataset import MSLRDataset
from dnn import DNN, ActorCritic, train
from metric import evaluate
# models
from crossentropy import train_crossentropy_one_epoch
from ddpg import train_ddpg_one_epoch
from lambdarank import train_lambdarank_one_epoch
from pgrank import train_pgrank_one_epoch
from reinforce import train_reinforce_one_epoch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', '-D', type=str, default=device, help='device to use')
    parser.add_argument('--dataset', '-d', type=str, help='path to dataset')
    # parser.add_argument('--model', '-m', type=str, choices=['dnn'], help='model to use')
    parser.add_argument('--algorithm', '-a', type=str, help='learning algorithm')
    # parser.add_argument('--checkpoint', '-c', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--features', '-f', type=int, default=136, help='number of features')
    parser.add_argument('--epoch', type=int, default=1000, help='number of epoch')
    parser.add_argument('--eval_epoch', type=int, default=10, help='evaluate every n epoch')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--sigma', type=float, default=1.0, help='sigma for lambdarank')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma for reinforce q discount')
    parser.add_argument('--tau', type=float, default=0.001, help='tau for reinforce soft update')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='Monte Carlo sample size for pgrank')

    args = parser.parse_args()
    args.checkpoint = f'runs/{args.algorithm}_{args.dataset.split("/")[-1]}_{int(time.time())}'
    model = {
        'crossentropy': DNN,
        'lambdarank': DNN,
        'pgrank': DNN,
        'reinforce': DNN,
        'ddpg': ActorCritic,
    }[args.algorithm](args.features).to(device)
    algorithm = {
        'crossentropy': train_crossentropy_one_epoch,
        'lambdarank': train_lambdarank_one_epoch,
        'pgrank': train_pgrank_one_epoch,
        'reinforce': train_reinforce_one_epoch,
        'ddpg': train_ddpg_one_epoch,
    }[args.algorithm]
    train_dataset = MSLRDataset(pathlib.Path(args.dataset) / 'train.txt', args.features)
    valid_dataset = MSLRDataset(pathlib.Path(args.dataset) / 'valid.txt', args.features)
    test_dataset = MSLRDataset(pathlib.Path(args.dataset) / 'test.txt', args.features)

    train(model, algorithm, train_dataset, valid_dataset, device, args)

    # load best model
    model.load(args.checkpoint)
    ndcg, err = evaluate(model, test_dataset, [1, 3, 5, 10], device)
    print(f'==== Test {args.algorithm} on {args.dataset} ====')
    print(
        f'NDCG@1: {ndcg[1]:.4f}, NDCG@3: {ndcg[3]:.4f}, NDCG@5: {ndcg[5]:.4f}, NDCG@10: {ndcg[10]:.4f}')
    print(f'ERR@1: {err[1]:.4f}, ERR@3: {err[3]:.4f}, ERR@5: {err[5]:.4f}, ERR@10: {err[10]:.4f}')
    print(f'model saved to {args.checkpoint}')
    pass
