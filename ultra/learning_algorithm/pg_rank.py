# -*- encoding: utf-8 -*-
# @Author: RenZihou

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ipdb

from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils


class PgRank(BaseAlgorithm):
    def __init__(self, data_set, exp_settings, forward_only=False):
        print('Build PgRank')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='softmax_cross_entropy',            # Select Loss function
            sample_size=10,  # Sample size for each state
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
        )
        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.device = torch.device('cuda') if self.is_cuda_avail else torch.device('cpu')
        self.train_summary = {}
        self.eval_summary = {}
        self.is_training = "is_train"
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.feature_size = data_set.feature_size

        self.model = self.create_model(self.feature_size)
        if self.is_cuda_avail:
            self.model = self.model.to(device=self.device)
        self.max_candidate_num = exp_settings['max_candidate_num']

        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0

        # Feeds for inputs.
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))

        self.optimizer_func = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, input_feed):
        self.global_step += 1
        self.model.train()
        self.create_input_feed(input_feed, self.rank_list_size)

        # Gradients and SGD update operation for training the model.

        train_output = self.ranking_model(self.model,
                                          self.rank_list_size)
        train_labels = self.labels
        train_output = torch.clamp(train_output, min=-10, max=10)
        train_probs = torch.softmax(train_output, dim=1)
        preds_sorted, preds_sorted_inds = torch.sort(train_output, dim=1, descending=True)
        labels_sorted_via_preds = torch.gather(self.labels, dim=1, index=preds_sorted_inds)
        ideally_sorted_labels, _ = torch.sort(self.labels, dim=1, descending=True)
        self.loss = torch.tensor(0, dtype=torch.float32, device=self.device,
                                 requires_grad=True)
        for _ in range(self.hparams.sample_size):
            with torch.no_grad():
                try:
                    ranking = torch.multinomial(train_probs, train_probs.shape[1], replacement=False)
                except RuntimeError:
                    ipdb.set_trace()
                ndcg = self.dcg(labels_sorted_via_preds) / self.dcg(ideally_sorted_labels)
            self.loss = self.loss + self.log_pi(train_output, ranking) * -ndcg

        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            loss_l2 = 0.0
            for p in params:
                loss_l2 += self.l2_loss(p)
            self.loss += self.hparams.l2_loss * loss_l2

        self.opt_step(self.optimizer_func, params)

        nn.utils.clip_grad_value_(train_labels, 1)
        print(" Loss %f at Global Step %d: " % (self.loss.item(), self.global_step))
        return self.loss.item(), None, self.train_summary

    def validation(self, input_feed, is_online_simulation=False):
        self.model.eval()
        self.create_input_feed(input_feed, self.max_candidate_num)
        with torch.no_grad():
            self.output = self.ranking_model(self.model,
                                             self.max_candidate_num)
            if not is_online_simulation:
                pad_removed_output = self.remove_padding_for_metric_eval(
                    self.docid_inputs, self.output)

                for metric in self.exp_settings['metrics']:
                    topn = self.exp_settings['metrics_topn']
                    metric_values = ultra.utils.make_ranking_metric_fn(
                        metric, topn)(self.labels, pad_removed_output, None)
                    for topn, metric_value in zip(topn, metric_values):
                        self.create_summary('%s_%d' % (metric, topn),
                                            '%s_%d' % (metric, topn), metric_value.item(), False)
        return None, self.output, self.eval_summary  # loss, outputs, summary.

    def dcg(self, labels):
        """Computes discounted cumulative gain (DCG).

        DCG =  SUM((2^label -1) / (log(1+rank))).

        Args:
         labels: The relevance `Tensor` of shape [batch_size, list_size]. For the
           ideal ranking, the examples are sorted by relevance in reverse order.
          weights: A `Tensor` of the same shape as labels or [batch_size, 1]. The
            former case is per-example and the latter case is per-list.

        Returns:
          A `Tensor` as the weighted discounted cumulative gain per-list. The
          tensor shape is [batch_size, 1].
        """
        list_size = labels.shape[1]
        position = torch.arange(1, list_size + 1, device=self.device, dtype=torch.float32)
        denominator = torch.log(position + 1)
        numerator = torch.pow(torch.tensor(2.0, device=self.device), labels.to(torch.float32)) - 1.0
        return torch.sum(ultra.utils.metrics._safe_div(numerator, denominator))

    def log_pi(self, score, ranking):
        """log(pi_theta(r | q))"""
        def log_sum_exp(x, dim=0):
            """numerically stable log sum exp"""
            s, _ = torch.max(x, dim=dim, keepdim=True)
            outputs = s + (x - s).exp().sum(dim=dim, keepdim=True).log()
            return outputs

        subtracts = torch.zeros_like(score, device=self.device)
        log_probs = torch.zeros_like(score, device=self.device)
        for i in range(score.shape[1]):
            pos_i = ranking[:, i]
            # l = log_sum_exp(score - subtracts, dim=1)
            # ipdb.set_trace()
            log_probs[:, [i]] = score.gather(dim=1, index=pos_i.unsqueeze(1)) - log_sum_exp(score - subtracts, dim=1)
            subtracts[:, pos_i] = score[:, pos_i] + 1e6
        return torch.sum(log_probs)


if __name__ == '__main__':
    pass
