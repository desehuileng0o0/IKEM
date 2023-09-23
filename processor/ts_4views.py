#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor


class TS_Processor(PT_Processor):
    """
        Processor for 3view-CrosSCLR Pretraining.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for [data1, data2], label, frame in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            frame = frame.long().to(self.dev, non_blocking=True)

            # forward这边就没有是不是cross的判断了
            output_jx, output_mx, output_bx, output_ax, mask_jx, mask_mx, mask_bx, mask_ax = self.model(data1, data2, frame, cross=True, topk=self.arg.topk, context=self.arg.context)
            if hasattr(self.model, 'module'):
                self.model.module.update_ptr(output_jx.size(0))
                self.model.module.update_ptr_teacher(output_jx.size(0))
            else:
                self.model.update_ptr(output_jx.size(0))
                self.model.update_ptr_teacher(output_jx.size(0))
            loss_jx = - (F.log_softmax(output_jx, dim=1) * mask_jx).sum(1) / mask_jx.sum(1)
            loss_mx = - (F.log_softmax(output_mx, dim=1) * mask_mx).sum(1) / mask_mx.sum(1)
            loss_bx = - (F.log_softmax(output_bx, dim=1) * mask_bx).sum(1) / mask_bx.sum(1)
            loss_ax = - (F.log_softmax(output_ax, dim=1) * mask_ax).sum(1) / mask_ax.sum(1)

            loss = (loss_jx + loss_mx + loss_bx + loss_ax) / 4
            loss = loss.mean()

            self.iter_info['loss'] = loss.data.item()
            loss_value.append(self.iter_info['loss'])

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)

        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--cross_epoch', type=int, default=1e6, help='the starting epoch of cross-view training')
        parser.add_argument('--context', type=str2bool, default=True, help='using context knowledge')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in cross-view training')
        # endregion yapf: enable

        return parser
