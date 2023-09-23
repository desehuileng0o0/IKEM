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


class SkeletonCLR_Processor(PT_Processor):
    """
        Processor for SkeletonCLR Pretraining.
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
            time_scale = 50 / frame

            if self.arg.view == 'joint':
                pass
            elif self.arg.view == 'motion':
                motion1 = torch.zeros_like(data1)
                motion2 = torch.zeros_like(data2)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]

                for i, f in enumerate(time_scale):
                    motion1[i, :, :, :, :] = f * motion1[i, :, :, :, :]
                    motion2[i, :, :, :, :] = f * motion2[i, :, :, :, :]
                data1 = motion1
                data2 = motion2
            elif self.arg.view == 'acceleration':
                motion1 = torch.zeros_like(data1)
                motion2 = torch.zeros_like(data2)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]
                
                for i, f in enumerate(time_scale):
                    motion1[i, :, :, :, :] = f * motion1[i, :, :, :, :]
                    motion2[i, :, :, :, :] = f * motion2[i, :, :, :, :]
                #print(motion1[:, :, 48, :, :])
                #print(motion1[:, :, 49, :, :])
                acceleration1 = torch .zeros_like(data1)
                acceleration1[:, :, :-2, :, :] = motion1[:, :, 1:-1, :, :] - motion1[:, :, :-2, :, :]
                
                #print(acceleration1[:, :, 49, :, :])
                #print(acceleration1[:, :, 48, :, :])
                #print(acceleration1[:, :, 47, :, :])
                acceleration2 = torch .zeros_like(data2)
                acceleration2[:, :, :-2, :, :] = motion2[:, :, 1:-1, :, :] - motion2[:, :, :-2, :, :]
                
                for i, f in enumerate(time_scale):
                    acceleration1[i, :, :, :, :] = f * acceleration1[i, :, :, :, :]
                    acceleration2[i, :, :, :, :] = f * acceleration2[i, :, :, :, :]

                data1 = acceleration1
                data2 = acceleration2
            elif self.arg.view == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 8), (23, 8), (24, 12), (25, 12)]
                
                bone1 = torch.zeros_like(data1)
                bone2 = torch.zeros_like(data2)

                for v1, v2 in Bone:
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]
                
                data1 = bone1
                data2 = bone2
            elif self.arg.view == 'rotation_axis':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 8), (23, 8), (24, 12), (25, 12)]
                Axis = [(1, 2), (2, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6), (8, 7), (9, 2),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 8), (23, 22), (24, 12), (25, 24)]
                
                bone1 = torch.zeros_like(data1)
                bone2 = torch.zeros_like(data2)

                for v1, v2 in Bone:
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]

                ra1 = torch.zeros_like(data1)
                ra2 = torch.zeros_like(data2)

                for b1, b2 in Axis:
                    ra1[:, 0:3, :, b1 - 1, :] = torch.cross(bone1[:, 0:3, :, b1 - 1, :],bone1[:, 0:3, :, b2 - 1, :])
                    ra2[:, 0:3, :, b1 - 1, :] = torch.cross(bone2[:, 0:3, :, b1 - 1, :],bone2[:, 0:3, :, b2 - 1, :])
                
                #data1 = F.normalize(ra1,dim=1)
                #data2 = F.normalize(ra2,dim=1)
                data1 = ra1
                data2 = ra2
            else:
                raise ValueError

            # forward
            output, target = self.model(data1, data2)
            if hasattr(self.model, 'module'):
                self.model.module.update_ptr(output.size(0))
            else:
                self.model.update_ptr(output.size(0))
            loss = self.loss(output, target)#这里直接用的交叉熵

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
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
        # endregion yapf: enable

        return parser
