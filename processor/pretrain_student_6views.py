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

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_motion_value = []
        loss_bone_value = []

        for [data1, data2], label, frame in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            frame = frame.long().to(self.dev, non_blocking=True)

            output_jsj, output_msj, output_bsj, output_asj, output_rsj, output_osj,\
            output_jsm, output_msm, output_bsm, output_asm, output_rsm, output_osm,\
            output_jsb, output_msb, output_bsb, output_asb, output_rsb, output_osb,\
                mask_jx, mask_mx, mask_bx, mask_ax, mask_rx, mask_ox= self.model(data1, data2, frame, cross=True, topk=self.arg.topk, context=self.arg.context)
            if hasattr(self.model, 'module'):
                self.model.module.update_ptr(output_jsj.size(0))
                self.model.module.update_ptr_teacher(output_jsj.size(0))
            else:
                self.model.update_ptr(output_jsj.size(0))
                self.model.update_ptr_teacher(output_jsj.size(0))
            loss_jsj = - (F.log_softmax(output_jsj, dim=1) * mask_jx).sum(1) / mask_jx.sum(1)
            loss_msj = - (F.log_softmax(output_msj, dim=1) * mask_mx).sum(1) / mask_mx.sum(1)
            loss_bsj = - (F.log_softmax(output_bsj, dim=1) * mask_bx).sum(1) / mask_bx.sum(1)
            loss_asj = - (F.log_softmax(output_asj, dim=1) * mask_ax).sum(1) / mask_ax.sum(1)
            loss_rsj = - (F.log_softmax(output_rsj, dim=1) * mask_rx).sum(1) / mask_rx.sum(1)
            loss_osj = - (F.log_softmax(output_osj, dim=1) * mask_ox).sum(1) / mask_ox.sum(1)

            loss_jsm = - (F.log_softmax(output_jsm, dim=1) * mask_jx).sum(1) / mask_jx.sum(1)
            loss_msm = - (F.log_softmax(output_msm, dim=1) * mask_mx).sum(1) / mask_mx.sum(1)
            loss_bsm = - (F.log_softmax(output_bsm, dim=1) * mask_bx).sum(1) / mask_bx.sum(1)
            loss_asm = - (F.log_softmax(output_asm, dim=1) * mask_ax).sum(1) / mask_ax.sum(1)
            loss_rsm = - (F.log_softmax(output_rsm, dim=1) * mask_rx).sum(1) / mask_rx.sum(1)
            loss_osm = - (F.log_softmax(output_osm, dim=1) * mask_ox).sum(1) / mask_ox.sum(1)

            loss_jsb = - (F.log_softmax(output_jsb, dim=1) * mask_jx).sum(1) / mask_jx.sum(1)
            loss_msb = - (F.log_softmax(output_msb, dim=1) * mask_mx).sum(1) / mask_mx.sum(1)
            loss_bsb = - (F.log_softmax(output_bsb, dim=1) * mask_bx).sum(1) / mask_bx.sum(1)
            loss_asb = - (F.log_softmax(output_asb, dim=1) * mask_ax).sum(1) / mask_ax.sum(1)
            loss_rsb = - (F.log_softmax(output_rsb, dim=1) * mask_rx).sum(1) / mask_rx.sum(1)
            loss_osb = - (F.log_softmax(output_osb, dim=1) * mask_ox).sum(1) / mask_ox.sum(1)

            loss = (loss_jsj + loss_msj + loss_bsj + loss_asj + loss_rsj + loss_osj) / 6.
            loss_motion = (loss_jsm + loss_msm + loss_bsm + loss_asm + loss_rsm + loss_osm) / 6.
            loss_bone = (loss_jsb + loss_msb + loss_bsb + loss_asb + loss_rsb + loss_osb) / 6.
            loss = loss.mean()
            loss_motion = loss_motion.mean()
            loss_bone = loss_bone.mean()

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_motion'] = loss_motion.data.item()
            self.iter_info['loss_bone'] = loss_bone.data.item()
            loss_value.append(self.iter_info['loss'])
            loss_motion_value.append(self.iter_info['loss_motion'])
            loss_bone_value.append(self.iter_info['loss_bone'])
            loss = loss + loss_motion + loss_bone

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
