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


class CrosSCLR_ViewsFusion_Processor(PT_Processor):
    """
        Processor for 3view-CrosSCLR Pretraining.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_motion_value = []
        loss_bone_value = []
        loss_acceleration_value = []
        loss_viewx_value = []

        for [data1, data2], label, frame in loader:#crop里面有random，后面再看，一个输入k一个输入q
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            frame = frame.long().to(self.dev, non_blocking=True)

            # forward
            if epoch <= self.arg.cross_epoch:#没到cross的时候
                output, output_motion, output_bone, output_acceleration, output_viewx, target = self.model(data1, data2, frame)#一个是q的输入，一个是k的输入
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                loss = self.loss(output, target)
                loss_motion = self.loss(output_motion, target)
                loss_bone = self.loss(output_bone, target)
                loss_acceleration = self.loss(output_acceleration, target)
                loss_viewx = self.loss(output_viewx, target)

                self.iter_info['loss'] = loss.data.item()
                self.iter_info['loss_motion'] = loss_motion.data.item()
                self.iter_info['loss_bone'] = loss_bone.data.item()
                self.iter_info['loss_acceleration'] = loss_acceleration.data.item()
                self.iter_info['loss_viewx'] = loss_viewx.data.item()
                loss_value.append(self.iter_info['loss'])
                loss_motion_value.append(self.iter_info['loss_motion'])
                loss_bone_value.append(self.iter_info['loss_bone'])
                loss_acceleration_value.append(self.iter_info['loss_acceleration'])
                loss_viewx_value.append(self.iter_info['loss_viewx'])
                loss = loss + loss_motion + loss_bone + loss_acceleration + loss_viewx
            else:
                output_jm, output_jb, output_mj, output_mb, output_bj, output_bm,\
                output_ja, output_aj, output_ma, output_am, output_ba, output_ab,\
                output_jx, output_mx, output_bx, output_ax,\
                mask_jm, mask_jb, mask_mj, mask_mb, mask_bj, mask_bm,\
                mask_ja, mask_aj, mask_ma, mask_am, mask_ba, mask_ab,\
                mask_jx, mask_mx, mask_bx, mask_ax\
                = self.model(data1, data2, frame, cross=True, topk=self.arg.topk, context=self.arg.context)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output_jm.size(0))
                else:
                    self.model.update_ptr(output_jm.size(0))
                loss_jm = - (F.log_softmax(output_jm, dim=1) * mask_jm).sum(1) / mask_jm.sum(1)
                loss_jb = - (F.log_softmax(output_jb, dim=1) * mask_jb).sum(1) / mask_jb.sum(1)
                loss_mj = - (F.log_softmax(output_mj, dim=1) * mask_mj).sum(1) / mask_mj.sum(1)
                loss_mb = - (F.log_softmax(output_mb, dim=1) * mask_mb).sum(1) / mask_mb.sum(1)
                loss_bj = - (F.log_softmax(output_bj, dim=1) * mask_bj).sum(1) / mask_bj.sum(1)
                loss_bm = - (F.log_softmax(output_bm, dim=1) * mask_bm).sum(1) / mask_bm.sum(1)
                loss_ja = - (F.log_softmax(output_ja, dim=1) * mask_ja).sum(1) / mask_ja.sum(1)
                loss_aj = - (F.log_softmax(output_aj, dim=1) * mask_aj).sum(1) / mask_aj.sum(1)
                loss_ma = - (F.log_softmax(output_ma, dim=1) * mask_ma).sum(1) / mask_ma.sum(1)
                loss_am = - (F.log_softmax(output_am, dim=1) * mask_am).sum(1) / mask_am.sum(1)
                loss_ba = - (F.log_softmax(output_ba, dim=1) * mask_ba).sum(1) / mask_ba.sum(1)
                loss_ab = - (F.log_softmax(output_ab, dim=1) * mask_ab).sum(1) / mask_ab.sum(1)

                loss_jx = - (F.log_softmax(output_jx, dim=1) * mask_jx).sum(1) / mask_jx.sum(1)
                loss_mx = - (F.log_softmax(output_mx, dim=1) * mask_mx).sum(1) / mask_mx.sum(1)
                loss_bx = - (F.log_softmax(output_bx, dim=1) * mask_bx).sum(1) / mask_bx.sum(1)
                loss_ax = - (F.log_softmax(output_ax, dim=1) * mask_ax).sum(1) / mask_ax.sum(1)

                loss = (loss_jm + loss_jb + loss_ja) / 3.
                loss_motion = (loss_mj + loss_mb + loss_ma) / 3.
                loss_bone = (loss_bj + loss_bm + loss_ba) / 3.
                loss_acceleration = (loss_ab + loss_aj + loss_am) / 3
                loss_viewx = (loss_jx + loss_mx + loss_bx + loss_ax) / 4
                loss = loss.mean()
                loss_motion = loss_motion.mean()
                loss_bone = loss_bone.mean()
                loss_acceleration = loss_acceleration.mean()
                loss_viewx = loss_viewx.mean()

                self.iter_info['loss'] = loss.data.item()
                self.iter_info['loss_motion'] = loss_motion.data.item()
                self.iter_info['loss_bone'] = loss_bone.data.item()
                self.iter_info['loss_acceleration'] = loss_acceleration.data.item()
                self.iter_info['loss_viewx'] = loss_viewx.data.item()
                loss_value.append(self.iter_info['loss'])
                loss_motion_value.append(self.iter_info['loss_motion'])
                loss_bone_value.append(self.iter_info['loss_bone'])
                loss_acceleration_value.append(self.iter_info['loss_acceleration'])
                loss_viewx_value.append(self.iter_info['loss_viewx'])
                loss = loss + loss_motion + loss_bone + loss_acceleration + loss_viewx

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)
            self.train_writer.add_scalar('batch_loss_motion', self.iter_info['loss_motion'], self.global_step)
            self.train_writer.add_scalar('batch_loss_bone', self.iter_info['loss_bone'], self.global_step)
            self.train_writer.add_scalar('batch_loss_acceleration', self.iter_info['loss_acceleration'], self.global_step)
            self.train_writer.add_scalar('batch_loss_viewx', self.iter_info['loss_viewx'], self.global_step)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.epoch_info['train_mean_loss_motion']= np.mean(loss_motion_value)
        self.epoch_info['train_mean_loss_bone']= np.mean(loss_bone_value)
        self.epoch_info['train_mean_loss_acceleration'] = np.mean(loss_acceleration_value)
        self.epoch_info['train_mean_loss_viewx'] = np.mean(loss_viewx_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.train_writer.add_scalar('loss_motion', self.epoch_info['train_mean_loss_motion'], epoch)
        self.train_writer.add_scalar('loss_bone', self.epoch_info['train_mean_loss_bone'], epoch)
        self.train_writer.add_scalar('loss_acceleration', self.epoch_info['train_mean_loss_acceleration'], epoch)
        self.train_writer.add_scalar('loss_viewx', self.epoch_info['train_mean_loss_viewx'], epoch)
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
