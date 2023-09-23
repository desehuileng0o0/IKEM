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


class CrosSCLR_6views_Processor(PT_Processor):
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
        loss_rotation_axis_value = []
        loss_omega_value = []

        for [data1, data2], label, frame in loader:#crop里面有random，后面再看，一个输入k一个输入q
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            frame = frame.long().to(self.dev, non_blocking=True)

            # forward
            if epoch <= self.arg.cross_epoch:#没到cross的时候
                output, output_motion, output_bone, output_acceleration, output_rotation_axis, output_omega, target = self.model(data1, data2, frame)#一个是q的输入，一个是k的输入
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                loss = self.loss(output, target)
                loss_motion = self.loss(output_motion, target)
                loss_bone = self.loss(output_bone, target)
                loss_acceleration = self.loss(output_acceleration, target)
                loss_rotation_axis = self.loss(output_rotation_axis, target)
                loss_omega = self.loss(output_omega, target)

                self.iter_info['loss'] = loss.data.item()
                self.iter_info['loss_motion'] = loss_motion.data.item()
                self.iter_info['loss_bone'] = loss_bone.data.item()
                self.iter_info['loss_acceleration'] = loss_acceleration.data.item()
                self.iter_info['loss_rotation_axis'] = loss_rotation_axis.data.item()
                self.iter_info['loss_omega'] = loss_omega.data.item()
                loss_value.append(self.iter_info['loss'])
                loss_motion_value.append(self.iter_info['loss_motion'])
                loss_bone_value.append(self.iter_info['loss_bone'])
                loss_acceleration_value.append(self.iter_info['loss_acceleration'])
                loss_rotation_axis_value.append(self.iter_info['loss_rotation_axis'])
                loss_omega_value.append(self.iter_info['loss_omega'])
                loss = loss + loss_motion + loss_bone + loss_acceleration + loss_rotation_axis +loss_omega
            else:
                output_jm, output_jb, output_mj, output_mb, output_bj, output_bm, output_ja, output_aj, output_ma, output_am, output_ba, output_ab,\
                    output_jr, output_rj, output_mr, output_rm, output_br, output_rb, output_ar, output_ra,\
                        output_jo, output_oj, output_mo, output_om, output_bo, output_ob, output_ao, output_oa, output_ro, output_or, \
                            mask_j, mask_m, mask_b, mask_a, mask_r, mask_o,\
                                output_j, output_m, output_b, output_a, output_r, output_o, targetc = self.model(data1, data2, frame, cross=True, topk=self.arg.topk, context=self.arg.context)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output_jm.size(0))
                else:
                    self.model.update_ptr(output_jm.size(0))
                
                loss_jm = - (F.log_softmax(output_jm, dim=1) * mask_m).sum(1) / mask_m.sum(1)
                loss_jb = - (F.log_softmax(output_jb, dim=1) * mask_b).sum(1) / mask_b.sum(1)
                loss_mj = - (F.log_softmax(output_mj, dim=1) * mask_j).sum(1) / mask_j.sum(1)
                loss_mb = - (F.log_softmax(output_mb, dim=1) * mask_b).sum(1) / mask_b.sum(1)
                loss_bj = - (F.log_softmax(output_bj, dim=1) * mask_j).sum(1) / mask_j.sum(1)
                loss_bm = - (F.log_softmax(output_bm, dim=1) * mask_m).sum(1) / mask_m.sum(1)
                loss_ja = - (F.log_softmax(output_ja, dim=1) * mask_a).sum(1) / mask_a.sum(1)
                loss_aj = - (F.log_softmax(output_aj, dim=1) * mask_j).sum(1) / mask_j.sum(1)
                loss_ma = - (F.log_softmax(output_ma, dim=1) * mask_a).sum(1) / mask_a.sum(1)
                loss_am = - (F.log_softmax(output_am, dim=1) * mask_m).sum(1) / mask_m.sum(1)
                loss_ba = - (F.log_softmax(output_ba, dim=1) * mask_a).sum(1) / mask_a.sum(1)
                loss_ab = - (F.log_softmax(output_ab, dim=1) * mask_b).sum(1) / mask_b.sum(1)

                loss_jr = - (F.log_softmax(output_jr, dim=1) * mask_r).sum(1) / mask_r.sum(1)
                loss_rj = - (F.log_softmax(output_rj, dim=1) * mask_j).sum(1) / mask_j.sum(1)
                loss_mr = - (F.log_softmax(output_mr, dim=1) * mask_r).sum(1) / mask_r.sum(1)
                loss_rm = - (F.log_softmax(output_rm, dim=1) * mask_m).sum(1) / mask_m.sum(1)
                loss_br = - (F.log_softmax(output_br, dim=1) * mask_r).sum(1) / mask_r.sum(1)
                loss_rb = - (F.log_softmax(output_rb, dim=1) * mask_b).sum(1) / mask_b.sum(1)
                loss_ar = - (F.log_softmax(output_ar, dim=1) * mask_r).sum(1) / mask_r.sum(1)
                loss_ra = - (F.log_softmax(output_ra, dim=1) * mask_a).sum(1) / mask_a.sum(1)

                loss_jo = - (F.log_softmax(output_jo, dim=1) * mask_o).sum(1) / mask_o.sum(1)
                loss_oj = - (F.log_softmax(output_oj, dim=1) * mask_j).sum(1) / mask_j.sum(1)
                loss_mo = - (F.log_softmax(output_mo, dim=1) * mask_o).sum(1) / mask_o.sum(1)
                loss_om = - (F.log_softmax(output_om, dim=1) * mask_m).sum(1) / mask_m.sum(1)
                loss_bo = - (F.log_softmax(output_bo, dim=1) * mask_o).sum(1) / mask_o.sum(1)
                loss_ob = - (F.log_softmax(output_ob, dim=1) * mask_b).sum(1) / mask_b.sum(1)
                loss_ao = - (F.log_softmax(output_ao, dim=1) * mask_o).sum(1) / mask_o.sum(1)
                loss_oa = - (F.log_softmax(output_oa, dim=1) * mask_a).sum(1) / mask_a.sum(1)
                loss_ro = - (F.log_softmax(output_ro, dim=1) * mask_o).sum(1) / mask_o.sum(1)
                loss_or = - (F.log_softmax(output_or, dim=1) * mask_r).sum(1) / mask_r.sum(1)

                loss = (loss_jm + loss_jb + loss_ja + loss_jr + loss_jo) / 5.
                loss_motion = (loss_mj + loss_mb + loss_ma + loss_mr + loss_mo) / 5.
                loss_bone = (loss_bj + loss_bm + loss_ba + loss_br + loss_bo) / 5.
                loss_acceleration = (loss_ab + loss_aj + loss_am + loss_ar + loss_ao) / 5.
                loss_rotation_axis = (loss_rj + loss_rm + loss_rb + loss_ra + loss_ro) / 5.
                loss_omega = (loss_oj + loss_om + loss_ob + loss_oa + loss_or) / 5.
                loss = loss.mean()# + self.loss(output_j, targetc)
                loss_motion = loss_motion.mean()# + self.loss(output_m, targetc)
                loss_bone = loss_bone.mean()# + self.loss(output_b, targetc)
                loss_acceleration = loss_acceleration.mean()# + self.loss(output_a, targetc)
                loss_rotation_axis = loss_rotation_axis.mean()# + self.loss(output_r, targetc)
                loss_omega = loss_omega.mean()# + self.loss(output_o, targetc)
                
                self.iter_info['loss'] = loss.data.item()
                self.iter_info['loss_motion'] = loss_motion.data.item()
                self.iter_info['loss_bone'] = loss_bone.data.item()
                self.iter_info['loss_acceleration'] = loss_acceleration.data.item()
                self.iter_info['loss_rotation_axis'] = loss_rotation_axis.data.item()
                self.iter_info['loss_omega'] = loss_omega.data.item()
                loss_value.append(self.iter_info['loss'])
                loss_motion_value.append(self.iter_info['loss_motion'])
                loss_bone_value.append(self.iter_info['loss_bone'])
                loss_acceleration_value.append(self.iter_info['loss_acceleration'])
                loss_rotation_axis_value.append(self.iter_info['loss_rotation_axis'])
                loss_omega_value.append(self.iter_info['loss_omega'])
                loss = loss + loss_motion + loss_bone + loss_acceleration + loss_rotation_axis + loss_omega

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
            self.train_writer.add_scalar('batch_loss_rotation_axis', self.iter_info['loss_rotation_axis'], self.global_step)
            self.train_writer.add_scalar('batch_loss_omega', self.iter_info['loss_omega'], self.global_step)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.epoch_info['train_mean_loss_motion']= np.mean(loss_motion_value)
        self.epoch_info['train_mean_loss_bone']= np.mean(loss_bone_value)
        self.epoch_info['train_mean_loss_acceleration'] = np.mean(loss_acceleration_value)
        self.epoch_info['train_mean_loss_rotation_axis'] = np.mean(loss_rotation_axis_value)
        self.epoch_info['train_mean_loss_omega'] = np.mean(loss_omega_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.train_writer.add_scalar('loss_motion', self.epoch_info['train_mean_loss_motion'], epoch)
        self.train_writer.add_scalar('loss_bone', self.epoch_info['train_mean_loss_bone'], epoch)
        self.train_writer.add_scalar('loss_acceleration', self.epoch_info['train_mean_loss_acceleration'], epoch)
        self.train_writer.add_scalar('loss_rotation_axis', self.epoch_info['train_mean_loss_rotation_axis'], epoch)
        self.train_writer.add_scalar('loss_omega', self.epoch_info['train_mean_loss_omega'], epoch)
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
