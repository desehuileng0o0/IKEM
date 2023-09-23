import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class SkeletonCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 8), (23, 8), (24, 12), (25, 12)]
        self.Axis = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 8), (23, 22), (24, 12), (25, 24)]

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels*6, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            self.encoder_q = base_encoder(in_channels=in_channels*6, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels*6, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient

            #for param_q, param_k in zip(self.views_fusion_q.parameters(), self.views_fusion_k.parameters()):
                #param_k.data.copy_(param_q.data)    # initialize
                #param_k.requires_grad = False       # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @torch.no_grad()
    def _momentum_update_key_vf(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.views_fusion_q.parameters(), self.views_fusion_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_qo, im_ko=None, frame=None, view='joint', cross=False, topk=1, context=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if cross:
            return self.cross_training(im_qo, im_ko, topk, context)
        time_scale = 50 / frame
        im_q_motion = torch.zeros_like(im_qo)#返回一个由标量0填充的张量
        im_q_motion[:, :, :-1, :, :] = im_qo[:, :, 1:, :, :] - im_qo[:, :, :-1, :, :]#第二帧到最后一帧减去第一帧到倒数第二帧
        for i, f in enumerate(time_scale):
            im_q_motion[i, :, :, :, :] = f * im_q_motion[i, :, :, :, :]

        im_q_acceleration = torch.zeros_like(im_qo)
        im_q_acceleration[:, :, :-2, :, :] = im_q_motion[:, :, 1:-1, :, :] - im_q_motion[:, :, :-2, :, :]
        for i, f in enumerate(time_scale):
            im_q_acceleration[i, :, :, :, :] = f * im_q_acceleration[i, :, :, :, :]

        im_q_bone = torch.zeros_like(im_qo)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_qo[:, :, :, v1 - 1, :] - im_qo[:, :, :, v2 - 1, :]#前一个关节的坐标减去后一个关节的坐标
        im_q_bone[:, 0:3, :, 20, :] = F.normalize(torch.cross((im_q_bone[:, 0:3, :, 1, :]-im_q_bone[:, 0:3, :, 8, :]), (im_q_bone[:, 0:3, :, 4, :]-im_q_bone[:, 0:3, :, 8, :])))

        im_q_rotation_axis = torch.zeros_like(im_qo)
        for b1, b2 in self.Axis:
            im_q_rotation_axis[:, 0:3, :, b1 - 1, :] = torch.cross(im_q_bone[:, 0:3, :, b1 - 1, :],im_q_bone[:, 0:3, :, b2 - 1, :])
        
        im_q_omega = torch.zeros_like(im_qo)
        theta_q = torch.zeros_like(im_qo)
        norm_q_bone = torch.zeros_like(im_q_bone)
        norm_q_bone = F.normalize(im_q_bone,dim=1)
        cp_q = torch.zeros_like(im_q_rotation_axis)#这是归一化后的Bone的叉积
        norm_q_rotation_axis = torch.zeros_like(im_q_rotation_axis)

        for b1, b2 in self.Axis:
            cp_q[:, 0:3, :, b1 - 1, :] = torch.cross(norm_q_bone[:, 0:3, :, b1 - 1, :],norm_q_bone[:, 0:3, :, b2 - 1, :])
        for b1, b2 in self.Axis:
            theta_q[:, 0, :, b1 - 1, :] = torch.atan2(torch.norm(cp_q[:, 0:3, :, b1 - 1, :],dim=1),torch.einsum('abcd,abcd->acd',norm_q_bone[:, 0:3, :, b1 - 1, :],norm_q_bone[:, 0:3, :, b2 - 1, :]))
        norm_q_rotation_axis = F.normalize(im_q_rotation_axis,dim=1)
                
        for i in range(3):
            im_q_omega[:, i, :-1, :, :] = (theta_q[:, 0, 1:, :, :] - theta_q[:, 0, :-1, :, :]) * norm_q_rotation_axis[:, i, :-1, :, :]

        for i, f in enumerate(time_scale):
            im_q_omega[i, :, :, :, :] = f * im_q_omega[i, :, :, :, :]
        
        im_q = torch.cat((im_qo, im_q_motion, im_q_bone, im_q_acceleration, im_q_rotation_axis, im_q_omega), 1)
        #im_q = im_q.permute(0, 2, 3, 4, 1).contiguous()
        #im_q = self.views_fusion_q(im_q)
        #im_q = im_q.permute(0, 4, 1, 2, 3).contiguous()
        #k和q不太一样，k的viewsfusion必须写在无梯度里
        if not self.pretrain:
            if view == 'all':
                return self.encoder_q(im_q)
            else:
                raise ValueError
        im_k_motion = torch.zeros_like(im_ko)
        im_k_motion[:, :, :-1, :, :] = im_ko[:, :, 1:, :, :] - im_ko[:, :, :-1, :, :]
        for i, f in enumerate(time_scale):
            im_k_motion[i, :, :, :, :] = f * im_k_motion[i, :, :, :, :]

        im_k_acceleration = torch .zeros_like(im_ko)
        im_k_acceleration[:, :, :-2, :, :] = im_k_motion[:, :, 1:-1, :, :] - im_k_motion[:, :, :-2, :, :]
        for i, f in enumerate(time_scale):
            im_k_acceleration[i, :, :, :, :] = f * im_k_acceleration[i, :, :, :, :]

        im_k_bone = torch.zeros_like(im_ko)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_ko[:, :, :, v1 - 1, :] - im_ko[:, :, :, v2 - 1, :]
        im_k_bone[:, 0:3, :, 20, :] = F.normalize(torch.cross((im_k_bone[:, 0:3, :, 1, :]-im_k_bone[:, 0:3, :, 8, :]), (im_k_bone[:, 0:3, :, 4, :]-im_k_bone[:, 0:3, :, 8, :])))

        im_k_rotation_axis = torch.zeros_like(im_ko)
        for b1, b2 in self.Axis:
            im_k_rotation_axis[:, 0:3, :, b1 - 1, :] = torch.cross(im_k_bone[:, 0:3, :, b1 - 1, :],im_k_bone[:, 0:3, :, b2 - 1, :])

        im_k_omega = torch.zeros_like(im_ko)
        theta_k = torch.zeros_like(im_ko)
        norm_k_bone = torch.zeros_like(im_k_bone)
        norm_k_bone = F.normalize(im_k_bone,dim=1)
        cp_k = torch.zeros_like(im_k_rotation_axis)#这是归一化后的Bone的叉积
        norm_k_rotation_axis = torch.zeros_like(im_k_rotation_axis)

        for b1, b2 in self.Axis:
            cp_k[:, 0:3, :, b1 - 1, :] = torch.cross(norm_k_bone[:, 0:3, :, b1 - 1, :],norm_k_bone[:, 0:3, :, b2 - 1, :])
        for b1, b2 in self.Axis:
            theta_k[:, 0, :, b1 - 1, :] = torch.atan2(torch.norm(cp_k[:, 0:3, :, b1 - 1, :],dim=1),torch.einsum('abcd,abcd->acd',norm_k_bone[:, 0:3, :, b1 - 1, :],norm_k_bone[:, 0:3, :, b2 - 1, :]))
        norm_k_rotation_axis = F.normalize(im_k_rotation_axis,dim=1)
                
        for i in range(3):
            im_k_omega[:, i, :-1, :, :] = (theta_k[:, 0, 1:, :, :] - theta_k[:, 0, :-1, :, :]) * norm_k_rotation_axis[:, i, :-1, :, :]

        for i, f in enumerate(time_scale):
            im_k_omega[i, :, :, :, :] = f * im_k_omega[i, :, :, :, :]
        
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            #self._momentum_update_key_vf()

            im_k = torch.cat((im_ko, im_k_motion, im_k_bone, im_k_acceleration, im_k_rotation_axis, im_k_omega), 1)
            #im_k = im_k.permute(0, 2, 3, 4, 1).contiguous()
            #im_k = self.views_fusion_k(im_k)
            #im_k = im_k.permute(0, 4, 1, 2, 3).contiguous()

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)# 128，1
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])# 128， 32768

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
        