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

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
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
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_q, im_k=None, frame=None, view='joint', cross=False, topk=1, context=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if cross:
            return self.cross_training(im_q, im_k, topk, context)

        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q)
            #multi-view input of skeletonclr 
            elif view == 'acceleration':
                time_scale = 50 / frame
                im_q_motion = torch.zeros_like(im_q)#返回一个由标量0填充的张量
                im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]#第二帧到最后一帧减去第一帧到倒数第二帧
                for i, f in enumerate(time_scale):
                    im_q_motion[i, :, :, :, :] = f * im_q_motion[i, :, :, :, :]
                im_q_acceleration = torch .zeros_like(im_q)
                im_q_acceleration[:, :, :-2, :, :] = im_q_motion[:, :, 1:-1, :, :] - im_q_motion[:, :, :-2, :, :]
                for i, f in enumerate(time_scale):
                    im_q_acceleration[i, :, :, :, :] = f * im_q_acceleration[i, :, :, :, :]
                
                return self.encoder_q(im_q_acceleration)
            elif view == 'joint+motion':
                im_q_motion = torch.zeros_like(im_q)#返回一个由标量0填充的张量
                im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]#第二帧到最后一帧减去第一帧到倒数第二帧
                return (self.encoder_q(im_q) + self.encoder_q(im_q_motion)) / 2.
            elif view == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 8), (23, 8), (24, 12), (25, 12)]
                im_q_bone = torch.zeros_like(im_q)
                for v1, v2 in Bone:
                    im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]
                return self.encoder_q(im_q_bone)
            elif view == 'rotation_axis':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 8), (23, 8), (24, 12), (25, 12)]
                self.Axis = [(1, 2), (2, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6), (8, 7), (9, 2),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 8), (23, 22), (24, 12), (25, 24)]
                im_q_bone = torch.zeros_like(im_q)
                for v1, v2 in Bone:
                    im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]
                im_q_rotation_axis = torch.zeros_like(im_q_bone)
                for b1, b2 in self.Axis:
                    im_q_rotation_axis[:, 0:3, :, b1 - 1, :] = torch.cross(im_q_bone[:, 0:3, :, b1 - 1, :],im_q_bone[:, 0:3, :, b2 - 1, :])
                #im_q_rotation_axis = F.normalize(im_q_rotation_axis, dim=1)
                return self.encoder_q(im_q_rotation_axis)

            else:
                raise ValueError
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

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
        #print(q.size())
        #print(k.size())
        #print(l_pos.size())
        #print(l_neg.size())
        #print(logits.size())
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
        