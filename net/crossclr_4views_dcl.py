import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class CrosSCLR(nn.Module):
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

        super().__init__()#继承父类构造函数中的内容
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=num_class,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_acceleration = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            #self.late_fusion = nn.Sequential(nn.Linear(4*num_class, num_class),
                                            #nn.ReLU())
            #self.late_fusion = nn.Linear(4*num_class, num_class)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            #为每个view的q和k都创建baseencoder。
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=hidden_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=hidden_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=hidden_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_k_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=hidden_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=hidden_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=hidden_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_acceleration = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=hidden_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_acceleration = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=hidden_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.q_batchdim = nn.Sequential(
                                            nn.ReLU(), 
                                            nn.Linear(256, 128))
            self.k_batchdim = nn.Sequential(
                                            nn.ReLU(), 
                                            nn.Linear(256, 128))
            self.featuredim1 = nn.Sequential(
                                        nn.ReLU(), 
                                        nn.Linear(256, 128))
            self.featuredim2 = nn.Sequential(
                                        nn.ReLU(), 
                                        nn.Linear(256, 128))
            self.q_motion_batchdim = nn.Sequential(
                                            nn.ReLU(), 
                                            nn.Linear(256, 128))
            self.k_motion_batchdim = nn.Sequential(
                                            nn.ReLU(), 
                                            nn.Linear(256, 128))
            self.motion_featuredim1 = nn.Sequential(
                                        nn.ReLU(), 
                                        nn.Linear(256, 128))
            self.motion_featuredim2 = nn.Sequential(
                                        nn.ReLU(), 
                                        nn.Linear(256, 128))
            self.q_bone_batchdim = nn.Sequential(
                                            nn.ReLU(), 
                                            nn.Linear(256, 128))
            self.k_bone_batchdim = nn.Sequential(
                                            nn.ReLU(), 
                                            nn.Linear(256, 128))
            self.bone_featuredim1 = nn.Sequential(
                                        nn.ReLU(), 
                                        nn.Linear(256, 128))
            self.bone_featuredim2 = nn.Sequential(
                                        nn.ReLU(), 
                                        nn.Linear(256, 128))
            self.q_acceleration_batchdim = nn.Sequential(
                                            nn.ReLU(), 
                                            nn.Linear(256, 128))
            self.k_acceleration_batchdim = nn.Sequential(
                                            nn.ReLU(), 
                                            nn.Linear(256, 128))
            self.acceleration_featuredim1 = nn.Sequential(
                                        nn.ReLU(), 
                                        nn.Linear(256, 128))
            self.acceleration_featuredim2 = nn.Sequential(
                                        nn.ReLU(), 
                                        nn.Linear(256, 128))

            if mlp:  # hack: brute-force replacement在每个全连接前又加了一层线性层
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)
                self.encoder_q_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_q_motion.fc)
                self.encoder_k_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_k_motion.fc)
                self.encoder_q_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_bone.fc)
                self.encoder_k_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_k_bone.fc)
                self.encoder_q_acceleration.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_acceleration.fc)
                self.encoder_k_acceleration.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_k_acceleration.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):#将可迭代的对象打包成一个个元组
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient
            for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False#初始化key的参数并且关闭梯度
            for param_q, param_k in zip(self.encoder_q_acceleration.parameters(), self.encoder_k_acceleration.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            for param_q, param_k in zip(self.q_batchdim.parameters(), self.k_batchdim.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.q_motion_batchdim.parameters(), self.k_motion_batchdim.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.q_bone_batchdim.parameters(), self.k_bone_batchdim.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.q_acceleration_batchdim.parameters(), self.k_acceleration_batchdim.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, self.K))#模型训练时不会更新，quene里现在存的都是随机数
            self.queue = F.normalize(self.queue, dim=0)#按列，也就是每个feature都除以这个位置的范数
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))#这个暂时不知道

            self.register_buffer("queue_motion", torch.randn(feature_dim, self.K))
            self.queue_motion = F.normalize(self.queue_motion, dim=0)
            self.register_buffer("queue_ptr_motion", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_bone", torch.randn(feature_dim, self.K))
            self.queue_bone = F.normalize(self.queue_bone, dim=0)
            self.register_buffer("queue_ptr_bone", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_acceleration", torch.randn(feature_dim, self.K))
            self.queue_acceleration = F.normalize(self.queue_acceleration, dim=0)
            self.register_buffer("queue_ptr_acceleration", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()#修饰器，以下操作不会进行梯度回传
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_motion(self):
        for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_bone(self):
        for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_acceleration(self):
        for param_q, param_k in zip(self.encoder_q_acceleration.parameters(), self.encoder_k_acceleration.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_batchdim(self):
        for param_q, param_k in zip(self.q_batchdim.parameters(), self.k_batchdim.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @torch.no_grad()
    def _momentum_update_key_motion_batchdim(self):
        for param_q, param_k in zip(self.q_motion_batchdim.parameters(), self.k_motion_batchdim.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @torch.no_grad()
    def _momentum_update_key_bone_batchdim(self):
        for param_q, param_k in zip(self.q_bone_batchdim.parameters(), self.k_bone_batchdim.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @torch.no_grad()
    def _momentum_update_key_acceleration_batchdim(self):
        for param_q, param_k in zip(self.q_acceleration_batchdim.parameters(), self.k_acceleration_batchdim.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_motion)
        gpu_index = keys.device.index
        self.queue_motion[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_bone(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_bone)
        gpu_index = keys.device.index
        self.queue_bone[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_acceleration(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_acceleration)
        gpu_index = keys.device.index
        self.queue_acceleration[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
        self.queue_ptr_motion[0] = (self.queue_ptr_motion[0] + batch_size) % self.K
        self.queue_ptr_bone[0] = (self.queue_ptr_bone[0] + batch_size) % self.K
        self.queue_ptr_acceleration[0] = (self.queue_ptr_acceleration[0] + batch_size) % self.K

    def forward(self, im_q, im_k=None, frame=None, view='all', cross=False, topk=1, context=True): 
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if cross:
            return self.cross_training(im_q, im_k, frame, topk, context)

        time_scale = 50 / frame
        im_q_motion = torch.zeros_like(im_q)#返回一个由标量0填充的张量
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]#第二帧到最后一帧减去第一帧到倒数第二帧
        for i, f in enumerate(time_scale):
            im_q_motion[i, :, :, :, :] = f * im_q_motion[i, :, :, :, :]

        im_q_acceleration = torch.zeros_like(im_q)
        im_q_acceleration[:, :, :-2, :, :] = im_q_motion[:, :, 1:-1, :, :] - im_q_motion[:, :, :-2, :, :]
        for i, f in enumerate(time_scale):
            im_q_acceleration[i, :, :, :, :] = f * im_q_acceleration[i, :, :, :, :]

        im_q_bone = torch.zeros_like(im_q)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]#前一个关节的坐标减去后一个关节的坐标

        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q)
            elif view == 'motion':
                return self.encoder_q_motion(im_q_motion)
            elif view == 'bone':
                return self.encoder_q_bone(im_q_bone)
            elif view == 'acceleration':
                return self.encoder_q_acceleration(im_q_acceleration)
            elif view == 'all':
                #a = self.encoder_q(im_q)
                #b = self.encoder_q_motion(im_q_motion)
                #c = self.encoder_q_bone(im_q_bone)
                #d = self.encoder_q_acceleration(im_q_acceleration)
                #series = torch.cat((a,b,c,d),dim=1)
                #fusion = self.late_fusion(series)
                #return fusion
                return (self.encoder_q(im_q) + self.encoder_q_motion(im_q_motion) + self.encoder_q_bone(im_q_bone) + self.encoder_q_acceleration(im_q_acceleration)) / 4.
            else:
                raise ValueError

        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]
        for i, f in enumerate(time_scale):
            im_k_motion[i, :, :, :, :] = f * im_k_motion[i, :, :, :, :]

        im_k_acceleration = torch .zeros_like(im_k)
        im_k_acceleration[:, :, :-2, :, :] = im_k_motion[:, :, 1:-1, :, :] - im_k_motion[:, :, :-2, :, :]
        for i, f in enumerate(time_scale):
            im_k_acceleration[i, :, :, :, :] = f * im_k_acceleration[i, :, :, :, :]

        im_k_bone = torch.zeros_like(im_k)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # compute query features
        q_j = self.encoder_q(im_q)  # queries: NxC
        q_j_batchdim = self.q_batchdim(q_j)
        q_j_batchdim = F.normalize(q_j_batchdim, dim=1)
        q_j_featuredim = self.featuredim1(q_j)
        q_j_normalized_featuredim = F.normalize(q_j_featuredim, dim=0)

        q_m = self.encoder_q_motion(im_q_motion)
        q_m_batchdim = self.q_motion_batchdim(q_m)
        q_m_batchdim = F.normalize(q_m_batchdim, dim=1)
        q_m_featuredim = self.motion_featuredim1(q_m)
        q_m_normalized_featuredim = F.normalize(q_m_featuredim, dim=0)

        q_b = self.encoder_q_bone(im_q_bone)
        q_b_batchdim = self.q_bone_batchdim(q_b)
        q_b_batchdim = F.normalize(q_b_batchdim, dim=1)
        q_b_featuredim = self.bone_featuredim1(q_b)
        q_b_normalized_featuredim = F.normalize(q_b_featuredim, dim=0)

        q_a = self.encoder_q_acceleration(im_q_acceleration)
        q_a_batchdim = self.q_acceleration_batchdim(q_a)
        q_a_batchdim = F.normalize(q_a_batchdim, dim=1)
        q_a_featuredim = self.acceleration_featuredim1(q_a)
        q_a_normalized_featuredim = F.normalize(q_a_featuredim, dim=0)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()
            self._momentum_update_key_encoder_acceleration()

            k_j = self.encoder_k(im_k)  # keys: NxC
            k_j_batchdim = self.k_batchdim(k_j)
            k_j_batchdim = F.normalize(k_j_batchdim, dim=1)
            kq_j = self.encoder_q(im_k)
            k_j_featuredim = self.featuredim2(kq_j)
            k_j_normalized_featuredim = F.normalize(k_j_featuredim, dim=0)

            k_m = self.encoder_k_motion(im_k_motion)
            k_m_batchdim = self.k_motion_batchdim(k_m)
            k_m_batchdim = F.normalize(k_m_batchdim,dim=1)
            kq_m = self.encoder_q_motion(im_k_motion)
            k_m_featuredim = self.motion_featuredim2(kq_m)
            k_m_normalized_featuredim = F.normalize(k_m_featuredim, dim=0)

            k_b = self.encoder_k_bone(im_k_bone)
            k_b_batchdim = self.k_bone_batchdim(k_b)
            k_b_batchdim = F.normalize(k_b_batchdim, dim=1)
            kq_b = self.encoder_q_bone(im_k_bone)
            k_b_featuredim = self.bone_featuredim2(kq_b)
            k_b_normalized_featuredim = F.normalize(k_b_featuredim, dim=0)

            k_a = self.encoder_k_acceleration(im_k_acceleration)
            k_a_batchdim = self.k_acceleration_batchdim(k_a)
            k_a_batchdim = F.normalize(k_a_batchdim, dim=1)
            kq_a = self.encoder_q_acceleration(im_k_acceleration)
            k_a_featuredim = self.acceleration_featuredim2(kq_a)
            k_a_normalized_featuredim = F.normalize(k_a_featuredim, dim=0)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_j_batchdim, k_j_batchdim]).unsqueeze(-1)#在dim=0的地方加上一个维度，这个一看就是n组向量的点积，因为一次进去是一个batch
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q_j_batchdim, self.queue.clone().detach()])#这个一看就是矩阵相乘

        l_pos_motion = torch.einsum('nc,nc->n', [q_m_batchdim, k_m_batchdim]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_m_batchdim, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_b_batchdim, k_b_batchdim]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_b_batchdim, self.queue_bone.clone().detach()])

        l_pos_acceleration = torch.einsum('nc,nc->n', [q_a_batchdim, k_a_batchdim]).unsqueeze(-1)
        l_neg_acceleration = torch.einsum('nc,ck->nk', [q_a_batchdim, self.queue_acceleration.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)#按维度1将两个张量拼接
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)
        logits_acceleration = torch.cat([l_pos_acceleration, l_neg_acceleration], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T
        logits_acceleration /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_j_batchdim)
        self._dequeue_and_enqueue_motion(k_m_batchdim)
        self._dequeue_and_enqueue_bone(k_b_batchdim)
        self._dequeue_and_enqueue_acceleration(k_a_batchdim)

        # loss featuredim joint
        l_j_pos_featuredim = torch.einsum('nc,nc->c', [q_j_normalized_featuredim, k_j_normalized_featuredim]).unsqueeze(-1)#进行完这一步操作后正对间的相似度转化为一列

        queue_j_featuredim = torch.cat([q_j_normalized_featuredim, k_j_normalized_featuredim], dim=1)#按列拼接

        l_j_q_neg_featuredim = torch.einsum('nc,nk->ck', [q_j_normalized_featuredim, queue_j_featuredim])
        #l_j_k_neg_featuredim = torch.einsum('nc,nk->ck', [k_j_normalized_featuredim, queue_j_featuredim])

        logits_j_q_featuredim = torch.zeros((l_j_q_neg_featuredim.shape[0], l_j_q_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        logits_j_q_featuredim = torch.cat([l_j_pos_featuredim, logits_j_q_featuredim], dim=1)
        #logits_j_k_featuredim = torch.zeros((l_j_k_neg_featuredim.shape[0], l_j_k_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        #logits_j_k_featuredim = torch.cat([l_j_pos_featuredim, logits_j_k_featuredim], dim=1)
        for i in range(128):
            if i == 0:
                logits_j_q_featuredim[i, 1:] = torch.cat([l_j_q_neg_featuredim[i,1:128].unsqueeze(0),l_j_q_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
                #logits_j_k_featuredim[i, 1:] = torch.cat([l_j_k_neg_featuredim[i,1:128].unsqueeze(0),l_j_k_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
            elif i == 127:
                logits_j_q_featuredim[i, 1:] = torch.cat([l_j_q_neg_featuredim[i,0:127].unsqueeze(0),l_j_q_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
                #logits_j_k_featuredim[i, 1:] = torch.cat([l_j_k_neg_featuredim[i,0:127].unsqueeze(0),l_j_k_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
            else:
                logits_j_q_featuredim[i, 1:] = torch.cat([l_j_q_neg_featuredim[i,0:i].unsqueeze(0),l_j_q_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_j_q_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
                #logits_j_k_featuredim[i, 1:] = torch.cat([l_j_k_neg_featuredim[i,0:i].unsqueeze(0),l_j_k_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_j_k_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
        #logits_j_featuredim = torch.cat([logits_j_q_featuredim, logits_j_k_featuredim], dim=0)
        #logits_cluster /= self.T

        # loss featuredim motion
        l_m_pos_featuredim = torch.einsum('nc,nc->c', [q_m_normalized_featuredim, k_m_normalized_featuredim]).unsqueeze(-1)#进行完这一步操作后正对间的相似度转化为一列

        queue_m_featuredim = torch.cat([q_m_normalized_featuredim, k_m_normalized_featuredim], dim=1)#按列拼接

        l_m_q_neg_featuredim = torch.einsum('nc,nk->ck', [q_m_normalized_featuredim, queue_m_featuredim])
        #l_m_k_neg_featuredim = torch.einsum('nc,nk->ck', [k_m_normalized_featuredim, queue_m_featuredim])

        logits_m_q_featuredim = torch.zeros((l_m_q_neg_featuredim.shape[0], l_m_q_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        logits_m_q_featuredim = torch.cat([l_m_pos_featuredim, logits_m_q_featuredim], dim=1)
        #logits_m_k_featuredim = torch.zeros((l_m_k_neg_featuredim.shape[0], l_m_k_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        #logits_m_k_featuredim = torch.cat([l_m_pos_featuredim, logits_m_k_featuredim], dim=1)
        for i in range(128):
            if i == 0:
                logits_m_q_featuredim[i, 1:] = torch.cat([l_m_q_neg_featuredim[i,1:128].unsqueeze(0),l_m_q_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
                #logits_m_k_featuredim[i, 1:] = torch.cat([l_m_k_neg_featuredim[i,1:128].unsqueeze(0),l_m_k_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
            elif i == 127:
                logits_m_q_featuredim[i, 1:] = torch.cat([l_m_q_neg_featuredim[i,0:127].unsqueeze(0),l_m_q_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
                #logits_m_k_featuredim[i, 1:] = torch.cat([l_m_k_neg_featuredim[i,0:127].unsqueeze(0),l_m_k_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
            else:
                logits_m_q_featuredim[i, 1:] = torch.cat([l_m_q_neg_featuredim[i,0:i].unsqueeze(0),l_m_q_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_m_q_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
                #logits_m_k_featuredim[i, 1:] = torch.cat([l_m_k_neg_featuredim[i,0:i].unsqueeze(0),l_m_k_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_m_k_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
        #logits_m_featuredim = torch.cat([logits_m_q_featuredim, logits_m_k_featuredim], dim=0)
        #logits_cluster /= self.T

        # loss featuredim bone
        l_b_pos_featuredim = torch.einsum('nc,nc->c', [q_b_normalized_featuredim, k_b_normalized_featuredim]).unsqueeze(-1)#进行完这一步操作后正对间的相似度转化为一列

        queue_b_featuredim = torch.cat([q_b_normalized_featuredim, k_b_normalized_featuredim], dim=1)#按列拼接

        l_b_q_neg_featuredim = torch.einsum('nc,nk->ck', [q_b_normalized_featuredim, queue_b_featuredim])
        #l_b_k_neg_featuredim = torch.einsum('nc,nk->ck', [k_b_normalized_featuredim, queue_b_featuredim])

        logits_b_q_featuredim = torch.zeros((l_b_q_neg_featuredim.shape[0], l_b_q_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        logits_b_q_featuredim = torch.cat([l_b_pos_featuredim, logits_b_q_featuredim], dim=1)
        #logits_b_k_featuredim = torch.zeros((l_b_k_neg_featuredim.shape[0], l_b_k_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        #logits_b_k_featuredim = torch.cat([l_b_pos_featuredim, logits_b_k_featuredim], dim=1)
        for i in range(128):
            if i == 0:
                logits_b_q_featuredim[i, 1:] = torch.cat([l_b_q_neg_featuredim[i,1:128].unsqueeze(0),l_b_q_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
                #logits_b_k_featuredim[i, 1:] = torch.cat([l_b_k_neg_featuredim[i,1:128].unsqueeze(0),l_b_k_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
            elif i == 127:
                logits_b_q_featuredim[i, 1:] = torch.cat([l_b_q_neg_featuredim[i,0:127].unsqueeze(0),l_b_q_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
                #logits_b_k_featuredim[i, 1:] = torch.cat([l_b_k_neg_featuredim[i,0:127].unsqueeze(0),l_b_k_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
            else:
                logits_b_q_featuredim[i, 1:] = torch.cat([l_b_q_neg_featuredim[i,0:i].unsqueeze(0),l_b_q_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_b_q_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
                #logits_b_k_featuredim[i, 1:] = torch.cat([l_b_k_neg_featuredim[i,0:i].unsqueeze(0),l_b_k_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_b_k_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
        #logits_b_featuredim = torch.cat([logits_b_q_featuredim, logits_b_k_featuredim], dim=0)
        #logits_cluster /= self.T

        # loss featuredim acceleration
        l_a_pos_featuredim = torch.einsum('nc,nc->c', [q_a_normalized_featuredim, k_a_normalized_featuredim]).unsqueeze(-1)#进行完这一步操作后正对间的相似度转化为一列

        queue_a_featuredim = torch.cat([q_a_normalized_featuredim, k_a_normalized_featuredim], dim=1)#按列拼接

        l_a_q_neg_featuredim = torch.einsum('nc,nk->ck', [q_a_normalized_featuredim, queue_a_featuredim])
        #l_a_k_neg_featuredim = torch.einsum('nc,nk->ck', [k_a_normalized_featuredim, queue_a_featuredim])

        logits_a_q_featuredim = torch.zeros((l_a_q_neg_featuredim.shape[0], l_a_q_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        logits_a_q_featuredim = torch.cat([l_a_pos_featuredim, logits_a_q_featuredim], dim=1)
        #logits_a_k_featuredim = torch.zeros((l_a_k_neg_featuredim.shape[0], l_a_k_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        #logits_a_k_featuredim = torch.cat([l_a_pos_featuredim, logits_a_k_featuredim], dim=1)
        for i in range(128):
            if i == 0:
                logits_a_q_featuredim[i, 1:] = torch.cat([l_a_q_neg_featuredim[i,1:128].unsqueeze(0),l_a_q_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
                #logits_a_k_featuredim[i, 1:] = torch.cat([l_a_k_neg_featuredim[i,1:128].unsqueeze(0),l_a_k_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
            elif i == 127:
                logits_a_q_featuredim[i, 1:] = torch.cat([l_a_q_neg_featuredim[i,0:127].unsqueeze(0),l_a_q_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
                #logits_a_k_featuredim[i, 1:] = torch.cat([l_a_k_neg_featuredim[i,0:127].unsqueeze(0),l_a_k_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
            else:
                logits_a_q_featuredim[i, 1:] = torch.cat([l_a_q_neg_featuredim[i,0:i].unsqueeze(0),l_a_q_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_a_q_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
                #logits_a_k_featuredim[i, 1:] = torch.cat([l_a_k_neg_featuredim[i,0:i].unsqueeze(0),l_a_k_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_a_k_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
        #logits_a_featuredim = torch.cat([logits_a_q_featuredim, logits_a_k_featuredim], dim=0)
        #logits_cluster /= self.T

        labels_featuredim = torch.zeros(logits_j_q_featuredim.shape[0], dtype=torch.long).cuda()#形状都一样，这里简化下
        '''
        #entropy of joint
        q_j_vnorm = torch.norm(q_j_featuredim, p=1, dim=0)
        k_j_vnorm = torch.norm(k_j_featuredim, p=1, dim=0)
        q_j_norm = torch.norm(q_j_vnorm, p=1)
        k_j_norm = torch.norm(k_j_vnorm, p=1)
        entropy_j = (q_j_vnorm / q_j_norm * torch.log(q_j_vnorm / q_j_norm)).sum(0) + (k_j_vnorm / k_j_norm * torch.log(k_j_vnorm / k_j_norm)).sum(0)
        #entropy of motion
        q_m_vnorm = torch.norm(q_m_featuredim, p=1, dim=0)
        k_m_vnorm = torch.norm(k_m_featuredim, p=1, dim=0)
        q_m_norm = torch.norm(q_m_vnorm, p=1)
        k_m_norm = torch.norm(k_m_vnorm, p=1)
        entropy_m = (q_m_vnorm / q_m_norm * torch.log(q_m_vnorm / q_m_norm)).sum(0) + (k_m_vnorm / k_m_norm * torch.log(k_m_vnorm / k_m_norm)).sum(0)
        #entropy of bone
        q_b_vnorm = torch.norm(q_b_featuredim, p=1, dim=0)
        k_b_vnorm = torch.norm(k_b_featuredim, p=1, dim=0)
        q_b_norm = torch.norm(q_b_vnorm, p=1)
        k_b_norm = torch.norm(k_b_vnorm, p=1)
        entropy_b = (q_b_vnorm / q_b_norm * torch.log(q_b_vnorm / q_b_norm)).sum(0) + (k_b_vnorm / k_b_norm * torch.log(k_b_vnorm / k_b_norm)).sum(0)
        #entropy of acceleration
        q_a_vnorm = torch.norm(q_a_featuredim, p=1, dim=0)
        k_a_vnorm = torch.norm(k_a_featuredim, p=1, dim=0)
        q_a_norm = torch.norm(q_a_vnorm, p=1)
        k_a_norm = torch.norm(k_a_vnorm, p=1)
        entropy_a = (q_a_vnorm / q_a_norm * torch.log(q_a_vnorm / q_a_norm)).sum(0) + (k_a_vnorm / k_a_norm * torch.log(k_a_vnorm / k_a_norm)).sum(0)
        '''
        return logits, logits_motion, logits_bone, logits_acceleration, labels,\
             logits_j_q_featuredim, logits_m_q_featuredim, logits_b_q_featuredim, logits_a_q_featuredim, labels_featuredim

    def cross_training(self, im_q, im_k, frame, topk=1, context=True):

        time_scale = 50/ frame
        im_q_motion = torch.zeros_like(im_q)
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]
        for i, f in enumerate(time_scale):
            im_q_motion[i, :, :, :, :] = f * im_q_motion[i, :, :, :, :]

        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]
        for i, f in enumerate(time_scale):
            im_k_motion[i, :, :, :, :] = f * im_k_motion[i, :, :, :, :]

        im_q_acceleration = torch.zeros_like(im_q)
        im_q_acceleration[:, :, :-2, :, :] = im_q_motion[:, :, 1:-1, :, :] - im_q_motion[:, :, :-2, :, :]
        for i, f in enumerate(time_scale):
            im_q_acceleration[i, :, :, :, :] = f * im_q_acceleration[i, :, :, :, :]

        im_k_acceleration = torch.zeros_like(im_k)
        im_k_acceleration[:, :, :-2, :, :] = im_k_motion[:, :, 1:-1, :, :] - im_k_motion[:, :, :-2, :, :]
        for i, f in enumerate(time_scale):
            im_k_acceleration[i, :, :, :, :] = f * im_k_acceleration[i, :, :, :, :]
    
        im_q_bone = torch.zeros_like(im_q)
        im_k_bone = torch.zeros_like(im_k)

        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # compute query features
        q_j = self.encoder_q(im_q)  # queries: NxC
        q_j_batchdim = self.q_batchdim(q_j)
        q_j_batchdim = F.normalize(q_j_batchdim, dim=1)
        q_j_featuredim = self.featuredim1(q_j)
        q_j_normalized_featuredim = F.normalize(q_j_featuredim, dim=0)

        q_m = self.encoder_q_motion(im_q_motion)
        q_m_batchdim = self.q_motion_batchdim(q_m)
        q_m_batchdim = F.normalize(q_m_batchdim, dim=1)
        q_m_featuredim = self.motion_featuredim1(q_m)
        q_m_normalized_featuredim = F.normalize(q_m_featuredim, dim=0)

        q_b = self.encoder_q_bone(im_q_bone)
        q_b_batchdim = self.q_bone_batchdim(q_b)
        q_b_batchdim = F.normalize(q_b_batchdim, dim=1)
        q_b_featuredim = self.bone_featuredim1(q_b)
        q_b_normalized_featuredim = F.normalize(q_b_featuredim, dim=0)

        q_a = self.encoder_q_acceleration(im_q_acceleration)
        q_a_batchdim = self.q_acceleration_batchdim(q_a)
        q_a_batchdim = F.normalize(q_a_batchdim, dim=1)
        q_a_featuredim = self.acceleration_featuredim1(q_a)
        q_a_normalized_featuredim = F.normalize(q_a_featuredim, dim=0)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            #self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            #self._momentum_update_key_encoder_bone()
            self._momentum_update_key_encoder_acceleration()

            k_j = self.encoder_k(im_k)  # keys: NxC
            k_j_batchdim = self.k_batchdim(k_j)
            k_j_batchdim = F.normalize(k_j_batchdim, dim=1)
            kq_j = self.encoder_q(im_k)
            k_j_featuredim = self.featuredim2(kq_j)
            k_j_normalized_featuredim = F.normalize(k_j_featuredim, dim=0)

            k_m = self.encoder_k_motion(im_k_motion)
            k_m_batchdim = self.k_motion_batchdim(k_m)
            k_m_batchdim = F.normalize(k_m_batchdim,dim=1)
            kq_m = self.encoder_q_motion(im_k_motion)
            k_m_featuredim = self.motion_featuredim2(kq_m)
            k_m_normalized_featuredim = F.normalize(k_m_featuredim, dim=0)

            k_b = self.encoder_k_bone(im_k_bone)
            k_b_batchdim = self.k_bone_batchdim(k_b)
            k_b_batchdim = F.normalize(k_b_batchdim, dim=1)
            kq_b = self.encoder_q_bone(im_k_bone)
            k_b_featuredim = self.bone_featuredim2(kq_b)
            k_b_normalized_featuredim = F.normalize(k_b_featuredim, dim=0)

            k_a = self.encoder_k_acceleration(im_k_acceleration)
            k_a_batchdim = self.k_acceleration_batchdim(k_a)
            k_a_batchdim = F.normalize(k_a_batchdim, dim=1)
            kq_a = self.encoder_q_acceleration(im_k_acceleration)
            k_a_featuredim = self.acceleration_featuredim2(kq_a)
            k_a_normalized_featuredim = F.normalize(k_a_featuredim, dim=0)

        l_pos = torch.einsum('nc,nc->n', [q_j_batchdim, k_j_batchdim]).unsqueeze(-1)#两组向量对应位置相乘,qk分别有128个128维向量，向量点乘得到128个相似度
        l_neg = torch.einsum('nc,ck->nk', [q_j_batchdim, self.queue.clone().detach()])#矩阵乘法，克隆张量并避免梯度传播，128个向量和bank里的32768个分别相乘

        l_pos_motion = torch.einsum('nc,nc->n', [q_m_batchdim, k_m_batchdim]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_m_batchdim, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_b_batchdim, k_b_batchdim]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_b_batchdim, self.queue_bone.clone().detach()])

        l_pos_acceleration = torch.einsum('nc,nc->n', [q_a_batchdim, k_a_batchdim]).unsqueeze(-1)
        l_neg_acceleration = torch.einsum('nc,ck->nk', [q_a_batchdim, self.queue_acceleration.clone().detach()])
        
        if context:
            l_context_jm = torch.einsum('nk,nk->nk', [l_neg, l_neg_motion])#这里是论文中的ss，两个view下和bank中embedding相似度的乘积
            l_context_jb = torch.einsum('nk,nk->nk', [l_neg, l_neg_bone])
            l_context_mb = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_bone])
            l_context_ja = torch.einsum('nk,nk->nk', [l_neg, l_neg_acceleration])
            l_context_ma = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_acceleration])
            l_context_ba = torch.einsum('nk,nk->nk', [l_neg_bone, l_neg_acceleration])

            logits_jm = torch.cat([l_pos, l_neg, l_context_jm], dim=1)#按行拼接
            logits_jb = torch.cat([l_pos, l_neg, l_context_jb], dim=1)
            logits_mj = torch.cat([l_pos_motion, l_neg_motion, l_context_jm], dim=1)
            logits_mb = torch.cat([l_pos_motion, l_neg_motion, l_context_mb], dim=1)
            logits_bj = torch.cat([l_pos_bone, l_neg_bone, l_context_jb], dim=1)
            logits_bm = torch.cat([l_pos_bone, l_neg_bone, l_context_mb], dim=1)
            logits_ja = torch.cat([l_pos, l_neg, l_context_ja], dim=1)
            logits_aj = torch.cat([l_pos_acceleration, l_neg_acceleration, l_context_ja], dim=1)
            logits_ma = torch.cat([l_pos_motion, l_neg_motion, l_context_ma], dim=1)
            logits_am = torch.cat([l_pos_acceleration, l_neg_acceleration, l_context_ma], dim=1)
            logits_ba = torch.cat([l_pos_bone, l_neg_bone, l_context_ba], dim=1)
            logits_ab = torch.cat([l_pos_acceleration, l_neg_acceleration, l_context_ba], dim=1)

        else:
            logits_jm = torch.cat([l_pos, l_neg], dim=1)
            logits_jb = torch.cat([l_pos, l_neg], dim=1)
            logits_mj = torch.cat([l_pos_motion, l_neg_motion], dim=1)
            logits_mb = torch.cat([l_pos_motion, l_neg_motion], dim=1)
            logits_bj = torch.cat([l_pos_bone, l_neg_bone], dim=1)
            logits_bm = torch.cat([l_pos_bone, l_neg_bone], dim=1)
            logits_ja = torch.cat([l_pos, l_neg], dim=1)
            logits_aj = torch.cat([l_pos_acceleration, l_neg_acceleration], dim=1)
            logits_ma = torch.cat([l_pos_motion, l_neg_motion], dim=1)
            logits_am = torch.cat([l_pos_acceleration, l_neg_acceleration], dim=1)
            logits_ba = torch.cat([l_pos_bone, l_neg_bone], dim=1)
            logits_ab = torch.cat([l_pos_acceleration, l_neg_acceleration], dim=1)

        logits_jm /= self.T
        logits_jb /= self.T
        logits_mj /= self.T
        logits_mb /= self.T
        logits_bj /= self.T
        logits_bm /= self.T
        logits_ja /= self.T
        logits_aj /= self.T
        logits_ma /= self.T
        logits_am /= self.T
        logits_ba /= self.T
        logits_ab /= self.T
        

        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_motion = torch.topk(l_neg_motion, topk, dim=1)
        _, topkdix_bone = torch.topk(l_neg_bone, topk, dim=1)
        _, topkdix_acceleration = torch.topk(l_neg_acceleration, topk, dim=1)

        topk_onehot_jm = torch.zeros_like(l_neg)
        topk_onehot_jb = torch.zeros_like(l_neg)
        topk_onehot_mj = torch.zeros_like(l_neg_motion)
        topk_onehot_mb = torch.zeros_like(l_neg_motion)
        topk_onehot_bj = torch.zeros_like(l_neg_bone)
        topk_onehot_bm = torch.zeros_like(l_neg_bone)
        topk_onehot_ja = torch.zeros_like(l_neg)
        topk_onehot_aj = torch.zeros_like(l_neg_acceleration)
        topk_onehot_ma = torch.zeros_like(l_neg_motion)
        topk_onehot_am = torch.zeros_like(l_neg_acceleration)
        topk_onehot_ba = torch.zeros_like(l_neg_bone)
        topk_onehot_ab = torch.zeros_like(l_neg_acceleration)

        topk_onehot_jm.scatter_(1, topkdix_motion, 1)
        topk_onehot_jb.scatter_(1, topkdix_bone, 1)
        topk_onehot_mj.scatter_(1, topkdix, 1)
        topk_onehot_mb.scatter_(1, topkdix_bone, 1)
        topk_onehot_bj.scatter_(1, topkdix, 1)
        topk_onehot_bm.scatter_(1, topkdix_motion, 1)
        topk_onehot_ja.scatter_(1, topkdix_acceleration, 1)
        topk_onehot_aj.scatter_(1, topkdix, 1)
        topk_onehot_ma.scatter_(1, topkdix_acceleration, 1)
        topk_onehot_am.scatter_(1, topkdix_motion, 1)
        topk_onehot_ba.scatter_(1, topkdix_acceleration, 1)
        topk_onehot_ab.scatter_(1, topkdix_bone, 1)
        

        if context:
            pos_mask_jm = torch.cat([torch.ones(topk_onehot_jm.size(0), 1).cuda(), topk_onehot_jm, topk_onehot_jm], dim=1)
            pos_mask_jb = torch.cat([torch.ones(topk_onehot_jb.size(0), 1).cuda(), topk_onehot_jb, topk_onehot_jb], dim=1)
            pos_mask_mj = torch.cat([torch.ones(topk_onehot_mj.size(0), 1).cuda(), topk_onehot_mj, topk_onehot_mj], dim=1)
            pos_mask_mb = torch.cat([torch.ones(topk_onehot_mb.size(0), 1).cuda(), topk_onehot_mb, topk_onehot_mb], dim=1)
            pos_mask_bj = torch.cat([torch.ones(topk_onehot_bj.size(0), 1).cuda(), topk_onehot_bj, topk_onehot_bj], dim=1)
            pos_mask_bm = torch.cat([torch.ones(topk_onehot_bm.size(0), 1).cuda(), topk_onehot_bm, topk_onehot_bm], dim=1)
            pos_mask_ja = torch.cat([torch.ones(topk_onehot_ja.size(0), 1).cuda(), topk_onehot_ja, topk_onehot_ja], dim=1)
            pos_mask_aj = torch.cat([torch.ones(topk_onehot_aj.size(0), 1).cuda(), topk_onehot_aj, topk_onehot_aj], dim=1)
            pos_mask_am = torch.cat([torch.ones(topk_onehot_am.size(0), 1).cuda(), topk_onehot_am, topk_onehot_am], dim=1)
            pos_mask_ma = torch.cat([torch.ones(topk_onehot_ma.size(0), 1).cuda(), topk_onehot_ma, topk_onehot_ma], dim=1)
            pos_mask_ba = torch.cat([torch.ones(topk_onehot_ba.size(0), 1).cuda(), topk_onehot_ba, topk_onehot_ba], dim=1)
            pos_mask_ab = torch.cat([torch.ones(topk_onehot_ab.size(0), 1).cuda(), topk_onehot_ab, topk_onehot_ab], dim=1)
        else:
            pos_mask_jm = torch.cat([torch.ones(topk_onehot_jm.size(0), 1).cuda(), topk_onehot_jm], dim=1)
            pos_mask_jb = torch.cat([torch.ones(topk_onehot_jb.size(0), 1).cuda(), topk_onehot_jb], dim=1)
            pos_mask_mj = torch.cat([torch.ones(topk_onehot_mj.size(0), 1).cuda(), topk_onehot_mj], dim=1)
            pos_mask_mb = torch.cat([torch.ones(topk_onehot_mb.size(0), 1).cuda(), topk_onehot_mb], dim=1)
            pos_mask_bj = torch.cat([torch.ones(topk_onehot_bj.size(0), 1).cuda(), topk_onehot_bj], dim=1)
            pos_mask_bm = torch.cat([torch.ones(topk_onehot_bm.size(0), 1).cuda(), topk_onehot_bm], dim=1)
            pos_mask_ja = torch.cat([torch.ones(topk_onehot_ja.size(0), 1).cuda(), topk_onehot_ja], dim=1)
            pos_mask_aj = torch.cat([torch.ones(topk_onehot_aj.size(0), 1).cuda(), topk_onehot_aj], dim=1)
            pos_mask_am = torch.cat([torch.ones(topk_onehot_am.size(0), 1).cuda(), topk_onehot_am], dim=1)
            pos_mask_ma = torch.cat([torch.ones(topk_onehot_ma.size(0), 1).cuda(), topk_onehot_ma], dim=1)
            pos_mask_ba = torch.cat([torch.ones(topk_onehot_ba.size(0), 1).cuda(), topk_onehot_ba], dim=1)
            pos_mask_ab = torch.cat([torch.ones(topk_onehot_ab.size(0), 1).cuda(), topk_onehot_ab], dim=1)

        self._dequeue_and_enqueue(k_j_batchdim)
        self._dequeue_and_enqueue_motion(k_m_batchdim)
        self._dequeue_and_enqueue_bone(k_b_batchdim)
        self._dequeue_and_enqueue_acceleration(k_a_batchdim)

        # loss featuredim joint
        l_j_pos_featuredim = torch.einsum('nc,nc->c', [q_j_normalized_featuredim, k_j_normalized_featuredim]).unsqueeze(-1)#进行完这一步操作后正对间的相似度转化为一列

        queue_j_featuredim = torch.cat([q_j_normalized_featuredim, k_j_normalized_featuredim], dim=1)#按列拼接

        l_j_q_neg_featuredim = torch.einsum('nc,nk->ck', [q_j_normalized_featuredim, queue_j_featuredim])
        #l_j_k_neg_featuredim = torch.einsum('nc,nk->ck', [k_j_normalized_featuredim, queue_j_featuredim])

        logits_j_q_featuredim = torch.zeros((l_j_q_neg_featuredim.shape[0], l_j_q_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        logits_j_q_featuredim = torch.cat([l_j_pos_featuredim, logits_j_q_featuredim], dim=1)
        #logits_j_k_featuredim = torch.zeros((l_j_k_neg_featuredim.shape[0], l_j_k_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        #logits_j_k_featuredim = torch.cat([l_j_pos_featuredim, logits_j_k_featuredim], dim=1)
        for i in range(128):
            if i == 0:
                logits_j_q_featuredim[i, 1:] = torch.cat([l_j_q_neg_featuredim[i,1:128].unsqueeze(0),l_j_q_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
                #logits_j_k_featuredim[i, 1:] = torch.cat([l_j_k_neg_featuredim[i,1:128].unsqueeze(0),l_j_k_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
            elif i == 127:
                logits_j_q_featuredim[i, 1:] = torch.cat([l_j_q_neg_featuredim[i,0:127].unsqueeze(0),l_j_q_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
                #logits_j_k_featuredim[i, 1:] = torch.cat([l_j_k_neg_featuredim[i,0:127].unsqueeze(0),l_j_k_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
            else:
                logits_j_q_featuredim[i, 1:] = torch.cat([l_j_q_neg_featuredim[i,0:i].unsqueeze(0),l_j_q_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_j_q_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
                #logits_j_k_featuredim[i, 1:] = torch.cat([l_j_k_neg_featuredim[i,0:i].unsqueeze(0),l_j_k_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_j_k_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
        #logits_j_featuredim = torch.cat([logits_j_q_featuredim, logits_j_k_featuredim], dim=0)
        #logits_cluster /= self.T

        # loss featuredim motion
        l_m_pos_featuredim = torch.einsum('nc,nc->c', [q_m_normalized_featuredim, k_m_normalized_featuredim]).unsqueeze(-1)#进行完这一步操作后正对间的相似度转化为一列

        queue_m_featuredim = torch.cat([q_m_normalized_featuredim, k_m_normalized_featuredim], dim=1)#按列拼接

        l_m_q_neg_featuredim = torch.einsum('nc,nk->ck', [q_m_normalized_featuredim, queue_m_featuredim])
        #l_m_k_neg_featuredim = torch.einsum('nc,nk->ck', [k_m_normalized_featuredim, queue_m_featuredim])

        logits_m_q_featuredim = torch.zeros((l_m_q_neg_featuredim.shape[0], l_m_q_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        logits_m_q_featuredim = torch.cat([l_m_pos_featuredim, logits_m_q_featuredim], dim=1)
        #logits_m_k_featuredim = torch.zeros((l_m_k_neg_featuredim.shape[0], l_m_k_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        #logits_m_k_featuredim = torch.cat([l_m_pos_featuredim, logits_m_k_featuredim], dim=1)
        for i in range(128):
            if i == 0:
                logits_m_q_featuredim[i, 1:] = torch.cat([l_m_q_neg_featuredim[i,1:128].unsqueeze(0),l_m_q_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
                #logits_m_k_featuredim[i, 1:] = torch.cat([l_m_k_neg_featuredim[i,1:128].unsqueeze(0),l_m_k_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
            elif i == 127:
                logits_m_q_featuredim[i, 1:] = torch.cat([l_m_q_neg_featuredim[i,0:127].unsqueeze(0),l_m_q_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
                #logits_m_k_featuredim[i, 1:] = torch.cat([l_m_k_neg_featuredim[i,0:127].unsqueeze(0),l_m_k_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
            else:
                logits_m_q_featuredim[i, 1:] = torch.cat([l_m_q_neg_featuredim[i,0:i].unsqueeze(0),l_m_q_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_m_q_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
                #logits_m_k_featuredim[i, 1:] = torch.cat([l_m_k_neg_featuredim[i,0:i].unsqueeze(0),l_m_k_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_m_k_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
        #logits_m_featuredim = torch.cat([logits_m_q_featuredim, logits_m_k_featuredim], dim=0)
        #logits_cluster /= self.T

        # loss featuredim bone
        l_b_pos_featuredim = torch.einsum('nc,nc->c', [q_b_normalized_featuredim, k_b_normalized_featuredim]).unsqueeze(-1)#进行完这一步操作后正对间的相似度转化为一列

        queue_b_featuredim = torch.cat([q_b_normalized_featuredim, k_b_normalized_featuredim], dim=1)#按列拼接

        l_b_q_neg_featuredim = torch.einsum('nc,nk->ck', [q_b_normalized_featuredim, queue_b_featuredim])
        #l_b_k_neg_featuredim = torch.einsum('nc,nk->ck', [k_b_normalized_featuredim, queue_b_featuredim])

        logits_b_q_featuredim = torch.zeros((l_b_q_neg_featuredim.shape[0], l_b_q_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        logits_b_q_featuredim = torch.cat([l_b_pos_featuredim, logits_b_q_featuredim], dim=1)
        #logits_b_k_featuredim = torch.zeros((l_b_k_neg_featuredim.shape[0], l_b_k_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        #logits_b_k_featuredim = torch.cat([l_b_pos_featuredim, logits_b_k_featuredim], dim=1)
        for i in range(128):
            if i == 0:
                logits_b_q_featuredim[i, 1:] = torch.cat([l_b_q_neg_featuredim[i,1:128].unsqueeze(0),l_b_q_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
                #logits_b_k_featuredim[i, 1:] = torch.cat([l_b_k_neg_featuredim[i,1:128].unsqueeze(0),l_b_k_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
            elif i == 127:
                logits_b_q_featuredim[i, 1:] = torch.cat([l_b_q_neg_featuredim[i,0:127].unsqueeze(0),l_b_q_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
                #logits_b_k_featuredim[i, 1:] = torch.cat([l_b_k_neg_featuredim[i,0:127].unsqueeze(0),l_b_k_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
            else:
                logits_b_q_featuredim[i, 1:] = torch.cat([l_b_q_neg_featuredim[i,0:i].unsqueeze(0),l_b_q_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_b_q_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
                #logits_b_k_featuredim[i, 1:] = torch.cat([l_b_k_neg_featuredim[i,0:i].unsqueeze(0),l_b_k_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_b_k_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
        #logits_b_featuredim = torch.cat([logits_b_q_featuredim, logits_b_k_featuredim], dim=0)
        #logits_cluster /= self.T

        # loss featuredim acceleration
        l_a_pos_featuredim = torch.einsum('nc,nc->c', [q_a_normalized_featuredim, k_a_normalized_featuredim]).unsqueeze(-1)#进行完这一步操作后正对间的相似度转化为一列

        queue_a_featuredim = torch.cat([q_a_normalized_featuredim, k_a_normalized_featuredim], dim=1)#按列拼接

        l_a_q_neg_featuredim = torch.einsum('nc,nk->ck', [q_a_normalized_featuredim, queue_a_featuredim])
        #l_a_k_neg_featuredim = torch.einsum('nc,nk->ck', [k_a_normalized_featuredim, queue_a_featuredim])

        logits_a_q_featuredim = torch.zeros((l_a_q_neg_featuredim.shape[0], l_a_q_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        logits_a_q_featuredim = torch.cat([l_a_pos_featuredim, logits_a_q_featuredim], dim=1)
        #logits_a_k_featuredim = torch.zeros((l_a_k_neg_featuredim.shape[0], l_a_k_neg_featuredim.shape[1]-2), dtype=torch.long).cuda()
        #logits_a_k_featuredim = torch.cat([l_a_pos_featuredim, logits_a_k_featuredim], dim=1)
        for i in range(128):
            if i == 0:
                logits_a_q_featuredim[i, 1:] = torch.cat([l_a_q_neg_featuredim[i,1:128].unsqueeze(0),l_a_q_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
                #logits_a_k_featuredim[i, 1:] = torch.cat([l_a_k_neg_featuredim[i,1:128].unsqueeze(0),l_a_k_neg_featuredim[i,129:256].unsqueeze(0)], dim=1)
            elif i == 127:
                logits_a_q_featuredim[i, 1:] = torch.cat([l_a_q_neg_featuredim[i,0:127].unsqueeze(0),l_a_q_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
                #logits_a_k_featuredim[i, 1:] = torch.cat([l_a_k_neg_featuredim[i,0:127].unsqueeze(0),l_a_k_neg_featuredim[i,128:255].unsqueeze(0)], dim=1)
            else:
                logits_a_q_featuredim[i, 1:] = torch.cat([l_a_q_neg_featuredim[i,0:i].unsqueeze(0),l_a_q_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_a_q_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
                #logits_a_k_featuredim[i, 1:] = torch.cat([l_a_k_neg_featuredim[i,0:i].unsqueeze(0),l_a_k_neg_featuredim[i,(i+1):(i+128)].unsqueeze(0),l_a_k_neg_featuredim[i,(i+129):256].unsqueeze(0)], dim=1)
        #logits_a_featuredim = torch.cat([logits_a_q_featuredim, logits_a_k_featuredim], dim=0)
        #logits_cluster /= self.T

        labels_featuredim = torch.zeros(logits_j_q_featuredim.shape[0], dtype=torch.long).cuda()#形状都一样，这里简化下,改回双向时把有的地方的q去掉
        '''
        #entropy of joint
        q_j_vnorm = torch.norm(q_j_featuredim, p=1, dim=0)
        k_j_vnorm = torch.norm(k_j_featuredim, p=1, dim=0)
        q_j_norm = torch.norm(q_j_vnorm, p=1)
        k_j_norm = torch.norm(k_j_vnorm, p=1)
        entropy_j = (q_j_vnorm / q_j_norm * torch.log(q_j_vnorm / q_j_norm)).sum(0) + (k_j_vnorm / k_j_norm * torch.log(k_j_vnorm / k_j_norm)).sum(0)
        #entropy of motion
        q_m_vnorm = torch.norm(q_m_featuredim, p=1, dim=0)
        k_m_vnorm = torch.norm(k_m_featuredim, p=1, dim=0)
        q_m_norm = torch.norm(q_m_vnorm, p=1)
        k_m_norm = torch.norm(k_m_vnorm, p=1)
        entropy_m = (q_m_vnorm / q_m_norm * torch.log(q_m_vnorm / q_m_norm)).sum(0) + (k_m_vnorm / k_m_norm * torch.log(k_m_vnorm / k_m_norm)).sum(0)
        #entropy of bone
        q_b_vnorm = torch.norm(q_b_featuredim, p=1, dim=0)
        k_b_vnorm = torch.norm(k_b_featuredim, p=1, dim=0)
        q_b_norm = torch.norm(q_b_vnorm, p=1)
        k_b_norm = torch.norm(k_b_vnorm, p=1)
        entropy_b = (q_b_vnorm / q_b_norm * torch.log(q_b_vnorm / q_b_norm)).sum(0) + (k_b_vnorm / k_b_norm * torch.log(k_b_vnorm / k_b_norm)).sum(0)
        #entropy of acceleration
        q_a_vnorm = torch.norm(q_a_featuredim, p=1, dim=0)
        k_a_vnorm = torch.norm(k_a_featuredim, p=1, dim=0)
        q_a_norm = torch.norm(q_a_vnorm, p=1)
        k_a_norm = torch.norm(k_a_vnorm, p=1)
        entropy_a = (q_a_vnorm / q_a_norm * torch.log(q_a_vnorm / q_a_norm)).sum(0) + (k_a_vnorm / k_a_norm * torch.log(k_a_vnorm / k_a_norm)).sum(0)
        '''
        return logits_jm, logits_jb, logits_mj, logits_mb, logits_bj, logits_bm, logits_ja, logits_aj, logits_ma, logits_am, logits_ba, logits_ab, \
            pos_mask_jm, pos_mask_jb, pos_mask_mj, pos_mask_mb, pos_mask_bj, pos_mask_bm, pos_mask_ja, pos_mask_aj, pos_mask_ma, pos_mask_am, pos_mask_ba, pos_mask_ab,\
            logits_j_q_featuredim, logits_m_q_featuredim, logits_b_q_featuredim, logits_a_q_featuredim, labels_featuredim
