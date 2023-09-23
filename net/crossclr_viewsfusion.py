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
            self.encoder_q_viewx = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.view_fusion = nn.Sequential(nn.Linear(12, 12),#跑pretrain的时候还是view，先配合一下。
                                            nn.Linear(12, 12),
                                            nn.ReLU(),
                                            nn.Linear(12, 12),
                                            nn.ReLU(),
                                            nn.Linear(12, 3))
            #self.late_fusion = nn.Sequential(nn.Linear(4*num_class, num_class),
                                            #nn.ReLU())
            #self.late_fusion = nn.Linear(4*num_class, num_class)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            #为每个view的q和k都创建baseencoder。
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
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_k_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_acceleration = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_acceleration = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_viewx = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.view_fusion = nn.Sequential(nn.Linear(12, 12),
                                            nn.Linear(12, 12),
                                            nn.ReLU(),
                                            nn.Linear(12, 12),
                                            nn.ReLU(),
                                            nn.Linear(12, 3))
            #self.emb_fusion = nn.Sequential(nn.Linear(4*feature_dim, feature_dim),
                                            #nn.ReLU())

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
                self.encoder_q_viewx.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_viewx.fc)

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

            self.register_buffer("queue_viewx", torch.randn(feature_dim, self.K))
            self.queue_viewx = F.normalize(self.queue_viewx, dim=0)
            self.register_buffer("queue_ptr_viewx", torch.zeros(1, dtype=torch.long))

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
    def _dequeue_and_enqueue_viewx(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_viewx)
        gpu_index = keys.device.index
        self.queue_viewx[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
        self.queue_ptr_motion[0] = (self.queue_ptr_motion[0] + batch_size) % self.K
        self.queue_ptr_bone[0] = (self.queue_ptr_bone[0] + batch_size) % self.K
        self.queue_ptr_acceleration[0] = (self.queue_ptr_acceleration[0] + batch_size) % self.K
        self.queue_ptr_viewx[0] = (self.queue_ptr_viewx[0] + batch_size) % self.K

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

        im_q_acceleration = torch .zeros_like(im_q)
        im_q_acceleration[:, :, :-2, :, :] = im_q_motion[:, :, 1:-1, :, :] - im_q_motion[:, :, :-2, :, :]
        for i, f in enumerate(time_scale):
            im_q_acceleration[i, :, :, :, :] = f * im_q_acceleration[i, :, :, :, :]

        im_q_bone = torch.zeros_like(im_q)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]#前一个关节的坐标减去后一个关节的坐标
        
        im_q_viewx = torch.cat((im_q, im_q_motion, im_q_bone, im_q_acceleration), 1)
        im_q_viewx = im_q_viewx.permute(0, 2, 3, 4, 1).contiguous()
        im_q_viewx = self.view_fusion(im_q_viewx)
        im_q_viewx = im_q_viewx.permute(0, 4, 1, 2, 3).contiguous()

        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q)
            elif view == 'motion':
                return self.encoder_q_motion(im_q_motion)
            elif view == 'bone':
                return self.encoder_q_bone(im_q_bone)
            elif view == 'acceleration':
                return self.encoder_q_acceleration(im_q_acceleration)
            elif view == 'viewx':
                return self.encoder_q_viewx(im_q_viewx)
            elif view == 'all':
                #a = self.encoder_q(im_q)
                #b = self.encoder_q_motion(im_q_motion)
                #c = self.encoder_q_bone(im_q_bone)
                #d = self.encoder_q_acceleration(im_q_acceleration)
                #series = torch.cat((a,b,c,d),dim=1)
                #fusion = self.late_fusion(series)
                #return fusion
                return (self.encoder_q(im_q) + self.encoder_q_motion(im_q_motion) + self.encoder_q_bone(im_q_bone) + self.encoder_q_acceleration(im_q_acceleration)) / 4.
            elif view == 'all+viewx':
                return (self.encoder_q(im_q) + self.encoder_q_motion(im_q_motion) + self.encoder_q_bone(im_q_bone) + self.encoder_q_acceleration(im_q_acceleration) + self.encoder_q_viewx(im_q_viewx)) / 5.
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
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)#第二个维度是坐标，归一化

        q_motion = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)

        q_bone = self.encoder_q_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)

        q_acceleration = self.encoder_q_acceleration(im_q_acceleration)
        q_acceleration = F.normalize(q_acceleration, dim=1)

        q_viewx = self.encoder_q_viewx(im_q_viewx)
        q_viewx = F.normalize(q_viewx, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()
            self._momentum_update_key_encoder_acceleration()

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

            k_acceleration = self.encoder_k_acceleration(im_k_acceleration)
            k_acceleration = F.normalize(k_acceleration, dim=1)

            k_viewx = (k + k_motion + k_bone + k_acceleration) / 4

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)#在dim=0的地方加上一个维度，这个一看就是n组向量的点积，因为一次进去是一个batch
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])#这个一看就是矩阵相乘

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

        l_pos_acceleration = torch.einsum('nc,nc->n', [q_acceleration, k_acceleration]).unsqueeze(-1)
        l_neg_acceleration = torch.einsum('nc,ck->nk', [q_acceleration, self.queue_acceleration.clone().detach()])

        l_pos_viewx = torch.einsum('nc,nc->n', [q_viewx, k_viewx]).unsqueeze(-1)
        l_neg_viewx = torch.einsum('nc,ck->nk', [q_viewx, self.queue_viewx.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)#按维度1将两个张量拼接
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)
        logits_acceleration = torch.cat([l_pos_acceleration, l_neg_acceleration], dim=1)
        logits_viewx = torch.cat([l_pos_viewx, l_neg_viewx], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T
        logits_acceleration /= self.T
        logits_viewx /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)
        self._dequeue_and_enqueue_acceleration(k_acceleration)
        self._dequeue_and_enqueue_viewx(k_viewx)
        #self.update_ptr(128)

        return logits, logits_motion, logits_bone, logits_acceleration,logits_viewx, labels

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

        im_q_acceleration = torch .zeros_like(im_q)
        im_q_acceleration[:, :, :-2, :, :] = im_q_motion[:, :, 1:-1, :, :] - im_q_motion[:, :, :-2, :, :]
        for i, f in enumerate(time_scale):
            im_q_acceleration[i, :, :, :, :] = f * im_q_acceleration[i, :, :, :, :]

        im_k_acceleration = torch .zeros_like(im_k)
        im_k_acceleration[:, :, :-2, :, :] = im_k_motion[:, :, 1:-1, :, :] - im_k_motion[:, :, :-2, :, :]
        for i, f in enumerate(time_scale):
            im_k_acceleration[i, :, :, :, :] = f * im_k_acceleration[i, :, :, :, :]
    
        im_q_bone = torch.zeros_like(im_q)
        im_k_bone = torch.zeros_like(im_k)

        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]
        
        im_q_viewx = torch.cat((im_q, im_q_motion, im_q_bone, im_q_acceleration), 1)
        im_q_viewx = im_q_viewx.permute(0, 2, 3, 4, 1).contiguous()
        im_q_viewx = self.view_fusion(im_q_viewx)
        im_q_viewx = im_q_viewx.permute(0, 4, 1, 2, 3).contiguous()

        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        q_motion = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)

        q_bone = self.encoder_q_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)

        q_acceleration = self.encoder_q_acceleration(im_q_acceleration)
        q_acceleration = F.normalize(q_acceleration, dim=1)

        q_viewx = self.encoder_q_viewx(im_q_viewx)
        q_viewx = F.normalize(q_viewx, dim=1)
        
        with torch.no_grad():  # no gradient to keys
            #self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            #self._momentum_update_key_encoder_bone()
            self._momentum_update_key_encoder_acceleration()

            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
            
            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

            k_acceleration = self.encoder_k_acceleration(im_k_acceleration)
            k_acceleration = F.normalize(k_acceleration, dim=1)

            k_viewx = (k + k_motion + k_bone + k_acceleration) / 4

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)#两组向量对应位置相乘,qk分别有128个128维向量，向量点乘得到128个相似度
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])#矩阵乘法，克隆张量并避免梯度传播，128个向量和bank里的32768个分别相乘

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

        l_pos_acceleration = torch.einsum('nc,nc->n', [q_acceleration, k_acceleration]).unsqueeze(-1)
        l_neg_acceleration = torch.einsum('nc,ck->nk', [q_acceleration, self.queue_acceleration.clone().detach()])

        l_pos_viewx = torch.einsum('nc,nc->n', [q_viewx, k_viewx]).unsqueeze(-1)
        l_neg_viewx = torch.einsum('nc,ck->nk', [q_viewx, self.queue_viewx.clone().detach()])
        
        if context:
            l_context_jm = torch.einsum('nk,nk->nk', [l_neg, l_neg_motion])#这里是论文中的ss，两个view下和bank中embedding相似度的乘积
            l_context_jb = torch.einsum('nk,nk->nk', [l_neg, l_neg_bone])
            l_context_mb = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_bone])
            l_context_ja = torch.einsum('nk,nk->nk', [l_neg, l_neg_acceleration])
            l_context_ma = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_acceleration])
            l_context_ba = torch.einsum('nk,nk->nk', [l_neg_bone, l_neg_acceleration])
            #其他视角向x传递知识时
            l_context_jx = torch.einsum('nk,nk->nk', [l_neg, l_neg_viewx])
            l_context_mx = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_viewx])
            l_context_bx = torch.einsum('nk,nk->nk', [l_neg_bone, l_neg_viewx])
            l_context_ax = torch.einsum('nk,nk->nk', [l_neg_acceleration, l_neg_viewx])

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
            #这里表述有问题，比如说jm实际上是m向j传递知识
            logits_jx = torch.cat([l_pos_viewx, l_neg_viewx, l_context_jx], dim=1)
            logits_mx = torch.cat([l_pos_viewx, l_neg_viewx, l_context_mx], dim=1)
            logits_bx = torch.cat([l_pos_viewx, l_neg_viewx, l_context_bx], dim=1)
            logits_ax = torch.cat([l_pos_viewx, l_neg_viewx, l_context_ax], dim=1)

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

            logits_jx = torch.cat([l_pos_viewx, l_neg_viewx], dim=1)
            logits_mx = torch.cat([l_pos_viewx, l_neg_viewx], dim=1)
            logits_bx = torch.cat([l_pos_viewx, l_neg_viewx], dim=1)
            logits_ax = torch.cat([l_pos_viewx, l_neg_viewx], dim=1)

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

        logits_jx /= self.T
        logits_mx /= self.T
        logits_bx /= self.T
        logits_ax /= self.T
        

        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_motion = torch.topk(l_neg_motion, topk, dim=1)
        _, topkdix_bone = torch.topk(l_neg_bone, topk, dim=1)
        _, topkdix_acceleration = torch.topk(l_neg_acceleration, topk, dim=1)
        #_, topkdix_viewx = torch.topk(l_neg_viewx, topk, dim=1) #不打算这个视角给其他视角输出

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

        topk_onehot_jx = torch.zeros_like(l_neg)
        topk_onehot_mx = torch.zeros_like(l_neg_motion)
        topk_onehot_bx = torch.zeros_like(l_neg_bone)
        topk_onehot_ax = torch.zeros_like(l_neg_acceleration)

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
        topk_onehot_ba.scatter_(1, topkdix_bone, 1)
        topk_onehot_ab.scatter_(1, topkdix_acceleration, 1)#指定轴方向对应位置全替换成1

        topk_onehot_jx.scatter_(1, topkdix, 1)
        topk_onehot_mx.scatter_(1, topkdix_motion, 1)
        topk_onehot_bx.scatter_(1, topkdix_bone, 1)
        topk_onehot_ax.scatter_(1, topkdix_acceleration, 1)


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

            pos_mask_jx = torch.cat([torch.ones(topk_onehot_jx.size(0), 1).cuda(), topk_onehot_jx, topk_onehot_jx], dim=1)
            pos_mask_mx = torch.cat([torch.ones(topk_onehot_mx.size(0), 1).cuda(), topk_onehot_mx, topk_onehot_mx], dim=1)
            pos_mask_bx = torch.cat([torch.ones(topk_onehot_bx.size(0), 1).cuda(), topk_onehot_bx, topk_onehot_bx], dim=1)
            pos_mask_ax = torch.cat([torch.ones(topk_onehot_ax.size(0), 1).cuda(), topk_onehot_ax, topk_onehot_ax], dim=1)
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

            pos_mask_jx = torch.cat([torch.ones(topk_onehot_jx.size(0), 1).cuda(), topk_onehot_jx], dim=1)
            pos_mask_mx = torch.cat([torch.ones(topk_onehot_mx.size(0), 1).cuda(), topk_onehot_mx], dim=1)
            pos_mask_bx = torch.cat([torch.ones(topk_onehot_bx.size(0), 1).cuda(), topk_onehot_bx], dim=1)
            pos_mask_ax = torch.cat([torch.ones(topk_onehot_ax.size(0), 1).cuda(), topk_onehot_ax], dim=1)

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)
        self._dequeue_and_enqueue_acceleration(k_acceleration)
        self._dequeue_and_enqueue_viewx(k_viewx)
        #self.update_ptr(128)

        return logits_jm, logits_jb, logits_mj, logits_mb, logits_bj, logits_bm, logits_ja, logits_aj, logits_ma, logits_am, logits_ba, logits_ab,\
                logits_jx, logits_mx, logits_bx, logits_ax,\
                pos_mask_jm, pos_mask_jb, pos_mask_mj, pos_mask_mb, pos_mask_bj, pos_mask_bm, pos_mask_ja, pos_mask_aj, pos_mask_ma, pos_mask_am, pos_mask_ba, pos_mask_ab,\
                pos_mask_jx, pos_mask_mx, pos_mask_bx, pos_mask_ax
