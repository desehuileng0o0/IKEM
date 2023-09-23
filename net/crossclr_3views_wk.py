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
                                                         self.encoder_q.fc)
                self.encoder_k_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_k.fc)
                self.encoder_q_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q.fc)
                self.encoder_k_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):#将可迭代的对象打包成一个个元组
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient
            for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False#初始化key的参数并且关闭梯度

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
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
        self.queue_ptr_motion[0] = (self.queue_ptr_motion[0] + batch_size) % self.K
        self.queue_ptr_bone[0] = (self.queue_ptr_bone[0] + batch_size) % self.K

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
            elif view == 'all':
                return (self.encoder_q(im_q) + self.encoder_q_motion(im_q_motion) + self.encoder_q_bone(im_q_bone)) / 3.
            else:
                raise ValueError

        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]
        for i, f in enumerate(time_scale):
            im_k_motion[i, :, :, :, :] = f * im_k_motion[i, :, :, :, :]

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

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)#在dim=0的地方加上一个维度，128个embedding和它们的正对分别相乘
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])#和bank里的embedding分别相乘计算相似度

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)#按维度1将两个张量拼接，左右放的
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return logits, logits_motion, logits_bone, labels

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
    
        im_q_bone = torch.zeros_like(im_q)
        im_k_bone = torch.zeros_like(im_k)

        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        q_motion = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)

        q_bone = self.encoder_q_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder_motion()

            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
            
            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])#cross的话到这里都是一样的
        
        if context:
            l_context_jm = torch.einsum('nk,nk->nk', [l_neg, l_neg_motion])
            l_context_jb = torch.einsum('nk,nk->nk', [l_neg, l_neg_bone])
            l_context_mb = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_bone])#这里计算公式中的ss
            

            logits_jm = torch.cat([l_pos, l_neg, l_context_jm], dim=1)
            logits_jb = torch.cat([l_pos, l_neg, l_context_jb], dim=1)
            logits_mj = torch.cat([l_pos_motion, l_neg_motion, l_context_jm], dim=1)
            logits_mb = torch.cat([l_pos_motion, l_neg_motion, l_context_mb], dim=1)
            logits_bj = torch.cat([l_pos_bone, l_neg_bone, l_context_jb], dim=1)
            logits_bm = torch.cat([l_pos_bone, l_neg_bone, l_context_mb], dim=1)#拼接，分别是计算某个loss所需的全部信息
            logits_jr = torch.cat([l_pos, l_neg], dim=1)
            logits_mr = torch.cat([l_pos_motion, l_neg_motion], dim=1)
            logits_br = torch.cat([l_pos_bone, l_neg_bone], dim=1)

        else:
            logits_jm = torch.cat([l_pos, l_neg], dim=1)
            logits_jb = torch.cat([l_pos, l_neg], dim=1)
            logits_mj = torch.cat([l_pos_motion, l_neg_motion], dim=1)
            logits_mb = torch.cat([l_pos_motion, l_neg_motion], dim=1)
            logits_bj = torch.cat([l_pos_bone, l_neg_bone], dim=1)
            logits_bm = torch.cat([l_pos_bone, l_neg_bone], dim=1)

        logits_jm /= self.T
        logits_jb /= self.T
        logits_mj /= self.T
        logits_mb /= self.T
        logits_bj /= self.T
        logits_bm /= self.T
        logits_jr /= self.T
        logits_mr /= self.T
        logits_br /= self.T

        _, topkdix = torch.topk(l_neg, topk, dim=1)#从bank中为这128个数据分别找到最相似的topk个样本，返回的只有每一行最大的数的index
        _, topkdix_motion = torch.topk(l_neg_motion, topk, dim=1)
        _, topkdix_bone = torch.topk(l_neg_bone, topk, dim=1)
        topkdix_rand = torch.rand_like(topkdix, dtype=float)
        topkdix_rand = torch.floor(topkdix_rand*128).long()

        topk_onehot_jm = torch.zeros_like(l_neg)
        topk_onehot_jb = torch.zeros_like(l_neg)
        topk_onehot_mj = torch.zeros_like(l_neg_motion)
        topk_onehot_mb = torch.zeros_like(l_neg_motion)
        topk_onehot_bj = torch.zeros_like(l_neg_bone)
        topk_onehot_bm = torch.zeros_like(l_neg_bone)#维度一致，全是0
        topk_onehot_jr = torch.zeros_like(l_neg)
        topk_onehot_mr = torch.zeros_like(l_neg)
        topk_onehot_br = torch.zeros_like(l_neg)

        topk_onehot_jm.scatter_(1, topkdix_motion, 1)
        topk_onehot_jb.scatter_(1, topkdix_bone, 1)
        topk_onehot_mj.scatter_(1, topkdix, 1)
        topk_onehot_mb.scatter_(1, topkdix_bone, 1)
        topk_onehot_bj.scatter_(1, topkdix, 1)
        topk_onehot_bm.scatter_(1, topkdix_motion, 1)#对应位置替换成1，onehot编码常用操作，官方例子难看，但实际应用好懂
        topk_onehot_jr.scatter_(1, topkdix_rand, 1)
        topk_onehot_mr.scatter_(1, topkdix_rand, 1)
        topk_onehot_br.scatter_(1, topkdix_rand, 1)

        if context:
            pos_mask_jm = torch.cat([torch.ones(topk_onehot_jm.size(0), 1).cuda(), topk_onehot_jm, topk_onehot_jm], dim=1)#128行，1列的1和对应的onehot按列拼接两次，看后面干什么了
            pos_mask_jb = torch.cat([torch.ones(topk_onehot_jb.size(0), 1).cuda(), topk_onehot_jb, topk_onehot_jb], dim=1)
            pos_mask_mj = torch.cat([torch.ones(topk_onehot_mj.size(0), 1).cuda(), topk_onehot_mj, topk_onehot_mj], dim=1)
            pos_mask_mb = torch.cat([torch.ones(topk_onehot_mb.size(0), 1).cuda(), topk_onehot_mb, topk_onehot_mb], dim=1)
            pos_mask_bj = torch.cat([torch.ones(topk_onehot_bj.size(0), 1).cuda(), topk_onehot_bj, topk_onehot_bj], dim=1)
            pos_mask_bm = torch.cat([torch.ones(topk_onehot_bm.size(0), 1).cuda(), topk_onehot_bm, topk_onehot_bm], dim=1)
            pos_mask_jr = torch.cat([torch.ones(topk_onehot_jr.size(0), 1).cuda(), topk_onehot_jr], dim=1)
            pos_mask_mr = torch.cat([torch.ones(topk_onehot_mr.size(0), 1).cuda(), topk_onehot_mr], dim=1)
            pos_mask_br = torch.cat([torch.ones(topk_onehot_br.size(0), 1).cuda(), topk_onehot_br], dim=1)
        else:
            pos_mask_jm = torch.cat([torch.ones(topk_onehot_jm.size(0), 1).cuda(), topk_onehot_jm], dim=1)
            pos_mask_jb = torch.cat([torch.ones(topk_onehot_jb.size(0), 1).cuda(), topk_onehot_jb], dim=1)
            pos_mask_mj = torch.cat([torch.ones(topk_onehot_mj.size(0), 1).cuda(), topk_onehot_mj], dim=1)
            pos_mask_mb = torch.cat([torch.ones(topk_onehot_mb.size(0), 1).cuda(), topk_onehot_mb], dim=1)
            pos_mask_bj = torch.cat([torch.ones(topk_onehot_bj.size(0), 1).cuda(), topk_onehot_bj], dim=1)
            pos_mask_bm = torch.cat([torch.ones(topk_onehot_bm.size(0), 1).cuda(), topk_onehot_bm], dim=1)

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return logits_jm, logits_jb, logits_mj, logits_mb, logits_bj, logits_bm, pos_mask_jm, pos_mask_jb, pos_mask_mj, pos_mask_mb, pos_mask_bj, pos_mask_bm,\
        logits_jr, logits_mr, logits_br, pos_mask_jr, pos_mask_mr, pos_mask_br
