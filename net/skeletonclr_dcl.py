import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class SkeletonCLR(nn.Module):#这个名字不用改的原因是config里面给的实际上是文件路径。
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
                                          hidden_dim=hidden_dim, num_class=hidden_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=hidden_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.q_instance = nn.Sequential(
                                            nn.ReLU(), 
                                            nn.Linear(256, 128))
            self.k_instance = nn.Sequential(
                                            nn.ReLU(), 
                                            nn.Linear(256, 128))
            self.cluster1 = nn.Sequential(
                                        nn.ReLU(), 
                                        nn.Linear(256, 128))
            self.cluster2 = nn.Sequential(
                                        nn.ReLU(), 
                                        nn.Linear(256, 128))
            #self.k_cluster = nn.Sequential(nn.ReLU(), 
                                            #nn.Linear(256, 128))

            if mlp:  # hack: brute-force replacement记得到时候把mlp关了
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
            for param_q, param_k in zip(self.q_instance.parameters(), self.k_instance.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient
            #for param_q, param_k in zip(self.q_cluster.parameters(), self.k_cluster.parameters()):
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
    def _momentum_update_key_instance(self):
        """
        Momentum update of the key instance
        """
        for param_q, param_k in zip(self.q_instance.parameters(), self.k_instance.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    #@torch.no_grad()
    #def _momentum_update_key_cluster(self):
        #"""
        #Momentum update of the key cluster
        #"""
        #for param_q, param_k in zip(self.q_cluster.parameters(), self.k_cluster.parameters()):
            #param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

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
            else:
                raise ValueError
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q_instance = self.q_instance(q)
        q_instance = F.normalize(q_instance, dim=1)
        q_ocluster = self.cluster1(q)
        q_cluster = F.normalize(q_ocluster, dim=0)
        #q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_instance()
            #self._momentum_update_key_cluster()

            k = self.encoder_k(im_k)  # keys: NxC
            k_instance = self.k_instance(k)
            k_instance = F.normalize(k_instance, dim=1)
            #k_ocluster = self.cluster2(k)
            #k_cluster = F.normalize(k_ocluster, dim=0)
            #for scl
            kq = self.encoder_q(im_k)
            k_ocluster = self.cluster2(kq)
            k_cluster = F.normalize(k_ocluster, dim=0)

            #k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_instance = torch.einsum('nc,nc->n', [q_instance, k_instance]).unsqueeze(-1)# 128，1
        # negative logits: NxK
        l_neg_instance = torch.einsum('nc,ck->nk', [q_instance, self.queue.clone().detach()])# 128， 32768

        # logits: Nx(1+K)
        logits_instance = torch.cat([l_pos_instance, l_neg_instance], dim=1)
        logits_instance /= self.T

        # labels: positive key indicators
        labels_instance = torch.zeros(logits_instance.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_instance)

        l_pos_cluster = torch.einsum('nc,nc->c', [q_cluster, k_cluster]).unsqueeze(-1)#进行完这一步操作后正对间的相似度转化为一列

        queue_cluster = torch.cat([q_cluster, k_cluster], dim=1)#按列拼接

        l_q_neg_cluster = torch.einsum('nc,nk->ck', [q_cluster, queue_cluster])
        l_k_neg_cluster = torch.einsum('nc,nk->ck', [k_cluster, queue_cluster])

        logits_q_cluster = torch.zeros((l_q_neg_cluster.shape[0], l_q_neg_cluster.shape[1]-2), dtype=torch.long).cuda()
        logits_q_cluster = torch.cat([l_pos_cluster, logits_q_cluster], dim=1)
        logits_k_cluster = torch.zeros((l_k_neg_cluster.shape[0], l_k_neg_cluster.shape[1]-2), dtype=torch.long).cuda()
        logits_k_cluster = torch.cat([l_pos_cluster, logits_k_cluster], dim=1)
        for i in range(128):
            if i == 0:
                logits_q_cluster[i, 1:] = torch.cat([l_q_neg_cluster[i,1:128].unsqueeze(0),l_q_neg_cluster[i,129:256].unsqueeze(0)], dim=1)
                logits_k_cluster[i, 1:] = torch.cat([l_k_neg_cluster[i,1:128].unsqueeze(0),l_k_neg_cluster[i,129:256].unsqueeze(0)], dim=1)
            elif i == 127:
                logits_q_cluster[i, 1:] = torch.cat([l_q_neg_cluster[i,0:127].unsqueeze(0),l_q_neg_cluster[i,128:255].unsqueeze(0)], dim=1)
                logits_k_cluster[i, 1:] = torch.cat([l_k_neg_cluster[i,0:127].unsqueeze(0),l_k_neg_cluster[i,128:255].unsqueeze(0)], dim=1)
            else:
                logits_q_cluster[i, 1:] = torch.cat([l_q_neg_cluster[i,0:i].unsqueeze(0),l_q_neg_cluster[i,(i+1):(i+128)].unsqueeze(0),l_q_neg_cluster[i,(i+129):256].unsqueeze(0)], dim=1)
                logits_k_cluster[i, 1:] = torch.cat([l_k_neg_cluster[i,0:i].unsqueeze(0),l_k_neg_cluster[i,(i+1):(i+128)].unsqueeze(0),l_k_neg_cluster[i,(i+129):256].unsqueeze(0)], dim=1)
        logits_cluster = torch.cat([logits_q_cluster, logits_k_cluster], dim=0)
        #logits_cluster /= self.T
        labels_cluster = torch.zeros(logits_cluster.shape[0], dtype=torch.long).cuda()
        '''
        #entropy
        q_vnorm = torch.norm(q_ocluster, p=1, dim=0)
        k_vnorm = torch.norm(k_ocluster, p=1, dim=0)
        q_norm = torch.norm(q_vnorm, p=1)
        k_norm = torch.norm(k_vnorm, p=1)
        q_p = q_vnorm / q_norm
        k_p = k_vnorm / k_norm
        log_q = torch.log(q_p)
        log_k = torch.log(k_p)
        entropy = (q_p * log_q).sum(0) + (k_p * log_k).sum(0)
        '''

        return logits_instance, labels_instance, logits_cluster, labels_cluster
        