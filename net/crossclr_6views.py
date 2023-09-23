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
                 edge_importance_weighting=True,
                 teacher_student=False, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()#继承父类构造函数中的内容
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.teacher_student = teacher_student
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 8), (23, 8), (24, 12), (25, 12)]
        self.Axis = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 8), (23, 22), (24, 12), (25, 24)]

        if not self.pretrain and not self.teacher_student:
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
            self.encoder_q_rotation_axis = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_omega = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
        elif self.pretrain and not self.teacher_student:
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
            self.encoder_q_rotation_axis = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_rotation_axis = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_omega = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_omega = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
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
                self.encoder_q_rotation_axis.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_rotation_axis.fc)
                self.encoder_k_rotation_axis.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_k_rotation_axis.fc)
                self.encoder_q_omega.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_omega.fc)
                self.encoder_k_omega.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_k_omega.fc)
                self.mlp2j = nn.Sequential(nn.Linear(dim_mlp, dim_mlp*3),
                                                       nn.ReLU(),
                                                       nn.Linear(dim_mlp*3, dim_mlp*3))
                self.mlp2m = nn.Sequential(nn.Linear(dim_mlp, dim_mlp*3),
                                                       nn.ReLU(),
                                                       nn.Linear(dim_mlp*3, dim_mlp*3))
                self.mlp2b = nn.Sequential(nn.Linear(dim_mlp, dim_mlp*3),
                                                       nn.ReLU(),
                                                       nn.Linear(dim_mlp*3, dim_mlp*3))
                self.mlp2a = nn.Sequential(nn.Linear(dim_mlp, dim_mlp*3),
                                                       nn.ReLU(),
                                                       nn.Linear(dim_mlp*3, dim_mlp*3))
                self.mlp2r = nn.Sequential(nn.Linear(dim_mlp, dim_mlp*3),
                                                       nn.ReLU(),
                                                       nn.Linear(dim_mlp*3, dim_mlp*3))
                self.mlp2o = nn.Sequential(nn.Linear(dim_mlp, dim_mlp*3),
                                                       nn.ReLU(),
                                                       nn.Linear(dim_mlp*3, dim_mlp*3))

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
            for param_q, param_k in zip(self.encoder_q_rotation_axis.parameters(), self.encoder_k_rotation_axis.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_omega.parameters(), self.encoder_k_omega.parameters()):
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

            self.register_buffer("queue_rotation_axis", torch.randn(feature_dim, self.K))
            self.queue_rotation_axis = F.normalize(self.queue_rotation_axis, dim=0)
            self.register_buffer("queue_ptr_rotation_axis", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_omega", torch.randn(feature_dim, self.K))
            self.queue_omega = F.normalize(self.queue_omega, dim=0)
            self.register_buffer("queue_ptr_omega", torch.zeros(1, dtype=torch.long))
        
        elif self.pretrain and self.teacher_student:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            #为每个view的q和k都创建baseencoder。
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_acceleration = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_rotation_axis = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_omega = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_student_j = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim*6,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_student_m = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim*6,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_student_b = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim*6,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            if mlp:  # hack: brute-force replacement在每个全连接前又加了一层线性层
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_q_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_q_motion.fc)
                self.encoder_q_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_bone.fc)
                self.encoder_q_acceleration.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_acceleration.fc)
                self.encoder_q_rotation_axis.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_rotation_axis.fc)
                self.encoder_q_omega.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_omega.fc)
                self.encoder_student_j.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp*3),
                                                       nn.ReLU(),
                                                       nn.Linear(dim_mlp*3, dim_mlp*3))
                self.encoder_student_m.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp*3),
                                                       nn.ReLU(),
                                                       nn.Linear(dim_mlp*3, dim_mlp*3))
                self.encoder_student_b.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp*3),
                                                       nn.ReLU(),
                                                       nn.Linear(dim_mlp*3, dim_mlp*3))
            for param_q in self.encoder_q.parameters():
                param_q.requires_grad = False
            for param_q in self.encoder_q_motion.parameters():
                param_q.requires_grad = False
            for param_q in self.encoder_q_bone.parameters():
                param_q.requires_grad = False
            for param_q in self.encoder_q_acceleration.parameters():
                param_q.requires_grad = False
            for param_q in self.encoder_q_rotation_axis.parameters():
                param_q.requires_grad = False
            for param_q in self.encoder_q_omega.parameters():
                param_q.requires_grad = False

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

            self.register_buffer("queue_rotation_axis", torch.randn(feature_dim, self.K))
            self.queue_rotation_axis = F.normalize(self.queue_rotation_axis, dim=0)
            self.register_buffer("queue_ptr_rotation_axis", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_omega", torch.randn(feature_dim, self.K))
            self.queue_omega = F.normalize(self.queue_omega, dim=0)
            self.register_buffer("queue_ptr_omega", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_teacher", torch.randn(feature_dim*6, self.K))#模型训练时不会更新，quene里现在存的都是随机数
            self.queue_teacher = F.normalize(self.queue_teacher, dim=0)#按列，也就是每个feature都除以这个位置的范数
            self.register_buffer("queue_ptr_teacher", torch.zeros(1, dtype=torch.long))#这个暂时不知道
        
        else:
            #le student
            self.encoder_student_j = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_student_m = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_student_b = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)

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
    def _momentum_update_key_encoder_rotation_axis(self):
        for param_q, param_k in zip(self.encoder_q_rotation_axis.parameters(), self.encoder_k_rotation_axis.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_omega(self):
        for param_q, param_k in zip(self.encoder_q_omega.parameters(), self.encoder_k_omega.parameters()):
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
    def _dequeue_and_enqueue_rotation_axis(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_rotation_axis)
        gpu_index = keys.device.index
        self.queue_rotation_axis[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_omega(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_omega)
        gpu_index = keys.device.index
        self.queue_omega[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_teacher(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_teacher)
        gpu_index = keys.device.index
        self.queue_teacher[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
        self.queue_ptr_motion[0] = (self.queue_ptr_motion[0] + batch_size) % self.K
        self.queue_ptr_bone[0] = (self.queue_ptr_bone[0] + batch_size) % self.K
        self.queue_ptr_acceleration[0] = (self.queue_ptr_acceleration[0] + batch_size) % self.K
        self.queue_ptr_rotation_axis[0] = (self.queue_ptr_rotation_axis[0] + batch_size) % self.K
        self.queue_ptr_omega[0] = (self.queue_ptr_omega[0] + batch_size) % self.K

    @torch.no_grad()
    def update_ptr_teacher(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr_teacher[0] = (self.queue_ptr_teacher[0] + batch_size) % self.K

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
        im_q_bone[:, 0:3, :, 20, :] = F.normalize(torch.cross((im_q_bone[:, 0:3, :, 1, :]-im_q_bone[:, 0:3, :, 8, :]), (im_q_bone[:, 0:3, :, 4, :]-im_q_bone[:, 0:3, :, 8, :])))

        im_q_rotation_axis = torch.zeros_like(im_q)
        for b1, b2 in self.Axis:
            im_q_rotation_axis[:, 0:3, :, b1 - 1, :] = torch.cross(im_q_bone[:, 0:3, :, b1 - 1, :],im_q_bone[:, 0:3, :, b2 - 1, :])
        
        im_q_omega = torch.zeros_like(im_q)
        theta_q = torch.zeros_like(im_q)
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
        '''
        if not self.pretrain and not self.teacher_student:
            if view == 'joint':
                return self.encoder_q(im_q)
            elif view == 'motion':
                return self.encoder_q_motion(im_q_motion)
            elif view == 'bone':
                return self.encoder_q_bone(im_q_bone)
            elif view == 'acceleration':
                return self.encoder_q_acceleration(im_q_acceleration)
            elif view == 'rotation_axis':
                return self.encoder_q_rotation_axis(im_q_rotation_axis)
            elif view == 'omega':
                return self.encoder_q_omega(im_q_omega)
            elif view == 'all':
                return (self.encoder_q(im_q) + self.encoder_q_motion(im_q_motion) + self.encoder_q_bone(im_q_bone) + self.encoder_q_acceleration(im_q_acceleration) + self.encoder_q_rotation_axis(im_q_rotation_axis) + self.encoder_q_omega(im_q_omega)) / 6.
            else:
                raise ValueError
        '''
        if not self.pretrain and not self.teacher_student:
            if view == 'joint':
                op,_ = self.encoder_q(im_q)
                return op
            elif view == 'motion':
                op,_ = self.encoder_q_motion(im_q_motion)
                return op
            elif view == 'bone':
                op,_ = self.encoder_q_bone(im_q_bone)
                return op
            elif view == 'acceleration':
                op,_ = self.encoder_q_acceleration(im_q_acceleration)
                return op
            elif view == 'rotation_axis':
                op,_ = self.encoder_q_rotation_axis(im_q_rotation_axis)
                return op
            elif view == 'omega':
                op,_ = self.encoder_q_omega(im_q_omega)
                return op
            elif view == 'all':
                op1,_ = self.encoder_q(im_q)
                op2,_ = self.encoder_q_motion(im_q_motion)
                op3,_ = self.encoder_q_bone(im_q_bone)
                op4,_ = self.encoder_q_acceleration(im_q_acceleration)
                op5,_ = self.encoder_q_rotation_axis(im_q_rotation_axis)
                op6,_ = self.encoder_q_omega(im_q_omega)
                return (op1+op2+op3+op4+op5+op6)/6.
            elif view == '3views':
                op1,_ = self.encoder_q(im_q)
                op2,_ = self.encoder_q_motion(im_q_motion)
                op3,_ = self.encoder_q_bone(im_q_bone)
                return (op1+op2+op3)/3.
            else:
                raise ValueError
            
        if not self.pretrain and self.teacher_student:
            #le student
            if view == 'all':
                op1,_ = self.encoder_student_j(im_q)
                op2,_ = self.encoder_student_m(im_q_motion)
                op3,_ = self.encoder_student_b(im_q_bone)
                return (op1+op2+op3)/3.
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
        im_k_bone[:, 0:3, :, 20, :] = F.normalize(torch.cross((im_k_bone[:, 0:3, :, 1, :]-im_k_bone[:, 0:3, :, 8, :]), (im_k_bone[:, 0:3, :, 4, :]-im_k_bone[:, 0:3, :, 8, :])))

        im_k_rotation_axis = torch.zeros_like(im_k)
        for b1, b2 in self.Axis:
            im_k_rotation_axis[:, 0:3, :, b1 - 1, :] = torch.cross(im_k_bone[:, 0:3, :, b1 - 1, :],im_k_bone[:, 0:3, :, b2 - 1, :])

        im_k_omega = torch.zeros_like(im_k)
        theta_k = torch.zeros_like(im_k)
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
        q,_ = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)#第二个维度是坐标，归一化

        q_motion,_ = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)

        q_bone,_ = self.encoder_q_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)

        q_acceleration,_ = self.encoder_q_acceleration(im_q_acceleration)
        q_acceleration = F.normalize(q_acceleration, dim=1)

        q_rotation_axis,_ = self.encoder_q_rotation_axis(im_q_rotation_axis)
        q_rotation_axis = F.normalize(q_rotation_axis, dim=1)

        q_omega,_ = self.encoder_q_omega(im_q_omega)
        q_omega = F.normalize(q_omega, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()
            self._momentum_update_key_encoder_acceleration()
            self._momentum_update_key_encoder_rotation_axis()
            self._momentum_update_key_encoder_omega()

            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            k_motion,_ = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone,_ = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

            k_acceleration,_ = self.encoder_k_acceleration(im_k_acceleration)
            k_acceleration = F.normalize(k_acceleration, dim=1)

            k_rotation_axis,_ = self.encoder_k_rotation_axis(im_k_rotation_axis)
            k_rotation_axis = F.normalize(k_rotation_axis, dim=1)

            k_omega,_ = self.encoder_k_omega(im_k_omega)
            k_omega = F.normalize(k_omega, dim=1)

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

        l_pos_rotation_axis = torch.einsum('nc,nc->n', [q_rotation_axis, k_rotation_axis]).unsqueeze(-1)
        l_neg_rotation_axis = torch.einsum('nc,ck->nk', [q_rotation_axis, self.queue_rotation_axis.clone().detach()])

        l_pos_omega = torch.einsum('nc,nc->n', [q_omega, k_omega]).unsqueeze(-1)
        l_neg_omega = torch.einsum('nc,ck->nk', [q_omega, self.queue_omega.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)#按维度1将两个张量拼接
        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)
        logits_acceleration = torch.cat([l_pos_acceleration, l_neg_acceleration], dim=1)
        logits_rotation_axis = torch.cat([l_pos_rotation_axis, l_neg_rotation_axis], dim=1)
        logits_omega = torch.cat([l_pos_omega, l_neg_omega], dim=1)

        # apply temperature
        logits /= self.T
        logits_motion /= self.T
        logits_bone /= self.T
        logits_acceleration /= self.T
        logits_rotation_axis /= self.T
        logits_omega /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)
        self._dequeue_and_enqueue_acceleration(k_acceleration)
        self._dequeue_and_enqueue_rotation_axis(k_rotation_axis)
        self._dequeue_and_enqueue_omega(k_omega)

        return logits, logits_motion, logits_bone, logits_acceleration, logits_rotation_axis, logits_omega, labels

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

        im_q_bone[:, 0:3, :, 20, :] = F.normalize(torch.cross((im_q_bone[:, 0:3, :, 1, :]-im_q_bone[:, 0:3, :, 8, :]), (im_q_bone[:, 0:3, :, 4, :]-im_q_bone[:, 0:3, :, 8, :])))
        im_k_bone[:, 0:3, :, 20, :] = F.normalize(torch.cross((im_k_bone[:, 0:3, :, 1, :]-im_k_bone[:, 0:3, :, 8, :]), (im_k_bone[:, 0:3, :, 4, :]-im_k_bone[:, 0:3, :, 8, :])))
        
        im_q_rotation_axis = torch.zeros_like(im_q)
        im_k_rotation_axis = torch.zeros_like(im_k)
        for b1, b2 in self.Axis:
            im_q_rotation_axis[:, 0:3, :, b1 - 1, :] = torch.cross(im_q_bone[:, 0:3, :, b1 - 1, :],im_q_bone[:, 0:3, :, b2 - 1, :])
            im_k_rotation_axis[:, 0:3, :, b1 - 1, :] = torch.cross(im_k_bone[:, 0:3, :, b1 - 1, :],im_k_bone[:, 0:3, :, b2 - 1, :])
        
        im_q_omega = torch.zeros_like(im_q)
        im_k_omega = torch.zeros_like(im_k)
        theta_q = torch.zeros_like(im_q)
        theta_k = torch.zeros_like(im_k)
        norm_q_bone = torch.zeros_like(im_q_bone)
        norm_q_bone = F.normalize(im_q_bone,dim=1)
        norm_k_bone = torch.zeros_like(im_k_bone)
        norm_k_bone = F.normalize(im_k_bone,dim=1)
        cp_q = torch.zeros_like(im_q_rotation_axis)
        cp_k = torch.zeros_like(im_k_rotation_axis)#这是归一化后的Bone的叉积
        norm_q_rotation_axis = torch.zeros_like(im_q_rotation_axis)
        norm_k_rotation_axis = torch.zeros_like(im_k_rotation_axis)

        for b1, b2 in self.Axis:
            cp_q[:, 0:3, :, b1 - 1, :] = torch.cross(norm_q_bone[:, 0:3, :, b1 - 1, :],norm_q_bone[:, 0:3, :, b2 - 1, :])
            cp_k[:, 0:3, :, b1 - 1, :] = torch.cross(norm_k_bone[:, 0:3, :, b1 - 1, :],norm_k_bone[:, 0:3, :, b2 - 1, :])
        for b1, b2 in self.Axis:
            theta_q[:, 0, :, b1 - 1, :] = torch.atan2(torch.norm(cp_q[:, 0:3, :, b1 - 1, :],dim=1),torch.einsum('abcd,abcd->acd',norm_q_bone[:, 0:3, :, b1 - 1, :],norm_q_bone[:, 0:3, :, b2 - 1, :]))
            theta_k[:, 0, :, b1 - 1, :] = torch.atan2(torch.norm(cp_k[:, 0:3, :, b1 - 1, :],dim=1),torch.einsum('abcd,abcd->acd',norm_k_bone[:, 0:3, :, b1 - 1, :],norm_k_bone[:, 0:3, :, b2 - 1, :]))
        norm_q_rotation_axis = F.normalize(im_q_rotation_axis,dim=1)
        norm_k_rotation_axis = F.normalize(im_k_rotation_axis,dim=1)
                
        for i in range(3):
            im_q_omega[:, i, :-1, :, :] = (theta_q[:, 0, 1:, :, :] - theta_q[:, 0, :-1, :, :]) * norm_q_rotation_axis[:, i, :-1, :, :]
            im_k_omega[:, i, :-1, :, :] = (theta_k[:, 0, 1:, :, :] - theta_k[:, 0, :-1, :, :]) * norm_k_rotation_axis[:, i, :-1, :, :]

        for i, f in enumerate(time_scale):
            im_q_omega[i, :, :, :, :] = f * im_q_omega[i, :, :, :, :]
            im_k_omega[i, :, :, :, :] = f * im_k_omega[i, :, :, :, :]
        
        
        if self.pretrain and self.teacher_student:
            #pretrain student
            #calculate features
            with torch.no_grad():
                q,_ = self.encoder_q(im_q)
                q = F.normalize(q, dim=1)
            
                q_motion,_ = self.encoder_q_motion(im_q_motion)
                q_motion = F.normalize(q_motion, dim=1)

                q_bone,_ = self.encoder_q_bone(im_q_bone)
                q_bone = F.normalize(q_bone, dim=1)

                q_acceleration,_ = self.encoder_q_acceleration(im_q_acceleration)
                q_acceleration = F.normalize(q_acceleration, dim=1)

                q_rotation_axis,_ = self.encoder_q_rotation_axis(im_q_rotation_axis)
                q_rotation_axis = F.normalize(q_rotation_axis, dim=1)

                q_omega,_ = self.encoder_q_omega(im_q_omega)
                q_omega = F.normalize(q_omega, dim=1)
                #calculate teacher feature
                #teacher = (q + q_motion + q_bone + q_acceleration + q_rotation_axis + q_omega) / 6.
                teacher = torch.cat((q, q_motion, q_bone, q_acceleration, q_rotation_axis, q_omega), 1)
                teacher = F.normalize(teacher, dim=1)

            #im_viewx = torch.cat((im_k, im_k_motion, im_k_bone, im_k_acceleration, im_k_rotation_axis, im_k_omega), 1)
            student_j,_ = self.encoder_student_j(im_k)
            student_j = F.normalize(student_j, dim=1)

            student_m,_ = self.encoder_student_m(im_k_motion)
            student_m = F.normalize(student_m, dim=1)

            student_b,_ = self.encoder_student_b(im_k_bone)
            student_b = F.normalize(student_b, dim=1)

            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])
            l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])
            l_neg_acceleration = torch.einsum('nc,ck->nk', [q_acceleration, self.queue_acceleration.clone().detach()])
            l_neg_rotation_axis = torch.einsum('nc,ck->nk', [q_rotation_axis, self.queue_rotation_axis.clone().detach()])
            l_neg_omega = torch.einsum('nc,ck->nk', [q_omega, self.queue_bone.clone().detach()])

            l_pos_sj = torch.einsum('nc,nc->n', [student_j, teacher]).unsqueeze(-1)
            l_neg_sj = torch.einsum('nc,ck->nk', [student_j, self.queue_teacher.clone().detach()])

            l_pos_sm = torch.einsum('nc,nc->n', [student_m, teacher]).unsqueeze(-1)
            l_neg_sm = torch.einsum('nc,ck->nk', [student_m, self.queue_teacher.clone().detach()])

            l_pos_sb = torch.einsum('nc,nc->n', [student_b, teacher]).unsqueeze(-1)
            l_neg_sb = torch.einsum('nc,ck->nk', [student_b, self.queue_teacher.clone().detach()])

            l_context_jsj = torch.einsum('nk,nk->nk', [l_neg, l_neg_sj])
            l_context_msj = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_sj])
            l_context_bsj = torch.einsum('nk,nk->nk', [l_neg_bone, l_neg_sj])
            l_context_asj = torch.einsum('nk,nk->nk', [l_neg_acceleration, l_neg_sj])
            l_context_rsj = torch.einsum('nk,nk->nk', [l_neg_rotation_axis, l_neg_sj])
            l_context_osj = torch.einsum('nk,nk->nk', [l_neg_omega, l_neg_sj])

            l_context_jsm = torch.einsum('nk,nk->nk', [l_neg, l_neg_sm])
            l_context_msm = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_sm])
            l_context_bsm = torch.einsum('nk,nk->nk', [l_neg_bone, l_neg_sm])
            l_context_asm = torch.einsum('nk,nk->nk', [l_neg_acceleration, l_neg_sm])
            l_context_rsm = torch.einsum('nk,nk->nk', [l_neg_rotation_axis, l_neg_sm])
            l_context_osm = torch.einsum('nk,nk->nk', [l_neg_omega, l_neg_sm])

            l_context_jsb = torch.einsum('nk,nk->nk', [l_neg, l_neg_sb])
            l_context_msb = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_sb])
            l_context_bsb = torch.einsum('nk,nk->nk', [l_neg_bone, l_neg_sb])
            l_context_asb = torch.einsum('nk,nk->nk', [l_neg_acceleration, l_neg_sb])
            l_context_rsb = torch.einsum('nk,nk->nk', [l_neg_rotation_axis, l_neg_sb])
            l_context_osb = torch.einsum('nk,nk->nk', [l_neg_omega, l_neg_sb])

            logits_jsj = torch.cat([l_pos_sj, l_neg_sj, l_context_jsj], dim=1)
            logits_msj = torch.cat([l_pos_sj, l_neg_sj, l_context_msj], dim=1)
            logits_bsj = torch.cat([l_pos_sj, l_neg_sj, l_context_bsj], dim=1)
            logits_asj = torch.cat([l_pos_sj, l_neg_sj, l_context_asj], dim=1)
            logits_rsj = torch.cat([l_pos_sj, l_neg_sj, l_context_rsj], dim=1)
            logits_osj = torch.cat([l_pos_sj, l_neg_sj, l_context_osj], dim=1)

            logits_jsm = torch.cat([l_pos_sm, l_neg_sm, l_context_jsm], dim=1)
            logits_msm = torch.cat([l_pos_sm, l_neg_sm, l_context_msm], dim=1)
            logits_bsm = torch.cat([l_pos_sm, l_neg_sm, l_context_bsm], dim=1)
            logits_asm = torch.cat([l_pos_sm, l_neg_sm, l_context_asm], dim=1)
            logits_rsm = torch.cat([l_pos_sm, l_neg_sm, l_context_rsm], dim=1)
            logits_osm = torch.cat([l_pos_sm, l_neg_sm, l_context_osm], dim=1)

            logits_jsb = torch.cat([l_pos_sb, l_neg_sb, l_context_jsb], dim=1)
            logits_msb = torch.cat([l_pos_sb, l_neg_sb, l_context_msb], dim=1)
            logits_bsb = torch.cat([l_pos_sb, l_neg_sb, l_context_bsb], dim=1)
            logits_asb = torch.cat([l_pos_sb, l_neg_sb, l_context_asb], dim=1)
            logits_rsb = torch.cat([l_pos_sb, l_neg_sb, l_context_rsb], dim=1)
            logits_osb = torch.cat([l_pos_sb, l_neg_sb, l_context_osb], dim=1)

            logits_jsj /= self.T
            logits_msj /= self.T
            logits_bsj /= self.T
            logits_asj /= self.T
            logits_rsj /= self.T
            logits_osj /= self.T
            logits_jsm /= self.T
            logits_msm /= self.T
            logits_bsm /= self.T
            logits_asm /= self.T
            logits_rsm /= self.T
            logits_osm /= self.T
            logits_jsb /= self.T
            logits_msb /= self.T
            logits_bsb /= self.T
            logits_asb /= self.T
            logits_rsb /= self.T
            logits_osb /= self.T

            _, topkdix = torch.topk(l_neg, topk, dim=1)
            _, topkdix_motion = torch.topk(l_neg_motion, topk, dim=1)
            _, topkdix_bone = torch.topk(l_neg_bone, topk, dim=1)
            _, topkdix_acceleration = torch.topk(l_neg_acceleration, topk, dim=1)
            _, topkdix_rotation_axis = torch.topk(l_neg_rotation_axis, topk, dim=1)
            _, topkdix_omega = torch.topk(l_neg_omega, topk, dim=1)

            topk_onehot_jx = torch.zeros_like(l_neg)
            topk_onehot_mx = torch.zeros_like(l_neg_motion)
            topk_onehot_bx = torch.zeros_like(l_neg_bone)
            topk_onehot_ax = torch.zeros_like(l_neg_acceleration)
            topk_onehot_rx = torch.zeros_like(l_neg_rotation_axis)
            topk_onehot_ox = torch.zeros_like(l_neg_omega)

            topk_onehot_jx.scatter_(1, topkdix, 1)
            topk_onehot_mx.scatter_(1, topkdix_motion, 1)
            topk_onehot_bx.scatter_(1, topkdix_bone, 1)
            topk_onehot_ax.scatter_(1, topkdix_acceleration, 1)
            topk_onehot_rx.scatter_(1, topkdix_rotation_axis, 1)
            topk_onehot_ox.scatter_(1, topkdix_omega, 1)

            pos_mask_jx = torch.cat([torch.ones(topk_onehot_jx.size(0), 1).cuda(), topk_onehot_jx, topk_onehot_jx], dim=1)
            pos_mask_mx = torch.cat([torch.ones(topk_onehot_mx.size(0), 1).cuda(), topk_onehot_mx, topk_onehot_mx], dim=1)
            pos_mask_bx = torch.cat([torch.ones(topk_onehot_bx.size(0), 1).cuda(), topk_onehot_bx, topk_onehot_bx], dim=1)
            pos_mask_ax = torch.cat([torch.ones(topk_onehot_ax.size(0), 1).cuda(), topk_onehot_ax, topk_onehot_ax], dim=1)
            pos_mask_rx = torch.cat([torch.ones(topk_onehot_rx.size(0), 1).cuda(), topk_onehot_rx, topk_onehot_rx], dim=1)
            pos_mask_ox = torch.cat([torch.ones(topk_onehot_ox.size(0), 1).cuda(), topk_onehot_ox, topk_onehot_ox], dim=1)

            self._dequeue_and_enqueue(q)
            self._dequeue_and_enqueue_motion(q_motion)
            self._dequeue_and_enqueue_bone(q_bone)
            self._dequeue_and_enqueue_acceleration(q_acceleration)
            self._dequeue_and_enqueue_rotation_axis(q_rotation_axis)
            self._dequeue_and_enqueue_omega(q_omega)
            self._dequeue_and_enqueue_teacher(teacher)

            return logits_jsj, logits_msj, logits_bsj, logits_asj, logits_rsj, logits_osj,\
                logits_jsm, logits_msm, logits_bsm, logits_asm, logits_rsm, logits_osm,\
                logits_jsb, logits_msb, logits_bsb, logits_asb, logits_rsb, logits_osb,\
                pos_mask_jx, pos_mask_mx, pos_mask_bx, pos_mask_ax, pos_mask_rx, pos_mask_ox



        q,qjf = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)
        qj_768 = self.mlp2j(qjf)
        qj_768 = F.normalize(qj_768, dim=1)

        q_motion,qmf = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)
        qm_768 = self.mlp2m(qmf)
        qm_768 = F.normalize(qm_768, dim=1)

        q_bone,qbf = self.encoder_q_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)
        qb_768 = self.mlp2b(qbf)
        qb_768 = F.normalize(qb_768, dim=1)

        q_acceleration,qaf = self.encoder_q_acceleration(im_q_acceleration)
        q_acceleration = F.normalize(q_acceleration, dim=1)
        qa_768 = self.mlp2a(qaf)
        qa_768 = F.normalize(qa_768, dim=1)

        q_rotation_axis,qrf = self.encoder_q_rotation_axis(im_q_rotation_axis)
        q_rotation_axis = F.normalize(q_rotation_axis, dim=1)
        qr_768 = self.mlp2r(qrf)
        qr_768 = F.normalize(qr_768, dim=1)

        q_omega,qof = self.encoder_q_omega(im_q_omega)
        q_omega = F.normalize(q_omega, dim=1)
        qo_768 = self.mlp2o(qof)
        qo_768 = F.normalize(qo_768, dim=1)
        
        with torch.no_grad():  # no gradient to keys
            #self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            #self._momentum_update_key_encoder_bone()
            self._momentum_update_key_encoder_acceleration()
            #self._momentum_update_key_encoder_rotation_axis()
            self._momentum_update_key_encoder_omega()

            k,_ = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
            
            k_motion,_ = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone,_ = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

            k_acceleration,_ = self.encoder_k_acceleration(im_k_acceleration)
            k_acceleration = F.normalize(k_acceleration, dim=1)

            k_rotation_axis,_ = self.encoder_k_rotation_axis(im_k_rotation_axis)
            k_rotation_axis = F.normalize(k_rotation_axis, dim=1)

            k_omega,_ = self.encoder_k_omega(im_k_omega)
            k_omega = F.normalize(k_omega, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)#两组向量对应位置相乘,qk分别有128个128维向量，向量点乘得到128个相似度
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])#矩阵乘法，克隆张量并避免梯度传播，128个向量和bank里的32768个分别相乘

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])

        l_pos_acceleration = torch.einsum('nc,nc->n', [q_acceleration, k_acceleration]).unsqueeze(-1)
        l_neg_acceleration = torch.einsum('nc,ck->nk', [q_acceleration, self.queue_acceleration.clone().detach()])

        l_pos_rotation_axis = torch.einsum('nc,nc->n', [q_rotation_axis, k_rotation_axis]).unsqueeze(-1)
        l_neg_rotation_axis = torch.einsum('nc,ck->nk', [q_rotation_axis, self.queue_rotation_axis.clone().detach()])

        l_pos_omega = torch.einsum('nc,nc->n', [q_omega, k_omega]).unsqueeze(-1)
        l_neg_omega = torch.einsum('nc,ck->nk', [q_omega, self.queue_omega.clone().detach()])
        
        if context:
            l_context_jm = torch.einsum('nk,nk->nk', [l_neg, l_neg_motion])#这里是论文中的ss，两个view下和bank中embedding相似度的乘积
            l_context_jb = torch.einsum('nk,nk->nk', [l_neg, l_neg_bone])
            l_context_mb = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_bone])
            l_context_ja = torch.einsum('nk,nk->nk', [l_neg, l_neg_acceleration])
            l_context_ma = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_acceleration])
            l_context_ba = torch.einsum('nk,nk->nk', [l_neg_bone, l_neg_acceleration])

            l_context_mr = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_rotation_axis])
            l_context_jr = torch.einsum('nk,nk->nk', [l_neg, l_neg_rotation_axis])
            l_context_br = torch.einsum('nk,nk->nk', [l_neg_bone, l_neg_rotation_axis])
            l_context_ar = torch.einsum('nk,nk->nk', [l_neg_acceleration, l_neg_rotation_axis])
            
            l_context_jo = torch.einsum('nk,nk->nk', [l_neg, l_neg_omega])
            l_context_mo = torch.einsum('nk,nk->nk', [l_neg_motion, l_neg_omega])
            l_context_bo = torch.einsum('nk,nk->nk', [l_neg_bone, l_neg_omega])
            l_context_ao = torch.einsum('nk,nk->nk', [l_neg_acceleration, l_neg_omega])
            l_context_ro = torch.einsum('nk,nk->nk', [l_neg_rotation_axis, l_neg_omega])

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

            logits_jr = torch.cat([l_pos, l_neg, l_context_jr], dim=1)
            logits_rj = torch.cat([l_pos_rotation_axis, l_neg_rotation_axis, l_context_jr], dim=1)
            logits_mr = torch.cat([l_pos_motion, l_neg_motion, l_context_mr], dim=1)
            logits_rm = torch.cat([l_pos_rotation_axis, l_neg_rotation_axis, l_context_mr], dim=1)
            logits_br = torch.cat([l_pos_bone, l_neg_bone, l_context_br], dim=1)
            logits_rb = torch.cat([l_pos_rotation_axis, l_neg_rotation_axis, l_context_br], dim=1)
            logits_ar = torch.cat([l_pos_acceleration, l_neg_acceleration, l_context_ar], dim=1)
            logits_ra = torch.cat([l_pos_rotation_axis, l_neg_rotation_axis, l_context_ar], dim=1)

            logits_jo = torch.cat([l_pos, l_neg, l_context_jo], dim=1)
            logits_oj = torch.cat([l_pos_omega, l_neg_omega, l_context_jo], dim=1)
            logits_mo = torch.cat([l_pos_motion, l_neg_motion, l_context_mo], dim=1)
            logits_om = torch.cat([l_pos_omega, l_neg_omega, l_context_mo], dim=1)
            logits_bo = torch.cat([l_pos_bone, l_neg_bone, l_context_bo], dim=1)
            logits_ob = torch.cat([l_pos_omega, l_neg_omega, l_context_bo], dim=1)
            logits_ao = torch.cat([l_pos_acceleration, l_neg_acceleration, l_context_ao], dim=1)
            logits_oa = torch.cat([l_pos_omega, l_neg_omega, l_context_ao], dim=1)
            logits_ro = torch.cat([l_pos_rotation_axis, l_neg_rotation_axis, l_context_ro], dim=1)
            logits_or = torch.cat([l_pos_omega, l_neg_omega, l_context_ro], dim=1)

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
            logits_jr = torch.cat([l_pos, l_neg], dim=1)
            logits_rj = torch.cat([l_pos_rotation_axis, l_neg_rotation_axis], dim=1)
            logits_mr = torch.cat([l_pos_motion, l_neg_motion], dim=1)
            logits_rm = torch.cat([l_pos_rotation_axis, l_neg_rotation_axis], dim=1)
            logits_br = torch.cat([l_pos_bone, l_neg_bone], dim=1)
            logits_rb = torch.cat([l_pos_rotation_axis, l_neg_rotation_axis], dim=1)
            logits_ar = torch.cat([l_pos_acceleration, l_neg_acceleration], dim=1)
            logits_ra = torch.cat([l_pos_rotation_axis, l_neg_rotation_axis], dim=1)

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
        logits_jr /= self.T
        logits_rj /= self.T
        logits_mr /= self.T
        logits_rm /= self.T
        logits_br /= self.T
        logits_rb /= self.T
        logits_ar /= self.T
        logits_ra /= self.T
        logits_jo /= self.T
        logits_oj /= self.T
        logits_mo /= self.T
        logits_om /= self.T
        logits_bo /= self.T
        logits_ob /= self.T
        logits_ao /= self.T
        logits_oa /= self.T
        logits_ro /= self.T
        logits_or /= self.T
        

        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_motion = torch.topk(l_neg_motion, topk, dim=1)
        _, topkdix_bone = torch.topk(l_neg_bone, topk, dim=1)
        _, topkdix_acceleration = torch.topk(l_neg_acceleration, topk, dim=1)
        _, topkdix_rotation_axis = torch.topk(l_neg_rotation_axis, topk, dim=1)
        _, topkdix_omega = torch.topk(l_neg_omega, topk, dim=1)
        #哪个视角要往外传知识就用哪个

        topk_onehot_j = torch.zeros_like(l_neg)
        topk_onehot_m = torch.zeros_like(l_neg_motion)
        topk_onehot_b = torch.zeros_like(l_neg_bone)
        topk_onehot_a = torch.zeros_like(l_neg_acceleration)
        topk_onehot_r = torch.zeros_like(l_neg_rotation_axis)
        topk_onehot_o = torch.zeros_like(l_neg_omega)

        topk_onehot_j.scatter_(1, topkdix, 1)
        topk_onehot_m.scatter_(1, topkdix_motion, 1)
        topk_onehot_b.scatter_(1, topkdix_bone, 1)
        topk_onehot_a.scatter_(1, topkdix_acceleration, 1)
        topk_onehot_r.scatter_(1, topkdix_rotation_axis, 1)
        topk_onehot_o.scatter_(1, topkdix_omega, 1)
        

        if context:
            pos_mask_j = torch.cat([torch.ones(topk_onehot_j.size(0), 1).cuda(), topk_onehot_j, topk_onehot_j], dim=1)
            pos_mask_m = torch.cat([torch.ones(topk_onehot_m.size(0), 1).cuda(), topk_onehot_m, topk_onehot_m], dim=1)
            pos_mask_b = torch.cat([torch.ones(topk_onehot_b.size(0), 1).cuda(), topk_onehot_b, topk_onehot_b], dim=1)
            pos_mask_a = torch.cat([torch.ones(topk_onehot_a.size(0), 1).cuda(), topk_onehot_a, topk_onehot_a], dim=1)
            pos_mask_r = torch.cat([torch.ones(topk_onehot_r.size(0), 1).cuda(), topk_onehot_r, topk_onehot_r], dim=1)
            pos_mask_o = torch.cat([torch.ones(topk_onehot_o.size(0), 1).cuda(), topk_onehot_o, topk_onehot_o], dim=1)
        else:
            pos_mask_j = torch.cat([torch.ones(topk_onehot_j.size(0), 1).cuda(), topk_onehot_j], dim=1)
            pos_mask_m = torch.cat([torch.ones(topk_onehot_m.size(0), 1).cuda(), topk_onehot_m], dim=1)
            pos_mask_b = torch.cat([torch.ones(topk_onehot_b.size(0), 1).cuda(), topk_onehot_b], dim=1)
            pos_mask_a = torch.cat([torch.ones(topk_onehot_a.size(0), 1).cuda(), topk_onehot_a], dim=1)
            pos_mask_r = torch.cat([torch.ones(topk_onehot_r.size(0), 1).cuda(), topk_onehot_r], dim=1)
            pos_mask_o = torch.cat([torch.ones(topk_onehot_o.size(0), 1).cuda(), topk_onehot_o], dim=1)

        #cat后的对比loss
        cat_k = torch.cat((k, k_motion, k_bone, k_acceleration, k_rotation_axis, k_omega), 1)
        cat_k = F.normalize(cat_k, dim=1)
        cat_queue = torch.cat((self.queue.clone().detach(),
                               self.queue_motion.clone().detach(),
                               self.queue_bone.clone().detach(),
                               self.queue_acceleration.clone().detach(),
                               self.queue_rotation_axis.clone().detach(),
                               self.queue_omega.clone().detach()),0)
        cat_queue = F.normalize(cat_queue, dim=0)
        lpj = torch.einsum('nc,nc->n', [qj_768, cat_k]).unsqueeze(-1)#这里teacher是所有k的拼接
        lnj = torch.einsum('nc,ck->nk', [qj_768, cat_queue])#负对是所有queue的拼接

        lpm = torch.einsum('nc,nc->n', [qm_768, cat_k]).unsqueeze(-1)#这里teacher是所有k的拼接
        lnm = torch.einsum('nc,ck->nk', [qm_768, cat_queue])#负对是所有queue的拼接

        lpb = torch.einsum('nc,nc->n', [qb_768, cat_k]).unsqueeze(-1)#这里teacher是所有k的拼接
        lnb = torch.einsum('nc,ck->nk', [qb_768, cat_queue])#负对是所有queue的拼接

        lpa = torch.einsum('nc,nc->n', [qa_768, cat_k]).unsqueeze(-1)#这里teacher是所有k的拼接
        lna = torch.einsum('nc,ck->nk', [qa_768, cat_queue])#负对是所有queue的拼接

        lpr = torch.einsum('nc,nc->n', [qr_768, cat_k]).unsqueeze(-1)#这里teacher是所有k的拼接
        lnr = torch.einsum('nc,ck->nk', [qr_768, cat_queue])#负对是所有queue的拼接

        lpo = torch.einsum('nc,nc->n', [qo_768, cat_k]).unsqueeze(-1)#这里teacher是所有k的拼接
        lno = torch.einsum('nc,ck->nk', [qo_768, cat_queue])#负对是所有queue的拼接


        logits_j = torch.cat([lpj, lnj], dim=1)
        logits_m = torch.cat([lpm, lnm], dim=1)
        logits_b = torch.cat([lpb, lnb], dim=1)
        logits_a = torch.cat([lpa, lna], dim=1)
        logits_r = torch.cat([lpr, lnr], dim=1)
        logits_o = torch.cat([lpo, lno], dim=1)

        logits_j /= self.T
        logits_m /= self.T
        logits_b /= self.T
        logits_a /= self.T
        logits_r /= self.T
        logits_o /= self.T

        labels = torch.zeros(logits_j.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)
        self._dequeue_and_enqueue_acceleration(k_acceleration)
        self._dequeue_and_enqueue_rotation_axis(k_rotation_axis)
        self._dequeue_and_enqueue_omega(k_omega)

        return logits_jm, logits_jb, logits_mj, logits_mb, logits_bj, logits_bm, logits_ja, logits_aj, logits_ma, logits_am, logits_ba, logits_ab,\
            logits_jr, logits_rj, logits_mr, logits_rm, logits_br, logits_rb, logits_ar, logits_ra,\
                logits_jo, logits_oj, logits_mo, logits_om, logits_bo, logits_ob, logits_ao, logits_oa, logits_ro, logits_or,\
                    pos_mask_j, pos_mask_m, pos_mask_b, pos_mask_a, pos_mask_r, pos_mask_o,\
                    logits_j, logits_m, logits_b, logits_a, logits_r, logits_o, labels
