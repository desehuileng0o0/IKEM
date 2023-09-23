import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.ntu_read_skeleton import read_xyz

#training_subjects = [
#    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
#]#参与训练的数据编号(受试者)
#training_cameras = [2, 3]#参与训练的摄像机

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,80, 81, 82, 
    83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
]
training_setups = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]

max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')#print等于这个方法加了个换行符
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")



def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    if ignored_sample_path != None:#如果给了错误数据的路径
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]#生成的所有需要忽略的文件的列表
    else:
        ignored_samples = []#不然就是空集
    print(ignored_samples)
    sample_name = []
    sample_label = []
    sample_frame = []
    dir_list = os.listdir(data_path)#返回指定目录下的文件和文件夹列表
    dir_list.sort()#排序
    for filename in dir_list:#对这一个文件做处理
        if filename in ignored_samples:
            print('pass')
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])#从文件名中把动作、测试者、摄像机ID提取出来。
        setup_id = int(
            filename[filename.find('S') + 1:filename.find('S') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)#判断现在的摄像机或者人物是不是在训练集里面，返回是TF
        elif benchmark == 'xsetup':
            istraining = (setup_id in training_setups)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining#训练时，如果这个.skeleton是训练集，训练的时候sample也就有他，评估的时候就不是它
        elif part == 'val':
            issample = not (istraining)#验证时，这个文件是训练集里的，那么sample就没它，不是训练集就有它
        else:
            raise ValueError()

        if issample:#就是决定这次采的数据里有没有它
            sample_name.append(filename)
            sample_label.append(action_class - 1)#根据训练的情况把采样集的文件名和动作类别都保存成列表

    #with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:#列表生成后，分别为训练和验证生成两个pkl文件
        #pickle.dump((sample_name, list(sample_label)), f)#将sample_name和label的ls存入f，持久保存对象
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(#内存映像文件，，允许将大文件分成小段进行独写
        '{}/{}_data.npy'.format(out_path, part),#用作数组数据缓存区的文件
        dtype='float32',
        mode='w+',#创建或覆盖现有文件进行独写
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))#所需的阵列形状

    for i, s in enumerate(sample_name):#遍历文件名
        print_toolbar(i * 1.0 / len(sample_label),#传入参数：当前进度、第几个sample、sample总数、bench、part
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))#打印进度条
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data#i表示第几个sample，第二维是3，应该是坐标，第三维应该是帧数，第四第五就分别是关节数和身体数了。
        sample_frame.append(data.shape[1])
        #print(data.shape[1])

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:#列表生成后，分别为训练和验证生成两个pkl文件
        pickle.dump((sample_name, list(sample_label), list(sample_frame)), f)#将sample_name和label的ls存入f，持久保存对象
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)
    end_toolbar()#关闭进度条，最重要的应该就是上面这句话了，显示了你创建的文件的架构


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='/lsdf/data/activity/NTU_RGBD/zipped/zipped_skeleton_csv/unzipped/nturgb+d_skeletons_120')
    parser.add_argument(
        '--ignored_sample_path',
        default='/home/ywei2/CrosSCLR/resource/NTU-RGB-D/NTU_RGBD120_samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/lsdf/data/activity/NTU_RGBD/zipped/zipped_skeleton_csv/TEST')#定义了库以及输出的文件夹

    benchmark = ['xsub', 'xsetup']#120
    #benchmark = ['xsub', 'xview']#两种评估方法
    part = ['train', 'val']#训练和评估
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)#处理数据
