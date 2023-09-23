#!/usr/bin/env python
import argparse
import sys
import os
import shutil
import zipfile
import time

# torchlight
import torchlight
from torchlight import import_class

from processor.processor import init_seed
init_seed(0)#设种子，保证同输入同输出

def save_src(target_path):
    code_root = os.getcwd()
    srczip = zipfile.ZipFile('./src.zip', 'w')
    for root, dirnames, filenames in os.walk(code_root):
            for filename in filenames:
                if filename.split('\n')[0].split('.')[-1] == 'py':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'yaml':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'ipynb':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
    srczip.close()
    save_path = os.path.join(target_path, 'src_%s.zip' % time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
    shutil.copy('./src.zip', save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['linear_evaluation'] = import_class('processor.linear_evaluation.LE_Processor')#中间这个就是processor名称
    processors['pretrain_crossclr_3views'] = import_class('processor.pretrain_crossclr_3views.CrosSCLR_3views_Processor')
    processors['pretrain_crossclr'] = import_class('processor.pretrain_crossclr.CrosSCLR_Processor')
    processors['pretrain_skeletonclr'] = import_class('processor.pretrain_skeletonclr.SkeletonCLR_Processor')
    processors['pretrain_skeletonclr_dcl'] = import_class('processor.pretrain_skeletonclr_dcl.SkeletonCLR_DCL_Processor')
    processors['pretrain_crossclr_4views'] = import_class('processor.pretrain_crossclr_4views.CrosSCLR_4views_Processor')
    processors['pretrain_crossclr_viewsfusion'] = import_class('processor.pretrain_crossclr_viewsfusion.CrosSCLR_ViewsFusion_Processor')
    processors['pretrain_crossclr_4views_dcl'] = import_class('processor.pretrain_crossclr_4views_dcl.CrosSCLR_4views_DCL_Processor')
    processors['pretrain_crossclr_3views_wk'] = import_class('processor.pretrain_crossclr_3views_wk.CrosSCLR_3views_WK_Processor')
    processors['pretrain_crossclr_5views'] = import_class('processor.pretrain_crossclr_5views.CrosSCLR_5views_Processor')
    processors['ts_4views'] = import_class('processor.ts_4views.TS_Processor')
    processors['pretrain_student_6views'] = import_class('processor.pretrain_student_6views.TS_Processor')
    processors['pretrain_skeletonclr_viewsfusion'] = import_class('processor.pretrain_skeletonclr_viewsfusion.SkeletonCLR_Viewsfusion_Processor')
    #processors['model_visualization_tsne'] = import_class('processor.model_visualization_tsne.Tsne_Processor')
    processors['pretrain_crossclr_6views'] = import_class('processor.pretrain_crossclr_6views.CrosSCLR_6views_Processor')
    processors['pretrain_student_3views'] = import_class('processor.pretrain_student_3views.TS_Processor')
    #processors['pretrain_mvclr'] = import_class('processor.pretrain_mvclr.MVCLR_Processor')
    # endregion yapf: enable

    # add sub-parser输入参数之前还要输一下subparser名字
    subparsers = parser.add_subparsers(dest='processor')#dest指定结果命名空间中使用的属性名称
    for k, p in processors.items():#p现在是个class，为每个processer建立一个parser？
        subparsers.add_parser(k, parents=[p.get_parser()])#为每个parser定义参数，会从processor里继承，再加上自己的args

    # read arguments把参数读进来
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]#读你指定的processor名
    p = Processor(sys.argv[2:])#根据--config实例化processor

    if p.arg.phase == 'train':
        # save src
        save_src(p.arg.work_dir)

    p.start()
