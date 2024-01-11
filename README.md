# IKEM  

This is the implementation of paper:  

Elevating Skeleton-Based Action Recognition with Efficient Multi-Modality Self-Supervision[[Paper](https://arxiv.org/pdf/2309.12009.pdf)]

Authors: Yiping Wei, [Kunyu Peng](https://www.researchgate.net/profile/Kunyu-Peng), [Alina Roitberg](https://www.researchgate.net/profile/Alina-Roitberg-2), [Jiaming Zhang](https://www.researchgate.net/profile/Jiaming-Zhang-10), [Junwei Zheng](https://www.researchgate.net/profile/Junwei-Zheng-4), [Ruiping Liu](https://www.researchgate.net/profile/Ruiping-Liu-7), [Yufan Chen](https://www.researchgate.net/profile/Yufan-Chen-27), [Kailun Yang](https://www.researchgate.net/profile/Kailun-Yang), [Rainer Stiefelhagen](https://www.researchgate.net/profile/Rainer-Stiefelhagen).  
## Update  
[Sep 2023] Code released
## Implementation guidance
Please refer to [CrosSCLR](https://github.com/LinguoLi/CrosSCLR) for environment requirements and installation.
### Data Preparation

We added frame numbers to the raw data for each action, so please reprocess the dataset using our code.

```
# Generate raw data
$ python tools/ntu_gendata.py --data_path <your path> --ignored_sample_path <your path> --out_folder <your path>

# preprocess
$ python feeder/preprocess_ntu.py --dataset_path <your path> --out_folder <your path>
```

### Model training and evaluation
Please modify the specific .yaml file to replace the training set and test set (evaluation protocol) you use.

```
# Pre-training of six-modality model
$ python main.py pretrain_crossclr_6views --config config/crossclr_6views/crossclr_6views_xview.yaml

# Evaluation of six-modality model
$ python main.py linear_evaluation --config config/crossclr_6views/le_crossclr_6views_xview.yaml

# Pre-training of three-modality student model(six-modality model as teacher model)
$ python main.py pretrain_student_6views --config config/crossclr_6views/ts_crossclr_6views.yaml

# Evaluation of the three-modality student model
$ python main.py linear_evaluation --config config/crossclr_6views/le_ts_crossclr_6views.yaml
```


## Abstract  
Self-supervised representation learning for human action recognition has developed rapidly in recent years. Most of the existing works are based on skeleton data while using a multimodality setup. These works overlooked the differences in performance among modalities, which led to the propagation of erroneous knowledge between modalities while only three fundamental modalities, i.e., joints, bones, and motions are used, hence no additional modalities are explored.  
In this work, we first propose an Implicit Knowledge Exchange Module (IKEM) which alleviates the propagation of erroneous knowledge between low-performance modalities. Then, we further propose three new modalities to enrich the complementary information between modalities. Finally, to maintain efficiency when introducing new modalities, we propose a novel teacher-student framework to distill the knowledge from the secondary modalities into the mandatory modalities considering the relationship constrained by anchors, positives, and negatives, named relational cross-modality knowledge distillation. The experimental results demonstrate the effectiveness of our approach, unlocking the efficient use of skeleton-based multimodality data.
## Method  
![sum drawio](https://github.com/desehuileng0o0/IKEM/assets/92596875/0dd7d0bd-28fa-4d43-8ad1-a48a4005d0ef)  
An overview of our pre-training model in red dashed box and teacher-student model in blue dashed box, where module (a) is the knowledge exchange module in CrosSCLR, module (b) is our proposed IKEM, and module (c) is the knowledge distillation module for our teacher-student model. All the modules in the figure use the update of the encoder from joint modality as an example.
## Acknowledgement  
* The framework of our code is based on [CrosSCLR](https://github.com/LinguoLi/CrosSCLR)
* [ST-GCN](https://github.com/yysijie/st-gcn)
* [NTURGB-D](https://github.com/shahroudy/NTURGB-D)
