

# 大纲

  - [筛选数据](#筛选数据)
  - [修改代码](#修改代码)
  - [按照python环境等](#按照python环境等)
      - [安装注意点⚠️](#安装注意点️)
  - [开始跑流程](#开始跑流程)
      - [train](#train)
      - [resume](#resume)
      - [Validation and Testing](#validation-and-testing)
  - [我的结果展示](#我的结果展示)

## 筛选数据


- 数据决定成败

[使用本人写的vqa筛选出跑代码的数据，详情可参考我的仓库->](https://github.com/zput/filter-vqa-data)


## 修改代码

> 为适配老师的数据集，根改load相关文件代码，去掉vg相关代码
>> 详情可参看git历史记录

## 按照python环境等

- 安装[conda](https://zhuanlan.zhihu.com/p/459607806)

```
export PATH="/home/username/miniconda/bin:$PATH"


wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.11-Linux-x86_64.sh

bash Anaconda3-2021.11-Linux-x86_64.sh

source ~/.bashrc

conda create -n trar python=3.7 -y

conda activate trar

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt
# wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz

```


#### 安装注意点⚠️

- ``` !pip install pyyaml==5.4.1 ```



## 开始跑流程


#### train

- ```python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --SPLIT='train+val'  ```

#### resume

- ```python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --RESUME=True --CKPT_V=str --CKPT_E=int```
  - ```python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --RESUME=True --CKPT_V=3950559 --CKPT_E=11```

#### Validation and Testing

- ```python3 run.py --RUN='val' --MODEL='trar' --DATASET='vqa' --CKPT_V=5652158 --CKPT_E=12```


## 我的结果展示


```
(trar) root@iZ0jld34l89hw61ntoxe34Z:~/TRAR-VQA# tree ckpts/
ckpts/
├── ckpt_3950559
│   ├── epoch1.pkl
│   ├── epoch10.pkl
│   ├── epoch11.pkl
│   ├── epoch12.pkl
│   ├── epoch13.pkl
│   ├── epoch2.pkl
│   ├── epoch3.pkl
│   ├── epoch4.pkl
│   ├── epoch5.pkl
│   ├── epoch6.pkl
│   ├── epoch7.pkl
│   ├── epoch8.pkl
│   └── epoch9.pkl
└── ckpt_5652158
    ├── epoch11.pkl
    ├── epoch12.pkl
    └── epoch13.pkl

2 directories, 16 files
(trar) root@iZ0jld34l89hw61ntoxe34Z:~/TRAR-VQA#
```

```
(trar) root@iZ0jld34l89hw61ntoxe34Z:~/TRAR-VQA# cat ./results/log/log_run_5652158.txt
{ BATCH_SIZE        }->64
{ BBOXFEAT_EMB_SIZE }->2048
{ BBOX_NORMALIZE    }->True
{ BINARIZE          }->False
{ CACHE_PATH        }->./results/cache
{ CKPTS_PATH        }->./ckpts
{ CKPT_EPOCH        }->10
{ CKPT_PATH         }->None
{ CKPT_VERSION      }->3950559
{ DATASET           }->vqa
{ DATA_PATH         }->{'vqa': './data/vqa', 'clevr': './data/clevr'}
{ DATA_ROOT         }->./data
{ DEVICES           }->[0]
{ DROPOUT_R         }->0.1
{ EVAL_BATCH_SIZE   }->32
{ EVAL_EVERY_EPOCH  }->True
{ FEATS_PATH        }->{'vqa': {'train': './data/vqa/feats/train2014', 'val': './data/vqa/feats/val2014', 'test': './data/vqa/feats/test2015'}, 'clevr': {'train': './data/clevr/feats/train', 'val': './data/clevr/feats/val', 'test': './data/clevr/feats/test'}}
{ FEAT_SIZE         }->{'vqa': {'FRCN_FEAT_SIZE': (64, 2048), 'BBOX_FEAT_SIZE': (100, 5)}, 'clevr': {'GRID_FEAT_SIZE': (196, 1024)}}
{ FF_SIZE           }->2048
{ FLAT_GLIMPSES     }->1
{ FLAT_MLP_SIZE     }->512
{ FLAT_OUT_SIZE     }->1024
{ GPU               }->0
{ GRAD_ACCU_STEPS   }->1
{ GRAD_NORM_CLIP    }->-1
{ HIDDEN_SIZE       }->512
{ IMG_SCALE         }->8
{ LAYER             }->6
{ LOG_PATH          }->./results/log
{ LOSS_FUNC         }->bce
{ LOSS_FUNC_NAME_DICT }->{'ce': 'CrossEntropyLoss', 'bce': 'BCEWithLogitsLoss', 'kld': 'KLDivLoss', 'mse': 'MSELoss'}
{ LOSS_FUNC_NONLINEAR }->{'ce': [None, 'flat'], 'bce': [None, None], 'kld': ['log_softmax', None], 'mse': [None, None]}
{ LOSS_REDUCTION    }->sum
{ LR_BASE           }->0.0001
{ LR_DECAY_LIST     }->[10, 12]
{ LR_DECAY_R        }->0.2
{ MAX_EPOCH         }->13
{ MODEL             }->trar
{ MODEL_USE         }->TRAR
{ MULTI_HEAD        }->8
{ NUM_WORKERS       }->8
{ N_GPU             }->1
{ OPT               }->Adam
{ OPT_PARAMS        }->{'betas': (0.9, 0.98), 'eps': 1e-09, 'weight_decay': 0, 'amsgrad': False}
{ ORDERS            }->[0, 1, 2, 3]
{ PIN_MEM           }->True
{ POOLING           }->avg
{ PRED_PATH         }->./results/pred
{ RAW_PATH          }->{'vqa': {'train': './data/vqa/raw/v2_OpenEnded_mscoco_train2014_questions.json', 'train-anno': './data/vqa/raw/v2_mscoco_train2014_annotations.json', 'val': './data/vqa/raw/v2_OpenEnded_mscoco_val2014_questions.json', 'val-anno': './data/vqa/raw/v2_mscoco_val2014_annotations.json', 'test': './data/vqa/raw/v2_OpenEnded_mscoco_test2015_questions.json'}, 'clevr': {'train': './data/clevr/raw/questions/CLEVR_train_questions.json', 'val': './data/clevr/raw/questions/CLEVR_val_questions.json', 'test': './data/clevr/raw/questions/CLEVR_test_questions.json'}}
{ RESULT_PATH       }->./results/result_test
{ RESUME            }->True
{ ROUTING           }->soft
{ RUN_MODE          }->train
{ SEED              }->5652158
{ SPLIT             }->{'train': 'train', 'val': 'val', 'test': 'test'}
{ SPLITS            }->{'vqa': {'train': 'train', 'val': 'val', 'test': 'test'}, 'clevr': {'train': '', 'val': 'val', 'test': 'test'}}
{ SUB_BATCH_SIZE    }->64
{ TASK_LOSS_CHECK   }->{'vqa': ['bce', 'kld'], 'clevr': ['ce']}
{ TAU_MAX           }->10
{ TAU_MIN           }->0.1
{ TAU_POLICY        }->1
{ TEST_SAVE_PRED    }->False
{ TRAIN_SPLIT       }->train
{ USE_AUX_FEAT      }->False
{ USE_BBOX_FEAT     }->False
{ USE_GLOVE         }->True
{ VERBOSE           }->True
{ VERSION           }->5652158
{ WARMUP_EPOCH      }->3
{ WORD_EMBED_SIZE   }->300
=====================================
nowTime: 2022-04-10 18:21:31
Epoch: 11, Loss: 83.59267324472403, Lr: 2e-05
Elapsed time: 2, Speed(s/batch): 0.5694633960723877

Overall Accuracy is: 23.38
other : 0.00 number : 0.00 yes/no : 71.92

=====================================
nowTime: 2022-04-10 18:21:37
Epoch: 12, Loss: 81.23818993506494, Lr: 2e-05
Elapsed time: 2, Speed(s/batch): 0.5361629962921143

Overall Accuracy is: 23.38
other : 0.00 number : 0.00 yes/no : 71.92

=====================================
nowTime: 2022-04-10 18:21:42
Epoch: 13, Loss: 79.75693739853897, Lr: 4.000000000000001e-06
Elapsed time: 2, Speed(s/batch): 0.5449070930480957

Overall Accuracy is: 23.38
other : 0.00 number : 0.00 yes/no : 71.92

Overall Accuracy is: 23.38
other : 0.00 number : 0.00 yes/no : 71.92

```















# TRAnsformer Routing Networks (TRAR)
This is an official implementation for ICCV 2021 paper ["TRAR: Routing the Attention Spans in Transformers for Visual Question Answering"](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_TRAR_Routing_the_Attention_Spans_in_Transformer_for_Visual_Question_ICCV_2021_paper.pdf). It currently includes the code for training TRAR on **VQA2.0** and **CLEVR** dataset. Our TRAR model for REC task is coming soon.

## Updates
- (2021/10/10) Release our TRAR-VQA project.
- (2021/08/31) Release our pretrained `CLEVR` TRAR model on `train` split: [TRAR CLEVR Pretrained Models](MODEL.md#CLEVR).
- (2021/08/18) Release our pretrained TRAR model on `train+val` split and `train+val+vg` split: [VQA-v2 TRAR Pretrained Models](MODEL.md#VQA-v2) 
- (2021/08/16) Release our `train2014`, `val2014` and `test2015` data. Please check our dataset setup page [DATA.md](DATA.md) for more details.
- (2021/08/15) Release our pretrained weight on `train` split. Please check our model page [MODEL.md](MODEL.md) for more details.
- (2021/08/13) The project page for TRAR is avaliable.

## Introduction
**TRAR vs Standard Transformer**
<p align="center">
	<img src="misc/trar_block.png" width="550">
</p>

**TRAR Overall**
<p align="center">
	<img src="misc/trar_overall.png" width="550">
</p>

## Table of Contents
0. [Installation](#Installation)
1. [Dataset setup](#Dataset-setup)
2. [Config Introduction](#Config-Introduction)
3. [Training](#Training)
4. [Validation and Testing](#Validation-and-Testing)
5. [Models](#Models)

### Installation
- Clone this repo
```bash
git clone https://github.com/rentainhe/TRAR-VQA.git
cd TRAR-VQA
```

- Create a conda virtual environment and activate it
```bash
conda create -n trar python=3.7 -y
conda activate trar
```

- Install `CUDA==10.1` with `cudnn7` following the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `Pytorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```
- Install [Spacy](https://spacy.io/) and initialize the [GloVe](https://github-releases.githubusercontent.com/84940268/9f4d5680-4fed-11e9-9dd2-988cce16be55?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210815%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210815T072922Z&X-Amz-Expires=300&X-Amz-Signature=1bd1bd4fc52057d8ac9eec7720e3dd333e63c234abead471c2df720fb8f04597&X-Amz-SignedHeaders=host&actor_id=48727989&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Den_vectors_web_lg-2.1.0.tar.gz&response-content-type=application%2Foctet-stream) as follows:
```bash
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```

### Dataset setup
see [DATA.md](DATA.md)

### Config Introduction
In [trar.yml](configs/vqa/trar.yml) config we have these specific settings for `TRAR` model
```
ORDERS: [0, 1, 2, 3]
IMG_SCALE: 8 
ROUTING: 'hard' # {'soft', 'hard'}
POOLING: 'attention' # {'attention', 'avg', 'fc'}
TAU_POLICY: 1 # {0: 'SLOW', 1: 'FAST', 2: 'FINETUNE'}
TAU_MAX: 10
TAU_MIN: 0.1
BINARIZE: False
```
- `ORDERS=list`, to set the local attention window size for routing.`0` for global attention.
- `IMG_SCALE=int`, which should be equal to the `image feature size` used for training. You should set `IMG_SCALE: 16` for `16 × 16` training features.
- `ROUTING={'hard', 'soft'}`, to set the `Routing Block Type` in TRAR model.
- `POOLING={'attention', 'avg', 'fc}`, to set the `Downsample Strategy` used in `Routing Block`.
- `TAU_POLICY={0, 1, 2}`, to set the `temperature schedule` in training TRAR when using `ROUTING: 'hard'`.
- `TAU_MAX=float`, to set the maximum temperature in training.
- `TAU_MIN=float`, to set the minimum temperature in training.
- `BINARIZE=bool`, binarize the predicted alphas (alphas: the prob of choosing one path), which means **during test time**, we only keep the maximum alpha and set others to zero. If `BINARIZE=False`, it will keep all of the alphas and get a weight sum of different routing predict result by alphas. **It won't influence the training time, just a small difference during test time**.

**Note that please set `BINARIZE=False` when `ROUTING='soft'`, it's no need to binarize the path prob in soft routing block.**

**`TAU_POLICY` visualization**

For `MAX_EPOCH=13` with `WARMUP_EPOCH=3` we have the following policy strategy:
<p align="center">
	<img src="misc/policy_visualization.png" width="550">
</p>

### Training
**Train model on VQA-v2 with default hyperparameters:**
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar'
```
and the training log will be seved to:
```
results/log/log_run_<VERSION>.txt
```
Args:
- `--DATASET={'vqa', 'clevr'}` to choose the task for training
- `--GPU=str`, e.g. `--GPU='2'` to train model on specific GPU device.
- `--SPLIT={'train', 'train+val', train+val+vg'}`, which combines different training datasets. The default training split is `train`.
- `--MAX_EPOCH=int` to set the total training epoch number.


**Resume Training**

Resume training from specific saved model weights
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --RESUME=True --CKPT_V=str --CKPT_E=int
```
- `--CKPT_V=str`: the specific checkpoint version
- `--CKPT_E=int`: the resumed epoch number

**Multi-GPU Training and Gradient Accumulation**
1. Multi-GPU Training:
Add `--GPU='0, 1, 2, 3...'` after the training scripts.
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --GPU='0,1,2,3'
```
The batch size on each GPU will be divided into `BATCH_SIZE/GPUs` automatically.

2. Gradient Accumulation:
Add `--ACCU=n` after the training scripts
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --ACCU=2
```
This makes the optimizer accumulate gradients for `n` mini-batches and update the model weights once. `BATCH_SIZE` should be divided by `n`.

### Validation and Testing
**Warning**: The args `--MODEL` and `--DATASET` should be set to the same values as those in the training stage.

**Validate on Local Machine**
Offline evaluation only support the evaluations on the `coco_2014_val` dataset now.
1. Use saved checkpoint
```bash
python3 run.py --RUN='val' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_V=str --CKPT_E=int
```

2. Use the absolute path
```bash
python3 run.py --RUN='val' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_PATH=str
```

**Online Testing**
All the evaluations on the `test` dataset of VQA-v2 and CLEVR benchmarks can be achieved as follows:
```bash
python3 run.py --RUN='test' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_V=str --CKPT_E=int
```

Result file are saved at:

`results/result_test/result_run_<CKPT_V>_<CKPT_E>.json`

You can upload the obtained result json file to [Eval AI](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to evaluate the scores.

### Models
Here we provide our pretrained model and log, please see [MODEL.md](MODEL.md)

## Acknowledgements
- [openvqa](https://github.com/MILVLG/openvqa)
- [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)

## Citation
if TRAR is helpful for your research or you wish to refer the baseline results published here, we'd really appreciate it if you could cite this paper:
```
@InProceedings{Zhou_2021_ICCV,
    author    = {Zhou, Yiyi and Ren, Tianhe and Zhu, Chaoyang and Sun, Xiaoshuai and Liu, Jianzhuang and Ding, Xinghao and Xu, Mingliang and Ji, Rongrong},
    title     = {TRAR: Routing the Attention Spans in Transformer for Visual Question Answering},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2074-2084}
}
```
