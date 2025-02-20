 # <p align=center> [NeurIPS 2024] EnsIR: An Ensemble Algorithm for Image Restoration via Gaussian Mixture Models</p>

<div align="center">
 
[![paper](https://img.shields.io/badge/EnsIR-paper-blue.svg)](https://arxiv.org/abs/2410.22959)
[![arXiv](https://img.shields.io/badge/EnsIR-arXiv-red.svg)](https://arxiv.org/abs/2410.22959)
[![](https://img.shields.io/badge/project-page-red.svg)](https://sunsean21.github.io/ensir.html)
[![poster](https://img.shields.io/badge/EnsIR-poster-green.svg)](https://sunsean21.github.io/resources/pdf/poster_neurips24_ensir.pdf)
[![Closed Issues](https://img.shields.io/github/issues-closed/sunshangquan/EnsIR)](https://github.com/sunshangquan/EnsIR/issues?q=is%3Aissue+is%3Aclosed) 
[![Open Issues](https://img.shields.io/github/issues/sunshangquan/EnsIR)](https://github.com/sunshangquan/EnsIR/issues) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fsunshangquan%2FEnsIR&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

---
>**EnsIR: An Ensemble Algorithm for Image Restoration via Gaussian Mixture Models**<br>  Shangquan Sun, Wenqi Ren, Zikun Liu, Hyunhee Park, Rui Wang, Xiaochun Cao<br> 
>Neural Information Processing Systems (NeurIPS 2024)

<details>
<summary><strong>Abstract</strong> (click to expand) </summary>
Image restoration has experienced significant advancements due to the development of deep learning. Nevertheless, it encounters challenges related to ill-posed problems, resulting in deviations between single model predictions and ground-truths. Ensemble learning, as a powerful machine learning technique, aims to address these deviations by combining the predictions of multiple base models. Most existing works adopt ensemble learning during the design of restoration models, while only limited research focuses on the inference-stage ensemble of pre-trained restoration models. Regression-based methods fail to enable efficient inference, leading researchers in academia and industry to prefer averaging as their choice for post-training ensemble. To address this, we reformulate the ensemble problem of image restoration into Gaussian mixture models (GMMs) and employ an expectation maximization (EM)-based algorithm to estimate ensemble weights for aggregating prediction candidates. We estimate the range-wise ensemble weights on a reference set and store them in a lookup table (LUT) for efficient ensemble inference on the test set. Our algorithm is model-agnostic and training-free, allowing seamless integration and enhancement of various pre-trained image restoration models. It consistently outperforms regression-based methods and averaging ensemble approaches on 14 benchmarks across 3 image restoration tasks, including super-resolution, deblurring and deraining. The codes and all estimated weights have been released in [Github](https://github.com/sunshangquan/EnsIR).
</details>

## :mega: Citation
If you use EnsIR, please consider citing:

    @misc{sun2024ensirensemblealgorithmimage,
      title={EnsIR: An Ensemble Algorithm for Image Restoration via Gaussian Mixture Models}, 
      author={Shangquan Sun and Wenqi Ren and Zikun Liu and Hyunhee Park and Rui Wang and Xiaochun Cao},
      year={2024},
      eprint={2410.22959},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.22959}, 
    }
    
---

## :rocket: News
* **2024.10.31**: [Arxiv Paper](https://arxiv.org/abs/2410.22959) is released.
* **2024.10.31**: Codes and pre-trained weights are released.
* **2024.09.30**: EnsIR is accepted by NeurIPS 2024.


## :pushpin: TODO

- [x] Release code
- [x] Release all weights 
- [x] Release manuscript 

## :jigsaw: Pre-inferenced Results <a id="dataset_download"></a>


We prepare the test sets of three tasks inlcuding super-resolution (SR), deblurring and deraining.

| SR | Deblurring | Deraining |
|:---------------:|:-----------------:|:-----------------:|
| [Baidu Disk](https://pan.baidu.com/s/1T-Mzy2fR5sMobNIYZRS0CA?pwd=d5bd) [```pin: d5bd```] | [Baidu Disk](https://pan.baidu.com/s/1XJZdJeCiFhE5mfjSsUus0g?pwd=egj8) [```pin: egj8```] | [Baidu Disk](https://pan.baidu.com/s/1B5rvISkq8qwvd9itpJe-Fw?pwd=vtyt) [```pin: vtyt```] |

File directory structure:

```
datasets/
│
├── SR/
│   ├── test
│   │   ├── gt 
│   │   │   ├── Set5
│   │   │   │   ├── baby.png
│   │   │   │   └── ...
│   │   │   ├── Set 14
│   │   │   │   ├── baboon.png
│   │   │   │   └── ...
│   │   ├── lq
│   │   │   ├── ...
│   │   ├── MambaIR
│   │   │   ├── ...
│   │   ├── ...  
│
├── deblurring/
│
└── deraining/
```

## :gear: Installation
```
pip install git+https://github.com/ldeecke/gmm-torch.git
```

## :hammer_and_wrench: Weight Estimation

1. Obtain the reference set, which has been already prepared in ```./dataset/reference_set/[TASK]```, where ```[TASK]``` is one of ```SR```, ```deblurring```, ```deraining```.

2. Prepare the test results of base models, input, and ground-truth images by downloading [Pre-inferenced Results](#dataset_download).

3. Estimate the ensemble weights by 

```
python ensemble.py --yaml_file opt/ensemble/[TASK]/[DATASET].yaml
```

where ```[DATASET]``` could be one of filename in the ```opt/[TASK]```. The estimated weights will be saved under ```./weights```.

### EXAMPLE:

```
python ensemble.py --yaml_file opt/ensemble/SR/Set5.yaml
```

## :balance_scale: Evaluation

1. Prepare the test results of base models, input, and ground-truth images by downloading [Pre-inferenced Results](#dataset_download).

```
python ensemble.py --yaml_file opt/ensemble/[TASK]/[DATASET].yaml
```

2. Change ```precompute_weight``` for assigning pre-estimated weights in the ```.yaml``` file and inference the ensemble weights by 

```
python ensemble.py --yaml_file opt/evaluate/[TASK]/[DATASET].yaml
```

where ```[DATASET]``` could be one of filename in the ```opt/[TASK]```. 

### EXAMPLE:

```
python ensemble.py --yaml_file opt/evaluate/SR/Set5.yaml
```

## :balance_scale: Test on other datasets

1. Prepare the test results of base models, input, and ground-truth images

2. Prepare the reference set of base models, input, and ground-truth images

3. Change the configurations in yaml file and run

```
python ensemble.py --yaml_file opt/ensemble/[TASK]/[DATASET].yaml
```



## :mailbox_with_mail: Contact 
If you have any question, please contact shangquansun@gmail.com

**Acknowledgment:** Part of codes is based on the [gmm_torch](https://github.com/ldeecke/gmm-torch). 

