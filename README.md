## Cost Volume Pyramid Based Depth Inference for Multi-View Stereo

[CVP-MVSNet](https://arxiv.org/abs/1912.08329) (CVPR 2020 Oral) is a cost volume pyramid based depth inference framework for Multi-View Stereo. 

CVP-MVSNet is compact, lightweight, fast in runtime and can  handle  high  resolution  images  to  obtain  high  quality depth map for 3D reconstruction.

If you find this project useful for your research, please cite:

```
@InProceedings{Yang_2020_CVPR,
    author = {Yang, Jiayu and Mao, Wei and Alvarez, Jose M. and Liu, Miaomiao},
    title = {Cost Volume Pyramid Based Depth Inference for Multi-View Stereo},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

## How to use

### 0. Pre-requisites

* Nvidia GPU with 11GB or more vRam.
* CUDA 10.1
* python3.6
* python2.7 for fusion script

### 1. Clone the source code

`git clone https://github.com/JiayuYANG/CVP-MVSNet.git`

### 2. Download testing dataset

Testing data(2G):

Download our pre-processed DTU testing data from [here](https://drive.google.com/file/d/1rX0EXlUL4prRxrRu2DgLJv2j7-tpUD4D/view?usp=sharing) and extract it to `CVP_MVSNet/dataset/dtu-test-1200`.

### 3. Install requirements

`cd CVP_MVSNet`

`pip3 install -r requirements.txt`

### 4. Generate depth map using our pre-trained model

`sh eval.sh`

When finished, you can find depth maps in `outputs_pretrained` folder.

### 5. Generate point clouds and reproduce DTU results


Check out Yao Yao's modified version of fusibile

`git clone https://github.com/YoYo000/fusibile`

Install fusibile by `cmake .` and `make`, which will generate the executable at`FUSIBILE_EXE_PATH`

Link fusibile executable into fusion folder (Note: You should modify FUSIBILE_EXE_PATH to the path to your fusibile executable)

`ln -s FUSIBILE_EXE_PATH CVP_MVSNet/fusion/fusibile`

Install extra dependencies

`pip2 install -r CVP_MVSNet/fusion/requirements_fusion.txt`

Use provided script to use fusibile to generate point clouds. 

`cd CVP_MVSNet/fusion/`

`sh fusion.sh`

Use provided script to move generated point clouds into `outputs_pretrained/dtu_eval` folder

`python2 fusibile_to_dtu_eval.py`

Evaluate the point clouds using the [DTU evaluation code](http://roboimagedata.compute.dtu.dk/?page_id=36).

The results should be like:

| Acc. (mm) | Comp. (mm) | Overall (mm) |
|-----------|------------|--------------|
| 0.296     | 0.406      | 0.351        |

### 6. Train your own model

Download training dataset from [here](https://drive.google.com/file/d/1_Nuud3lRGaN_DOkeTNOvzwxYa2z2YRbX/view?usp=sharing) and extract it to `CVP-MVSNet/datasets/dtu-train-128`.

Modify training parameters in `train.sh` script.

Start training

`sh train.sh`


## Acknowledgment

This repository is partly based on the [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch) repository by Xiaoyang Guo. Many thanks to Xiaoyang Guo for the great code!

This repository is inspired by the [MVSNet](https://github.com/YoYo000/MVSNet) by Yao Yao et al. Many thanks to Yao Yao and his mates for the great paper and great code! 
