# 面向交通标志检测系统的隐蔽物理对抗样本研究

## Introduction

本项目为**面向交通标志检测系统的隐蔽物理对抗样本研究**（**Stealthy Physical Adversarial Attacks against Traffic Sign Recognition Systems**）的代码实现仓库。

其中，examples目录下为本项目提供的实际效果视频，code目录下为项目本身的代码。



## Prepare

本项目由于针对Yolov5，DINO等目标检测模型进行攻击，所以在配置本项目的代码环境前，需要先配置相应的目标检测模型的训练与推理环境。这两个项目对应的地址为：

[Yolov5](https://github.com/ultralytics/yolov5)

[DINO](https://github.com/IDEA-Research/DINO)

code目录中主要提供了针对Yolov5的训练代码，所以这里也提供一份我们使用TT100k数据集训练得到的Yolov5的[权重文件](https://pan.quark.cn/s/32fadf58e017)，方便读者可以直接使用本仓库中提供的code文件进行快速的复现。

如果想要测试在DINO上的训练效果，那么用户可以选择自行训练一个基于DINO的TSR并对拉取到的DINO仓库下的util文件夹内容进行修改，以保证从对抗样本的反向传播链不会断裂。

同时，由于训练过程中需要使用到TT100K数据集中的无标志牌图片，这里也需要在训练前下载TT00k数据集，其地址为：

[Tsinghua_Tencent_100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/)