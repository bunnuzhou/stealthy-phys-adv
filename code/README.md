# Usage

在完成Yolov5的环境配置并将TT100K的部分无标志牌数据下载并解压至某个文件夹。

将我们提供的权重文件Yolov5x.pt文件放置在该目录下，进行训练：

`pip install -r requirements.txt`

`python train.py  `

该训练文件中，提供了一些训练过程中需要的环境设置:

img_path 用于指示之前下载的TT100k无标志牌数据的文件夹位置

`--img_path the_path_to_TT100kNosign`

log_path 用于指示训练过程中记录的loss tensorboard数据存储的位置

`--log_path the_path_to_store_tensorboard_loss`

color 用于指示最终视频中需要的平均混合的白色灰度值

 `--color the_averaged_illimination`

victim 用于指示本次训练的攻击对象，实验过程中需要的dark/white图片放置在template_speedlimit文件夹，本仓库中提供了我们实拍得到的一些图片进行训练，读者也可以自己拍摄相应的图片放置在该文件夹下，命名方式为攻击对象-环境亮度-brig/dark，pl30-100-brig.png

 `--victim pl30/pl70/white`

light 用于指示当前需要训练哪个亮度下的对抗样本序列，这里template_speedlimit文件夹下提供了100/300/500亮度下拍摄的70，30限速牌，直接使用我们的代码可以将light设置为100/300/500

`--light 100/300/500`

attack 用于指示攻击类型用来设置相应的损失函数计算方式，包含untargeted, targeted, disappear, creation四种

`--attack untargeted, targeted, disappear, creation`

训练最终会输出训练得到的三帧到light/victim目录下，比如这里我们也提供了100/pl30等多个我们实验过程中输出的结果。攻击过程中将会利用这三帧计算出相应的第四帧并使用opencv合成为一个高帧率视频进行物理测试。

ps. 该目录下我们还提供了一个hard_constraint.py文件，该文件中提供了使用SLSQP方法对最终帧画面进行约束的实现，由于该步骤耗时较长，所以我们将这部分内容从train.py文件中抹去，有兴趣进行物理实验复现的读者可以将这部分内容加上。