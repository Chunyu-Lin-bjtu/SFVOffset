# SFVOffset
场景视点偏移改善激光雷达点云分割,该代码在SqueezeSeg(https://github.com/BichenWuUCB/SqueezeSeg)基础上修改完成。

## 环境安装
'pip install requirements.txt' 

## 训练与测试
训练
'''
CUDA_VISIBLE_DEVICES=0 python ./src/train.py --dataset=KITTI --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl --data_path=./data/ --image_set="train" --train_dir="./log/train" --net="squeezeSeg" --max_steps=100000 --summary_step=100 --checkpoint_step=1000 --gpu=0
'''

测试
'''
python ./src/eval.py --dataset=KITTI --data_path=./data/ --imageset="val" --eval_dir="./log/myeval_val/" --checkpoint_path="./log/train" --net="squeezeSeg" --gpu=0
'''
