# 关于训练
CUDA_VISIBLE_DEVICES=0 python ./src/train.py --dataset=KITTI --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl --data_path=./data/ --image_set="train" --train_dir="./log/train" --net="squeezeSeg" --max_steps=25000 --summary_step=100 --checkpoint_step=1000 --gpu=0

# 关于测试
./scripts/eval.sh -gpu 1 -image_set val -log_dir ./log/
# 关于偏移视角生成文件my_cls_counter.py的说明
本文件用于检测每一张图片的检测结果,并保存对应的idx,iou(car,pedestrain,cyclist)
命令为: ./scripts/my_cls_counter.sh -gpu 0 -image_set train -log_dir ./log_remove/log_0_9
补充说明,其中检测的索引文件放在 SQUEEZESEG_/data/ImageSet下;写入的文件放在SQUEEZESEG_/src/val.txt

如果要让生成的npy文件为6通道数的数据,则在./scripts/my_cls_counter.sh中的"python ./src/my_cls_counter.py ..."中的"my_cls_counter.py"换成"my_cls_counter_6channels.py"