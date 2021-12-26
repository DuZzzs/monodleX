### 测试
使用作者提供的预训练模型在kitti上进行测试，
运行：
```angular2html
# 新建文件夹，将预训练模型放在work_dirs/kitti_debug/checkpoints/kitti_epoch_140.pth
cd experiments/example
bash test.sh
```
结果如下：
```angular2html
2021-08-18 13:59:07,635   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.1217, 88.3670, 79.8853
bev  AP:31.2712, 24.7619, 23.4836
3d   AP:23.7493, 20.7087, 17.9959
aos  AP:89.09, 87.18, 78.04
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:95.9642, 91.8784, 84.7531
bev  AP:25.8910, 20.8330, 18.1531
3d   AP:18.2593, 14.5657, 12.9989
aos  AP:94.80, 90.55, 82.54
Car AP@0.70, 0.50, 0.50:
bbox AP:90.1217, 88.3670, 79.8853
bev  AP:61.6387, 50.2435, 44.7139
3d   AP:57.7730, 44.3736, 42.4333
aos  AP:89.09, 87.18, 78.04
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:95.9642, 91.8784, 84.7531
bev  AP:61.4324, 47.3653, 41.9808
3d   AP:56.0393, 42.8401, 38.6675
aos  AP:94.80, 90.55, 82.54

```