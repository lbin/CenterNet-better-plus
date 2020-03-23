# CenterNet-better-plus

This repo is implemented based on [detectron2](https://github.com/facebookresearch/detectron2) and [CenterNet-better](https://github.com/FateScript/CenterNet-better/edit/master/README.md)

## Requirements

- Python >= 3.6
- PyTorch >= 1.4
- torchvision that matches the PyTorch installation.
- OpenCV
- pycocotools

```shell
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

- GCC >= 4.9

```shell
gcc --version
```

- detectron2

```shell
pip install -U 'git+https://github.com/facebookresearch/detectron2.git'
```

### Training

```shell
python train_net.py --num-gpus 8 --config-file configs/centernet_r_18_C4_1x.yaml
```

### Testing and Evaluation

```shell
python train_net.py --num-gpus 8 --config-file configs/centernet_r_18_C4_1x.yaml --eval-only MODEL.WEIGHTS model_0007999.pth
```

## Performance

This repo use less training time to get a better performance, it nearly spend half training time and get 1~2 pts higher mAP compared with the old repo. Here is the table of performance.

Backbone ResNet-50

| Code             | mAP  |
| ---------------- | ---- |
| ours             |      |
| centernet-better | 35.1 |

Backbone ResNet-18
| Code             | mAP  |
| ---------------- | ---- |
| ours             | 29.7 |
| centernet-better | 29.8 |


## What\'s comming

- [ ] Support DLA backbone
- [ ] Support Hourglass backbone
- [ ] Support KeyPoints dataset

## Acknowledgement

- [detectron2](https://github.com/facebookresearch/detectron2)
- [CenterNet](https://github.com/xingyizhou/CenterNet)
- [CenterNet-better](https://github.com/FateScript/CenterNet-better)
