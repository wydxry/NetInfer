# ResNet-ONNX-TensorRT

Module

```shell
pip install onnxruntime-gpu
```

Lib Version
```shell
Windows 11
GPU 4090
CUDA 12.6
cuDnn 8.9.7.29
TensorRT 10.5.0.18
PyTorch 2.4.1
Python 3.10
```

## Test Result
```
Onnx (GPU):
resnet10 FPS 1104.949544 
resnet18 FPS  516.008626
resnet50 FPS  333.727555
resnet10PE128 FPS 503.101748 
resnet18PE128 FPS  352.149057
resnet50PE128 FPS  219.475096
resnet10PE64 FPS  956.631219
resnet18PE64 FPS  571.736885 
resnet50PE64 FPS  350.914636

TensorRT:
resnet10 FPS 2438.268527
resnet18 FPS 1703.828298
resnet50 FPS 974.808511 
resnet10PE128 FPS 1073.175663 
resnet18PE128 FPS  655.425035 
resnet50PE128 FPS  380.379059
resnet10PE64 FPS  1989.452322 
resnet18PE64 FPS  1149.313923 
resnet50PE64 FPS  672.243645
```
