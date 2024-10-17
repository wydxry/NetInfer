# ResNet-ONNX-TensorRT

Test ResNet infer time in ONNX and TensorRT.

## Hard & Soft Wave and version

```shell
Windows 11
GPU 4090
CUDA 12.6
cuDnn 8.9.7.29
TensorRT 10.5.0.18
PyTorch 2.4.1
Python 3.10
```

CUDA 12.6

- [CUDA Toolkit 12.6 Update 2 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

cuDNN 8.9.7

- https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip/

TensorRT 10.5.0

- https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.5.0/zip/TensorRT-10.5.0.18.Windows.win10.cuda-12.6.zip

- ```Shell
  pip install tensorrt-10.5.0-cp310-none-win_amd64.whl
  ```

PyTorch 2.4.1

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

Other

```shell
pip install onnx
pip install onnxruntime-gpu
pip install pycuda
pip install tensorrt
```

PS:

CUDA, cuDNN, and TensorRT should add path to the sys path, TensorRT's lib and include files should be copy to CUDA folders.

## How to use

1. Net to onnx

   ```shell
   python convert_onnx.py
   ```

2. infer time in onnx

   ```shell
   python infer_onnx.py
   ```

3. onnx to tensorrt after step 1 or step 2

   ```shell
   trtexec --onnx=xxx.onnx --saveEngine=xxx.engine
   ```

4. infer time in tensorrt after step 3

   ```shell
   python infer_tensorrt.py
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