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
Pycuda 2024.1.2
onnx 1.17.0
onnxruntime-gpu 1.19.2 (Python)
onnxruntime-gpu 1.18.0 (C++)
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
Onnx Runtime 1.18.0 (C++)

- https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-win-x64-gpu-cuda12-1.18.0.zip

PS: C++ version 1.18.0 is different from python version 1.19.2, cause it confilt with cuDnn 8.9.7.

Other

```shell
pip install onnx
pip install onnxruntime-gpu
pip install pycuda
pip install tensorrt
```

PS:

CUDA, cuDNN, OnnxRuntime and TensorRT should add path to the sys path, TensorRT's lib and include files should be copy to CUDA folders.

## How to use (Python)

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
   # fp32
   trtexec --onnx=xxx.onnx --saveEngine=xxx.engine
   
   # fp16
   trtexec --onnx=xxx.onnx --saveEngine=xxx.engine --fp16

   # int8
   trtexec --onnx=xxx.onnx --saveEngine=xxx.engine --int8
   ```

4. infer time in tensorrt after step 3

   ```shell
   python infer_tensorrt.py
   ```
5. infer time in torch (cpu & gpu)
   ```shell
   python infer_torch.py
   ```

## How to use (C++)
1. download libs and add related path.
2. get xxx.onnx and xxx.engine
3. run infer_onnx.cpp
4. run infer_tensorrt.cpp

PS:

CUDA, cuDNN, OnnxRuntime and TensorRT's lib and include files should be added to the project path, such as cudart.lib etc.
   
## Test Result

### Python

```
Torch (CPU):
ResNet10 FPS 113.58191
ResNet18 FPS 68.064344 
ResNet50 FPS 40.726698
ResNet10_PE64 FPS 70.718973 
ResNet18_PE64 FPS 36.965400 
ResNet50_PE64 FPS 26.087974
ResNet10_PE128 FPS 29.20346
ResNet18_PE128 FPS 18.42818
ResNet50_PE128 FPS 9.810414

Torch (GPU):
ResNet10 FPS 1094.2849
ResNet18 FPS 623.55040
ResNet50 FPS 328.40845
ResNet10_PE64 FPS 509.50153
ResNet18_PE64 FPS 264.40072
ResNet50_PE64 FPS 405.38294
ResNet10_PE128 FPS 749.59062790
ResNet18_PE128 FPS 524.28793446
ResNet50_PE128 FPS 140.03763848

Onnx (GPU):
resnet10 FPS 1104.949544 
resnet18 FPS 516.008626
resnet50 FPS 333.727555
resnet10PE64 FPS 956.631219
resnet18PE64 FPS 571.736885 
resnet50PE64 FPS 350.914636
resnet10PE128 FPS 503.101748 
resnet18PE128 FPS 352.149057
resnet50PE128 FPS 219.475096

TensorRT:
resnet10 FPS 2438.268527
resnet18 FPS 1703.828298
resnet50 FPS 974.808511
resnet10PE64 FPS 1989.452322 
resnet18PE64 FPS 1149.313923 
resnet50PE64 FPS 672.243645
resnet10PE128 FPS 1073.175663 
resnet18PE128 FPS 655.425035 
resnet50PE128 FPS 380.379059
```

### C++
```
Onnx (GPU):
resnet10 FPS 1222.05
resnet18 FPS 586.27
resnet50 FPS 390.137
resnet10PE64 FPS 887.784
resnet18PE64 FPS 459.432
resnet50PE64 FPS 321.543
resnet10PE128 FPS 592.557
resnet18PE128 FPS 359.803
resnet50PE128 FPS 239.143

TensorRT:
resnet10 FPS 3371.54
resnet18 FPS 2070.82
resnet50 FPS 1015.33
resnet10PE64 FPS 2294.1
resnet18PE64 FPS 1252.82
resnet50PE64 FPS 687.805
resnet10PE128 FPS 1094.69
resnet18PE128 FPS 652.742
resnet50PE128 FPS 386.25

TensorRT (FP16):
ResNet10_FP16 FPS 6854.01
ResNet18_FP16 FPS 4488.33
ResNet50_FP16 FPS 2579.31
ResNet10_PE64_FP16 FPS 5260.39
ResNet18_PE64_FP16 FPS 3106.55
ResNet50_PE64_FP16 FPS 1780.31
ResNet10_PE128_FP16 FPS 3228.93
ResNet18_PE128_FP16 FPS 2048.34
ResNet50_PE128_FP16 FPS 1118.19 
```

