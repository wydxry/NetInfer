# -- coding: utf-8 --
# @Time : 2024/10/15 11:07
# @Author : Zeng Li


import torch
import torch.nn as nn
import torchvision.models as models
import torch.onnx
import time
import onnxruntime as ort
import numpy as np
import ResNet
import ResNet_PE128
import ResNet_PE64

# pip install onnxruntime-gpu

# 1. 准备ResNet模型
# 使用预训练的模型
# model = ResNet.ResNet10(num_classes=30, channels=2)
# model = ResNet.ResNet18(num_classes=30, channels=2)
# model = ResNet.ResNet50(num_classes=30, channels=2)
# model = ResNet_PE128.ResNet10PE_128(img_channel=2, img_size=128)
# model = ResNet_PE128.ResNet18PE_128(img_channel=2, img_size=128)
# model = ResNet_PE128.ResNet50PE_128(img_channel=2, img_size=128)
# model = ResNet_PE64.ResNet10PE_64(img_channel=2, img_size=128)
# model = ResNet_PE64.ResNet18PE_64(img_channel=2, img_size=128)
model = ResNet_PE64.ResNet50PE_64(img_channel=2, img_size=128)
model.eval()  # 切换到评估模式

providers = ['CUDAExecutionProvider']

# 2. 转换为ONNX格式
# 创建一个虚拟输入张量，用于导出模型
dummy_input = torch.randn(1, 2, 128, 128)  # batch_size=1, 3通道, 224x224图像
onnx_file_path = 'model.onnx'

# 导出模型
torch.onnx.export(model, dummy_input, onnx_file_path,
                  export_params=True,
                  opset_version=11,  # 选择一个适当的opset版本
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print(f"Model saved as {onnx_file_path}")

# 3. 加载ONNX模型
ort_session = ort.InferenceSession(onnx_file_path, providers=providers)

# 获取输入和输出的名称
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# 4. 测试推理速度
# 准备输入数据
# 这里我们仍然使用虚拟数据，但在实际应用中应使用真实数据
input_data = np.random.randn(1, 2, 128, 128).astype(np.float32)

# 进行多次推理以获取平均时间
num_warmup_runs = 10
num_inference_runs = 1000

rt_outputs = ort_session.run([output_name], {input_name: input_data})

# Warm-up runs
for _ in range(num_warmup_runs):
    ort_outputs = ort_session.run([output_name], {input_name: input_data})

# Measure inference time
total_time = 0
for _ in range(num_inference_runs):
    start_time = time.perf_counter()
    ort_outputs = ort_session.run([output_name], {input_name: input_data})
    end_time = time.perf_counter()
    total_time += (end_time - start_time)

average_time_per_inference = total_time / num_inference_runs
print(f"Average inference time per run: {average_time_per_inference:.6f} seconds")
FPS = 1.0 / average_time_per_inference
print(f"Average FPS: {FPS:.6f} ")

# Measure inference time
total_time = 0
for _ in range(num_inference_runs):
    start_time = time.time()
    ort_outputs = ort_session.run([output_name], {input_name: input_data})
    end_time = time.time()
    total_time += (end_time - start_time)

average_time_per_inference = total_time / num_inference_runs
print(f"Average inference time per run: {average_time_per_inference:.6f} seconds")
FPS = 1.0 / average_time_per_inference
print(f"Average FPS: {FPS:.6f} ")

# 打印输出形状（可选）
print(f"Output shape: {ort_outputs[0].shape}")