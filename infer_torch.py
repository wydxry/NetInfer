# -- coding: utf-8 --
# @Time : 2024/10/24 11:07
# @Author : Zeng Li
from random import random

import torch
import ResNet
import ResNet_PE128
import ResNet_PE64
import time


model = ResNet.ResNet10(num_classes=30, channels=2)
# model = ResNet.ResNet18(num_classes=30, channels=2)
# model = ResNet.ResNet50(num_classes=30, channels=2)

# model = ResNet_PE64.ResNet10PE_64(img_channel=2, img_size=128)
# model = ResNet_PE64.ResNet18PE_64(img_channel=2, img_size=128)
# model = ResNet_PE64.ResNet50PE_64(img_channel=2, img_size=128)

# model = ResNet_PE128.ResNet10PE_128(img_channel=2, img_size=128)
# model = ResNet_PE128.ResNet18PE_128(img_channel=2, img_size=128)
# model = ResNet_PE128.ResNet50PE_128(img_channel=2, img_size=128)

model_name = "ResNet10.pth"

torch.save(model.state_dict(), model_name)

model.load_state_dict(torch.load(model_name))

model.eval()  # 切换到评估模式

img_channels = 2
output_size = 30
img_size = 128
iterations = 1000

random_input = torch.randn(1, img_channels, img_size, img_size)

def infer_gpu(random_input):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    random_input = random_input.to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(10):
        output = model(random_input)
        # print(output.shape)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 测速
    times = torch.zeros(iterations)  # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

def infer_cpu(random_input):
    model.eval()
    # Measure inference time
    total_time = 0
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = model(random_input)
            end_time = time.perf_counter()
            total_time += (end_time - start_time)

    average_time_per_inference = total_time / iterations
    print(f"Average inference time per run: {average_time_per_inference:.6f} seconds")
    FPS = 1.0 / average_time_per_inference
    print(f"Average FPS: {FPS:.6f} ")

if __name__ == '__main__':
    infer_cpu(random_input)
    infer_gpu(random_input)