# -- coding: utf-8 --
# @Time : 2024/10/16 11:07
# @Author : Zeng Li


import numpy as np
import torchvision.models as models
import torch
import ResNet
import ResNet_PE64
import ResNet_PE128

# model = models.resnet50(pretrained=False)
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

output_onnx = "ResNet50_PE64.onnx"

# Generate input tensor with random values
input_tensor = torch.rand(1, 2, 128, 128)

# Export torch model to ONNX
print("Exporting ONNX model {}".format(output_onnx))
torch.onnx.export(model, input_tensor, output_onnx,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    verbose=False)