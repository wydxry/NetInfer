# -- coding: utf-8 --
# @Time : 2024/10/15 11:07
# @Author : Zeng Li


import torch
import torch.nn as nn

# https://arxiv.org/pdf/1512.03385.pdf

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for _ in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet10(num_classes, channels=3):
    return ResNet(Bottleneck, [1, 1, 1, 1], num_classes, channels)


def ResNet12(num_classes, channels=3):
    return ResNet(Bottleneck, [2, 1, 1, 1], num_classes, channels)


def ResNet14(num_classes, channels=3):
    return ResNet(Bottleneck, [2, 2, 1, 1], num_classes, channels)


def ResNet16(num_classes, channels=3):
    return ResNet(Bottleneck, [2, 2, 2, 1], num_classes, channels)


def ResNet18(num_classes, channels=3):
    return ResNet(Bottleneck, [2, 2, 2, 2], num_classes, channels)


def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)


def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)


def save_model_test():
    model = ResNet10(30, 2)
    image = torch.randn((1, 2, 128, 128))
    model.eval()
    traced_script_module = torch.jit.trace(model, image)
    traced_script_module.save("./model_0927_10.pt")

    model = ResNet18(30, 2)
    image = torch.randn((1, 2, 128, 128))
    model.eval()
    traced_script_module = torch.jit.trace(model, image)
    traced_script_module.save("./model_0927_18.pt")

    model = ResNet50(30, 2)
    image = torch.randn((1, 2, 128, 128))
    model.eval()
    traced_script_module = torch.jit.trace(model, image)
    traced_script_module.save("./model_0927_50.pt")


def test():
    img_channels = 2
    output_size = 30
    img_size = 128
    iterations = 1000  # 重复计算的轮次
    model = ResNet14(output_size, img_channels)
    print("model: ", model)
    for m in model.modules():
        print(m)

    device = torch.device("cuda:0")
    model.to(device)

    random_input = torch.randn(1, img_channels, img_size, img_size).to(device)
    random_input = torch.randn(1, img_channels, img_size, img_size).to(device)
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
            # print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))


if __name__ == "__main__":
    # save_model_test()

    # 测试网络推理速度
    test()
    # Inference time: 8.705185, FPS: 114.87406727046132