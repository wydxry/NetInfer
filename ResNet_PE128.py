# -- coding: utf-8 --
# @Time : 2024/10/15 11:07
# @Author : Zeng Li


import torch
import torch.nn as nn
import torch.nn.functional as F


# conv1 7 x 7 64 stride=2
def Conv1(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(
            channel_in,
            channel_out,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False
        ),
        nn.BatchNorm2d(channel_out),
        # 会改变输入数据的值
        # 节省反复申请与释放内存的空间与时间
        # 只是将原来的地址传递，效率更好
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
    )


# 构建ResNet50-101-152的网络基础模块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 构建 1x1, 3x3, 1x1的核心卷积块
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # 采用1x1的kernel，构建shout cut
        # 注意这里除了第一个bottleblock之外，都需要下采样，所以步长要设置为stride=2
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Embedding(nn.Module):
    def __init__(self, image_channels=6, image_size=340, patch_size=2, dim=32):
        super(Embedding, self).__init__()
        self.num_patches = (image_size // patch_size)  # Patch数量 128
        self.patch_conv = nn.Conv2d(image_channels, dim, patch_size, patch_size)  # 使用卷积将图像划分成Patches
        # self.pos_emb = nn.Parameter(torch.zeros(batch_size, dim, self.num_patches, self.num_patches))  # position embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, dim, self.num_patches, self.num_patches))

    def forward(self, x):
        x = self.patch_conv(x)  # (B, 32, 128, 128)
        # print(x.shape)
        # x = x + self.pos_emb    # (1, 32, 64, 64)
        # pos_emb = self.pos_emb.squeeze()
        # pos_emb = torch.tile(self.pos_emb, (x.shape[0], 1, 1, 1))  #待确认
        pos_emb = self.pos_emb
        # print("pos_emb.shape", pos_emb.shape)
        # print("x.shape", x.shape)
        x = torch.concat([x, pos_emb], dim=1)  # (1, 32, 128, 128)
        # print("x.shape", x.shape)
        # print(x.shape)    # torch.Size([B, 64, 128, 128])
        return x


# 搭建ResNet模板块
class ResNet(nn.Module):
    def __init__(self, image_channels, image_size, block, num_blocks, num_classes=30):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.embedding = Embedding(image_channels=image_channels, image_size=image_size, patch_size=1, dim=32)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 逐层搭建ResNet
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512 * block.expansion * 4, num_classes)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # Xavier参数初始化
        # 基本思想：保持输入和输出的方差一致
        # 这样就避免了所有输出值都趋向于0
        # 这是通用的方法，适用于任何激活函数
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # layers = [ ] 是一个列表
        # 通过下面的for循环遍历配置列表，可以得到一个由 卷积操作、池化操作等 组成的一个列表layers
        # return nn.Sequential(*layers)，即通过nn.Sequential函数将列表通过非关键字参数的形式传入(列表layers前有一个星号)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = self.embedding(x)  # 5, 64, 128, 128
        # print(out.shape)
        out = F.relu(self.bn1(self.conv1(out)))  # 5, 64, 128, 128
        # print(out.shape)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # 5, 2048, 16, 16
        # print(out.shape)
        out = F.avg_pool2d(out, 16)  # 5, 2048, 1, 1
        # print(out.shape)
        out = out.view(out.size(0), -1)  # 5, 2048
        # print(out.shape)
        out = self.linear(out)
        return out


def ResNet10PE_128(img_channel, img_size):
    return ResNet(img_channel, img_size, Bottleneck, [1, 1, 1, 1])


def ResNet12PE_128(img_channel, img_size):
    return ResNet(img_channel, img_size, Bottleneck, [2, 1, 1, 1])


def ResNet14PE_128(img_channel, img_size):
    return ResNet(img_channel, img_size, Bottleneck, [2, 2, 1, 1])


def ResNet16PE_128(img_channel, img_size):
    return ResNet(img_channel, img_size, Bottleneck, [2, 2, 2, 1])


def ResNet18PE_128(img_channel, img_size):
    return ResNet(img_channel, img_size, Bottleneck, [2, 2, 2, 2])


def ResNet50PE_128(img_channel, img_size):
    return ResNet(img_channel, img_size, Bottleneck, [3, 4, 6, 3])


def ResNet101PE_128(img_channel, img_size):
    return ResNet(img_channel, img_size, Bottleneck, [3, 4, 23, 3])


def ResNet152PE_128(img_channel, img_size):
    return ResNet(img_channel, img_size, Bottleneck, [3, 8, 36, 3])


def test():
    img_channels = 2
    output_size = 30
    img_size = 128
    iterations = 1000  # 重复计算的轮次
    model = ResNet50PE_128(img_channels, img_size)
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


# def test2():
#     img_channels = 2
#     output_size = 30
#     img_size = 128
#     iterations = 1000   # 重复计算的轮次
#     model = ResNet50PE_128(img_channels, img_size)

#     # create a quantized model instance
#     model = torch.ao.quantization.quantize_dynamic(
#         model,  # the original model
#         {torch.nn.Linear},  # a set of layers to dynamically quantize
#         dtype=torch.qint8)  # the target dtype for quantized weights

#     print("model: ", model)
#     for m in model.modules():
#         print(m)

#     device = torch.device("cuda:0")
#     model.to(device)

#     random_input = torch.randn(1, img_channels, img_size, img_size).to(device)
#     random_input = torch.randn(1, img_channels, img_size, img_size).to(device)
#     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

#     # GPU预热
#     for _ in range(10):
#         output = model(random_input)
#         # print(output.shape)

#     # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
#     torch.cuda.synchronize()

#     # 测速
#     times = torch.zeros(iterations)     # 存储每轮iteration的时间
#     with torch.no_grad():
#         for iter in range(iterations):
#             starter.record()
#             _ = model(random_input)
#             ender.record()
#             # 同步GPU时间
#             torch.cuda.synchronize()
#             curr_time = starter.elapsed_time(ender) # 计算时间
#             times[iter] = curr_time
#             # print(curr_time)

#     mean_time = times.mean().item()
#     print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))


def save_model_test():
    model = ResNet10PE_128(2, 128)
    image = torch.randn((1, 2, 128, 128))
    model.eval()
    traced_script_module = torch.jit.trace(model, image)
    traced_script_module.save("./model_0927_10_PE.pt")

    model = ResNet18PE_128(2, 128)
    image = torch.randn((1, 2, 128, 128))
    model.eval()
    traced_script_module = torch.jit.trace(model, image)
    traced_script_module.save("./model_0927_18_PE.pt")

    model = ResNet50PE_128(2, 128)
    image = torch.randn((1, 2, 128, 128))
    model.eval()
    traced_script_module = torch.jit.trace(model, image)
    traced_script_module.save("./model_0927_50_PE.pt")


if __name__ == '__main__':
    # save_model_test()

    # 测试网络推理速度
    test()
    # Inference time: 8.001419, FPS: 124.97783100455585
    # test2()