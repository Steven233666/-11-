# -*- coding: utf-8 -*-
"""PlantCNN：全项目唯一的模型定义。

原文档把 FruitCNN 在 3/4/5/6/7 五个脚本里各复制一份（因为 "1.xxx.py" 不是合法模块名，
不能互相 import）。代价是改网络结构必须五个文件同步改，否则加载权重报 size mismatch。
本项目改为统一从 model.py 导入——文件名带数字不影响 import 这个独立模块。
"""

import torch
import torch.nn as nn


class PlantCNN(nn.Module):
    """4 层卷积 + 2 层全连接的轻量 CNN，输入固定 3×128×128。

    结构：
        Conv(3→32)+Pool  → [32, 64, 64]
        Conv(32→64)+Pool → [64, 32, 32]
        Conv(64→128)+Pool→ [128, 16, 16]
        Conv(128→128)+Pool→[128, 8, 8]
        Flatten(8192) → Dropout → Linear(256) → Dropout → Linear(num_classes)
    """

    def __init__(self, num_classes: int = 5, img_size: int = 128):
        super().__init__()
        if img_size != 128:
            raise ValueError(
                f"当前结构按 {img_size}x{img_size} 输入硬编码了展平维度；"
                "若改输入尺寸，需同步修改下方的 128*8*8"
            )
        self.num_classes = num_classes
        self.img_size = img_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(num_classes: int = 5, img_size: int = 128) -> PlantCNN:
    return PlantCNN(num_classes=num_classes, img_size=img_size)


def build_resnet18(num_classes: int = 5, pretrained: bool = False):
    """迁移学习版本：ImageNet 预训练的 ResNet18，只把最后的 fc 换成自己的类别数。

    pretrained=True 时首次运行会下载约 45MB 权重到 torch 缓存目录。
    """
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def build_by_arch(arch: str = "cnn", num_classes: int = 5, img_size: int = 128):
    """按 checkpoint 里记录的 arch 字段建模型，推理时才能知道该用哪个结构。"""
    arch = (arch or "cnn").lower()
    if arch in ("resnet18", "resnet"):
        return build_resnet18(num_classes=num_classes, pretrained=False)
    return build_model(num_classes=num_classes, img_size=img_size)


if __name__ == "__main__":
    net = build_model(5)
    x = torch.zeros(2, 3, 128, 128)
    print(net)
    print("输出维度：", tuple(net(x).shape))
    print("参数量：", sum(p.numel() for p in net.parameters()))
