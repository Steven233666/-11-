# -*- coding: utf-8 -*-
"""数据加载公共模块：带清洗的 ImageFolder + 训练/评估单轮函数。

为什么单独抽出来：3.train_cnn.py 和 6.distillation_cnn.py 需要完全相同的数据管线，
如果各写一份，很容易出现"训练增强不一致 / 类别顺序不一致"这种隐蔽问题。
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import plant_config as cfg


class SafeImageFolder(Dataset):
    """带清洗的 ImageFolder。

    爬来的图里一定有坏文件、灰度图、非图片：构建时逐个 verify + load 校验，
    校验不过的直接丢弃；读取时强制转 RGB，避免 Conv2d 通道数不匹配。
    """

    def __init__(self, root: str, transform):
        self.root = root
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d))])
        if not self.classes:
            raise RuntimeError(f"{root} 下没有类别文件夹")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        self.skipped = 0

        for cls in self.classes:
            folder = os.path.join(root, cls)
            for name in os.listdir(folder):
                path = os.path.join(folder, name)
                if not os.path.isfile(path):
                    continue
                if os.path.splitext(name)[1].lower() not in cfg.VALID_EXTS:
                    continue
                if self._valid(path):
                    self.samples.append((path, self.class_to_idx[cls]))
                else:
                    self.skipped += 1

    @staticmethod
    def _valid(path: str) -> bool:
        try:
            with Image.open(path) as im:
                im.verify()                 # 查文件头
            with Image.open(path) as im:
                im.load()                   # 真解码，截断文件在这里暴露
            return True
        except Exception:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with open(path, "rb") as f:
                image = Image.open(f).convert("RGB")
            return self.transform(image), label
        except Exception:
            return torch.zeros(3, cfg.IMG_SIZE, cfg.IMG_SIZE), label


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """跑一轮训练或评估，返回 (平均损失, 准确率)。"""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, ncols=90, leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()       # 1. 清空上批梯度
                loss.backward()             # 2. 反向传播
                optimizer.step()            # 3. 更新参数

            total_loss += loss.item() * images.size(0)   # 按样本数加权
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.3f}")

    return total_loss / max(total, 1), correct / max(total, 1)
