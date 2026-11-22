# -*- coding: utf-8 -*-
"""花卉分类项目 · 全局配置

所有脚本（1~7）统一从这里读取类别、路径、超参数和预处理参数。
原文档把模型结构和参数在 7 个脚本里各写一份，改一处要改七处；
这里抽成公共模块，避免"训练用一套参数、推理用另一套"这类经典事故。
"""

import os

# ---------------------------------------------------------------- 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "plant_images")            # 原始图片：plant_images/<类别>/*.jpg
SPLIT_DIR = os.path.join(BASE_DIR, "plant_images_split")    # 划分后：train/ test/
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEMO_DIR = os.path.join(BASE_DIR, "images")                 # 演示用图片
LOG_DIR = os.path.join(BASE_DIR, "logs")

TEACHER_MODEL_PATH = os.path.join(MODEL_DIR, "plant_classification_model.pth")
STUDENT_MODEL_PATH = os.path.join(MODEL_DIR, "distilled_plant_model.pth")
RESNET_MODEL_PATH = os.path.join(MODEL_DIR, "plant_resnet18_model.pth")   # 可选：迁移学习版本
USER_INFO_PATH = os.path.join(BASE_DIR, "user_info.txt")

# ---------------------------------------------------------------- 类别
# 5 类常见花卉。注意：ImageFolder 会按文件夹名的 Unicode 编码排序，
# 实际训练出的顺序是 ['兰花', '向日葵', '玫瑰', '菊花', '郁金香']，
# 以 checkpoint 里保存的 class_names 为准，不要按本列表的顺序去解读输出下标。
CLASSES = ["兰花", "向日葵", "玫瑰", "菊花", "郁金香"]

# 每类用于爬虫的检索词（可多个，轮流翻页，提高召回和多样性）
CLASS_KEYWORDS = {
    "兰花": ["兰花", "蝴蝶兰", "兰花 盆栽"],
    "向日葵": ["向日葵", "向日葵花海", "向日葵 花朵"],
    "玫瑰": ["玫瑰花", "红玫瑰", "玫瑰花束"],
    "菊花": ["菊花", "菊花 花朵", "秋菊"],
    "郁金香": ["郁金香", "郁金香花海", "郁金香 花朵"],
}

TARGET_PER_CLASS = 150        # 每类目标图片数
MIN_SIDE = 100               # 短边小于该值的图丢弃（缩略图太小，画质差）
MIN_BYTES = 6 * 1024         # 小于 6KB 视为占位图/坏图

# ---------------------------------------------------------------- 训练超参
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
NUM_WORKERS = 0              # Windows 下必须为 0，否则 DataLoader 多进程会报错
SEED = 42

# ImageNet 均值方差：训练、测试、推理三处必须完全一致
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 知识蒸馏超参（已按标准公式修正：alpha 取 (0,1)，硬损失权重为 1-alpha）
TEMPERATURE = 4.0
ALPHA = 0.7
DISTILL_EPOCHS = 50

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dirs():
    for d in (RAW_DIR, SPLIT_DIR, MODEL_DIR, DEMO_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def eval_transform():
    """评估/推理用的预处理（无任何随机增强）。"""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def train_transform():
    """训练用的预处理：先缩放到稍大尺寸再随机裁剪，效果比直接 Resize 后裁剪更好。"""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
