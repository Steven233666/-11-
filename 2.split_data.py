# -*- coding: utf-8 -*-
"""② 训练集 / 测试集划分（逐类别 8:2）

用法：python 2.split_data.py [--ratio 0.8] [--clean]

要点：
  - 固定随机种子 42，保证每次划分结果一致、实验可复现；
  - 逐类别划分，每类都有 8:2，避免整体打乱后某类全进训练集；
  - 用 os.listdir + 后缀小写取一次，而不是"每种后缀 glob 一次"——
    Windows 的 glob 会做 normcase（转小写），*.jpg 和 *.JPG 匹配同一批文件，
    累加后数量翻倍。
"""

import argparse
import os
import random
import shutil

import plant_config as cfg


def list_images(folder: str) -> list:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, name) for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name))
            and os.path.splitext(name)[1].lower() in cfg.VALID_EXTS]


def split_class(cls: str, ratio: float, clean: bool):
    src = os.path.join(cfg.RAW_DIR, cls)
    images = list_images(src)
    if not images:
        print(f"  「{cls}」没有图片，跳过")
        return 0, 0

    random.shuffle(images)                       # 种子已在 main 中固定
    n_train = int(len(images) * ratio)
    n_train = max(1, min(n_train, len(images) - 1)) if len(images) > 1 else len(images)
    train_imgs, test_imgs = images[:n_train], images[n_train:]

    for subset, items in (("train", train_imgs), ("test", test_imgs)):
        dst_dir = os.path.join(cfg.SPLIT_DIR, subset, cls)
        if clean and os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir, exist_ok=True)
        for p in items:
            shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))

    print(f"  「{cls}」：总 {len(images)} 张 → 训练集 {len(train_imgs)} 张 | 测试集 {len(test_imgs)} 张")
    return len(train_imgs), len(test_imgs)


def main():
    parser = argparse.ArgumentParser(description="划分训练/测试集")
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--clean", action="store_true", help="先清空旧的 train/test 目录")
    args = parser.parse_args()

    if not os.path.isdir(cfg.RAW_DIR):
        raise SystemExit(f"未找到原始数据目录：{cfg.RAW_DIR}\n请先运行 1.download_data.py")

    random.seed(cfg.SEED)
    print("========== 开始划分训练集/测试集（8:2）==========")
    tr = te = 0
    classes = sorted([d for d in os.listdir(cfg.RAW_DIR)
                      if os.path.isdir(os.path.join(cfg.RAW_DIR, d))])
    for cls in classes:
        a, b = split_class(cls, args.ratio, args.clean)
        tr += a
        te += b

    print("\n========== 数据集划分完成 ==========")
    print(f"整体统计：训练集 {tr} 张 | 测试集 {te} 张")
    if tr < 200:
        print("警告：样本偏少，准确率会很飘，建议每类至少 100 张")
    print(f"数据目录：{cfg.SPLIT_DIR}")


if __name__ == "__main__":
    main()
