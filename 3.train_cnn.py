# -*- coding: utf-8 -*-
"""③ 花卉 CNN 训练与评估

用法：python 3.train_cnn.py [--epochs 20] [--batch-size 32] [--lr 0.001]

产出：
  models/plant_classification_model.pth   （教师模型，界面默认读取）
  training_history.png                    （准确率 + 损失曲线）
  confusion_matrix.png                    （混淆矩阵）
  prediction_示例.png                      （测试集抽样预测可视化）
"""

import argparse
import copy
import os
import random
import time

import matplotlib
matplotlib.use("Agg")                        # 只出图不弹窗，无 GUI 环境也不会报错
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

import plant_config as cfg
from data_utils import SafeImageFolder, run_epoch
from model import build_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------------ 可视化
def plot_history(history: dict, save_path: str):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, history["train_acc"], "o-", label="训练准确率", color="#3b7dd8")
    axes[0].plot(epochs, history["test_acc"], "s-", label="测试准确率", color="#e8843c")
    axes[0].set_title("准确率曲线")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_loss"], "o-", label="训练损失", color="#3b7dd8")
    axes[1].plot(epochs, history["test_loss"], "s-", label="测试损失", color="#e8843c")
    axes[1].set_title("损失曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: list, save_path: str):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("测试集混淆矩阵")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(model, dataset, class_names, device, save_path: str, per_class: int = 2):
    """从测试集每类抽 2 张，标注真实类别 / 预测类别 / 置信度。"""
    picked = []
    for i in range(len(class_names)):
        idxs = [k for k, (_, y) in enumerate(dataset.samples) if y == i][:per_class]
        picked.extend(idxs)
    if not picked:
        return

    cols = (len(picked) + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6.4))
    axes = np.atleast_2d(axes)
    model.eval()
    for ax, idx in zip(axes.flatten(), picked):
        path, label = dataset.samples[idx]
        image = Image.open(path).convert("RGB")
        tensor = cfg.eval_transform()(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(tensor), 1)[0]
        conf, pred = torch.max(prob, 0)
        ok = class_names[pred.item()] == class_names[label]
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"真:{class_names[label]} 预:{class_names[pred.item()]} {conf.item() * 100:.1f}%",
                     color="#2e7d32" if ok else "#c62828", fontsize=9)
    for ax in axes.flatten()[len(picked):]:
        ax.axis("off")
    fig.suptitle("测试集抽样预测结果（绿=正确，红=错误）", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def confusion_matrix(model, loader, num_classes, device):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    model.eval()
    for images, labels in loader:
        preds = model(images.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(labels.numpy(), preds):
            cm[t, p] += 1
    return cm


# ------------------------------------------------------------------ 主流程
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    args = parser.parse_args()

    train_dir = os.path.join(cfg.SPLIT_DIR, "train")
    test_dir = os.path.join(cfg.SPLIT_DIR, "test")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise SystemExit("未找到划分后的数据集目录！请先运行 2.split_data.py")

    cfg.ensure_dirs()
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备：{device}")

    train_ds = SafeImageFolder(train_dir, cfg.train_transform())
    test_ds = SafeImageFolder(test_dir, cfg.eval_transform())
    class_names = train_ds.classes
    if test_ds.classes != class_names:
        raise SystemExit(f"训练/测试集类别不一致：{train_ds.classes} vs {test_ds.classes}")
    print(f"类别（ImageFolder 按 Unicode 排序）：{class_names}")
    print(f"样本：训练 {len(train_ds)} 张，测试 {len(test_ds)} 张"
          + (f"（已丢弃无效图片 {train_ds.skipped + test_ds.skipped} 张）"
             if train_ds.skipped + test_ds.skipped else ""))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=cfg.NUM_WORKERS)

    model = build_model(num_classes=len(class_names), img_size=cfg.IMG_SIZE).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc, best_state, best_epoch = 0.0, None, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer,
                                    desc=f"Epoch {epoch}/{args.epochs} [Train]")
        te_loss, te_acc = run_epoch(model, test_loader, criterion, device,
                                    desc=f"Epoch {epoch}/{args.epochs} [Test ]")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc, best_epoch = te_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:>2} | 训练 loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"测试 loss {te_loss:.4f} acc {te_acc:.4f}")

    print(f"\n训练结束，耗时 {(time.time() - started) / 60:.1f} 分钟")
    print(f"最佳测试准确率：{best_acc:.4f}（第 {best_epoch} 个 epoch）")

    if best_state is not None:
        model.load_state_dict(best_state)
    _, final_acc = run_epoch(model, test_loader, criterion, device, desc="最终评估")

    # 保存 checkpoint：class_names 必须一起存，否则推理只能拿到 0~4 的下标
    torch.save({
        "arch": "cnn",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        "final_accuracy": final_acc,
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "img_size": cfg.IMG_SIZE,
        "history": history,
    }, cfg.TEACHER_MODEL_PATH)
    print(f"模型已保存：{cfg.TEACHER_MODEL_PATH}")

    try:
        plot_history(history, os.path.join(cfg.BASE_DIR, "training_history.png"))
        cm = confusion_matrix(model, test_loader, len(class_names), device)
        plot_confusion(cm, class_names, os.path.join(cfg.BASE_DIR, "confusion_matrix.png"))
        plot_predictions(model, test_ds, class_names, device,
                         os.path.join(cfg.BASE_DIR, "prediction_示例.png"))
        print("已生成 training_history.png / confusion_matrix.png / prediction_示例.png")
    except Exception as e:
        print(f"绘图失败（不影响模型）：{e}")
        print("\nEpoch | train_acc | test_acc | train_loss | test_loss")
        for i in range(len(history["train_acc"])):
            print(f"{i + 1:>5} | {history['train_acc'][i]:.4f} | {history['test_acc'][i]:.4f} | "
                  f"{history['train_loss'][i]:.4f} | {history['test_loss'][i]:.4f}")
        cm = confusion_matrix(model, test_loader, len(class_names), device)

    print("\n各类别测试准确率：")
    for i, name in enumerate(class_names):
        n = cm[i].sum()
        acc = cm[i, i] / n if n else 0
        print(f"  {name:>4}：{acc:.2%}（{cm[i, i]}/{n}）")


if __name__ == "__main__":
    main()
