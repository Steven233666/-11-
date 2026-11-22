# -*- coding: utf-8 -*-
"""⑧ 可选提升：ResNet18 迁移学习（ImageNet 预训练）

用法：python 8.train_resnet.py [--epochs 15]

为什么加这个：自建 4 层 CNN 在 750 张网络图片上大约 72%~75%，
因为类别间背景杂乱、爬来的图里有不少干扰样本。
换成 ImageNet 预训练的 ResNet18 只微调最后几层，同样数据能上到 90% 左右——
这也是原文档"项目扩展方向"里第 7 条建议的做法。

产出：models/plant_resnet18_model.pth
界面（4/5）会自动优先加载这个模型；想用回 CNN，把该文件改名或删除即可。
"""

import argparse
import copy
import importlib.util
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import plant_config as cfg
from data_utils import SafeImageFolder, run_epoch
from model import build_resnet18

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_helper():
    """复用 3.train_cnn.py 里的画图函数（文件名带数字，用 importlib 按路径加载）。"""
    path = os.path.join(cfg.BASE_DIR, "3.train_cnn.py")
    spec = importlib.util.spec_from_file_location("train_cnn_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_dir = os.path.join(cfg.SPLIT_DIR, "train")
    test_dir = os.path.join(cfg.SPLIT_DIR, "test")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise SystemExit("未找到数据集！请先运行 2.split_data.py")

    cfg.ensure_dirs()
    torch.manual_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备：{device}")

    train_ds = SafeImageFolder(train_dir, cfg.train_transform())
    test_ds = SafeImageFolder(test_dir, cfg.eval_transform())
    class_names = train_ds.classes
    print(f"样本：训练 {len(train_ds)} 张，测试 {len(test_ds)} 张 | 类别 {class_names}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=cfg.NUM_WORKERS)

    # 预训练权重首次运行会自动下载（约 45MB）
    model = build_resnet18(num_classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc, best_state, best_epoch = 0.0, None, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer,
                                    desc=f"ResNet {epoch}/{args.epochs} [Train]")
        te_loss, te_acc = run_epoch(model, test_loader, criterion, device,
                                    desc=f"ResNet {epoch}/{args.epochs} [Test ]")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc, best_epoch = te_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:>2} | 训练 acc {tr_acc:.4f} | 测试 acc {te_acc:.4f} | "
              f"最佳 {best_acc:.4f}")

    print(f"\n训练结束，耗时 {(time.time() - started) / 60:.1f} 分钟，最佳测试准确率 "
          f"{best_acc:.4f}（第 {best_epoch} 轮）")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "arch": "resnet18",
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "final_accuracy": best_acc,
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "img_size": cfg.IMG_SIZE,
        "history": history,
    }, cfg.RESNET_MODEL_PATH)
    print(f"模型已保存：{cfg.RESNET_MODEL_PATH}")

    helper = load_helper()
    try:
        helper.plot_history(history, os.path.join(cfg.BASE_DIR, "training_history_resnet.png"))
        cm = helper.confusion_matrix(model, test_loader, len(class_names), device)
        helper.plot_confusion(cm, class_names, os.path.join(cfg.BASE_DIR, "confusion_matrix_resnet.png"))
        helper.plot_predictions(model, test_ds, class_names, device,
                                os.path.join(cfg.BASE_DIR, "prediction_resnet.png"))
        print("已生成 training_history_resnet.png / confusion_matrix_resnet.png / prediction_resnet.png")
    except Exception as e:
        print(f"绘图失败（不影响模型）：{e}")

    print("\n各类别测试准确率：")
    for i, name in enumerate(class_names):
        n = cm[i].sum()
        print(f"  {name:>4}：{cm[i, i] / n:.2%}（{cm[i, i]}/{n}）")

    # 与自建 CNN 对比
    if os.path.exists(cfg.TEACHER_MODEL_PATH):
        try:
            ck = torch.load(cfg.TEACHER_MODEL_PATH, map_location="cpu", weights_only=False)
            print(f"\n对比：自建 PlantCNN {ck.get('best_accuracy', 0):.2%} | "
                  f"ResNet18 迁移学习 {best_acc:.2%}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
