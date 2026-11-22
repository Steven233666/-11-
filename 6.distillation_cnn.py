# -*- coding: utf-8 -*-
"""⑥ 知识蒸馏训练（教师 → 学生）

用法：python 6.distillation_cnn.py [--epochs 50] [--temperature 4] [--alpha 0.7]

前置条件：
  models/plant_classification_model.pth（教师）已训练完成
  plant_images_split/ 已划分

与原始文档的差异（重要）：
  文档里的实现是 total = alpha*soft + 1.0*hard，且 alpha=2.0，软损失权重远大于硬损失，
  这不是标准写法。这里按标准公式修正：
      Loss = alpha * T^2 * KL(p_teacher^T || p_student^T) + (1 - alpha) * CE(student, label)
  默认 T=4、alpha=0.7，可自行改参数做对比实验。
"""

import argparse
import copy
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import plant_config as cfg
from data_utils import SafeImageFolder, run_epoch
from model import build_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class DistillationLoss(nn.Module):
    """软标签 KL + 硬标签交叉熵。

    T 越大，教师输出的分布越平滑，"像玫瑰但更像菊花"这类暗知识越明显；
    乘 T^2 是为了补偿除以 T 后在梯度上出现的 1/T 因子，让软硬损失保持同一量级。
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        hard_loss = self.ce_loss(student_logits, labels)
        total = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
        return total, soft_loss, hard_loss


def train_one_epoch(student, teacher, loader, loss_fn, optimizer, device, desc=""):
    student.train()
    teacher.eval()
    total_loss = 0.0
    correct, total = 0, 0
    from tqdm import tqdm

    pbar = tqdm(loader, desc=desc, ncols=90, leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher(images)     # 教师只前向，不建计算图

        student_logits = student(images)
        loss, soft_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (student_logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.3f}")

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.DISTILL_EPOCHS)
    parser.add_argument("--temperature", type=float, default=cfg.TEMPERATURE)
    parser.add_argument("--alpha", type=float, default=cfg.ALPHA)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    args = parser.parse_args()

    if not os.path.exists(cfg.TEACHER_MODEL_PATH):
        raise SystemExit("未找到教师模型！请先运行 3.train_cnn.py")
    test_dir = os.path.join(cfg.SPLIT_DIR, "test")
    if not os.path.isdir(test_dir):
        raise SystemExit("未找到划分后的数据集！请先运行 2.split_data.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备：{device}")

    try:
        ckpt = torch.load(cfg.TEACHER_MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:                     # 老版本 torch 没有 weights_only 参数
        ckpt = torch.load(cfg.TEACHER_MODEL_PATH, map_location=device)
    class_names = list(ckpt["class_names"])
    num_classes = len(class_names)

    # ---------------- 数据集（必须与教师训练时同一类别顺序）
    train_ds = SafeImageFolder(os.path.join(cfg.SPLIT_DIR, "train"), cfg.train_transform())
    test_ds = SafeImageFolder(test_dir, cfg.eval_transform())
    if train_ds.classes != class_names or test_ds.classes != class_names:
        raise SystemExit(f"类别顺序与教师模型不一致！\n教师：{class_names}\n当前：{train_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS)

    # ---------------- 教师：冻结
    teacher = build_model(num_classes=num_classes).to(device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    _, teacher_acc = run_epoch(teacher, test_loader, criterion, device, desc="教师评估")
    print(f"教师模型测试准确率：{teacher_acc:.4f}（{teacher_acc:.2%}）")

    # ---------------- 学生：同结构
    student = build_model(num_classes=num_classes).to(device)
    loss_fn = DistillationLoss(args.temperature, args.alpha)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    best_acc, best_state, best_epoch = 0.0, None, 0
    acc_curve, loss_curve = [], []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            student, teacher, train_loader, loss_fn, optimizer, device,
            desc=f"蒸馏 {epoch}/{args.epochs} [Train]")
        _, te_acc = run_epoch(student, test_loader, criterion, device,
                              desc=f"蒸馏 {epoch}/{args.epochs} [Test ]")
        scheduler.step()

        acc_curve.append(te_acc)
        loss_curve.append(tr_loss)

        if te_acc > best_acc:
            best_acc, best_epoch = te_acc, epoch
            best_state = copy.deepcopy(student.state_dict())

        print(f"Epoch {epoch:>2} | 平均训练损失 {tr_loss:.4f} | 学生测试 acc {te_acc:.4f} | "
              f"教师基准 {teacher_acc:.4f}")

    print(f"\n蒸馏结束，耗时 {(time.time() - started) / 60:.1f} 分钟")
    print(f"最优学生准确率 {best_acc:.4f}（第 {best_epoch} 轮），教师 {teacher_acc:.4f}")
    if best_acc >= teacher_acc:
        print(f"提升 {(best_acc - teacher_acc) * 100:.2f} 个百分点")
    else:
        print(f"未超过教师，差 {(teacher_acc - best_acc) * 100:.2f} 个百分点（可调低 T 或 alpha 重试）")

    if best_state is not None:
        student.load_state_dict(best_state)

    torch.save({
        "epoch": best_epoch,
        "model_state_dict": student.state_dict(),
        "best_test_acc": best_acc,
        "final_accuracy": best_acc,
        "class_names": class_names,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "teacher_acc": teacher_acc,
        "img_size": cfg.IMG_SIZE,
        "history": {"test_acc": acc_curve, "train_loss": loss_curve},
    }, cfg.STUDENT_MODEL_PATH)
    print(f"蒸馏模型已保存：{cfg.STUDENT_MODEL_PATH}")

    # ---------------- 曲线
    try:
        fig, ax1 = plt.subplots(figsize=(8, 4.6))
        epochs = range(1, len(acc_curve) + 1)
        ax1.plot(epochs, acc_curve, "o-", color="#4a90d9", label="学生测试准确率")
        ax1.axhline(teacher_acc, color="#e8843c", linestyle="--", label=f"教师基准 {teacher_acc:.2%}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("知识蒸馏训练曲线（T=%.1f, alpha=%.2f）" % (args.temperature, args.alpha))
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.BASE_DIR, "distillation_history.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("已生成 distillation_history.png")
    except Exception as e:
        print(f"绘图失败（不影响模型）：{e}")


if __name__ == "__main__":
    main()
