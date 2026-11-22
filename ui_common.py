# -*- coding: utf-8 -*-
"""公共 UI / 推理工具

被 4、5、7 三个界面脚本共用：
  - 登录窗口（读 user_info.txt，支持 admin:123456 / admin / 123456 / admin 123456 三种写法）
  - 模型加载（从 checkpoint 读 class_names，避免下标与类别错位）
  - 单张图 top-k 预测
  - 摄像头打开（DSHOW 优先 + 预热丢首帧）
"""

import os
import re
import tkinter as tk

import cv2
from PIL import Image
import torch

import plant_config as cfg
from model import build_by_arch


# ------------------------------------------------------------------ 账号
def load_user_info(path: str = None) -> dict:
    """读取账号文件，统一成 {用户名: 密码}。用户名忽略大小写，密码区分大小写。"""
    path = path or cfg.USER_INFO_PATH
    users = {}
    if not os.path.exists(path):
        return users
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[:\s/]+", line)
            if len(parts) >= 2 and parts[0] and parts[1]:
                users[parts[0].lower()] = parts[1]
    return users


def ask_login(title: str = "花卉分类系统 - 登录") -> bool:
    """弹一个独立登录窗口，登录成功返回 True。"""
    users = load_user_info()
    if not users:
        # 没有账号文件也不阻塞流程，直接放行（仅提示）
        print("提示：未找到 user_info.txt，跳过登录校验")
        return True

    result = {"ok": False}
    win = tk.Tk()
    win.title(title)
    win.geometry("380x220")
    win.resizable(False, False)

    tk.Label(win, text="花 卉 图 像 分 类 系 统", font=("Microsoft YaHei", 16, "bold")).pack(pady=14)

    frame = tk.Frame(win)
    frame.pack(pady=6)
    tk.Label(frame, text="用户名：", font=("Microsoft YaHei", 11)).grid(row=0, column=0, pady=6, sticky="e")
    entry_user = tk.Entry(frame, width=20, font=("Microsoft YaHei", 11))
    entry_user.grid(row=0, column=1, pady=6)
    tk.Label(frame, text="密  码：", font=("Microsoft YaHei", 11)).grid(row=1, column=0, pady=6, sticky="e")
    entry_pass = tk.Entry(frame, width=20, show="*", font=("Microsoft YaHei", 11))
    entry_pass.grid(row=1, column=1, pady=6)

    tip = tk.Label(win, text="默认账号 admin / 123456", fg="#888888", font=("Microsoft YaHei", 9))
    tip.pack()

    def do_login(event=None):
        name = entry_user.get().strip().lower()
        pwd = entry_pass.get().strip()
        if users.get(name) == pwd:
            result["ok"] = True
            win.destroy()
        else:
            tip.config(text="用户名或密码错误，请重试", fg="#d9534f")

    def cancel(event=None):
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=12)
    tk.Button(btn_frame, text="登 录", width=10, font=("Microsoft YaHei", 10),
              bg="#4a90d9", fg="white", command=do_login).pack(side=tk.LEFT, padx=8)
    tk.Button(btn_frame, text="退 出", width=10, font=("Microsoft YaHei", 10),
              command=cancel).pack(side=tk.LEFT, padx=8)

    win.bind("<Return>", do_login)
    win.protocol("WM_DELETE_WINDOW", cancel)
    entry_user.focus_set()
    win.mainloop()
    return result["ok"]


# ------------------------------------------------------------------ 模型
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str = None, device=None):
    """加载 checkpoint，返回 (model, class_names, device)。

    model_path 为空时自动挑选：有 ResNet18 迁移学习模型就优先用它，否则用 CNN 教师模型。
    找不到任何模型时返回 model=None，界面仍可打开，只提示无法预测。
    """
    if model_path is None:
        model_path = cfg.RESNET_MODEL_PATH if os.path.exists(cfg.RESNET_MODEL_PATH) \
            else cfg.TEACHER_MODEL_PATH
    device = device or get_device()

    if not os.path.exists(model_path):
        print(f"未找到模型文件：{model_path}")
        return None, cfg.CLASSES, device

    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:                     # 老版本 torch 没有 weights_only 参数
        ckpt = torch.load(model_path, map_location=device)

    class_names = list(ckpt.get("class_names") or cfg.CLASSES)
    arch = ckpt.get("arch", "cnn")
    model = build_by_arch(arch, num_classes=len(class_names), img_size=cfg.IMG_SIZE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    acc = ckpt.get("final_accuracy") or ckpt.get("best_test_acc")
    if acc is not None:
        try:
            print(f"已加载模型：{os.path.basename(model_path)} | 类别 {class_names} | 准确率 {acc:.2%}")
        except (TypeError, ValueError):
            print(f"已加载模型：{os.path.basename(model_path)} | 类别 {class_names}")
    return model, class_names, device


def predict_topk(model, image, class_names, device, k: int = 3):
    """对 PIL.Image 做单张预测，返回 [(类别, 置信度%), ...]，按概率降序。"""
    if image is None or model is None:
        return []
    img = image.convert("RGB")
    tensor = cfg.eval_transform()(img).unsqueeze(0).to(device)   # 单张也要凑出 batch 维
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    k = min(k, len(class_names))
    values, indices = torch.topk(probs, k)
    return [(class_names[i], round(v * 100, 2)) for v, i in zip(values.tolist(), indices.tolist())]


def predict_label(model, image, class_names, device):
    top = predict_topk(model, image, class_names, device, k=1)
    return top[0] if top else ("未知", 0.0)


# ------------------------------------------------------------------ 摄像头
def open_camera(index: int = 0, width: int = 700, height: int = 380, warmup_reads: int = 5):
    """打开摄像头：DSHOW 优先，MSMF 兜底，并丢掉首帧未就绪的无效帧。

    'can't grab frame. Error: -2147483638'(0x8000000A) 就是首帧未就绪的典型报错，
    不是硬件坏了，预热几次就能拿到有效帧。
    """
    if not hasattr(cv2, "VideoCapture"):
        return None
    backends = [b for b in (getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None)) if b]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok = False
        for _ in range(warmup_reads):
            ret, frame = cap.read()
            if ret and frame is not None:
                ok = True
                break
        if ok:
            return cap
        cap.release()
    return None


def frame_to_pil(frame):
    """OpenCV 的 BGR 帧转 PIL RGB 图（少这一步就会出现"蓝色玫瑰"这种误判）。"""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def bind_center(win, width, height):
    win.update_idletasks()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    x, y = max(0, (sw - width) // 2), max(0, (sh - height) // 3)
    win.geometry(f"{width}x{height}+{x}+{y}")
