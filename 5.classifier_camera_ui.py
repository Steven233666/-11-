# -*- coding: utf-8 -*-
"""⑤ 花卉分类界面（摄像头版）

用法：python 5.classifier_camera_ui.py

线程模型（本文件是"正确示范"）：
  子线程只负责 cap.read() 并把帧塞进容量 1 的队列；
  所有 Tk 控件操作（包括 PhotoImage 创建）都在主线程，用 root.after 定时刷新。
  子线程里直接 PhotoImage + label.config 的写法，在 Windows 上典型表现是画面全黑。
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

import plant_config as cfg
import ui_common

DISPLAY_W, DISPLAY_H = 700, 400


class CameraApp:
    def __init__(self, root, model, class_names, device):
        self.root = root
        self.model = model
        self.class_names = class_names
        self.device = device

        self.cap = None
        self.is_camera_open = False
        self.camera_thread = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_frame = None            # 原始 BGR 帧，用于拍照预测
        self.current_image = None            # 本地图片（PIL）
        self.pending_photo = None

        root.title("花卉图像分类系统 · 摄像头")
        root.geometry("1060x760")
        root.configure(bg="#f5f7fa")
        ui_common.bind_center(root, 1060, 760)

        tk.Label(root, text="花 卉 实 时 识 别", font=("Microsoft YaHei", 22, "bold"),
                 bg="#f5f7fa", fg="#2c3e50").pack(pady=(14, 2))
        tk.Label(root, text=f"支持类别：{' / '.join(class_names)}", font=("Microsoft YaHei", 10),
                 bg="#f5f7fa", fg="#7f8c8d").pack()

        body = tk.Frame(root, bg="#f5f7fa")
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        left = tk.Frame(body, bg="black", bd=1, relief=tk.SOLID)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.display_label = tk.Label(left, text="摄像头未打开", bg="#212121", fg="#9e9e9e",
                                      font=("Microsoft YaHei", 13))
        self.display_label.pack(padx=6, pady=6)

        right = tk.Frame(body, width=320, bg="white", bd=1, relief=tk.SOLID)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(14, 0))
        right.pack_propagate(False)
        tk.Label(right, text="识别结果", font=("Microsoft YaHei", 13, "bold"),
                 bg="white", fg="#2c3e50").pack(pady=(20, 6))
        self.result_label = tk.Label(right, text="—", font=("Microsoft YaHei", 22, "bold"),
                                     bg="white", fg="#4a90d9", wraplength=290, justify=tk.CENTER)
        self.result_label.pack(pady=6)
        self.conf_label = tk.Label(right, text="", font=("Microsoft YaHei", 11),
                                   bg="white", fg="#7f8c8d", justify=tk.LEFT, wraplength=280)
        self.conf_label.pack()
        tk.Label(right, text="提示：让花朵占据画面中央，\n光线充足时识别更准。",
                 font=("Microsoft YaHei", 9), bg="white", fg="#b0bec5",
                 justify=tk.LEFT).pack(side=tk.BOTTOM, pady=16, padx=10)

        btns = tk.Frame(root, bg="#f5f7fa")
        btns.pack(pady=(0, 14))
        self.btn_open = tk.Button(btns, text="打开摄像头", width=13, font=("Microsoft YaHei", 11),
                                  bg="#4a90d9", fg="white", relief=tk.FLAT,
                                  command=self.open_camera)
        self.btn_open.pack(side=tk.LEFT, padx=8)
        self.btn_shot = tk.Button(btns, text="拍照预测", width=12, font=("Microsoft YaHei", 11),
                                  bg="#5cb85c", fg="white", relief=tk.FLAT,
                                  state=tk.DISABLED, command=self.snapshot_predict)
        self.btn_shot.pack(side=tk.LEFT, padx=8)
        self.btn_local = tk.Button(btns, text="选择本地图片", width=13, font=("Microsoft YaHei", 11),
                                   bg="#7f8c8d", fg="white", relief=tk.FLAT,
                                   command=self.choose_local)
        self.btn_local.pack(side=tk.LEFT, padx=8)
        self.btn_close = tk.Button(btns, text="关闭摄像头", width=13, font=("Microsoft YaHei", 11),
                                   bg="#d9534f", fg="white", relief=tk.FLAT,
                                   state=tk.DISABLED, command=self.close_camera_click)
        self.btn_close.pack(side=tk.LEFT, padx=8)

        self.status = tk.Label(root, text="就绪", bg="#dfe6ec", fg="#607d8b", anchor="w",
                               font=("Microsoft YaHei", 9))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        root.protocol("WM_DELETE_WINDOW", self.on_close)

        if model is None:
            self.status.config(text="未找到模型文件，请先运行 3.train_cnn.py")
            self.btn_shot.config(state=tk.DISABLED)

    # ---------------------------------------------------------- 摄像头
    def open_camera(self):
        if self.is_camera_open:
            return
        self.status.config(text="正在打开摄像头…")
        self.root.update_idletasks()
        cap = ui_common.open_camera(0, DISPLAY_W, DISPLAY_H)
        if cap is None:
            messagebox.showerror("打开失败",
                                 "无法打开摄像头。\n\n排查顺序：\n"
                                 "1. 关闭其它正在使用摄像头的程序（微信/腾讯会议等）\n"
                                 "2. Windows 设置 → 隐私 → 相机，允许应用访问\n"
                                 "3. 笔记本可能有多个摄像头，试试索引 1\n"
                                 "4. 更新摄像头驱动")
            self.status.config(text="摄像头打开失败")
            return
        self.cap = cap
        self.is_camera_open = True
        self.camera_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.camera_thread.start()
        self.btn_open.config(state=tk.DISABLED)
        self.btn_close.config(state=tk.NORMAL)
        self.btn_shot.config(state=tk.NORMAL if self.model else tk.DISABLED)
        self.status.config(text="摄像头已打开")
        self.root.after(30, self._refresh_display)

    def _read_loop(self):
        """子线程：只读帧、只入队，绝不碰任何 Tk 控件。"""
        while self.is_camera_open and self.cap is not None:
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret, frame = False, None
            if not ret or frame is None:
                self.is_camera_open = False
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.root.after(0, self._reset_camera_ui)
                break

            self.current_frame = frame                       # 原图，拍照预测用
            small = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            try:
                self.frame_queue.put_nowait(small)
            except queue.Full:                               # 队列满就丢旧帧，保证实时性
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(small)

    def _refresh_display(self):
        """主线程：定时取帧并刷新，约 33 FPS。"""
        if not self.is_camera_open:
            return
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            photo = ImageTk.PhotoImage(image=ui_common.frame_to_pil(frame))   # ★ 主线程创建
            self.display_label.config(image=photo, text="")
            self.display_label.photo = photo
        self.root.after(30, self._refresh_display)

    def _reset_camera_ui(self):
        self.is_camera_open = False
        self.cap = None
        self.display_label.config(image="", text="摄像头未打开")
        self.display_label.photo = None
        self.btn_open.config(state=tk.NORMAL)
        self.btn_close.config(state=tk.DISABLED)
        self.btn_shot.config(state=tk.DISABLED)
        self.status.config(text="摄像头已断开")

    def close_camera_click(self):
        self._release_camera()
        self._reset_camera_ui()
        self.status.config(text="摄像头已关闭")

    def _release_camera(self):
        self.is_camera_open = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    # ---------------------------------------------------------- 预测
    def snapshot_predict(self):
        if self.model is None:
            messagebox.showwarning("提示", "模型未加载，无法预测")
            return
        if self.current_frame is None:
            messagebox.showwarning("提示", "还没有画面，请先打开摄像头")
            return
        frame = self.current_frame.copy()
        image = ui_common.frame_to_pil(frame)               # BGR → RGB
        self._show_result(image)

    def choose_local(self):
        path = filedialog.askopenfilename(
            title="选择花卉图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("错误", f"图片打不开：{e}")
            return
        self.current_image = image
        disp = image.copy()
        disp.thumbnail((DISPLAY_W, DISPLAY_H))
        photo = ImageTk.PhotoImage(disp)
        self.display_label.config(image=photo, text="")
        self.display_label.photo = photo
        self.status.config(text=f"已载入：{os.path.basename(path)}")
        if self.model is not None:
            self._show_result(image)

    def _show_result(self, image):
        top = ui_common.predict_topk(self.model, image, self.class_names, self.device, k=3)
        if not top:
            return
        name, conf = top[0]
        self.result_label.config(text=name, fg="#2e7d32")
        lines = [f"{n}  {c:.2f}%" for n, c in top]
        self.conf_label.config(text="\n".join(lines))
        self.status.config(text=f"识别：{name}（{conf:.2f}%）")

    # ---------------------------------------------------------- 关闭
    def on_close(self):
        self._release_camera()
        self.root.destroy()


def main():
    if not ui_common.ask_login("花卉分类系统 - 登录"):
        return
    model, class_names, device = ui_common.load_model(cfg.TEACHER_MODEL_PATH)
    root = tk.Tk()
    CameraApp(root, model, class_names, device)
    root.mainloop()


if __name__ == "__main__":
    main()
