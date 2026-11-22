# -*- coding: utf-8 -*-
"""⑦ 集成预测界面（教师 + 学生，支持摄像头）

用法：python 7.distillation_predictor.py

三种模式：
  教师      单模型基线
  学生      蒸馏后的模型
  集成      两个模型 logits 取平均后再 softmax（不是"两个概率取平均"，前者保留尺度信息）
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import torch
from PIL import Image, ImageTk
from torch.utils.data import DataLoader

import plant_config as cfg
import ui_common
from data_utils import SafeImageFolder

DISPLAY_W, DISPLAY_H = 760, 430


class EnsembleApp:
    def __init__(self, root, teacher, student, class_names, device):
        self.root = root
        self.teacher = teacher
        self.student = student
        self.class_names = class_names
        self.device = device

        self.cap = None
        self.is_camera_open = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_frame = None
        self.current_image = None
        self.eval_queue = queue.Queue()

        root.title("花卉图像分类 · 集成预测")
        root.geometry("1140x800")
        root.configure(bg="#f5f7fa")
        ui_common.bind_center(root, 1140, 800)

        tk.Label(root, text="花 卉 分 类 · 教 师 / 学 生 / 集 成", font=("Microsoft YaHei", 20, "bold"),
                 bg="#f5f7fa", fg="#2c3e50").pack(pady=(14, 2))
        tk.Label(root, text=f"类别：{' / '.join(class_names)}", font=("Microsoft YaHei", 10),
                 bg="#f5f7fa", fg="#7f8c8d").pack()

        body = tk.Frame(root, bg="#f5f7fa")
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 左：画面
        left = tk.Frame(body, bg="#212121", bd=1, relief=tk.SOLID)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.display_label = tk.Label(left, text="请选择图片或打开摄像头", bg="#212121", fg="#9e9e9e",
                                      font=("Microsoft YaHei", 13))
        self.display_label.pack(padx=6, pady=6)

        # 右：控制 + 结果
        right = tk.Frame(body, width=330, bg="white", bd=1, relief=tk.SOLID)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(14, 0))
        right.pack_propagate(False)

        tk.Label(right, text="预测模式", font=("Microsoft YaHei", 11, "bold"),
                 bg="white", fg="#37474f").pack(pady=(16, 4))
        self.mode_var = tk.StringVar(value="ensemble")
        for value, text in (("ensemble", "集成模型（教师+学生）"),
                            ("teacher", "仅教师模型"),
                            ("student", "仅学生模型")):
            tk.Radiobutton(right, text=text, variable=self.mode_var, value=value,
                           bg="white", font=("Microsoft YaHei", 10),
                           activebackground="white", anchor="w",
                           state=tk.NORMAL if self._mode_available(value) else tk.DISABLED
                           ).pack(fill=tk.X, padx=26)

        ttk.Separator(right, orient="horizontal").pack(fill=tk.X, padx=16, pady=10)

        tk.Label(right, text="识别结果", font=("Microsoft YaHei", 11, "bold"), bg="white",
                 fg="#37474f").pack()
        self.result_label = tk.Label(right, text="—", font=("Microsoft YaHei", 22, "bold"),
                                     bg="white", fg="#4a90d9", wraplength=290, justify=tk.CENTER)
        self.result_label.pack(pady=4)
        self.conf_label = tk.Label(right, text="", font=("Microsoft YaHei", 10), bg="white",
                                   fg="#607d8b", justify=tk.LEFT, wraplength=280)
        self.conf_label.pack()

        self.progress = ttk.Progressbar(right, mode="indeterminate", length=250)

        # 底部按钮
        btns = tk.Frame(root, bg="#f5f7fa")
        btns.pack(pady=(0, 12))
        self.btn_local = tk.Button(btns, text="选择本地图片", width=13, font=("Microsoft YaHei", 11),
                                   bg="#4a90d9", fg="white", relief=tk.FLAT, command=self.choose_local)
        self.btn_local.pack(side=tk.LEFT, padx=6)
        self.btn_cam = tk.Button(btns, text="打开摄像头", width=12, font=("Microsoft YaHei", 11),
                                 bg="#5cb85c", fg="white", relief=tk.FLAT, command=self.open_camera)
        self.btn_cam.pack(side=tk.LEFT, padx=6)
        self.btn_shot = tk.Button(btns, text="执行预测", width=12, font=("Microsoft YaHei", 11),
                                  bg="#f0ad4e", fg="white", relief=tk.FLAT, command=self.predict)
        self.btn_shot.pack(side=tk.LEFT, padx=6)
        self.btn_eval = tk.Button(btns, text="评估准确率", width=13, font=("Microsoft YaHei", 11),
                                  bg="#7f8c8d", fg="white", relief=tk.FLAT,
                                  command=self.eval_ensemble_accuracy)
        self.btn_eval.pack(side=tk.LEFT, padx=6)
        self.btn_close = tk.Button(btns, text="关闭摄像头", width=12, font=("Microsoft YaHei", 11),
                                   bg="#95a5a6", fg="white", relief=tk.FLAT,
                                   state=tk.DISABLED, command=self.close_camera_click)
        self.btn_close.pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(root, text="就绪", bg="#dfe6ec", fg="#607d8b", anchor="w",
                               font=("Microsoft YaHei", 9))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _mode_available(self, mode):
        if mode == "ensemble":
            return self.teacher is not None and self.student is not None
        if mode == "teacher":
            return self.teacher is not None
        return self.student is not None

    # -------------------------------------------------------------- 摄像头
    def open_camera(self):
        if self.is_camera_open:
            return
        self.status.config(text="正在打开摄像头…")
        self.root.update_idletasks()
        cap = ui_common.open_camera(0, DISPLAY_W, DISPLAY_H)
        if cap is None:
            messagebox.showerror("打开失败", "摄像头打不开，请检查是否被其它程序占用或隐私权限。")
            self.status.config(text="摄像头打开失败")
            return
        self.cap = cap
        self.is_camera_open = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        self.btn_cam.config(state=tk.DISABLED)
        self.btn_close.config(state=tk.NORMAL)
        self.status.config(text="摄像头已打开")
        self.root.after(30, self._refresh_display)

    def _read_loop(self):
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
            self.current_frame = frame
            small = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            try:
                self.frame_queue.put_nowait(small)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(small)

    def _refresh_display(self):
        if not self.is_camera_open:
            return
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            photo = ImageTk.PhotoImage(image=ui_common.frame_to_pil(frame))
            self.display_label.config(image=photo, text="")
            self.display_label.photo = photo
        self.root.after(30, self._refresh_display)

    def _reset_camera_ui(self):
        self.is_camera_open = False
        self.cap = None
        self.display_label.config(image="", text="请选择图片或打开摄像头")
        self.display_label.photo = None
        self.btn_cam.config(state=tk.NORMAL)
        self.btn_close.config(state=tk.DISABLED)
        self.status.config(text="摄像头已断开")

    def close_camera_click(self):
        self.is_camera_open = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self._reset_camera_ui()

    # -------------------------------------------------------------- 图片
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

    # -------------------------------------------------------------- 预测
    def predict(self):
        if self.current_frame is not None and self.is_camera_open:
            image = ui_common.frame_to_pil(self.current_frame)
        elif self.current_image is not None:
            image = self.current_image
        else:
            messagebox.showwarning("提示", "请先选择本地图片或打开摄像头！")
            return

        mode = self.mode_var.get()
        tensor = cfg.eval_transform()(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            t_logits = self.teacher(tensor) if self.teacher is not None else None
            s_logits = self.student(tensor) if self.student is not None else None
            if mode == "ensemble" and t_logits is not None and s_logits is not None:
                logits = (t_logits + s_logits) / 2.0
                used = "集成"
            elif mode == "teacher" and t_logits is not None:
                logits, used = t_logits, "教师"
            elif mode == "student" and s_logits is not None:
                logits, used = s_logits, "学生"
            else:
                logits = t_logits if t_logits is not None else s_logits
                used = "教师" if t_logits is not None else "学生"
            probs = torch.softmax(logits, dim=1)[0]

        k = min(3, len(self.class_names))
        values, indices = torch.topk(probs, k)
        top = [(self.class_names[i], v * 100) for v, i in zip(values.tolist(), indices.tolist())]

        self.result_label.config(text=top[0][0])
        detail = [f"模式：{used}", f"置信度：{top[0][1]:.2f}%", ""]
        detail += [f"{name}  {conf:.2f}%" for name, conf in top]
        self.conf_label.config(text="\n".join(detail))
        self.status.config(text=f"[{used}] 预测：{top[0][0]}（{top[0][1]:.2f}%）")

    # -------------------------------------------------------------- 集成本体评估
    def eval_ensemble_accuracy(self):
        test_dir = os.path.join(cfg.SPLIT_DIR, "test")
        if not os.path.isdir(test_dir):
            messagebox.showwarning("提示", "找不到测试集，请先运行 2.split_data.py")
            return
        if self.teacher is None and self.student is None:
            messagebox.showwarning("提示", "没有可用模型")
            return

        self.btn_eval.config(state=tk.DISABLED)
        self.progress.pack(pady=8)
        self.progress.start(12)
        self.status.config(text="正在后台评估…")
        threading.Thread(target=self._eval_worker, args=(test_dir,), daemon=True).start()
        self.root.after(100, self._poll_eval_result)

    def _eval_worker(self, test_dir):
        try:
            ds = SafeImageFolder(test_dir, cfg.eval_transform())
            loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.NUM_WORKERS)
            t_ok = s_ok = e_ok = total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    t_logits = self.teacher(images) if self.teacher is not None else None
                    s_logits = self.student(images) if self.student is not None else None
                    if t_logits is not None:
                        t_ok += (t_logits.argmax(1) == labels).sum().item()
                    if s_logits is not None:
                        s_ok += (s_logits.argmax(1) == labels).sum().item()
                    if t_logits is not None and s_logits is not None:
                        e_ok += (((t_logits + s_logits) / 2).argmax(1) == labels).sum().item()
                    total += labels.size(0)
            self.eval_queue.put(("ok", t_ok, s_ok, e_ok, total))
        except Exception as e:
            self.eval_queue.put(("err", str(e), 0, 0, 0))

    def _poll_eval_result(self):
        try:
            result = self.eval_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_eval_result)
            return
        self.progress.stop()
        self.progress.pack_forget()
        self.btn_eval.config(state=tk.NORMAL)
        if result[0] == "err":
            self.status.config(text="评估失败")
            messagebox.showerror("评估失败", result[1])
            return
        _, t_ok, s_ok, e_ok, total = result
        lines = [f"测试集样本：{total} 张"]
        if total:
            if self.teacher is not None:
                lines.append(f"教师：{t_ok / total:.2%}（{t_ok}/{total}）")
            if self.student is not None:
                lines.append(f"学生：{s_ok / total:.2%}（{s_ok}/{total}）")
            if self.teacher is not None and self.student is not None:
                lines.append(f"集成：{e_ok / total:.2%}（{e_ok}/{total}）")
        self.status.config(text="评估完成")
        messagebox.showinfo("集成准确率", "\n".join(lines))

    def on_close(self):
        self.is_camera_open = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.root.destroy()


def main():
    if not ui_common.ask_login("花卉分类系统 - 登录"):
        return

    teacher, t_classes, device = ui_common.load_model(cfg.TEACHER_MODEL_PATH)
    student, s_classes, _ = ui_common.load_model(cfg.STUDENT_MODEL_PATH)

    if teacher is not None and student is not None:
        if t_classes != s_classes:
            messagebox.showerror("错误", f"两个模型的类别顺序不一致！\n教师：{t_classes}\n学生：{s_classes}\n"
                                         "集成结果会完全错乱，请重新训练学生模型。")
            return
        class_names = t_classes
    else:
        class_names = t_classes if teacher is not None else s_classes

    root = tk.Tk()
    EnsembleApp(root, teacher, student, class_names, device)
    root.mainloop()


if __name__ == "__main__":
    main()
