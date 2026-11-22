# -*- coding: utf-8 -*-
"""④ 花卉分类界面（本地图片版）

用法：python 4.classifier_ui.py
默认读取 models/plant_classification_model.pth，训练完直接运行即可。

界面要点：
  - 登录窗口与主窗口是两个独立 Tk，登录成功后 destroy 再建主窗口，避免事件循环冲突；
  - 显示区用 pack_propagate(False) 固定尺寸，否则大图会把结果区挤出窗口；
  - PhotoImage 必须用 self.image_label.photo 持有引用，否则被回收后显示空白。
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

import plant_config as cfg
import ui_common


class ClassifierApp:
    def __init__(self, root, model, class_names, device):
        self.root = root
        self.model = model
        self.class_names = class_names
        self.device = device
        self.current_image = None
        self.current_path = None

        root.title("花卉图像分类系统 · 本地图片")
        root.geometry("1000x700")
        root.configure(bg="#f5f7fa")
        ui_common.bind_center(root, 1000, 700)

        tk.Label(root, text="花 卉 图 像 分 类", font=("Microsoft YaHei", 22, "bold"),
                 bg="#f5f7fa", fg="#2c3e50").pack(pady=(16, 2))
        tk.Label(root, text=f"支持类别：{' / '.join(class_names)}", font=("Microsoft YaHei", 10),
                 bg="#f5f7fa", fg="#7f8c8d").pack()

        body = tk.Frame(root, bg="#f5f7fa")
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=12)

        # ---------------- 左：图片显示区
        left = tk.Frame(body, bg="white", bd=1, relief=tk.SOLID)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_frame = tk.Frame(left, width=560, height=400, bg="#eceff1")
        self.image_frame.pack(padx=12, pady=12)
        self.image_frame.pack_propagate(False)          # ★ 固定显示区尺寸
        self.image_label = tk.Label(self.image_frame, text="请选择一张花卉图片",
                                    bg="#eceff1", fg="#90a4ae", font=("Microsoft YaHei", 12))
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.path_label = tk.Label(left, text="未选择文件", bg="white", fg="#607d8b",
                                   font=("Microsoft YaHei", 9), wraplength=540)
        self.path_label.pack(pady=(0, 10))

        # ---------------- 右：结果区
        right = tk.Frame(body, width=360, bg="white", bd=1, relief=tk.SOLID)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(16, 0))
        right.pack_propagate(False)

        tk.Label(right, text="预测结果", font=("Microsoft YaHei", 13, "bold"),
                 bg="white", fg="#2c3e50").pack(pady=(18, 4))
        self.result_label = tk.Label(right, text="—", font=("Microsoft YaHei", 20, "bold"),
                                     bg="white", fg="#4a90d9", wraplength=320, justify=tk.CENTER)
        self.result_label.pack(pady=4)
        self.conf_label = tk.Label(right, text="", font=("Microsoft YaHei", 11),
                                   bg="white", fg="#7f8c8d")
        self.conf_label.pack()

        tk.Label(right, text="各类别置信度", font=("Microsoft YaHei", 10),
                 bg="white", fg="#95a5a6").pack(pady=(18, 2))
        self.bar_canvas = tk.Canvas(right, width=310, height=110, bg="white", highlightthickness=0)
        self.bar_canvas.pack(pady=4)

        self.tip_label = tk.Label(right, text="", font=("Microsoft YaHei", 9),
                                  bg="white", fg="#b0bec5", wraplength=300)
        self.tip_label.pack(side=tk.BOTTOM, pady=12)

        # ---------------- 底部按钮
        btns = tk.Frame(root, bg="#f5f7fa")
        btns.pack(pady=(0, 18))
        self.btn_choose = tk.Button(btns, text="选择花卉图片", width=16, font=("Microsoft YaHei", 11),
                                    bg="#4a90d9", fg="white", relief=tk.FLAT,
                                    command=self.choose_image)
        self.btn_choose.pack(side=tk.LEFT, padx=10)

        self.btn_predict = tk.Button(btns, text="开始预测", width=14, font=("Microsoft YaHei", 11),
                                     bg="#5cb85c", fg="white", relief=tk.FLAT,
                                     state=tk.DISABLED, command=self.predict)
        self.btn_predict.pack(side=tk.LEFT, padx=10)

        tk.Button(btns, text="清 空", width=10, font=("Microsoft YaHei", 11),
                  bg="#95a5a6", fg="white", relief=tk.FLAT,
                  command=self.reset).pack(side=tk.LEFT, padx=10)

        self.status = tk.Label(root, text="就绪" if model is not None else "未找到模型，无法预测",
                               bg="#dfe6ec", fg="#607d8b", anchor="w",
                               font=("Microsoft YaHei", 9))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        if model is None:
            self.tip_label.config(text="models/plant_classification_model.pth 不存在，\n"
                                       "请先运行 3.train_cnn.py 训练模型。")
            self.btn_predict.config(state=tk.DISABLED)

    # -------------------------------------------------------------- 交互
    def choose_image(self):
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
        self.current_path = path
        self.show_image(image)
        self.path_label.config(text=os.path.basename(path))
        self.result_label.config(text="待预测", fg="#4a90d9")
        self.conf_label.config(text="")
        self.bar_canvas.delete("all")
        if self.model is not None:
            self.btn_predict.config(state=tk.NORMAL)
        self.status.config(text="已选择图片，点击「开始预测」")

    def show_image(self, image):
        image = image.copy()
        image.thumbnail((540, 380))                    # 等比缩到显示区内
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo, text="")
        self.image_label.photo = photo                 # ★ 必须持有引用

    def predict(self):
        if self.current_image is None or self.model is None:
            messagebox.showwarning("提示", "请先选择一张图片")
            return
        top = ui_common.predict_topk(self.model, self.current_image,
                                     self.class_names, self.device, k=3)
        if not top:
            return
        name, conf = top[0]
        self.result_label.config(text=name, fg="#2e7d32")
        self.conf_label.config(text=f"置信度 {conf:.2f}%")
        self.draw_bars(top)
        self.status.config(text=f"预测完成：{name}（{conf:.2f}%）")

    def draw_bars(self, items):
        self.bar_canvas.delete("all")
        colors = ["#4a90d9", "#7fb3e3", "#c5d9ee"]
        for i, (name, conf) in enumerate(items):
            y = 10 + i * 34
            self.bar_canvas.create_text(6, y + 9, text=name, anchor="w",
                                        font=("Microsoft YaHei", 9), fill="#37474f")
            x0, x1 = 62, 300
            self.bar_canvas.create_rectangle(x0, y, x1, y + 16, fill="#eceff1", outline="")
            width = max(1, int((x1 - x0) * conf / 100))
            self.bar_canvas.create_rectangle(x0, y, x0 + width, y + 16,
                                             fill=colors[i % len(colors)], outline="")
            self.bar_canvas.create_text(x1 + 0, y + 8, text=f"{conf:.1f}%", anchor="w",
                                        font=("Microsoft YaHei", 8), fill="#607d8b")
            self.bar_canvas.create_text(x1 + 22, y + 8, text=f"{conf:.1f}%", anchor="w",
                                        font=("Microsoft YaHei", 8), fill="#607d8b")

    def reset(self):
        self.current_image = None
        self.current_path = None
        self.image_label.config(image="", text="请选择一张花卉图片")
        self.image_label.photo = None
        self.path_label.config(text="未选择文件")
        self.result_label.config(text="—", fg="#4a90d9")
        self.conf_label.config(text="")
        self.bar_canvas.delete("all")
        self.btn_predict.config(state=tk.DISABLED)
        self.status.config(text="就绪")


def main():
    if not ui_common.ask_login("花卉分类系统 - 登录"):
        return
    model, class_names, device = ui_common.load_model(cfg.TEACHER_MODEL_PATH)
    root = tk.Tk()
    ClassifierApp(root, model, class_names, device)
    root.mainloop()


if __name__ == "__main__":
    main()
