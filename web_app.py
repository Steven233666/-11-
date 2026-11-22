# -*- coding: utf-8 -*-
"""⑨ Web 端：Flask + 浏览器（用户端 / 管理端）

用法：双击 run_web.bat，或执行
      .\\.venv\\Scripts\\python.exe web_app.py
然后浏览器打开 http://127.0.0.1:5000

账号：admin / 123456（管理端）   user / 123456（用户端）

设计说明：
  - 模型只在进程启动后第一次请求时加载一次（_MODELS 缓存），之后所有请求复用；
  - 默认加载 models/plant_resnet18_model.pth，没有则回落到教师 CNN；
  - 每次识别的结果写进 SQLite（web_data.db），缩略图存到 web/static/uploads/；
  - 会话用 Flask 自带的签名 Cookie（session），密码校验复用 user_info.txt。
"""

import io
import os
import sqlite3
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from functools import wraps

from flask import (Flask, g, jsonify, redirect, render_template, request, session,
                   url_for)
from PIL import Image

import plant_config as cfg
import ui_common

BASE = cfg.BASE_DIR
DB_PATH = os.path.join(BASE, "web_data.db")
UPLOAD_DIR = os.path.join(BASE, "web", "static", "uploads")
MAX_BYTES = 10 * 1024 * 1024

app = Flask(__name__,
            template_folder=os.path.join(BASE, "web", "templates"),
            static_folder=os.path.join(BASE, "web", "static"))
app.secret_key = "flowervision-local-demo-2026"
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

_MODELS = {}
_MODEL_INFO = {}


# ------------------------------------------------------------------ 数据库
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            username TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            elapsed_ms REAL NOT NULL,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()


# ------------------------------------------------------------------ 模型
def get_main_model():
    """进程内只加载一次，避免每个请求都读 45 MB 权重。"""
    if "main" not in _MODELS:
        _MODELS["main"] = ui_common.load_model()
    return _MODELS["main"]


def get_model_infos():
    """管理端展示用：三个模型的元信息，读一次缓存。"""
    if _MODEL_INFO:
        return _MODEL_INFO
    import torch

    entries = [
        ("ResNet18 迁移学习", cfg.RESNET_MODEL_PATH, "生产推荐"),
        ("CNN 教师模型", cfg.TEACHER_MODEL_PATH, "基线"),
        ("蒸馏学生模型", cfg.STUDENT_MODEL_PATH, "实验"),
    ]
    main_path = cfg.RESNET_MODEL_PATH if os.path.exists(cfg.RESNET_MODEL_PATH) \
        else cfg.TEACHER_MODEL_PATH

    for name, path, tag in entries:
        info = {"name": name, "tag": tag, "file": os.path.basename(path),
                "exists": os.path.exists(path), "acc": None, "epochs": None,
                "arch": None, "classes": None, "size_mb": None,
                "active": os.path.normpath(path) == os.path.normpath(main_path)}
        if info["exists"]:
            info["size_mb"] = round(os.path.getsize(path) / 1024 / 1024, 1)
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                info["acc"] = ckpt.get("best_accuracy", ckpt.get("best_test_acc"))
                info["epochs"] = ckpt.get("epochs") or ckpt.get("epoch")
                info["arch"] = ckpt.get("arch", "cnn")
                info["classes"] = ckpt.get("class_names")
            except Exception:
                pass
        _MODEL_INFO[name] = info
    return _MODEL_INFO


def dataset_stats():
    """原始数据 + 划分后数据的逐类数量。"""
    rows = []
    classes = sorted([d for d in os.listdir(cfg.RAW_DIR)
                      if os.path.isdir(os.path.join(cfg.RAW_DIR, d) )]) \
        if os.path.isdir(cfg.RAW_DIR) else []
    for cls in classes:
        def count(*parts):
            d = os.path.join(*parts)
            if not os.path.isdir(d):
                return 0
            return len([f for f in os.listdir(d)
                        if os.path.splitext(f)[1].lower() in cfg.VALID_EXTS])
        rows.append({
            "cls": cls,
            "raw": count(cfg.RAW_DIR, cls),
            "train": count(cfg.SPLIT_DIR, "train", cls),
            "test": count(cfg.SPLIT_DIR, "test", cls),
        })
    return rows


# ------------------------------------------------------------------ 登录
def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapper


def admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            return redirect(url_for("user_home"))
        return view(*args, **kwargs)
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        users = ui_common.load_user_info()
        if not users:                       # 没有账号文件时给一个默认管理员
            users = {"admin": "123456"}
        name = (request.form.get("username") or "").strip().lower()
        pwd = (request.form.get("password") or "").strip()
        if users.get(name) == pwd:
            session["user"] = name
            session["role"] = "admin" if name == "admin" else "user"
            return redirect(url_for("admin_home" if session["role"] == "admin" else "user_home"))
        error = "用户名或密码错误，请重试"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def root():
    if not session.get("user"):
        return redirect(url_for("login"))
    return redirect(url_for("admin_home" if session.get("role") == "admin" else "user_home"))


# ------------------------------------------------------------------ 用户端
@app.route("/predict")
@login_required
def user_home():
    model, class_names, device = get_main_model()
    active = os.path.basename(cfg.RESNET_MODEL_PATH if os.path.exists(cfg.RESNET_MODEL_PATH)
                              else cfg.TEACHER_MODEL_PATH)
    return render_template("user.html", class_names=class_names,
                           model_ready=model is not None,
                           active_model=active,
                           device=str(device))


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "没有收到图片"}), 400

    data = file.read()
    if not data:
        return jsonify({"ok": False, "error": "图片为空"}), 400
    if len(data) > MAX_BYTES:
        return jsonify({"ok": False, "error": "图片超过 10MB"}), 413

    model, class_names, device = get_main_model()
    if model is None:
        return jsonify({"ok": False, "error": "模型文件缺失，请先训练或放入 models/"}), 503

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return jsonify({"ok": False, "error": "图片无法解码，换一张试试"}), 400

    started = time.time()
    top = ui_common.predict_topk(model, image, class_names, device, k=3)
    elapsed_ms = round((time.time() - started) * 1000, 1)
    if not top:
        return jsonify({"ok": False, "error": "预测失败"}), 500

    # 存一张缩略图给历史记录用
    thumb_name = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.jpg"
    thumb_img = image.copy()
    thumb_img.thumbnail((260, 260))
    thumb_img.save(os.path.join(UPLOAD_DIR, thumb_name), "JPEG", quality=82)

    db = get_db()
    db.execute(
        "INSERT INTO records (ts, username, label, confidence, elapsed_ms, image_path) "
        "VALUES (?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session["user"],
         top[0][0], top[0][1], elapsed_ms, f"uploads/{thumb_name}"))
    db.commit()

    return jsonify({
        "ok": True,
        "label": top[0][0],
        "confidence": top[0][1],
        "top": [{"label": n, "confidence": c} for n, c in top],
        "elapsed_ms": elapsed_ms,
        "model": os.path.basename(cfg.RESNET_MODEL_PATH if os.path.exists(cfg.RESNET_MODEL_PATH)
                                  else cfg.TEACHER_MODEL_PATH),
    })


@app.route("/api/history")
@login_required
def api_history():
    db = get_db()
    rows = db.execute(
        "SELECT ts, label, confidence, elapsed_ms, image_path FROM records "
        "WHERE username = ? ORDER BY id DESC LIMIT 12", (session["user"],)).fetchall()
    total = db.execute("SELECT COUNT(*) c FROM records WHERE username = ?",
                       (session["user"],)).fetchone()["c"]
    return jsonify({"total": total, "items": [dict(r) for r in rows]})


# ------------------------------------------------------------------ 管理端
@app.route("/admin")
@admin_required
def admin_home():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM records").fetchone()["c"]
    today = db.execute("SELECT COUNT(*) c FROM records WHERE ts LIKE ?",
                       (datetime.now().strftime("%Y-%m-%d") + "%",)).fetchone()["c"]
    avg_ms = db.execute("SELECT AVG(elapsed_ms) a FROM records").fetchone()["a"] or 0
    users = len(ui_common.load_user_info())

    dist = db.execute("SELECT label, COUNT(*) c FROM records GROUP BY label "
                      "ORDER BY c DESC").fetchall()
    dist = [dict(r) for r in dist]
    max_c = max([d["c"] for d in dist], default=1)

    history = db.execute(
        "SELECT ts, username, label, confidence, elapsed_ms, image_path FROM records "
        "ORDER BY id DESC LIMIT 30").fetchall()

    models = list(get_model_infos().values())
    dataset = dataset_stats()

    return render_template("admin.html",
                           total=total, today=today,
                           avg_ms=round(avg_ms, 1), users=users,
                           dist=dist, max_c=max_c, history=history,
                           models=models, dataset=dataset,
                           class_names=cfg.CLASSES)


# ------------------------------------------------------------------ 模板全局变量
@app.context_processor
def inject_nav():
    """导航栏用的轻量信息：当前模型 + 本账号识别总数。"""
    active = None
    try:
        active = next((m for m in get_model_infos().values() if m["active"]), None)
    except Exception:
        pass
    total = 0
    if session.get("user"):
        try:
            total = get_db().execute("SELECT COUNT(*) c FROM records WHERE username = ?",
                                     (session["user"],)).fetchone()["c"]
        except Exception:
            pass
    return {"nav_model": active, "nav_total": total, "nav_classes": len(cfg.CLASSES)}


# ------------------------------------------------------------------ 启动
def main():
    init_db()
    print("=" * 56)
    print("  花卉图像分类 · Web 端")
    print("  用户端：http://127.0.0.1:5000   (user / 123456)")
    print("  管理端：http://127.0.0.1:5000/admin (admin / 123456)")
    print("  关闭服务：在此窗口按 Ctrl+C")
    print("=" * 56)
    if os.environ.get("FV_NO_BROWSER") != "1":
        threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
