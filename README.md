# 花卉图像分类系统 · 项目说明（README）

基于你上传的水果识别工程文档改造的**植物（花卉）分类**版本，代码在 `E:\plant_recognition`，已在本机完整跑通：从零采集 750 张图 → 训练 → 蒸馏 → 界面可运行。

**5 个类别**：兰花、向日葵、玫瑰、菊花、郁金香

## 实测结果（本机 CPU 跑出来的真实数字）

| 模型 | 结构 | 训练轮数 | 测试集准确率 |
|---|---|---|---|
| 教师模型 | 自建 4 层 CNN（2.34M 参数） | 40 | **72.67%** |
| 学生模型 | 同结构 + 知识蒸馏（T=4, α=0.7） | 40 | **73.33%** |
| 提升版 | ResNet18 迁移学习（ImageNet 预训练） | 15 | **90.67%** |

测试集 150 张（每类 30 张）。ResNet18 逐类：兰花 100%、向日葵 90%、玫瑰 86.67%、菊花 90%、郁金香 86.67%。

> 自建 CNN 只有 72%，主要原因是网络爬来的图里有不少干扰样本（菊花类里混进了菊花茶、菊花展全景，郁金香类里有大片花海远景）。换成预训练 ResNet18 后同样的数据直接到 90.67%，这条也正好对应原文档"项目扩展方向"里的第 7 条建议。

## 快速开始

三条命令（在 `E:\plant_recognition` 下，用项目自带的虚拟环境）：

```powershell
.\.venv\Scripts\python.exe 2.split_data.py      # 数据已采好，重新划分 8:2
.\.venv\Scripts\python.exe 3.train_cnn.py       # 训练教师模型（约 10 分钟 CPU）
.\.venv\Scripts\python.exe 4.classifier_ui.py   # 打开分类界面，账号 admin / 123456
```

更省事的方式：双击 `run_all.bat` 一次跑完采集→划分→训练→蒸馏；双击 `run_ui.bat`（同 `运行界面.bat`）选择要打开的界面；双击 `run_web.bat` 打开网页端（浏览器访问 http://127.0.0.1:5000）。

单独启动网页端：

```powershell
.\.venv\Scripts\python.exe web_app.py
```

## 目录结构

```
E:\plant_recognition\
├── plant_config.py            # 全局配置：类别、路径、超参、预处理（所有脚本共用）
├── model.py                   # PlantCNN + ResNet18 定义（全项目唯一一份）
├── data_utils.py              # 带清洗的 ImageFolder + 单轮训练/评估
├── ui_common.py               # 登录窗口、模型加载、top-k 预测、摄像头打开
│
├── 1.download_data.py         # ① 百度图片采集（含 Cookie 预热、坏图过滤）
├── 2.split_data.py            # ② 逐类 8:2 划分
├── 3.train_cnn.py             # ③ 训练教师 CNN，出曲线/混淆矩阵/预测图
├── 4.classifier_ui.py         # ④ 本地图片分类界面（登录 + top3 置信度条）
├── 5.classifier_camera_ui.py  # ⑤ 摄像头实时识别界面
├── 6.distillation_cnn.py      # ⑥ 知识蒸馏（教师 → 学生）
├── 7.distillation_predictor.py# ⑦ 集成预测界面（教师/学生/集成三模式 + 准确率评估）
├── 8.train_resnet.py          # ⑧ 可选：ResNet18 迁移学习（准确率提升到 90%+）
├── web_app.py                 # ⑨ 网页端（Flask：用户端 + 管理端）
├── web/templates/             # 网页模板：login / user / admin
├── web/static/                # 样式表 + 识别缩略图（uploads/）
├── web_data.db                # 网页端识别记录（SQLite，自动创建）
├── single_data_download.py    # 单类别爬虫（调试用）
├── run_all.bat / run_ui.bat / run_web.bat  # 一键跑通 / 一键开界面 / 一键开网页端
├── requirements.txt
│
├── plant_images/<类别>/        # 原始图：5 类 × 150 张 = 750 张
├── plant_images_split/
│   ├── train/<类别>/           # 每类 120 张
│   └── test/<类别>/            # 每类 30 张
├── images/                    # 5 张演示图（每类一张）
├── models/
│   ├── plant_classification_model.pth   # 教师 CNN（72.67%）
│   ├── distilled_plant_model.pth        # 蒸馏学生（73.33%）
│   ├── plant_resnet18_model.pth         # ResNet18（90.67%，界面优先加载）
│   └── plant_classification_model_20ep_backup.pth
├── training_history.png        # 教师训练曲线
├── confusion_matrix.png        # 教师混淆矩阵
├── prediction_示例.png          # 教师抽样预测（6 张）
├── distillation_history.png    # 蒸馏曲线（含教师基准线）
├── training_history_resnet.png / confusion_matrix_resnet.png / prediction_resnet.png
├── docs/                       # 说明文档
└── logs/                       # 运行日志
```

## 界面说明

三个界面都要先登录，`user_info.txt` 里预置了 `admin:123456` 和 `user:123456`（用户名不分大小写，密码区分）。

- `4.classifier_ui.py`：选图 → 预测，右侧显示类别、置信度和 Top-3 条形图。
- `5.classifier_camera_ui.py`：打开摄像头 → 实时预览 → 拍照预测。子线程只读帧入队，主线程刷界面（避免 Tkinter 黑屏）。
- `7.distillation_predictor.py`：教师 / 学生 / 集成三种模式，支持本地图与摄像头，还能后台评估测试集准确率。

**模型选择逻辑**：`ui_common.load_model()` 会优先加载 `models/plant_resnet18_model.pth`；把这个文件改名或删掉，就自动回落到教师 CNN（72.67%）。想固定用某个模型，可以传路径。

## 网页端（Web）

双击 `run_web.bat`（或执行 `.\.venv\Scripts\python.exe web_app.py`），会自动打开浏览器；地址 `http://127.0.0.1:5000`，同样用 `user_info.txt` 里的账号登录。关闭服务：在命令行窗口按 `Ctrl+C`。

- **用户端**（`user` / `123456`）：上传图片或调用浏览器摄像头拍照 → 识别 → 显示类别、置信度和 Top-3 概率条，右侧滚动显示本人最近 12 次识别记录（含缩略图）。
- **管理端**（`admin` / `123456`）：累计/今日识别次数、平均响应耗时、模型版本与准确率对比表（ResNet18 90.67% / 学生 73.33% / 教师 72.67%）、数据集逐类统计、识别类别分布、最近 30 条识别日志。
- 权限：`admin` 视为管理员可进管理端，其他账号访问 `/admin` 会被跳回用户端；未登录一律跳到登录页。
- 实现要点：模型在进程内只加载一次（首屏约 2~3 秒，之后单张推理约 20 ms）；识别记录写入 SQLite `web_data.db`，缩略图存 `web/static/uploads/`；请求 `http://127.0.0.1` 属于浏览器认可的安全上下文，所以摄像头无需 HTTPS 也能用。

## 换成别的植物怎么办

改 `plant_config.py` 里的两个地方，然后重跑 1→2→3（或 8）：

```python
CLASSES = ["兰花", "向日葵", "玫瑰", "菊花", "郁金香"]     # 你的类别（目录名就是类别名）
CLASS_KEYWORDS = {                                       # 每类 1~3 个检索词
    "兰花": ["兰花", "蝴蝶兰", "兰花 盆栽"],
    ...
}
```

类别文件夹名会按 Unicode 排序成为模型的类别顺序，改完重训即可，checkpoint 里存了 `class_names`，界面会自动跟着变。

## 本机环境（已装好）

```
Python       3.13.18（项目内 .venv）
torch        2.15.0+cpu
torchvision  0.29.2+cpu
opencv       5.1.0
Pillow       13.0.0
numpy        2.5.4
matplotlib   3.11.2
flask        3.1.3
设备         CPU（CUDA 不可用）
```

两个环境上的注意点：

1. 本机 Python 运行时缺 tcl8.6 数据文件，直接建 Tk 窗口会报 `Can't find a usable init.tcl`。已在 `.venv\Lib\site-packages\sitecustomize.py` 里自动指向 `E:\python3.13\tcl`，不用手动设环境变量。
2. Windows 下 `num_workers` 必须为 0（已在配置里固定），否则 DataLoader 多进程会报错。

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 双击 bat 出现一串「'xxx' 不是内部或外部命令」乱码 | bat 用 UTF-8 存中文，cmd 按 GBK 解码，中文行被当成命令 | 已把两个 bat 改成纯英文内容（`run_ui.bat`、`run_all.bat`），直接双击即可 |
| 爬虫一张都下不到、接口返回 47 字节 | 没带 Cookie 直接打 acjson，被判定爬虫 | 代码已内置先访问搜索页预热；仍失败就降低频率或过段时间再跑 |
| 下载的图很多但不是花 | 关键词命中花茶/花展/花海远景 | 换更精确的检索词，或手工删掉 `plant_images/<类别>/` 里明显不对的图后重跑 2、3 |
| `size mismatch` | 换了输入尺寸或类别数 | 输入固定 128（对应 `128*8*8`）；类别数以 checkpoint 的 `class_names` 为准 |
| 界面点预测没反应 | 没选图/没模型 | 先选图；`models/` 下要有 `.pth` |
| 图片显示空白 | `PhotoImage` 被回收 | 代码已用 `label.photo = photo` 持有引用 |
| 摄像头打不开或全黑 | 被占用 / 隐私权限 / 索引不对 / 子线程创建了 PhotoImage | 关掉微信腾讯会议等；设置里允许相机；试索引 1；代码已是主线程刷新 |
| 第二次运行摄像头打不开 | 上次没释放 | 关窗口走 `on_close`（已调 `cap.release()`） |
| 训练准确率 100%、测试只有 70% | 数据噪声 + 过拟合 | 清理数据；或直接用 `8.train_resnet.py` |
| 网页端提示 `Address already in use` / 端口被占 | 5000 端口被别的程序（或上一次没关的窗口）占用 | 关掉旧窗口重开；或改 `web_app.py` 末尾 `port=5000` 为 5001 |
| 网页摄像头打不开 | 浏览器未授权、被其他程序占用 | 地址栏摄像头图标里允许访问；关掉微信/会议软件再试；也可先上传图片 |

## 后续可做

- 把 ResNet18 换成 ConvNeXt / EfficientNet 再比一轮；
- 导出 ONNX，做离线 App 或树莓派端识别；
- 加"未知类别"拒识（置信度低于阈值就报"不是这 5 类"）；
- 换成病虫害、树叶、绿植等类别，只需改配置重跑。

## 代码仓库说明

仓库里放的是代码、文档与少量演示素材，**不带**虚拟环境、模型权重和爬来的数据集（由 `.gitignore` 排除），克隆后按"快速开始"重跑即可生成：

| 已入库 | 未入库（可重新生成 / 体积大） |
|---|---|
| 9 个 Python 脚本、`web_app.py`、`web/`、全部 bat | `.venv/`（`python -m venv` 重建） |
| `README.md`、`docs/` 说明文档 | `models/*.pth`（重跑 3 / 6 / 8 号脚本生成，约 105 MB） |
| 曲线、混淆矩阵、预测图（PNG） | `plant_images/`、`plant_images_split/`（重跑 1、2 号脚本，约 118 MB） |
| `images/` 5 张演示图、`web/static/demo/` | `logs/`、`web_data.db`、`web/static/uploads/`（运行产物） |
| `requirements.txt`、`user_info.txt`、`拓展.txt` | |

如果确实需要把权重一起发布，用 Git LFS（单个 `.pth` 最大 45 MB，未超 GitHub 100 MB 单文件上限，但建议 LFS）：

```powershell
git lfs install
git lfs track "models/*.pth"
git add .gitattributes models
git commit -m "add model weights via lfs"
```
