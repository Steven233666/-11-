# -*- coding: utf-8 -*-
"""单类别爬虫（调试用）：只采一个类别，页数更多。

用法：
  python single_data_download.py            # 默认采「玫瑰」，10 页
  python single_data_download.py 兰花 5      # 指定类别与页数

说明：文件名以数字开头不是合法模块名，不能 `import 1.download_data`，
这里用 importlib 按文件路径动态加载——也是"想复用带数字前缀脚本"的正规做法。
"""

import importlib.util
import os
import sys

import plant_config as cfg


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cls = sys.argv[1] if len(sys.argv) > 1 else "玫瑰"
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if cls not in cfg.CLASS_KEYWORDS:
        raise SystemExit(f"未知类别：{cls}，可选：{list(cfg.CLASS_KEYWORDS)}")

    cfg.ensure_dirs()
    mod = load_module(os.path.join(cfg.BASE_DIR, "1.download_data.py"), "download_data")
    spider = mod.BaiduImageSpider(cls, os.path.join(cfg.RAW_DIR, cls),
                                  target=cfg.TARGET_PER_CLASS)
    spider.run(cfg.CLASS_KEYWORDS[cls], max_pages=pages)


if __name__ == "__main__":
    main()
