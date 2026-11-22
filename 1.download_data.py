# -*- coding: utf-8 -*-
"""① 花卉图片采集（百度图片 JSON 接口）

用法：python 1.download_data.py [--per-class 150] [--class 玫瑰]

要点：
  1. 直接打百度图片的 acjson 接口，不用解析网页 HTML；
  2. 接口必带 tn=resultjson_com&ala=0&nojc=0&newReq=1，否则被判定爬虫；
  3. 新版返回体 data 是"字典"，图片在 data['images']，字段是小写 thumburl/objurl；
  4. 只保存真正能解码的图片（Content-Type 校验 + PIL 解码 + 重新压成 RGB JPEG），
     避免把 HTML 错误页当 .jpg 存下来污染训练集。
"""

import argparse
import io
import json
import os
import re
import time

import requests
from PIL import Image

import plant_config as cfg

API = "https://image.baidu.com/search/acjson"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"),
    "Referer": "https://image.baidu.com/search/index?tn=baiduimage&word=",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "text/plain, */*; q=0.01",
}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class BaiduImageSpider:
    def __init__(self, word: str, save_dir: str, target: int = 150,
                 min_side: int = cfg.MIN_SIDE):
        self.word = word
        self.save_dir = save_dir
        self.target = target
        self.min_side = min_side
        os.makedirs(save_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.seen_urls = set()
        self.seen_bytes = set()
        self.count = self._existing_count()
        self.failed = 0
        self.skipped = 0

    # ------------------------------------------------------------ 工具
    def _existing_count(self) -> int:
        if not os.path.isdir(self.save_dir):
            return 0
        return len([n for n in os.listdir(self.save_dir)
                    if os.path.splitext(n)[1].lower() in cfg.VALID_EXTS])

    def _next_index(self) -> int:
        """接着已有编号往后写，重复运行不会覆盖旧图。"""
        idx = 1
        while os.path.exists(os.path.join(self.save_dir, f"{self.word}_{idx}.jpg")):
            idx += 1
        return idx

    @staticmethod
    def fix_image_url(url: str) -> str:
        if not url:
            return ""
        url = url.replace("\\/", "/").strip()
        if url.startswith("//"):
            return "https:" + url
        if url.startswith("http://"):
            return "https://" + url[7:]
        if not url.startswith("http"):
            return ""
        return url

    # ------------------------------------------------------------ 抓链接
    def warm_up(self, keyword: str):
        """先访问一次搜索结果页拿到 Cookie。

        不做这一步，直接打 acjson 只返回 47 字节的 {"antiFlag":1,...}——
        表现就是"接口 200、但 0 张图"。
        """
        try:
            self.session.get("https://image.baidu.com/search/index",
                             params={"tn": "baiduimage", "word": keyword},
                             headers={"Referer": "https://image.baidu.com/"},
                             timeout=20)
        except requests.RequestException:
            pass

    @staticmethod
    def _unescape(raw: str) -> str:
        """把 JSON 字符串里的 \\u0026、\\/ 还原成正常字符。"""
        try:
            return json.loads('"%s"' % raw)
        except Exception:
            return raw.replace("\\/", "/").replace("\\u0026", "&")

    def _parse_json_urls(self, text: str) -> list:
        try:
            payload = json.loads(text)
        except Exception:
            return []
        data = payload.get("data")
        items = data.get("images") if isinstance(data, dict) else data
        urls = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            for key in ("thumburl", "middleURL", "objurl", "hoverURL", "objURL"):
                fixed = self.fix_image_url(item.get(key) or "")
                if fixed:
                    urls.append(fixed)
                    break                      # 同一张图只取一个地址，优先 thumburl
        return urls

    def _parse_regex_urls(self, text: str) -> list:
        """JSON 解析失败时的兜底：直接正则抠字段。"""
        urls = []
        for key in ("thumburl", "middleURL", "objurl", "hoverURL", "objURL"):
            for raw in re.findall(r'"%s"\s*:\s*"(.*?)"' % key, text):
                fixed = self.fix_image_url(self._unescape(raw))
                if fixed:
                    urls.append(fixed)
            if urls:
                break
        return urls

    def get_image_urls(self, keyword: str, page: int) -> list:
        params = {
            "tn": "resultjson_com",
            "word": keyword,
            "pn": page * 30,       # 偏移量，不是页码
            "rn": 30,
            "ala": 0,
            "nojc": 0,
            "newReq": 1,
            "ie": "utf-8",
        }

        text = ""
        for attempt in range(2):
            try:
                resp = self.session.get(API, params=params, timeout=20)
                resp.raise_for_status()
                text = resp.text
            except requests.RequestException as e:
                print(f"  [请求失败] {keyword} 第{page + 1}页：{e}")
                return []
            if len(text) > 500:                # 正常的响应有几十 KB
                break
            print(f"  [接口被拦] {keyword} 第{page + 1}页，重新预热 Cookie 后重试")
            self.warm_up(keyword)
            time.sleep(1.5)

        if not text or "Forbid spider access" in text or '"antiFlag":1' in text.replace(" ", ""):
            print(f"  [被反爬拦截] {keyword} 第{page + 1}页，跳过")
            return []

        urls = self._parse_json_urls(text) or self._parse_regex_urls(text)
        return urls

    # ------------------------------------------------------------ 下载
    def download_image(self, url: str) -> bool:
        if url in self.seen_urls:
            return False
        self.seen_urls.add(url)
        try:
            resp = self.session.get(url, timeout=10, stream=True,
                                    headers={"User-Agent": HEADERS["User-Agent"],
                                             "Referer": "https://image.baidu.com/"})
            if resp.status_code != 200:
                self.failed += 1
                return False
            if "image" not in resp.headers.get("Content-Type", "").lower():
                self.failed += 1
                return False
            buf = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    buf.write(chunk)
                    if buf.tell() > MAX_IMAGE_BYTES:
                        raise ValueError("图片过大")
        except (requests.RequestException, ValueError):
            self.failed += 1
            return False

        data = buf.getvalue()
        if len(data) < cfg.MIN_BYTES or hash(data) in self.seen_bytes:
            self.skipped += 1
            return False
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()                       # 真正解码，破坏文件在这里会抛异常
                if min(im.size) < self.min_side:
                    self.skipped += 1
                    return False
                im = im.convert("RGB")
                if max(im.size) > 900:          # 统一压到 900px 内，省磁盘也够 128 训练
                    im.thumbnail((900, 900))
        except Exception:
            self.failed += 1
            return False

        self.seen_bytes.add(hash(data))
        idx = self._next_index()
        path = os.path.join(self.save_dir, f"{self.word}_{idx}.jpg")
        try:
            im.save(path, "JPEG", quality=88)
        except Exception:
            self.failed += 1
            return False
        self.count += 1
        return True

    # ------------------------------------------------------------ 主循环
    def run(self, keywords: list, max_pages: int = 10):
        print(f"开始采集「{self.word}」（已有 {self.count} 张，目标 {self.target} 张）")
        for kw in keywords:
            self.warm_up(kw)                 # 每个关键词先取一次 Cookie
            time.sleep(0.8)
            for page in range(max_pages):
                if self.count >= self.target:
                    break
                urls = self.get_image_urls(kw, page)
                if not urls:
                    break                    # 该关键词翻到底或被拦，换下一个
                for url in urls:
                    if self.count >= self.target:
                        break
                    ok = self.download_image(url)
                    if ok and self.count % 20 == 0:
                        print(f"  已下载 {self.count}/{self.target}")
                    time.sleep(0.25)         # 单张间隔，避免触发风控
                time.sleep(1.0)              # 翻页间隔
        print(f"「{self.word}」完成：新增至 {self.count} 张 "
              f"（失败 {self.failed} 张，跳过 {self.skipped} 张）")


def main():
    parser = argparse.ArgumentParser(description="花卉图片采集")
    parser.add_argument("--per-class", type=int, default=cfg.TARGET_PER_CLASS)
    parser.add_argument("--class", dest="only", default=None, help="只采某一类，例如 玫瑰")
    parser.add_argument("--max-pages", type=int, default=10)
    args = parser.parse_args()

    cfg.ensure_dirs()
    classes = [args.only] if args.only else cfg.CLASSES
    started = time.time()
    for cls in classes:
        if cls not in cfg.CLASS_KEYWORDS:
            print(f"跳过未知类别：{cls}")
            continue
        spider = BaiduImageSpider(cls, os.path.join(cfg.RAW_DIR, cls), target=args.per_class)
        spider.run(cfg.CLASS_KEYWORDS[cls], max_pages=args.max_pages)

    print("\n========== 采集结果 ==========")
    total = 0
    for cls in classes:
        d = os.path.join(cfg.RAW_DIR, cls)
        n = len([x for x in os.listdir(d)]) if os.path.isdir(d) else 0
        total += n
        print(f"{cls:>4}：{n} 张")
    print(f"合计：{total} 张，耗时 {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
