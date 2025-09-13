#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
self.py_demo
从脚本同目录下的 demo/ 读取 *.txt，组织为：
{'demo': {'xx.py': 'content', ...}}
用法：
  python self.py_demo
  python self.py_demo --dir demo --out demo.json
"""

from pathlib import Path
import json
import argparse
import sys

def build_json(demo_dir: Path, src_ext: str = ".txt") -> dict:
    """构造 {'demo': {'xx.py': 'content', ...}} 结构"""
    inner = {}
    for p in sorted(demo_dir.glob(f"*{src_ext}")):
        if not p.is_file():
            continue
        key = f"{p.stem}.py"  # xx.txt -> xx.py
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = p.read_text(encoding="utf-8", errors="replace")
        inner[key] = content
    return {"demo": inner}

def read():
    # parser = argparse.ArgumentParser(description="Pack demo/*.txt into JSON")
    # parser.add_argument("--dir", default="demo", help="源目录（相对脚本目录；默认: demo）")
    # parser.add_argument("--ext", default=".txt", help="源文件扩展名（默认: .txt）")
    # parser.add_argument("--out", default="", help="输出文件路径（相对脚本目录；留空仅打印）")
    # args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # 解析源目录：相对则基于脚本目录
    src_path = Path("demo")
    demo_dir = src_path if src_path.is_absolute() else (script_dir / src_path)
    demo_dir = demo_dir.resolve()

    if not demo_dir.exists() or not demo_dir.is_dir():
        print(f"[ERROR] 目录不存在或不是目录: {demo_dir}", file=sys.stderr)
        sys.exit(1)

    data = build_json(demo_dir, ".txt")
    text = json.dumps(data, ensure_ascii=False, indent=2)

    # 打印到 stdout
    print(text)

    # 可选写入文件：相对则写到脚本目录
    # if args.out:
    #     out_path = Path(args.out)
    #     if not out_path.is_absolute():
    #         out_path = script_dir / out_path
    #     out_path.write_text(text, encoding="utf-8")
    #     print(f"[INFO] 已写入: {out_path}", file=sys.stderr)
    
    return data

read()
