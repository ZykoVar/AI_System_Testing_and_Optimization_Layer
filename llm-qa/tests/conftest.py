"""pytest 配置：无需安装即可导入 src/llmqa。"""
import sys
from pathlib import Path

# 将 src 目录插入 sys.path 最前，未安装包时也能 import llmqa
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
