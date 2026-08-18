"""pytest 配置：无需安装即可导入 src/llmqa。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
