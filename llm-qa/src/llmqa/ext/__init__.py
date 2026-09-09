"""可插拔成熟工具适配层（ext）。

全部懒加载：未安装对应依赖时构造抛 LLMError（附安装指引），离线 CI 不受影响。
激活方式：pip install -e ".[ext]"

状态区分（避免过度声明）：
- LiteLLMClient           → 可用适配（模型网关，kind: litellm）
- Retriever 协议           → 可用契约（真实向量库接入 RAGHarness，见 harnesses/rag）
- RagasJudgeBackend       → 接入骨架（adapter contract）：证明可插拔性，
                             ragas.evaluate 绑定需按锁定版本补全（见模块 docstring）
- Langfuse Adapter         → 完整映射实现（fixture 验证；未经真实平台实测）
- LangSmith Adapter        → 轨迹归一骨架（映射需按所用平台导出结构补全）
"""
from llmqa.ext.litellm_backend import LiteLLMClient
from llmqa.ext.ragas_backend import RagasJudgeBackend
from llmqa.ext.trajectory_adapters import (
    ADAPTERS,
    LangfuseAdapter,
    LangSmithAdapter,
    adapt_trajectory,
)

__all__ = [
    "LiteLLMClient",
    "RagasJudgeBackend",
    "LangSmithAdapter",
    "LangfuseAdapter",
    "adapt_trajectory",
    "ADAPTERS",
]
