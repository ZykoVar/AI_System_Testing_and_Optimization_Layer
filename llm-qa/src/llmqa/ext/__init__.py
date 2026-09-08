"""可插拔成熟工具适配层（ext）。

这里只放"把成熟评测工具接到本项目契约"的薄适配器，全部懒加载：
未安装对应依赖时构造抛 LLMError（附安装指引），离线 CI 完全不受影响。

- litellm_backend: LiteLLM 客户端（100+ 模型统一接口/重试/成本核算）
- ragas_backend: Ragas 指标（faithfulness 等）作为 Judge 后端

激活方式：pip install -e ".[ext]"
"""
from llmqa.ext.litellm_backend import LiteLLMClient
from llmqa.ext.ragas_backend import RagasJudgeBackend

__all__ = ["LiteLLMClient", "RagasJudgeBackend"]
