"""Agent 测试 Harness：受控的工具调用执行环（Agent Loop）。

测试关注点：
- 工具选择是否正确、参数是否符合 JSON Schema、多步规划是否合理；
- 会话状态/记忆是否正确传递；
- 护栏是否生效：循环检测、预算上限（迭代/token）、工具白名单、危险工具拒绝；
- 工具观测可注入（tool_result_override），用于间接提示注入测试。
"""
from __future__ import annotations

import inspect
import time
from typing import Any, Callable

from pydantic import BaseModel, Field

from llmqa.clients.base import LLMClient, Message, TokenUsage, ToolCall


class AgentAbortError(RuntimeError):
    """Agent 执行被护栏中止。"""

    def __init__(self, reason: str, *, abort_reason: str = ""):
        # abort_reason 缺省时退化为人类可读 reason，保证 trace 总有一个机器可读的停机原因。
        self.abort_reason = abort_reason or reason
        super().__init__(reason)


class LoopDetected(AgentAbortError):
    """重复工具调用循环。"""
    def __init__(self, tool_name: str, times: int):
        super().__init__("检测到工具 {} 连续重复调用 {} 次".format(tool_name, times),
                         abort_reason="loop_detected")


class BudgetExceeded(AgentAbortError):
    """迭代或 token 预算耗尽。"""
    def __init__(self, message: str):
        super().__init__(message, abort_reason="budget_exceeded")


class ToolPolicyViolation(AgentAbortError):
    """调用未注册/未授权工具。"""
    def __init__(self, tool_name: str):
        super().__init__("模型调用了不允许的工具: {}".format(tool_name),
                         abort_reason="tool_policy_violation")


class Tool(BaseModel):
    """Agent 可用工具。handler 同步/异步均可，返回值转为字符串观测。"""
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON Schema
    handler: Any = None    # 工具实现，可同步/异步；返回值一律转成字符串观测
    dangerous: bool = False
    allowlist_only: bool = False    # 仅当显式加入 allowed_tools 才可调用

    def schema(self) -> dict[str, Any]:
        """生成 OpenAI function-calling 风格的工具描述。"""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters or {"type": "object", "properties": {}},
        }}

    async def invoke(self, arguments: dict[str, Any]) -> str:
        """调用工具并统一返回字符串观测；handler 异常不向上抛，而是转成错误观测。"""
        if self.handler is None:
            return "[工具 {} 无 handler]".format(self.name)
        try:
            result = self.handler(**arguments)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as e:  # noqa: BLE001 —— 工具异常转观测
            return "[工具异常] {}: {}".format(type(e).__name__, e)


class ToolResult(BaseModel):
    """一次工具调用的结果观测，供断言检查模型实际看到的工具输出。"""

    tool_call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    error: str | None = None


class AgentStep(BaseModel):
    """一轮 Agent 迭代：模型文本、工具调用与停止原因。"""

    index: int
    assistant_text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None


class AgentTrace(BaseModel):
    """一次 Agent 运行的完整轨迹，可直接用于断言。"""
    task: str
    success: bool = False
    final_answer: str = ""
    iterations: int = 0
    steps: list[AgentStep] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    abort_reason: str | None = None    # 非空表示被护栏中止（loop/budget/policy），而非正常完成
    total_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0

    @property
    def tool_call_names(self) -> list[str]:
        """按调用顺序收集全部工具名，便于断言"调用了哪些工具"。"""
        return [tc.name for step in self.steps for tc in step.tool_calls]


class AgentHarness:
    """受控 Agent 执行环：提供工具注册、循环/预算/白名单护栏与轨迹记录。"""

    def __init__(self, client: LLMClient, tools: list[Tool], *,
                 system_prompt: str = "", max_iterations: int = 8,
                 max_tokens_per_call: int = 512, max_total_tokens: int | None = None,
                 allowed_tools: list[str] | None = None,
                 stop_on_repeated_calls: int = 3, temperature: float = 0.0):
        self.client = client
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens_per_call = max_tokens_per_call
        self.max_total_tokens = max_total_tokens
        # 注意：allowed_tools 必须先于 add_tool 初始化（add_tool 会读取它）
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.stop_on_repeated_calls = stop_on_repeated_calls
        self.temperature = temperature
        self.tools: dict[str, Tool] = {}
        for t in tools:
            self.add_tool(t)

    def add_tool(self, tool: Tool) -> None:
        """按名称注册工具；同名后注册会覆盖先前定义。"""
        self.tools[tool.name] = tool

    def _tool_schemas(self) -> list[dict[str, Any]]:
        """把所有已注册工具转成 function-calling 描述列表。"""
        return [t.schema() for t in self.tools.values()]

    async def run(self, task: str, *,
                  session_history: list[Message] | None = None,
                  tool_result_override: Callable[[ToolCall], str] | None = None,
                  variables: dict[str, str] | None = None) -> AgentTrace:
        """执行一次 Agent 任务。

        tool_result_override: 注入工具观测（安全测试用，模拟被污染的检索/工具结果）。
        """
        start = time.perf_counter()
        system = self.system_prompt
        if variables:
            for k, v in variables.items():
                system = system.replace("{{" + k + "}}", str(v))
        messages: list[Message] = []
        if system:
            messages.append(Message.system(system))
        messages.extend(session_history or [])
        messages.append(Message.user(task))
        trace = AgentTrace(task=task)
        total_tokens = TokenUsage()
        repeated = 0
        last_calls: list[tuple[str, str]] = []

        for i in range(self.max_iterations):
            resp = await self.client.generate(
                messages, temperature=self.temperature,
                max_tokens=self.max_tokens_per_call, tools=self._tool_schemas())
            total_tokens.prompt_tokens += resp.usage.prompt_tokens
            total_tokens.completion_tokens += resp.usage.completion_tokens
            trace.total_usage = total_tokens
            step = AgentStep(index=i, assistant_text=resp.text,
                             tool_calls=resp.tool_calls, finish_reason=resp.finish_reason)
            trace.steps.append(step)
            trace.iterations = i + 1

            # token 预算在每轮生成后立即检查，避免模型继续消耗额度。
            if self.max_total_tokens and total_tokens.total_tokens > self.max_total_tokens:
                trace.abort_reason = "budget_exceeded"
                break
            if not resp.tool_calls:
                trace.success = True
                trace.final_answer = resp.text
                break

            # 循环检测：连续相同 (name, args) 调用
            current_calls = [(tc.name, str(sorted(tc.arguments.items()))) for tc in resp.tool_calls]
            if current_calls == last_calls:
                repeated += 1
            else:
                repeated = 1
            last_calls = current_calls
            if repeated >= self.stop_on_repeated_calls:
                trace.abort_reason = "loop_detected"
                break

            # 策略检查：调用未注册/未授权工具 → 立即中止（护栏生效）
            violation = False
            for tc in resp.tool_calls:
                tool = self.tools.get(tc.name)
                denied = (tool is None
                          or (self.allowed_tools and tc.name not in self.allowed_tools)
                          or (tool.allowlist_only
                              and (self.allowed_tools is None or tc.name not in self.allowed_tools)))
                if denied:
                    trace.abort_reason = "tool_policy_violation"
                    violation = True
                    break
            if violation:
                break
            messages.append(Message.assistant(content=resp.text, tool_calls=resp.tool_calls))
            for tc in resp.tool_calls:
                tool = self.tools[tc.name]
                if tool_result_override is not None:
                    output = tool_result_override(tc)
                else:
                    output = await tool.invoke(tc.arguments)
                trace.tool_results.append(ToolResult(
                    tool_call_id=tc.id, name=tc.name,
                    arguments=tc.arguments, output=output))
                messages.append(Message.tool(content=output, tool_call_id=tc.id, name=tc.name))
        else:
            # for 未被 break：迭代次数耗尽仍无终态，归因于预算耗尽（覆盖任何残留原因）。
            trace.abort_reason = trace.abort_reason or "budget_exceeded"
        trace.latency_ms = (time.perf_counter() - start) * 1000
        return trace
