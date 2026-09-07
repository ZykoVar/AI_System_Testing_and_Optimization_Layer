# llm-qa 便捷入口（Linux/macOS/WSL；Windows 用 scripts/*.ps1）
PY ?= python

.PHONY: install demo list smoke all-mock prompts-scan lint test gate help

install:            ## 安装（含 dev 依赖）
	$(PY) -m pip install -e ".[dev]"

demo:               ## 运行内置演示套件（Mock）
	llmqa demo

list:               ## 列出全部内置用例
	llmqa run --list

smoke:              ## Mock 冒烟（全套件 smoke 标签）
	llmqa run --tag smoke --no-color

all-mock:           ## 全量 Mock 回归
	llmqa run --no-color

prompts-scan:       ## Prompt 契约校验 + 注入扫描（L0 门禁）
	llmqa prompts validate
	llmqa prompts scan

lint:               ## 用例 id 唯一性 lint（L0 门禁）
	$(PY) scripts/lint_ids.py

test:               ## 框架自测（pytest）
	$(PY) -m pytest -q

gate:               ## 严重级门禁（解析最新 report.json）
	$(PY) scripts/gate.py reports

help:               ## 帮助
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "%-16s %s\n", $$1, $$2}'
