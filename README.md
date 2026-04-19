# Smart Knowledge Hub

基于多阶段检索增强生成（RAG）与模型上下文协议（MCP）的智能问答与知识检索框架。

## Quick Start

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate   # macOS/Linux

# 安装依赖
pip install -e ".[dev]"

# 验证安装
python -c "import src.mcp_server; import src.core; import src.ingestion; import src.libs; import src.observability"
```

## Project Structure

```
config/          — 配置文件 (settings.yaml, prompts/)
src/             — 源代码
  mcp_server/    — MCP Server 层
  core/          — 核心业务逻辑
  ingestion/     — 数据摄取 Pipeline
  libs/          — 可插拔抽象层
  observability/ — 可观测性 & Dashboard
scripts/         — CLI 脚本
tests/           — 测试
```
