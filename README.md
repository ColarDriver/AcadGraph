# AcadGraph

AcadGraph 是一个面向论文处理的命令行工具。下面只保留怎么用。

## 1) 安装

```bash
cd AcadGraph
python -m venv .venv
source .venv/bin/activate
pip install -e ".[pdf,dev]"
```

不处理 PDF 也可以用：

```bash
pip install -e .
```

## 2) 配置环境变量

```bash
cp .env.example .env
```

然后编辑 `.env`，至少填好以下项：

- `LLM_API_BASE` `LLM_API_KEY` `LLM_MODEL`
- `EMBEDDING_API_BASE` `EMBEDDING_API_KEY` `EMBEDDING_MODEL` `EMBEDDING_DIM`
- `NEO4J_URI` `NEO4J_USER` `NEO4J_PASSWORD`
- `QDRANT_HOST` `QDRANT_PORT`

## 3) 启动依赖服务

```bash
docker-compose up -d
```

## 4) 初始化数据库

```bash
acadgraph init
```

## 5) 导入论文

```bash
# 单篇 PDF
acadgraph add ./paper.pdf --title "My Paper" --year 2024 --venue "ICLR"

# 单篇文本
acadgraph add ./paper.txt --title "My Paper"

# 批量导入目录中的 PDF
acadgraph batch ./papers --batch-size 5 --ext .pdf
```

## 6) 查询、审计、统计

```bash
# 查询竞争空间
acadgraph query "attention mechanism for long sequences" --mode competition -k 10

# 生成 Gap 语句
acadgraph query "sparse attention with linear complexity" --mode gap

# 增强召回
acadgraph query "efficient transformer" --mode recall -k 10

# 审计某篇论文的 claim-evidence
acadgraph audit <paper_id>

# 查看图谱统计
acadgraph stats
```

## 7) 查看帮助

```bash
acadgraph --help
acadgraph <command> --help
```
