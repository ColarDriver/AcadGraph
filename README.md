# AcadGraph — 三层证据链知识图谱

> **Three-Layer Evidence-Chain Knowledge Graph for Academic Papers**

一套面向论文全文的知识图谱系统，支撑 idea2paper 的 Gap Mining（新颖性挖掘）和 Evidence Auditing（证据审计）任务。

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                 Query / Reasoning Layer                   │
│  Gap Miner | Evidence Auditor | Novelty Checker | Recall │
└───────────┬──────────────────────────────┬───────────────┘
            │                              │
     ┌──────▼──────┐               ┌──────▼──────┐
     │   Neo4j     │               │   Qdrant    │
     │ (Graph DB)  │◄──Anchor──►   │ (Vector DB) │
     │ 结构化推理   │   Edges       │ 语义检索     │
     └──────┬──────┘               └──────┬──────┘
            │                              │
   ┌────────┼──────────────────────────────┼────────┐
   │ Layer 3│  论证层 (Argumentation)      │        │
   │  PROBLEM → GAP → CORE_IDEA → CLAIM → EVIDENCE │
   ├────────┼──────────────────────────────┼────────┤
   │ Layer 2│  发展脉络层 (Evolution)      │        │
   │  CITES_AS_BASELINE | EVOLVES_FROM | CITES_FOR_*│
   ├────────┼──────────────────────────────┼────────┤
   │ Layer 1│  语义实体层 (Semantic)       │        │
   │  METHOD | DATASET | METRIC | TASK | MODEL      │
   └────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆并安装
cd AcadGraph
cp .env.example .env
# 编辑 .env 填入你的 LLM/Embedding API 配置

pip install -e ".[dev]"
```

### 2. 启动数据库

```bash
docker-compose up -d
```

### 3. 初始化 Schema

```bash
acadgraph init
```

### 4. 添加论文

```bash
# 添加单篇论文 (PDF)
acadgraph add paper.pdf --title "My Paper" --year 2024 --venue "ICLR"

# 添加单篇论文 (文本文件)
acadgraph add paper.txt --title "My Paper"

# 批量添加
acadgraph batch ./papers_dir/ --batch-size 5
```

### 5. 查询

```bash
# 找竞争文献空间
acadgraph query "attention mechanism for long sequences" --mode competition

# 生成 Gap Statement
acadgraph query "sparse attention with linear complexity" --mode gap

# 增强召回
acadgraph query "efficient transformer" --mode recall

# 审计论文证据链
acadgraph audit <paper_id>

# 查看统计
acadgraph stats
```

## 📁 项目结构

```
src/acadgraph/
├── config.py                  # 配置管理
├── llm_client.py             # LLM 调用封装
├── embedding_client.py        # Embedding 调用封装
├── cli.py                    # CLI 入口
└── kg/
    ├── schema.py              # 所有数据结构定义
    ├── paper_parser.py        # 全文解析 (PDF/Text → Sections)
    ├── layer1_entities.py     # Layer 1: 语义实体抽取 (7 类)
    ├── layer2_evolution.py    # Layer 2: 引用关系 + 技术演化链
    ├── layer3_argumentation.py # Layer 3: 论证链 (3-Pass Pipeline)
    ├── incremental_builder.py # 增量构建管道
    ├── query_engine.py        # 推理查询引擎
    ├── storage/
    │   ├── neo4j_store.py     # Neo4j 图存储
    │   └── qdrant_store.py    # Qdrant 向量存储
    └── prompts/
        ├── layer1_extraction.py
        ├── layer2_citation.py
        ├── layer3_zoning.py    # Pass 1: 修辞角色
        ├── layer3_schema.py    # Pass 2: 结构化抽取
        └── layer3_evidence.py  # Pass 3: 证据链接
```

## 🔑 关键设计决策

1. **自研核心，借鉴思想**：没有直接使用 LightRAG 或 GraphRAG，因为两者都不支持我们需要的强类型三层 Schema、论证链建模和带意图引用关系。但借鉴了 LightRAG 的增量更新思想和 GraphRAG 的社区检测思想。

2. **Neo4j 为主，Qdrant 为辅**：结构化推理（引用链、论证链多跳遍历）用 Neo4j，语义检索（相似实体/claim）用 Qdrant。

3. **3-Pass LLM Pipeline**：Pass 1 修辞角色分类 → Pass 2 结构化抽取 (PROBLEM/GAP/CLAIM) → Pass 3 证据链接与数值一致性检查。

4. **增量更新**：每篇新论文 ~10 次 LLM 调用入图，不需要全量重建。

5. **Claim-Evidence Ledger**：核心审计工具，像审稿人一样检查每个 P0 级 claim 是否有充分的实验证据支持。

## 📋 License

MIT
