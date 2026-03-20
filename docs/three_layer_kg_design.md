# 三层证据链知识图谱系统 — 完整设计方案

## 一、问题诊断：当前系统的核心缺陷

### 当前状态
- **数据源**：仅使用 title + abstract + review，未利用论文全文
- **实体类型**：只有 Pattern / Idea / Domain / Paper / Review 五类节点
- **关系类型**：仅有 Paper→Pattern、Paper→Idea、Paper→Domain 等简单归属关系
- **缺失的关键能力**：
  1. 无引用关系建模（谁引了谁、为什么引）
  2. 无时间演化链（技术如何从A演变到B）
  3. 无论证链（PROBLEM→GAP→CLAIM→EVIDENCE 闭环）
  4. 无全文 section 级解析
  5. 无增量更新机制（每次全量重建）

### 目标
构建一套 **三层证据链知识图谱**，覆盖论文全文的结构化解析，支撑 idea2paper 的推理任务（Gap Mining + Evidence Auditing），并支持增量构建。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Query / Reasoning Layer                   │
│  Gap Miner | Evidence Auditor | Novelty Checker | Recall    │
└───────────┬────────────────────────────────┬────────────────┘
            │                                │
     ┌──────▼──────┐                  ┌──────▼──────┐
     │   Neo4j     │                  │   Qdrant    │
     │ (Graph DB)  │◄────Anchor──────►│ (Vector DB) │
     │ 结构化推理   │    Edges        │ 语义检索     │
     └──────┬──────┘                  └──────┬──────┘
            │                                │
   ┌────────┼────────────────────────────────┼────────┐
   │ Layer 3│ 论证层 (Argumentation)         │        │
   │  PROBLEM → GAP → CORE_IDEA → CLAIM → EVIDENCE   │
   ├────────┼────────────────────────────────┼────────┤
   │ Layer 2│ 发展脉络层 (Evolution)         │        │
   │  CITES_AS_BASELINE | EVOLVES_FROM | CITES_FOR_*  │
   ├────────┼────────────────────────────────┼────────┤
   │ Layer 1│ 语义实体层 (Semantic Entities) │        │
   │  METHOD | DATASET | METRIC | TASK | MODEL        │
   └─────────────────────────────────────────────────┘
            ▲
            │ Full-Text Parsing Pipeline
   ┌────────┴─────────┐
   │  PDF → marker    │
   │  Section Zoning  │
   │  3-Pass LLM      │
   └──────────────────┘
```

---

## 三、实现步骤（共 8 个模块）

### 模块 1: 全文解析管道 (`src/idea2paper/kg/paper_parser.py`)

**功能**：PDF → 结构化 Sections → Markdown

```python
class PaperParser:
    """论文全文解析器"""

    def parse(self, pdf_path: str) -> ParsedPaper:
        """
        Returns:
            ParsedPaper with sections:
            - abstract, introduction, related_work,
            - method, experiments, limitation, conclusion
            - tables: List[Table], figures: List[Figure]
            - references: List[Reference]
        """

    def parse_from_text(self, full_text: str) -> ParsedPaper:
        """从已有文本（如 OpenReview/arXiv HTML）解析"""
```

**技术选型**：
- 主力: `marker` (preserves tables/formulas)
- Fallback: `pymupdf4llm`
- Section 切分: 基于正则 + LLM 辅助的 heading 识别

**输出 Schema**:
```python
@dataclass
class ParsedPaper:
    paper_id: str
    title: str
    year: int
    venue: str
    sections: Dict[str, str]  # section_name → text
    tables: List[TableData]    # structured table extraction
    figures: List[FigureRef]   # figure captions + references
    references: List[Reference]  # parsed bibliography
    raw_text: str
```

---

### 模块 2: Layer 1 — 语义实体抽取 (`src/idea2paper/kg/layer1_entities.py`)

**功能**：从论文各 section 抽取 7 类学术实体

**实体类型**:
| 实体类型 | 抽取来源 | 属性 |
|---------|---------|------|
| METHOD | Method/Intro | name, description, category, components |
| DATASET | Experiments | name, domain, size, task_type, url |
| METRIC | Experiments | name, higher_is_better, domain |
| TASK | Abstract/Intro | name, description, domain |
| MODEL | Method/Experiments | name, architecture, params, pretrained |
| FRAMEWORK | Method | name, description, components |
| CONCEPT | All sections | name, definition, domain |

**实现方式**：
```python
class Layer1Extractor:
    """语义实体抽取器 — 基于 LightRAG 风格的 schema-constrained extraction"""

    def extract(self, parsed_paper: ParsedPaper) -> Layer1Result:
        """
        Per-section extraction:
        - Abstract+Intro → TASK, CONCEPT, high-level METHOD
        - Method → METHOD (detailed), FRAMEWORK, MODEL
        - Experiments → DATASET, METRIC, MODEL (baselines)
        - Related Work → METHOD (prior), CONCEPT
        """

    def deduplicate(self, entities: List[Entity]) -> List[Entity]:
        """Cross-paper entity deduplication via embedding similarity + name matching"""
```

**Neo4j Schema**:
```cypher
CREATE CONSTRAINT FOR (m:Method) REQUIRE m.entity_id IS UNIQUE;
CREATE CONSTRAINT FOR (d:Dataset) REQUIRE d.entity_id IS UNIQUE;
-- ... 同理其他实体类型

// 实体关联
(:Method)-[:APPLIED_ON]->(:Task)
(:Method)-[:EVALUATED_ON]->(:Dataset)
(:Method)-[:MEASURED_BY]->(:Metric)
(:Method)-[:USES]->(:Framework)
(:Method)-[:OUTPERFORMS]->(:Method)
(:Paper)-[:PROPOSES]->(:Method)
(:Paper)-[:INTRODUCES]->(:Dataset)
```

**Qdrant 同步**：每个实体生成 embedding 存入 Qdrant collection `layer1_entities`

---

### 模块 3: Layer 2 — 引用关系与技术演化 (`src/idea2paper/kg/layer2_evolution.py`)

**功能**：建立带意图的引用边 + 时间演化链

**引用意图分类（6类）**:
| 引用意图 | 含义 | 抽取信号 |
|---------|------|---------|
| CITES_FOR_PROBLEM | 引用以说明问题背景 | Introduction 中的引用 |
| CITES_AS_BASELINE | 作为对比基线引用 | Experiments 中的引用 |
| CITES_FOR_FOUNDATION | 构建于此工作之上 | Method 中的引用 |
| CITES_AS_COMPARISON | 水平对比相关工作 | Related Work 中的引用 |
| CITES_FOR_THEORY | 理论基础引用 | Method/Intro 中理论引用 |
| EVOLVES_FROM | 技术演化关系 | 明确声明"基于X改进" |

**实现**:
```python
class Layer2EvolutionBuilder:
    """引用关系 + 技术演化链构建"""

    def classify_citations(self, parsed_paper: ParsedPaper) -> List[CitationEdge]:
        """
        Step 1: 从各 section 提取引用上下文 (citation context)
        Step 2: LLM 分类引用意图
        Step 3: 生成带意图的引用边
        """

    def build_evolution_chains(self, papers: List[Paper]) -> List[EvolutionChain]:
        """
        基于引用意图 + 时间排序，构建技术演化链:
        MethodA (2020) --EVOLVES_FROM--> MethodB (2021) --EVOLVES_FROM--> MethodC (2023)
        """

    def enrich_from_apis(self, paper_id: str) -> List[CitationEdge]:
        """从 Semantic Scholar / OpenAlex 补充引用数据"""
```

**Neo4j Schema**:
```cypher
// 引用边（带意图）
(:Paper)-[:CITES {intent: "BASELINE", context: "...", section: "experiments"}]->(:Paper)

// 技术演化边（带时间）
(:Method)-[:EVOLVES_FROM {year_from: 2020, year_to: 2023, delta: "added X"}]->(:Method)

// 时间索引
CREATE INDEX paper_year FOR (p:Paper) ON (p.year)
```

---

### 模块 4: Layer 3 — 论证链构建 (`src/idea2paper/kg/layer3_argumentation.py`)

**这是核心创新层** — 建模论文的完整论证链

**论证链结构**:
```
PROBLEM → GAP → CORE_IDEA → CLAIM₁ → EVIDENCE₁
                           → CLAIM₂ → EVIDENCE₂
                           → CLAIM₃ → EVIDENCE₃
                                    ↗ BASELINE (comparison)
                                    ↗ LIMITATION
```

**节点定义**:

| 节点类型 | 属性 | 抽取来源 |
|---------|------|---------|
| PROBLEM | description, scope, importance_signal | Abstract + Intro |
| GAP | failure_mode, constraint, prior_methods_failing | Intro + Related Work |
| CORE_IDEA | mechanism, novelty_type, key_innovation | Abstract + Method |
| CLAIM | text, type (novelty/performance/robustness/efficiency/theory/generality), severity (P0/P1/P2) | Title + Abstract + Intro + Conclusion |
| EVIDENCE | type (experiment/ablation/theorem/case_study), result_summary, datasets, metrics, tables, figures | Experiments + Appendix |
| BASELINE | method_name, paper_ref, performance | Experiments |
| LIMITATION | text, scope, acknowledged_by_author | Limitation + Conclusion |

**三遍 LLM 抽取**:
```python
class Layer3ArgumentationExtractor:
    """论证链抽取器 — 3-Pass Pipeline"""

    def pass1_argumentative_zoning(self, parsed_paper: ParsedPaper) -> ZonedPaper:
        """
        Pass 1: 修辞角色分类
        对每个段落标注角色: Motivation / Background / Contribution /
        Method / Result / Comparison / Limitation / Future
        """

    def pass2_schema_extraction(self, zoned_paper: ZonedPaper) -> ArgumentationGraph:
        """
        Pass 2: Schema-Constrained JSON 抽取
        从 Intro+Abstract 抽取: PROBLEM, GAP, CORE_IDEA
        从 Title+Abstract+Intro+Conclusion 拆解 atomic CLAIMS
        """

    def pass3_evidence_linking(self, argumentation: ArgumentationGraph,
                                parsed_paper: ParsedPaper) -> ArgumentationGraph:
        """
        Pass 3: Evidence-Claim 链接
        从 Experiments section 抽取 EVIDENCE 节点
        建立 CLAIM → EVIDENCE 的 supports/partially_supports/refutes/unverifiable 边
        核对数值一致性 (text claims vs table numbers)
        """
```

**Claim 拆解示例**:
```json
{
  "claims": [
    {
      "text": "Our method achieves SOTA on all 5 benchmarks",
      "type": "performance",
      "severity": "P0",
      "source_section": "abstract"
    },
    {
      "text": "The proposed module improves robustness under distribution shift",
      "type": "robustness",
      "severity": "P1",
      "source_section": "introduction"
    }
  ]
}
```

**Neo4j Schema**:
```cypher
// 论证链
(:Paper)-[:HAS_PROBLEM]->(:Problem)
(:Problem)-[:HAS_GAP]->(:Gap)
(:Gap)-[:ADDRESSED_BY]->(:CoreIdea)
(:CoreIdea)-[:MAKES_CLAIM]->(:Claim)
(:Claim)-[:SUPPORTED_BY {strength: "full"|"partial"|"refuted"|"unverifiable"}]->(:Evidence)
(:Evidence)-[:USES_BASELINE]->(:Baseline)
(:Paper)-[:HAS_LIMITATION]->(:Limitation)

// Claim-Evidence Ledger 视图
(:Claim {severity: "P0"})-[:SUPPORTED_BY {strength: "partial"}]->(:Evidence)
// → 这就是审稿人最关心的：P0 级 claim 只有 partial support
```

---

### 模块 5: 存储层 — Neo4j + Qdrant 双驱 (`src/idea2paper/kg/storage/`)

#### 5.1 Neo4j 存储 (`neo4j_store.py`)

```python
class Neo4jKGStore:
    """Neo4j 图存储 — 管理三层知识图谱的结构化数据"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

    # Layer 1
    def upsert_entity(self, entity: Entity) -> str: ...
    def upsert_relation(self, src_id: str, dst_id: str, rel_type: str, props: dict) -> None: ...

    # Layer 2
    def add_citation(self, citing: str, cited: str, intent: str, context: str) -> None: ...
    def get_evolution_chain(self, method_id: str) -> List[EvolutionStep]: ...

    # Layer 3
    def store_argumentation(self, paper_id: str, arg_graph: ArgumentationGraph) -> None: ...
    def get_claim_evidence_ledger(self, paper_id: str) -> ClaimEvidenceLedger: ...

    # 查询
    def find_nearest_competitors(self, paper_id: str, k: int = 10) -> List[Paper]: ...
    def get_gap_context(self, problem_id: str) -> GapContext: ...
    def traverse_evidence_chain(self, claim_id: str) -> EvidenceChain: ...
```

**Neo4j 索引**:
```cypher
// 全文索引 (for search)
CREATE FULLTEXT INDEX entity_names FOR (n:Method|Dataset|Task|Metric|Model) ON EACH [n.name];
CREATE FULLTEXT INDEX claim_text FOR (n:Claim) ON EACH [n.text];

// 时间索引
CREATE INDEX paper_year FOR (p:Paper) ON (p.year);

// 复合索引
CREATE INDEX claim_severity FOR (c:Claim) ON (c.severity, c.type);
```

#### 5.2 Qdrant 存储 (`qdrant_store.py`)

```python
class QdrantKGStore:
    """Qdrant 向量存储 — 语义检索层"""

    COLLECTIONS = {
        "layer1_entities": {  # METHOD, DATASET, METRIC, TASK 等实体的 embedding
            "vector_size": 4096,  # Qwen3-Embedding-8B
            "distance": "Cosine"
        },
        "layer2_citations": {  # 引用上下文的 embedding
            "vector_size": 4096,
            "distance": "Cosine"
        },
        "layer3_claims": {  # Claim 文本的 embedding
            "vector_size": 4096,
            "distance": "Cosine"
        },
        "layer3_evidence": {  # Evidence 描述的 embedding
            "vector_size": 4096,
            "distance": "Cosine"
        },
        "paper_sections": {  # 论文各 section 的 embedding
            "vector_size": 4096,
            "distance": "Cosine"
        }
    }

    def upsert_embedding(self, collection: str, id: str, vector: List[float],
                         payload: dict) -> None: ...
    def search_similar(self, collection: str, query_vector: List[float],
                       k: int, filters: dict = None) -> List[SearchResult]: ...
```

---

### 模块 6: 增量构建管道 (`src/idea2paper/kg/incremental_builder.py`)

**核心设计**：新论文到达时，只处理新论文，不重建已有图谱

```python
class IncrementalKGBuilder:
    """增量知识图谱构建器"""

    def __init__(self, neo4j_store: Neo4jKGStore, qdrant_store: QdrantKGStore):
        self.neo4j = neo4j_store
        self.qdrant = qdrant_store
        self.parser = PaperParser()
        self.l1_extractor = Layer1Extractor()
        self.l2_builder = Layer2EvolutionBuilder()
        self.l3_extractor = Layer3ArgumentationExtractor()

    async def add_paper(self, paper_source: PaperSource) -> BuildResult:
        """
        增量添加一篇论文到知识图谱

        Pipeline:
        1. Parse → ParsedPaper
        2. Layer1: Extract entities → Neo4j + Qdrant
        3. Layer2: Classify citations → Neo4j (edges)
        4. Layer3: Extract argumentation chain → Neo4j + Qdrant
        5. Cross-layer linking (entity ↔ claim ↔ evidence)
        6. Update evolution chains if new method detected
        """

    async def add_papers_batch(self, papers: List[PaperSource],
                                batch_size: int = 10) -> BatchBuildResult:
        """批量增量添加（并发处理，共享 dedup 缓存）"""

    def _check_duplicate(self, paper_id: str) -> bool:
        """检查论文是否已存在于图谱中"""

    def _cross_layer_link(self, paper_id: str):
        """
        跨层关联:
        - Layer1 METHOD ↔ Layer3 CLAIM (方法声明了什么)
        - Layer1 DATASET ↔ Layer3 EVIDENCE (在哪个数据集上验证)
        - Layer2 CITATION ↔ Layer3 GAP (引用如何支撑 gap)
        """
```

**增量更新策略**:
```
新论文到达
  ├── 1. 解析全文 → ParsedPaper
  ├── 2. 实体去重 (embedding similarity > 0.95 → merge)
  │     └── Neo4j: MERGE ON entity_id
  │     └── Qdrant: upsert with same point_id
  ├── 3. 引用边追加 (append-only)
  │     └── Neo4j: CREATE (:Paper)-[:CITES]->(:Paper)
  ├── 4. 论证链写入 (per-paper, idempotent)
  │     └── Neo4j: MERGE on (paper_id, claim_hash)
  └── 5. 演化链更新 (如有新 EVOLVES_FROM 关系)
        └── Neo4j: 检测并追加演化边
```

---

### 模块 7: 推理查询引擎 (`src/idea2paper/kg/query_engine.py`)

**功能**：供 idea2paper pipeline 调用的统一查询接口

```python
class KGQueryEngine:
    """知识图谱推理查询引擎 — 连接 Neo4j + Qdrant"""

    # === Gap Mining 查询 ===

    def find_competition_space(self, idea: str, k: int = 20) -> CompetitionSpace:
        """
        构建竞争文献空间:
        1. Qdrant: 语义检索最相似的论文 (embedding)
        2. Neo4j: 引文网络扩展 (co-citation, bibliographic coupling)
        3. 合并去重，返回最危险的 k 篇近邻
        """

    def structural_alignment(self, idea: str, competitors: List[Paper]) -> NoveltyMap:
        """
        结构化对齐:
        将 idea 和每篇竞争论文投射到统一坐标：
        [Problem, Setting, Assumption, Mechanism, Supervision, Metric, FailureMode]
        检测真正的差异点
        """

    def generate_gap_statement(self, novelty_map: NoveltyMap) -> GapStatement:
        """
        生成可被证伪的 gap 声明:
        "现有方法在 [Setting S] 下可以解决 [Problem P]，
         但在 [Constraint F] 下仍然失败，
         因为它们缺少 [Mechanism M]；
         据检索，尚无方法同时满足 [A, B, C]。"
        """

    # === Evidence Auditing 查询 ===

    def get_claim_evidence_ledger(self, paper_id: str) -> ClaimEvidenceLedger:
        """
        获取论文的 Claim-Evidence 台账:
        | Claim | Type | Severity | Required Evidence | Actual Evidence | Support Status |
        """

    def verify_claims_against_literature(self, claims: List[Claim]) -> List[VerificationResult]:
        """
        外部验证:
        1. Qdrant 检索相关 claims 和 evidence
        2. 查找支持/反驳/收缩当前 claim 的外部证据
        """

    # === Recall 增强查询 ===

    def enhanced_recall(self, idea: str, config: RecallConfig) -> List[Pattern]:
        """
        增强版三路召回 (替代现有 RecallSystem):
        Path 1: Idea → Qdrant(similar entities) → Neo4j(entity→paper→pattern)
        Path 2: Idea → Qdrant(similar claims) → Neo4j(claim→paper→pattern)
        Path 3: Idea → Neo4j(evolution chain) → temporally related patterns
        Path 4: 现有三路召回 (保留兼容)
        """

    # === 技术演化查询 ===

    def get_method_evolution(self, method_name: str) -> EvolutionTimeline:
        """获取方法的演化时间线: A(2019) → B(2021) → C(2023)"""

    def get_research_trends(self, domain: str, year_range: Tuple[int, int]) -> TrendReport:
        """获取领域研究趋势（基于 Leiden 社区分析）"""
```

---

### 模块 8: 与现有 Pipeline 集成

**改造点**:

1. **`recall_system.py` 增强** — 新增 Path 5 (KG-based recall):
```python
# 在 RecallSystem.__init__ 中
self.kg_engine = KGQueryEngine(neo4j_store, qdrant_store)

# 新增路径
def _recall_path5_kg(self, idea_text: str) -> List[PatternScore]:
    """KG 增强召回: 利用论证链和演化链找到更精准的 pattern"""
    competition = self.kg_engine.find_competition_space(idea_text)
    # ... 转换为 PatternScore
```

2. **`story_generator.py` 增强** — 利用 gap statement:
```python
# 在生成 story 前，先查 gap
gap = self.kg_engine.generate_gap_statement(novelty_map)
# 将 gap 注入到 story generation prompt 中
```

3. **`verifier.py` 增强** — 利用 claim-evidence ledger:
```python
# 验证生成的论文是否有完整的证据链
ledger = self.kg_engine.get_claim_evidence_ledger(paper_id)
unsupported = [c for c in ledger if c.support_status != "full"]
```

---

## 四、文件结构

```
src/idea2paper/kg/
├── __init__.py
├── paper_parser.py           # 模块1: 全文解析
├── layer1_entities.py        # 模块2: 语义实体抽取
├── layer2_evolution.py       # 模块3: 引用关系与演化链
├── layer3_argumentation.py   # 模块4: 论证链构建
├── storage/
│   ├── __init__.py
│   ├── neo4j_store.py        # 模块5a: Neo4j 存储
│   └── qdrant_store.py       # 模块5b: Qdrant 存储
├── incremental_builder.py    # 模块6: 增量构建管道
├── query_engine.py           # 模块7: 推理查询引擎
├── schema.py                 # 所有节点/边的 dataclass 定义
└── prompts/
    ├── layer1_extraction.py  # Layer1 抽取 prompt
    ├── layer2_citation.py    # Layer2 引用意图分类 prompt
    ├── layer3_zoning.py      # Layer3 Pass1 修辞角色 prompt
    ├── layer3_schema.py      # Layer3 Pass2 结构化抽取 prompt
    └── layer3_evidence.py    # Layer3 Pass3 证据链接 prompt
```

---

## 五、实施路线

| Phase | 工作内容 | 依赖 | 交付物 |
|-------|---------|------|--------|
| **Phase 1** | schema.py + storage/ (Neo4j + Qdrant) | Neo4j/Qdrant 部署 | 存储层可用 |
| **Phase 2** | paper_parser.py (全文解析) | marker 安装 | Section 切分可用 |
| **Phase 3** | layer1_entities.py | Phase 1+2 | 实体抽取 + 存储 |
| **Phase 4** | layer2_evolution.py | Phase 1+3, Semantic Scholar API | 引用+演化链 |
| **Phase 5** | layer3_argumentation.py (核心) | Phase 1+2 | 论证链抽取 |
| **Phase 6** | incremental_builder.py | Phase 3+4+5 | 增量构建可用 |
| **Phase 7** | query_engine.py | Phase 1-6 | 推理接口可用 |
| **Phase 8** | Pipeline 集成 | Phase 7 | recall/story/verify 增强 |

---

## 六、关键设计决策

1. **Neo4j 为主 Qdrant 为辅**：结构化推理用 Neo4j（引用链、论证链需要多跳遍历），语义检索用 Qdrant（相似实体/claim 检索）
2. **LightRAG 风格的增量更新**：每篇新论文 ~10 次 LLM 调用即可入图，不需要全量重建
3. **论证链是 append-only**：历史论文的 claim-evidence 关系不会被后来的论文修改，只会追加新的交叉验证边
4. **向后兼容**：现有 nodes_*.json 和 RecallSystem 继续工作，KG 作为增强层叠加
5. **Section 级粒度**：不是把整篇论文当一个节点，而是每个 section 的内容都有对应的实体/claim/evidence 锚点
