# PaperGraph — 异质文献图导航与算法可视化

> 数算B课程大作业 · Vibe Coding文献导航 · 5个自实现图算法 · 交互式可视化

## 项目背景

Vibe Coding（AI辅助编程）是2024-2025年学术界和工业界的热门话题。作为数算B课程的课上汇报主题，我需要从约50篇相关论文中快速找到核心文献、理解研究脉络、定位知识空隙。

传统文献导航方式（Google Scholar搜索、引用链追踪）只能看到线性引用关系，无法发现**跨关键词、跨作者、跨研究社区**的隐式关联。本项目构建一个**异质文献图**，将论文、作者、关键词作为不同类型节点，通过5种边类型建立多维关联，并用自实现的图算法提供4个科研辅助功能。

**核心问题**：如何在大量文献中快速定位核心论文、发现研究社区、规划阅读路径、识别知识空隙？

**解决方案**：异质图 + 5个自实现算法 + 交互式可视化动画。

---

## 核心算法

> 所有算法均自实现，未使用 networkx、igraph 等外部图库。底层使用 Python dict + list + deque + heapq，完全对应课程知识点。

### 1. 类型过滤BFS (`src/graph/algorithms/traversal.py`)

**课程对应**：BFS层序遍历（lecture W09-12, 3.1节）

**自实现要点**：
- 使用 `collections.deque` 作为队列，`popleft()` 保证O(1)出队
- 扩展：遍历时只走指定边类型/节点类型的邻居（异质图的类型约束）
- 方法：`type_filtered_bfs(graph, start, edge_types, node_types, max_depth)`

**复杂度**：O(V + E)，每个节点和边最多访问一次

**应用**：从一篇论文出发，探索其"研究邻域"——只看引用链、只看同关键词论文

### 2. 多权重Dijkstra (`src/graph/algorithms/dijkstra.py`)

**课程对应**：Dijkstra堆优化版（lecture W09-12, 最短路径章节）

**自实现要点**：
- 使用 `heapq` 实现优先队列，每次取出距离最小节点
- 扩展：边权重 = base_weight × weight_profile[edge_type]，不同边类型权重不同
  - cite边权重低(0.5)：引用链是自然的阅读过渡
  - shared_keyword权重高(1.5)：关键词跳跃需要更多背景知识
  - shared_author权重最高(2.0)：跨作者切换跨度最大
- 方法：`multi_weight_dijkstra(graph, start, end, weight_profile)`

**复杂度**：O(E log V)，堆优化保证每次relaxation效率

**应用**：阅读路径规划——从论文A到论文B的最优阅读顺序

### 3. 异质PageRank (`src/graph/algorithms/pagerank.py`)

**课程对应**：不在课内大纲，但基于图的邻接表迭代，本质是BFS式的邻居扩散。创新点：不同边类型有不同转移概率权重。

**自实现要点**：
- 初始化所有节点 score = 1/N
- 每轮迭代：score(u) = (1-d)/N + d × Σ(score(v) × P(v→u))
- P(v→u) 取决于边类型和v的出度：type_weight / total_type_weighted_out_degree
- 阻尼系数 d=0.85，收敛条件 max_change < 1e-6

**复杂度**：O(k × (V + E))，k为迭代次数（通常<20）

**应用**：论文影响力排名——被更多重要论文引用/关联的论文分数更高

### 4. 连通分量 + 桥边检测 (`src/graph/algorithms/components.py`)

**课程对应**：
- 连通分量：BFS（lecture, 3.1节）
- 桥边检测：Tarjan DFS的disc/low数组扩展（lecture, DFS扩展应用）

**自实现要点**：
- 连通分量：BFS遍历所有节点，每个连通分量 = 一个可达集
- 桥边检测：DFS遍历时维护 `disc[u]`（发现时间）和 `low[u]`（最远回溯时间）
  - 若 low[v] > disc[u]，则边 u-v 是桥边（删掉后图将分裂）
- 方法：`connected_components(graph, node_type_filter, edge_type_filter)`
- 方法：`bridge_edges(graph, node_type_filter, edge_type_filter)`

**复杂度**：O(V + E)，DFS单次遍历即可同时求出所有桥边

**应用**：研究社区发现（连通分量） + 知识空隙定位（桥边 = 跨社区的脆弱连接）

### 5. 元路径子图提取 (`src/graph/algorithms/metapath.py`)

**课程对应**：受限DFS/BFS，类型约束版遍历

**自实现要点**：
- 定义元路径（如 PAP = Paper-Author-Paper, PKP = Paper-Keyword-Paper）
- 沿元路径逐步遍历，每步只走符合路径类型约束的边
- 最终收集到达的target节点，计算出现频次作为关联强度

**复杂度**：取决于元路径长度和中间节点出度

**应用**：相关论文推荐——同作者(PAP)或同关键词(PKP)的论文

---

## 算法动画系统 (`src/graph/algorithms/traces.py`)

每个算法提供逐步执行轨迹（trace），前端动画播放时显示三层信息：

1. **意义层**：算法对文献导航的实际含义（如"BFS第1层：近邻——直接引用的论文"）
2. **摘要层**：关键发现（如"社区1: 16篇论文"）
3. **算法层**：DSA知识点（如"Tarjan DFS: low[v]>disc[u] → 桥边条件"）

---

## 项目结构

```
PaperGraph/
├── src/graph/               # 图数据结构 + 5个算法
│   ├── hgraph.py            # 异质图数据结构（dict+list，无networkx）
│   ├── builder.py           # 图构建器
│   ├── node.py / edge.py    # 节点/边类
│   └── algorithms/
│       ├── traversal.py     # BFS
│       ├── dijkstra.py      # Dijkstra
│       ├── pagerank.py      # PageRank
│       ├── components.py    # 连通分量 + 桥边
│       ├── metapath.py      # 元路径
│       └── traces.py        # 算法执行轨迹（动画数据）
├── src/parser/              # 论文数据获取与处理
│   ├── fetch_arxiv.py       # arXiv元数据抓取
│   ├── download_pdfs.py     # PDF下载
│   ├── pdf_to_md.py         # PDF→Obsidian Markdown转换
│   ├── update_verified.py   # 交叉验证更新
│   └── cleanup_papers.py    # 数据清理
├── src/recommender/         # 科研辅助功能封装
│   ├── community.py         # 研究社区发现
│   ├── gaps.py              # 知识空隙定位
│   ├── reading_path.py      # 阅读路径规划
│   └── related_papers.py    # 相关论文推荐
├── data/                    # papers.json（27篇论文元数据）
├── pdfs/                    # 24篇论文PDF原文
├── obsidian_vault/          # 27篇论文Obsidian笔记 + 5个域笔记
├── static/                  # 前端可视化
│   ├── index.html           # 交互式Canvas可视化
│   ├── index_viewer.html    # 自包含HTML（嵌入所有数据）
│   └── data/                # 预计算JSON数据（16个文件）
├── docs/                    # 文档
│   ├── vibe_coding_survey.md # 图结构引导的Vibe Coding综述
│   ├── graph_full.png        # 全图截图
│   ├── graph_papers.png      # 论文子图截图
│   └── pagerank_heatmap.png  # PageRank热力图截图
├── compute_results.py       # 预计算脚本（生成所有JSON + viewer）
├── tests/                   # 算法测试
│   └── test_algorithms.py   # 5个算法的完整性测试
├── .github/workflows/       # GitHub Actions自动部署
├── README.md                # 本文件
├── LICENSE                  # MIT
└── requirements.txt         # numpy, pymupdf, requests
```

---

## 运行指南

### 环境要求

- Python 3.10+
- pip

### 步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成所有预计算数据和可视化页面
python compute_results.py

# 3. 预览可视化
# 直接在浏览器中打开 static/index_viewer.html
# 或通过VS Code打开该文件

# 4. 运行算法测试
PYTHONPATH=. python3 tests/test_algorithms.py
```

### GitHub Pages自动部署

推送代码到GitHub后，`.github/workflows/deploy.yml` 会自动：
1. 运行 `compute_results.py` 生成数据
2. 将 `static/` 目录部署到 GitHub Pages

---

## 数据说明

`data/papers.json` 包含27篇真实论文，所有arXiv ID经过PDF内容交叉验证。其中24篇已下载PDF原文并转换为Obsidian Markdown笔记。每篇论文标注：
- `confidence`: "verified"（PDF已验证）或 "partial"（仅元数据验证）

---

## 综述成果

基于图算法的计算结果，撰写了一篇图结构引导的Vibe Coding综述（`docs/vibe_coding_survey.md`）：

- **PageRank** 确定B1(HumanEval)为核心锚点，从最有影响力的论文开始阅读
- **连通分量** 发现研究社区不是按领域划分，而是按问题束聚合（代码生成及其下游影响）
- **桥边检测** 定位10条脆弱跨域连接，如评测→安全(B1→E3)、基准→智能体(D1→D2)
- **元路径** 追踪作者/关键词关联，发现UIUC Lingming Zhang组跨越代码生成与自动修复

综述的核心结论：Vibe Coding不是单一概念，而是三个结构性悖论——更快但更不安全(C2 vs C10)、更多代码但更少理解(C3 vs C8)、知道跑通但不知道安全(B1 vs E3)。图结构直接贡献了这些洞察，因为悖论的对冲关系来自PageRank并列节点与桥边的拓扑结构。

---

## AI工具声明

| 部分 | AI辅助 | 自主实现 |
|------|---------|----------|
| 5个核心算法逻辑 | AI辅助编写框架代码 | 算法选型、数据结构选择、异质图扩展设计 |
| 图数据结构(hgraph.py) | AI辅助编写 | 邻接表+反向邻接表设计、类型索引设计 |
| 前端可视化 | AI辅助编写Canvas渲染和交互 | 物理模拟参数调优、算法动画叙事设计、交互bug修复 |
| 论文元数据验证 | AI辅助搜索arXiv ID | PDF内容交叉验证、编造论文识别与删除 |
| 数据完整性 | AI辅助批量搜索 | 判断哪些论文是编造的，决定删除策略 |
| PDF→MD转换 | AI辅助运行转换脚本 | 决定转换策略、判断哪些论文需补PDF |
| Vibe Coding综述 | AI辅助撰写文字 | 综述框架设计、三个悖论的提炼、图-结论对应关系的论证 |
| README文档 | AI辅助起草 | 内容审核、算法课程对应关系的确认 |

核心算法的逻辑架构、异质图类型约束扩展、动画叙事三层设计（意义/摘要/知识点）、综述的悖论提炼均为自主思考。

使用的AI工具：Claude Code (Anthropic)

---

## License

MIT License
