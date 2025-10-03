# SocLitGen - 社会科学文献综述自动生成系统

基于大语言模型的智能文献综述生成框架，专为社会科学研究需求设计，实现从主题分解到完整综述的全流程自动化。

## 项目概述

SocLitGen 是一个综合性的 AI 驱动文献综述生成系统，能够自动完成研究主题分解、文献检索、内容生成、表格制作、引用匹配等全部流程，生成符合学术规范的结构化文献综述。

## 核心功能

- **智能选题分解**：自动将研究主题分解为 2-3 个互补的子议题
- **混合检索机制**：结合向量检索和倒排索引的四步循环检索
- **LLM 精准筛选**：使用思维链提示进行文献相关性判断
- **结构化内容生成**：围绕竞争性理论视角组织文献
- **自动表格生成**：遵循社会科学期刊惯例创建结构化表格
- **APA 引用管理**：精确的引用匹配和参考文献列表生成
- **双语支持**：中文生成，可选英文翻译
- **语言比例控制**：支持设定中英文文献比例
- **增量保存**：所有步骤实时保存到 MongoDB，支持断点续传

## 项目结构
```
SocLitGen/
├── agent/                                    主要代理模块
│   ├── citation_formatting.py              引用格式化
│   ├── literature_collection.py            文献收集
│   ├── literature_review_system.py         主系统
│   ├── overview_commentary.py              总论述评生成
│   ├── review_generation.py                综述生成
│   ├── table_generation.py                 表格生成
│   └── topic_planning.py                   选题规划
│
├── components/                              工具组件
│   ├── check_state.py                      状态检查
│   ├── config.py                           配置管理
│   ├── editor.py                           引用检查
│   ├── es_search.py                        ES 检索接口
│   └── llm_call.py                         LLM 调用接口
│
├── literature_preprocessing/                文献预处理
│   ├── element_extraction_and_summary.py   元素提取与摘要
│   ├── es_index_builder.py                 ES 索引构建
│   ├── literature_classification.py        文献分类
│   └── research_config.xlsx                研究配置
│
├── main.py                                  FastAPI 服务器
├── service.py                               业务逻辑层
├── model.py                                 数据模型
├── requirements.txt                         项目依赖
└── .env                                     环境变量配置
```
## 模块功能详解

### 1. Agent 模块 (agent/)

#### topic_planning.py - 选题规划
使用 few-shot 提示引导 LLM 分解研究主题，确保子议题的概念独立性和研究方向的具体性。

**核心方法**：
- `decompose_topic()` - 分解主题为 2-3 个子议题
- `validate_subtopics()` - 验证子议题质量

#### literature_collection.py - 文献收集

**四步循环检索机制**：
1. **查询生成**：基于 LLM 的智能关键词生成
2. **混合检索**：向量索引 + 倒排索引融合
3. **LLM 筛选**：两阶段相关性判断（概念 → 方向）
4. **迭代检索**：使用已检索文献标题扩展查询

**核心方法**：
- `collect_literature()` - 并行收集所有子议题文献
- `collect_single_subtopic_literature()` - 单个子议题收集
- `collect_single_subtopic_literature_by_weight()` - 按语言比例收集

**特性**：
- 支持语言过滤和比例控制
- 并行处理（最多 10 个线程）
- 全局去重机制

#### review_generation.py - 综述生成

使用思维链（CoT）推理将文献分类为竞争性理论视角，生成三种类型的段落。

**核心方法**：
- `generate_paragraphs_only()` - 并行生成所有段落
- `generate_paragraph()` - 生成单个段落（concept/theory/findings）
- `refine_paragraph_list()` - 并行精炼段落

**特性**：
- 长文模式支持（2500+ 汉字）
- 自动扩充内容
- 段落连贯性优化

#### table_generation.py - 表格生成

基于段落内容生成 Markdown 格式的结构化表格。

**核心方法**：
- `generate_tables_only()` - 并行生成所有表格
- `generate_table()` - 生成单个表格

#### overview_commentary.py - 总论述评

生成宏观层面的总论和述评。

**核心方法**：
- `generate_overview_and_critique()` - 并行生成总论和述评

#### citation_formatting.py - 引用格式化

精确的 APA 第 7 版格式引用管理。

**核心方法**：
- `transform_to_apa_format()` - 转换为 APA 格式
- `extract_cited_references()` - 提取实际被引用的参考文献

#### literature_review_system.py - 主系统

整个系统的核心协调器，负责统筹所有 Agent、组装文本、增量保存。

**核心方法**：
- `generate_review()` - 主入口方法
- `generate_review_structure_0()` - 标准结构生成
- `generate_review_structure_1()` - 语言分离结构生成

**工作流程**：
1. 选题规划 → 保存
2. 文献收集 → 保存
3. 段落生成 → 保存
4. 表格生成 → 保存
5. 主题组装 → 保存
6. 总论述评 → 保存
7. 引用提取 → 保存
8. 完整文本 → 保存
9. 翻译（可选）→ 保存

### 2. Components 模块 (components/)

#### config.py - 配置管理
集中管理所有配置项，从 `.env` 文件加载环境变量。

#### llm_call.py - LLM 调用接口
统一的多提供商 LLM 调用接口，支持 OpenAI、Anthropic、智谱 AI、Doubao、SiliconFlow、Qwen、DeepSeek，具有自动 fallback 机制。

#### es_search.py - Elasticsearch 检索
混合检索实现（向量 + 文本），支持多种 chunk_type 和语言过滤。

#### editor.py - 引用检查工具
验证引用在文本中的存在性，支持多种引用格式匹配。

### 3. Literature Preprocessing 模块 (literature_preprocessing/)

文献预处理模块负责文献理解和结构化摘要生成，包括文献分类（10 种研究类型）、元素提取（9 种核心元素）、摘要生成和 Elasticsearch 索引构建。

### 4. 服务层

#### main.py - FastAPI 服务器
RESTful API 服务器，提供生成综述和查询状态两个端点。

#### service.py - 业务逻辑
封装业务逻辑，包括异步生成综述和任务状态管理。

#### model.py - 数据模型
Pydantic 数据验证模型定义。

## 安装部署

### 环境要求
```
- Python 3.8+
- MongoDB
- Elasticsearch 8.x
- LLM API 密钥（至少一个提供商）
```
### 安装步骤

**1. 克隆项目**
```
git clone https://github.com/liu-zhanyu/SocLitGen.git
cd soclitgen
```
**2. 安装依赖**
```
pip install -r requirements.txt
```
**3. 配置环境变量**

创建 `.env` 文件：
```
# ==================== MongoDB Configuration ====================
MONGO_HOST=your_mongo_host
MONGO_PORT=27017
MONGO_USER=your_mongo_user
MONGO_PASSWORD=your_mongo_password
DB=data_analysis_v2
COLLECTION=article_info_v2

# ==================== Elasticsearch Configuration ====================
ES_HOST=http://your_es_host:9200
ES_USER=elastic
ES_PASSWORD=your_es_password
DEFAULT_INDEX=your_default_index

# ==================== Knowledge Base IDs ====================
KB_ID_SUMMARY=your_summary_kb_id

# ==================== Default ES Search Parameters ====================
DEFAULT_TOP_K=30
DEFAULT_VECTOR_WEIGHT=0.7
DEFAULT_TEXT_WEIGHT=0.3
DEFAULT_CHUNK_TYPE=["summary"]

# ==================== BGE API Configuration ====================
BGE_API_URL=https://api.siliconflow.cn/v1/embeddings
BGE_API_TOKEN_1=your_bge_token_1
BGE_API_TOKEN_2=your_bge_token_2

# ==================== LLM API Keys ====================
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Ark (Doubao)
ARK_API_KEY=your_ark_api_key

# SiliconFlow
SILICONFLOW_API_KEY=your_siliconflow_api_key

# ZhipuAI (智谱AI)
ZHIPUAI_API_KEY=your_zhipuai_api_key

# Qwen (通义千问)
QWEN_API_KEY=your_qwen_api_key

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key
```
**4. 启动服务**
```
python main.py
```
服务将在 `http://localhost:8001` 启动。

## 联系方式

如有问题或需要支持，请提交 Issue 或联系：zyliu22@m.fudan.edu.cn