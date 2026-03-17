# AI Search Copilot — 企业级项目文档

## 1. 项目概述

**项目名称**：AI Search Copilot  
**定位**：面向 boutique / mid-market executive search firm 的 AI 驱动搜索操作系统，覆盖 role intake、candidate research、market mapping、ranking、briefing 等核心环节，为顾问和 researcher 提供可解释、可审计的智能助手。

**目标**：

- 把传统高端猎头流程拆解为**可编排的 agent/workflow**，而不是单次 prompt。
- 在不引入真实 PII 的前提下，构建**贴近生产形态的原型**，可在 regulated financial services 环境中平滑演进。
- 为后续接入真实 CRM / email / document store 留出**清晰的 integration 边界与适配层**。

---

## 2. 业务背景与核心场景

### 2.1 目标用户

- Executive search firm 内部：
  - Partner / Consultant：负责 BD 与客户关系
  - Researcher / Associate：负责 candidate mapping 与 market research
  - Management / Compliance：关心审计、风险与效率

### 2.2 核心用户故事

- **Role Intake**：将客户需求 / JD 解析为结构化 `role_spec`，供检索与打分使用。
- **Candidate Research & Market Mapping**：从内部知识库 + demo 候选池中检索/聚合关键信息，为顾问提供 market view。
- **Explainable Ranking**：对候选人进行**可解释打分**（规则分 + LLM 解释层），生成 shortlist/longlist。
- **Client Brief**：生成内部搜索备忘录 / 对外 briefing draft，由顾问审批后再发给客户。
- **Auditability**：所有重要调用保留 request_id、prompt version、model、来源文档、审批状态等元数据。

---

## 3. 系统总体架构

### 3.1 技术栈

- **后端**：Python, FastAPI, Pydantic
- **LLM 层**：Anthropic Claude (`anthropic` SDK)，支持 mock / real 双模式
- **Agent / Workflow**：LangGraph（规划中），当前共享状态 & 顺序编排封装在 `app/workflows/`
- **检索**：ChromaDB（`PersistentClient`，内存 fallback）、简易 rule-based ranking
- **前端**：Streamlit（4 步向导式 UI）
- **DevOps**：`requirements.txt`、`.env`、本地运行；后续可接 Docker / CI

### 3.2 模块分层

- `app/main.py`：FastAPI 应用入口与路由注册。
- `app/api/`：HTTP API 层（health, search, brief）。
- `app/services/`：领域服务层（role parsing、candidate 搜索、ranking、brief 生成）。
- `app/llm/`：LLM 客户端封装与 prompt 定义。
- `app/retrieval/`：embedding store（Chroma）与检索逻辑。
- `app/workflows/`：面向 use-case 的编排（如 candidate_search_graph）。
- `app/core/`：通用能力（异常、PII 处理、audit 日志）。
- `app/repositories/`：简单的 file-backed 存储（brief repo）。
- `app/ui/`：Streamlit 前端（step-based wizard）。
- `data/`：demo 数据、向量库、audit 与 brief 存储。
- `scripts/`：数据初始化 / ingest 工具。

---

## 4. 关键子系统与组件说明

### 4.1 API 层（app/api）

- **routes_health.py**
  - `GET /health`：简单存活检测。
- **routes_search.py**
  - `POST /search/intake`：Role Intake + RAG 检索。
  - `POST /search/candidates`：基于 demo pool 的候选人检索。
  - `POST /search/rank`：规则 + 打分逻辑。
  - `POST /search/run`：端到端 agent 流程（intake → retrieve → candidates → rank → brief → critique）。
- **routes_brief.py**
  - `POST /brief/generate`：生成 markdown brief，并持久化到 brief repo。
  - `POST /brief/approve/{brief_id}`：审批 brief。
  - `GET /brief/{brief_id}`：查看 brief。
  - `GET /brief/{brief_id}/export`：仅在 approved=true 时允许导出（模拟审批 gate）。

### 4.2 LLM 封装（app/llm）

- **ClaudeClient**：
  - 读取 `ANTHROPIC_API_KEY`，若不存在则使用 **mock** 策略保证系统可离线运行。
  - 提供统一的 `generate_text(system_prompt, user_prompt, ...)` 接口。
- **prompts.py**：
  - `ROLE_PARSER_SYSTEM_PROMPT`：将 JD 转结构化 role spec，要求返回纯 JSON。
  - `BRIEF_GENERATOR_SYSTEM_PROMPT`：生成内部 executive search briefing note。
  - 对每个 prompt 维护 `PROMPT_ID` 与 `PROMPT_VERSION`，用于审计和后续 prompt versioning。

### 4.3 Workflow / Agent 编排（app/workflows）

- **SearchState**（TypedDict）：共享状态字段包括 request_id、raw_user_input、parsed_role、retrieval_context、candidate_pool、ranking_results、brief_draft、critique_feedback 等。
- **candidate_search_graph.run_workflow**：
  - 生成 request_id；
  - 解析 role（JobParserService）；
  - 调用 PlannerAgent 生成步骤列表（目前为 rule-based planner）；
  - RAG 检索（Retriever + VectorStore）；
  - 载入 & 过滤候选池；
  - RankingService 打分；
  - BriefService 生成 markdown；
  - CritiqueAgent 做初步检查；
  - 通过 AuditLogger 写入 audit.jsonl。

### 4.4 前端 UI（app/ui）

- **streamlit_app.py**：
  - 注册基础布局与侧边 Workflow 时间线（👉 当前、✅ 已完成、⬜ 未完成）。
- **pages/1_role_intake.py**：
  - 左侧输入 JD，右侧 API 设置与提示；
  - 解析结果展示为指标卡片 + skills 列表 + RAG context 展开 card；
  - 提供 “下一步：Candidate Search →” 导航。
- **pages/2_candidate_search.py**：
  - DataFrame 表格展示候选人列表，侧边 detail card；
  - 提供返回/下一步导航。
- **pages/3_market_map.py**：
  - Top firms 柱状图 + DataFrame；
  - RAG firm profiles 折叠展示；
  - 提供返回/下一步导航。
- **pages/4_client_brief.py**：
  - 候选人多选 + Ranking 表格；
  - brief 生成、审批、导出按钮；
  - brief markdown 预览。

---

## 5. 数据模型与存储设计

### 5.1 领域模型（部分）

- **Candidate**：
  - 基本身份信息、当前职务、地理位置；
  - `sectors`、`functions`、`summary`、`evidence`、`source_urls`；
  - `confidence_score`：数据质量可信度。
- **RoleSpec**（由 LLM 生成的字典）：
  - `title` / `seniority` / `sector` / `location`；
  - `required_skills` / `preferred_skills`；
  - `search_keywords` / `disqualifiers`；
  - `_prompt`：prompt id 与 version；
  - `_parse_error`（可选）：标记解析兜底情况。
- **BriefDocument**（API 层用 dict + brief repo 结构化）：
  - `brief_id`、`markdown`、`generated_at`、`citations`；
  - `approved`、`approved_at`（在 repo 中维护）。

### 5.2 存储

- **Demo 数据**：
  - `data/raw/sample_candidates/candidates.json`
  - `data/raw/sample_firm_profiles/firm_profiles.json`
- **向量库**：
  - `data/vector_db/`（Chroma 持久化目录，支持内存 fallback）
- **审计**：
  - `data/audit.jsonl`（一行一个 AuditEvent）
- **Brief 存储**：
  - `data/processed/briefs/{brief_id}.json`（file-backed repo）

> 企业级化建议：实际部署时应替换为 SQL/NoSQL 数据库与集中式日志系统，并将 demo 数据、运行产物与真实数据层分离。

---

## 6. LLM / RAG / Ranking 设计

### 6.1 LLM 设计

- **Role Parser**：
  - 输入：raw JD / client request；
  - 输出：严格 JSON 的 `RoleSpec`；
  - 容错：`_safe_json_extract` 处理 code-fence、混杂文本；解不出时 fallback 到 `Unparsed Role` 且 `_parse_error=true`，避免 API 500。
- **Brief Generator**：
  - 输入：role spec + ranked candidates + RAG context；
  - 输出：markdown brief，包含固定章节与 citations；
  - 支持 prompt cache 场景（通过 prompt id/version 区分稳定指令前缀）。

### 6.2 RAG 设计

- **文档类型**：firm profiles（demo）；
- **流程**：JSON → ingest → Chroma collection `docs`；
- **检索**：
  - query：`search_keywords` 拼接或 role title；
  - where：优先按 sector 过滤；如无结果则回退为无过滤查询；
  - 结果：`doc_id`、`text`、`metadata`（sector、region、company、source 等）。

### 6.3 Ranking 设计（当前）

- **确定性维度**：
  - `skill_match`：required_skills 与候选人 summary/sectors 的交集；
  - `sector_relevance`：有无 sector 标记；
  - `location_alignment`：基于字符串匹配 location；
  - `seniority_match`、`functional_similarity`、`stability_signal`：当前为固定基线值。
- **综合得分**：
  ```
  Final Score =
    0.30 × skill_match +
    0.20 × seniority_match +
    0.20 × sector_relevance +
    0.15 × functional_similarity +
    0.10 × location_alignment +
    0.05 × stability_signal
  ```
- **输出**：fit_score、各子维度得分、`reasoning` 与 `risks` 列表。

> 企业级建议：将 Ranking 拆成独立的策略引擎（可 JSON 配置权重与规则），并支持 A/B test 与 evaluator 集成。

---

## 7. 安全、隐私与合规

### 7.1 PII 处理

- `app/core/pii.py`：简单的 email / phone masking；
- 审计与日志中不记录原始 PII 文本；
- Demo 仓库中所有候选人与公司数据均为**高仿真合成数据**，不包含真实个人/客户信息。

### 7.2 审计与可追溯性

- **AuditEvent 结构**：
  - `event_type`、`request_id`、`payload`、`ts`；
  - 关键事件：request_received、role_parsed（含 prompt）、plan_created、context_retrieved、candidates_collected、candidates_ranked、brief_generated、brief_critique。
- **Brief 审批**：
  - 生成 brief 时写入 repo（approved=false）；
  - `POST /brief/approve/{id}` 将其状态置为 approved，并写入 `approved_at`；
  - `GET /brief/{id}/export` 仅在 approved=true 时返回，模拟人审 gate。

### 7.3 模型使用合规

- 所有调用通过后端服务端；前端/浏览器不暴露 API key；
- 支持 mock mode 以在无网络或未获批准的环境中演示架构与流程。

---

## 8. 运维与可观测性（当前与规划）

### 8.1 当前

- **本地启动 / 手工运维**：
  - FastAPI + Uvicorn；
  - Streamlit 作为独立前端进程。
- **日志**：
  - `logging.basicConfig` 控制台输出；
  - audit 以 `jsonl` 形式存磁盘。

### 8.2 未来企业级化建议

- 接入集中日志（ELK 或云厂商日志服务），审计与应用日志分 index；
- 增加 metrics（Prometheus / OpenTelemetry）：
  - 每个 workflow 的 latency、LLM token 使用量、RAG 命中率等；
- 引入简单的 feature flag / config service 管理：
  - 切换 mock/real 模式；
  - 控制某些 agent/step 的开启与否。

---

## 9. 部署与环境

### 9.1 运行环境

- Python 3.12+
- 推荐通过 venv / Conda 管理虚拟环境
- `.env` 管理敏感配置（API key、数据库 / 向量库路径等）

### 9.2 部署形态（建议）

- **单机 PoC**：FastAPI + Streamlit 同机运行（当前形态）。
- **小规模内部环境**：
  - FastAPI 部署到容器 / PaaS（如 Cloud Run / ECS）
  - Streamlit 改造为 Next.js 或 React SPA，通过 Nginx / API Gateway 访问 FastAPI。
- **安全边界**：
  - 前端只访问后端 API；
  - 后端通过 VPC/Private Link 访问模型与数据服务。

---

## 10. 路线图与扩展方向

- **短期（PoC → internal alpha）**
  - 强化 Ranking 与 reason 输出；
  - 完善审计结构（含人审 identity、模型版本）；
  - 增加 evaluation 脚本与 demo benchmark。
- **中期（agent orchestration & integrations）**
  - 使用 LangGraph 显式编排 planner / retriever / ranker / brief / critique 各节点；
  - 引入 CRM / email / doc store 的 adapter 层接口（mock + 文档级契约）。
- **长期（企业级投产）**
  - 替换 demo 数据为脱敏生产数据，完成端到端渗透测试与合规评审；
  - 增加多租户支持、权限控制与配额管理；
  - 搭建指标看板（performance / quality / usage）支撑持续运营。
