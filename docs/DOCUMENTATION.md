# 项目技术文档（中文）

## 1. 项目概述
- 针对任意分类标签数据集，用户指定“OK”取值，其余自动标记为“KO”，再完成统计分析、判别特征挖掘、可视化，并通过聊天界面交互。
- 支持自然语言请求生成时域与频域图、统计指标与判别特征报告，结果在 GUI 中展示。
- 可选本地或云端 LLM 后端，结合 AutoGluon 与传统模型完成特征重要性与分类评估。

## 2. 系统架构
- 分层关系（上下游）：
	- 表示层（UI）：Streamlit 负责配置/展示，不含业务逻辑。
	- 业务/代理层：agent_core 负责意图解析与工具编排，调用 llm_interface（可选 LLM）、conversation 管理上下文。
	- 分析/模型层：statistical_analyzer、feature_importance、model_trainer 提供可调用的分析与训练能力。
	- 数据处理层：loader/preprocessor 完成数据读取、清洗、OK/KO 标签构造与特征准备，为上层提供 DataFrame。
	- 可视化工具：plotting_tools（位于 agent 模块下的工具集），为代理层/分析层提供可复用绘图函数，结果回传 UI（不单独成层）。
- 调用方向：UI → 代理层 →（数据处理 / 分析 / 模型 / 可视化）；LLM 仅在代理层按需参与意图增强或回答。

## 3. 设计选择
### 3.1 统计分析
- 描述性：均值/中位数/众数、std/var、min/max、分位数、偏度/峰度，快速把握分布形态（详见 [docs/ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)）。
- 显著性检验（α=0.05）：
	- 数值：T 检验（均值差，正态假设）、Mann-Whitney U（位置差，非参数更稳健）、KS（分布形状差异）。
	- 分类：卡方检验（分布独立性）。
- 效应量：Cohen's d（数值，衡量差异大小）、Cramér's V（分类，衡量关联强度），用于判断工程上是否“重要”。
- 排名策略：
	- 数值特征：score = -log10(p_MW) × |Cohen's d| × difference_ratio（均值相对差），兼顾显著性与实际差异。
	- 分类特征：score = -log10(p_χ²) × Cramér's V，兼顾显著性与关联强度。
	- 以 score 降序输出 feature_ranking 供 UI 展示与后续模型训练。

### 3.2 特征选择与判别
- AutoGluon 进行集成特征重要性；详见 [docs/FEATURE_IMPORTANCE_ANALYZER.md](FEATURE_IMPORTANCE_ANALYZER.md) 获取排除列、训练流程与输出结构说明。

### 3.3 模型训练
- 特征来源可选：统计分析排名或 AutoGluon 排名，均可作为后续训练的特征子集来源。
- 模型对比：Logistic、SVM(RBF)、Decision Tree、Random Forest；支持多组 top-N 特征组合交叉训练，观察不同特征来源/特征数对性能的影响。

### 3.4 AI 代理
- 多后端 LLM；规则优先的意图解析，LLM 仅作回退；工具返回结构化 summary+figure，避免幻觉。
- 支持请求：统计摘要、特征重要性、时序图、FFT 频谱、分布对比、相关热图、特征对比。

### 3.5 GUI
- 五步配置向导：加载数据 → 选择标签/定义 OK → 预处理（缺失/编码/缩放） → AI 设置（LLM 后端、开关） → 完成确认。
- 跨页状态共享：核心对象存于 Session State（processed_df、analysis_results、training_results 等），确保不同页面查看与训练使用同一份数据与分析结果；向导产出的配置/结果也写入 Session State 供后续页面直接复用。
- 页面组成（与代码对应）：
	- 配置向导（app_pages/configuration.py）
	- 数据总览（app_pages/data_overview.py）
	- 数据分析（app_pages/data_analysis.py）
	- 高级分析（app_pages/advanced_analysis.py）
	- 模型训练（app_pages/model_training.py）
	- AI Agent Chat（app_pages/ai_agent.py）

## 4. 功能实现
### 4.1 数据处理
- 输入：CSV（第一行为表头），需包含可选的时间列与一个标签列；用户在 UI 选择“标签列 + OK 取值集合”，其余值自动标记为 KO。
- 标签生成：构造 `OK_KO_Label`（二分类），生成后删除原标签列以防泄漏；自动对标签做 Label Encoding。
- 缺失值：策略可选 mean/median/mode/drop/forward_fill，默认 mean；支持对数值/分类分别应用。
- 缩放：可选 Standard/MinMax，只有勾选时才对数值列缩放；类别列保持不变。
- 列过滤：时间索引列（如 `time`, `timestamp`）在建模与 AutoGluon 中会被过滤；可配置保留其他特征。
- 最小示例：`input.csv(time, sensor1, sensor2, label)` → 处理后 `processed_df` 包含 `{sensor1, sensor2, OK_KO_Label}`（必要时附加编码列）。

### 4.2 统计分析
- 输入：processed_df（含 OK_KO_Label），数值/分类分开处理；支持自定义 alpha=0.05（默认）。
- 检验选择：数值用 T/Mann-Whitney/KS，分类用卡方；自动根据列类型分派。
- 输出字段：p_value、effect_size（Cohen's d 或 Cramér's V）、difference_ratio（均值相对差）、score（用于排序）；以降序生成 feature_ranking。
- 异常处理：无法计算的列（常量、缺失过多）会跳过并给出提示。

### 4.3 特征重要性（AutoGluon）
- 过滤：去除时间索引列、标签列，仅保留可训练特征。
- 训练：使用 TabularPredictor 分类固定预设（UI 不暴露超参），支持 GPU/CPU 自适应；可在代码中扩展 exclude/keep 列。
- 输出：feature importance 排名、leaderboard、best_model、训练耗时；可用于后续 top-N 选择。

### 4.4 模型训练
- 输入：processed_df + feature ranking（来自统计分析或 AutoGluon）。
- 特征子集：按多组 top-N 交叉（示例：5/10/20/50，可在代码调整）；每组与多模型组合训练。
- 模型集：Logistic、SVM(RBF)、Decision Tree、Random Forest；可在 model_trainer 扩展。
- 输出：性能表（Accuracy/F1/Recall/ROC-AUC）、最佳模型记录、特征数 vs 指标曲线、模型对比条图，可扩展 ROC/混淆矩阵绘制。

### 4.5 可视化
- plot_time_series：自动探测时间列（若不存在用索引），支持多列叠加。
- plot_frequency_spectrum：FFT 取正频段，提取主导频率 top-k（默认 5）。
- plot_distribution_comparison：直方/KDE/箱/小提琴对比 OK/KO，自动选择合适的分箱数。
- plot_feature_comparison：散点/箱/小提琴多特征对比，可传入特征列表。
- plot_correlation_heatmap：皮尔逊相关热力图，支持相关系数阈值高亮。

## 5. 实验评估（建议撰写）
- 数据集描述：样本量、特征、OK/KO 分布、时间轴字段。
- 统计检验结果：p 值、效应量、Top-N 判别特征。
- AutoGluon 结果：leaderboard、特征重要性、训练耗时。
- 传统模型对比：Accuracy/F1/Recall/ROC-AUC；特征数对性能曲线；混淆矩阵。
- 可视化示例：时序、FFT、分布对比、相关热图、特征对比。

## 6. 结果与讨论（建议撰写）
- 关键发现：最具判别力传感器/特征；时域 vs 频域的贡献。
- 模型对比：AutoGluon 与传统模型的优势/劣势。
- 实际意义与业务解读。

## 7. 局限性
- 数据依赖：需足够 OK/KO 样本且质量良好。
- 泛化：跨数据集迁移待验证。
- 资源：AutoGluon 训练耗时；本地 LLM 需内存/磁盘。
- LLM 解析：复杂查询可能需更强模型。

## 8. 未来改进
- 支持更多格式（Parquet/Excel/JSON）；多分类与异常检测。
- 深度学习时序模型；增量/在线学习；分布式训练。
- 报告自动生成（PDF/HTML）；多语言 UI；协作/会话持久化。
- 更智能的意图解析与主动分析建议。

