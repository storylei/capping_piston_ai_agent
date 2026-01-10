# 分析方法与页面逻辑说明（Statistical & ML Feature Importance Guide)

本指南系统说明项目中的统计分析与机器学习特征重要性方法、关键指标含义、特征排名计算逻辑，以及高级分析页面的交互与状态管理。

## 1. 总览

- 统计分析器：见 [src/analysis/statistical_analyzer.py](../src/analysis/statistical_analyzer.py)
  - 单列逐一分析 OK/KO 两组差异，输出基础统计、显著性检验、效应量。
- ML 特征重要性：见 [src/analysis/feature_importance.py](../src/analysis/feature_importance.py)
  - 基于 AutoGluon 训练分类模型，使用置换法计算特征重要性并生成排行榜与模型榜单。
- 高级分析页面：见 [src/app/app_pages/advanced_analysis.py](../src/app/app_pages/advanced_analysis.py)
  - 按用户选择（统计/ML/组合）执行分析，并清空旧的 `analysis_results` 后再运行。

会话状态对象在 [src/app/main.py](../src/app/main.py) 初始化：
- `st.session_state.analysis_engine`: `StatisticalAnalyzer`
- `st.session_state.ml_analyzer`: `FeatureImportanceAnalyzer`

## 2. 数值特征分析（_analyze_numerical_features）

针对每个数值列（逐列循环），对 OK/KO 两组分别计算以下基础统计（函数 `_calculate_basic_stats()`）：
- **count**：样本数量（非空值个数）
- **mean**：均值（平均数）
- **median**：中位数（50% 分位）
- **mode**：众数（出现次数最多的值；若有多个，取第一个）
- **std**：标准差，衡量波动/离散程度
- **var**：方差，标准差的平方
- **min / max**：最小值 / 最大值
- **q1**：第一四分位数（25% 分位）
- **q3**：第三四分位数（75% 分位）
- **skewness**：偏度，衡量分布左右偏斜（>0 右偏，<0 左偏，≈0 接近对称）
- **kurtosis**：峰度，衡量分布尾部/尖峭程度（>0 比正态尖、厚尾；<0 比正态平、薄尾）

示例（偏度/峰度直觉）：
- 右偏（skewness > 0）：大部分值在较小端，少数极大值拖右尾（如收入分布）
- 左偏（skewness < 0）：大部分值在较大端，少数极小值拖左尾（如高分为主的考试分数）
- 峰度高（>0）：中间尖，两端厚尾（如收益率常有暴涨暴跌）
- 峰度低（<0）：中间平，两端薄尾（如接近均匀分布）

差异比例（difference_ratio）用于规范化均值差：

$$difference\_ratio = \frac{|\mu_{OK} - \mu_{KO}|}{\max(|\mu_{OK}|, |\mu_{KO}|, 10^{-8})}$$

## 3. 显著性检验（_perform_statistical_tests）

对每个数值列进行以下三种检验，均以 `p_value < 0.05` 作为显著阈值：
- **T 检验（ttest_ind）**：假设正态分布，检验均值是否不同。
- **Mann-Whitney U（非参数）**：基于秩，稳健检验两组分布位置差异（不要求正态）。
- **Kolmogorov-Smirnov（KS）**：非参数，比较两组累积分布的最大距离（形状整体差异）。

容错：各检验均在 `try-except` 中，失败时返回 `nan` 与 `significant=False`，避免中断整体分析。

## 4. 效应量（Cohen's d）

用于衡量两组均值差异的标准化大小（不依赖样本量）：

$$d = \frac{\bar{x}_{OK} - \bar{x}_{KO}}{s_{pooled}}$$

其中合并标准差：

$$s_{pooled} = \sqrt{\frac{(n_{OK}-1)\sigma_{OK}^{2} + (n_{KO}-1)\sigma_{KO}^{2}}{n_{OK}+n_{KO}-2}}$$

参考量级：`0.2` 小效应、`0.5` 中效应、`0.8` 大效应、`>1.2` 极大效应。

为何需要效应量：p 值在大样本下容易“显著”，但实际差可能很小；Cohen's d 反映工程上的**实际重要性**。项目中以 |d| 与 p 值共同决定特征排名。

## 5. 分类特征分析（_analyze_categorical_features）

逐列计算：
- **分布**：`value_counts(normalize=True)` 得到各类别比例（OK/KO 各自）
- **列联表**：拼接 OK/KO 后用 `pd.crosstab` 构建（行为组、列为类别）
- **卡方检验**：`stats.chi2_contingency(table)`，返回 `chi2_stat`、`p_value`、`dof`、`expected`，判断两组分布是否显著不同
- **Cramér's V**（效应量）：

$$V = \sqrt{\frac{\chi^2}{n\cdot (k-1)}}\quad \text{其中 } k = \min(\text{行数}, \text{列数})$$

V 取值含义：`0.0` 无关联，`0.1~0.3` 弱，`0.3~0.5` 中，`>0.5` 强。

## 6. 特征排名（_rank_features_by_significance）

统一以“显著性 × 效应量 × 差异程度”的思想对两个类型进行评分，并按降序排序：

- **数值特征**：

$$composite\_score = \big(-\log_{10}(p_{MW})\big) \times |d| \times difference\_ratio$$

其中 $p_{MW}$ 为 Mann-Whitney U 的 p 值（非参数更稳健）。

- **分类特征**：

$$composite\_score = \big(-\log_{10}(p_{\chi^2})\big) \times V$$

最终输出 `feature_ranking` 列表，供页面展示与后续模型使用。

## 7. 高级分析页面逻辑

页面文件：见 [src/app/app_pages/advanced_analysis.py](../src/app/app_pages/advanced_analysis.py)

- “运行分析”按钮点击时会**先清空**旧的 `st.session_state.analysis_results`，确保不残留上次结果。
- 根据用户选择的 `analysis_types`：
  - 仅“Statistical Tests”：运行统计分析
  - 仅“Machine Learning Feature Importance”：运行 ML 特征重要性
  - “Combined Analysis”：两者都运行；若统计无 `feature_ranking`，以 ML 排名作为回退
- 结果展示：按选择动态生成标签页（组合/统计/ML），对应表格与条形图可视化。

## 8. 会话状态对象

- 初始化位置：[src/app/main.py](../src/app/main.py)
- 对象列表：
  - `st.session_state.analysis_engine`：统计分析器 `StatisticalAnalyzer`
  - `st.session_state.ml_analyzer`：ML 特征重要性分析器 `FeatureImportanceAnalyzer`
  - 其它状态（如数据、配置进度、AI Agent）参见 [docs/SESSION_STATE.md](./SESSION_STATE.md)

## 9. 快速使用示例

```python
# 统计分析
results_stat = st.session_state.analysis_engine.analyze_all_features(processed_df)

# ML 特征重要性
results_ml = st.session_state.ml_analyzer.analyze_feature_importance(
    df=processed_df,
    target_col='OK_KO_Label',
    time_limit=120,
    preset='medium_quality'
)

# 合并并设置回退排名
analysis_results = results_stat
analysis_results['ml_feature_importance'] = results_ml
if not analysis_results.get('feature_ranking'):
    fi = results_ml.get('feature_importance', {})
    ranking_ml = fi.get('feature_ranking')
    if ranking_ml:
        analysis_results['feature_ranking'] = ranking_ml

st.session_state['analysis_results'] = analysis_results
```

## 10. 设计原则与实践建议

- 显著性（p 值）与效应量（Cohen's d / Cramér's V）**同时考虑**，避免“统计显著但工程不重要”。
- 非参数检验（Mann-Whitney、KS）在分布未知或含极端值时更稳健。
- 逐列分析与统一评分，方便跨类型特征的可比性与排序。
- 每次分析前清空旧结果，保持 UI 与状态的一致性。
