# FeatureImportanceAnalyzer 模块说明（AutoGluon 版本）

本文档详细说明 [src/analysis/feature_importance.py](../src/analysis/feature_importance.py) 中 `FeatureImportanceAnalyzer` 的职责、数据流、方法与使用方式。

## 模块概览
- 目标：使用 AutoGluon 对二分类数据（OK/KO）进行自动训练，并计算特征重要性、生成模型榜单与最佳模型信息。
- 依赖：`autogluon.tabular`（未安装将抛出 ImportError）、`pandas`、`numpy`。
- 标签列：默认 `OK_KO_Label`，必须存在于输入 DataFrame 中。

## 输入与预处理
- 输入：预处理后的 `DataFrame`，包含标签列与特征列。
- 列排除：
  - 来自 `df.attrs['exclude_from_ml']` 的排除列表（可在预处理阶段注入）。
  - 常见时间/索引列模式：`time_cycles`、`time_cycle`、`cycle`、`timestamp`、`index`。
- 处理方式：训练前复制为 `df_ml`，并按上述规则删除不适合用于 ML 的列（保留标签列）。

## 训练与重要性计算流程
方法：`analyze_feature_importance(df, target_col='OK_KO_Label', time_limit=120, preset='medium_quality', save_path=None)`
- 训练参数：
  - `time_limit` 控制训练时间（秒）。
  - `preset` 控制质量与耗时的取舍（如 `best_quality`、`high_quality`、`medium_quality`、`good_quality`、`fast_training`）。
  - `eval_metric='accuracy'`。
  - `save_path`：模型目录，默认 `autogluon_models_temp`（也会写入 `self.model_path`）。
- 特征重要性：调用 `predictor.feature_importance(df_ml)` 返回包含 `importance` 列的 DataFrame；转换为：
  - `importance_scores`（DataFrame 的字典形式）。
  - `feature_ranking`（列表：`feature`、`importance`、`rank`）。
- 模型榜单：`predictor.leaderboard(df_ml)` 返回各模型表现，保存为记录列表。
- 最佳模型：取榜单第一名的关键指标（`model`、`score_val`、可能的 `score_test`、`pred_time_val`、`fit_time`）。
- 结果缓存：写入 `self.feature_importances`，结构见下。

## 结果结构
`self.feature_importances` 典型结构：
- `feature_names`：参与训练的特征列（不含标签列）。
- `data_shape`：训练数据的形状 `(rows, cols)`。
- `class_distribution`：标签列类别计数。
- `feature_importance`：
  - `importance_scores`：完整字典化结果。
  - `feature_ranking`：排序后的简明列表（`feature`、`importance`、`rank`）。
- `model_leaderboard`：榜单记录列表（每行一个模型）。
- `best_model`：第一名模型的名称与关键指标。
- `training_time`：训练耗时（秒）。

## 方法一览
- `analyze_feature_importance(...)`：训练 + 计算重要性 + 榜单 + 最佳模型，并缓存。
- `get_top_features(n=10)`：返回前 `n` 个特征（来自 `feature_ranking`）。
- `get_best_model()`：返回 `(model_name, score_val)`；若尚无结果返回 `(None, 0.0)`。
- `get_leaderboard()`：将缓存的榜单记录转为 `DataFrame`（无结果返空表）。
- `predict(df)`：使用已训练的 `predictor` 做预测（输入不应包含标签列）。
- `predict_proba(df)`：返回类别概率预测（同上要求）。
- `save_model(path=None)`：保存训练好的 `predictor`（若未指定 `path`，沿用 `self.model_path`）。
- `load_model(path)`：从目录加载 `predictor` 到内存并记录 `self.model_path`（当前代码中未被调用，但保留以备复用）。

## 注意事项
- 数据泄露风险：当前用同一 `df_ml` 进行训练与 `leaderboard` 评估，指标可能偏乐观；生产场景建议拆分验证/测试集或设置 `tuning_data`。
- 标签列校验：若缺少 `target_col` 或数据为空会抛错；调用前确保预处理完成。
- 列排除一致性：排除逻辑依赖 `df.attrs['exclude_from_ml']` 与时间/索引模式；确保预处理环节正确设置。
- AutoGluon 安装：未安装会抛 ImportError；需在环境中安装并匹配版本。
- 模型目录：默认 `autogluon_models_temp`（在 `.gitignore` 中忽略）；不需要持久化时可以删除，训练时会自动重新创建。

## 快速使用示例
```python
from src.analysis.feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()
results = analyzer.analyze_feature_importance(df, target_col='OK_KO_Label', time_limit=120)

top10 = analyzer.get_top_features(10)
best_model_name, best_score = analyzer.get_best_model()
leaderboard_df = analyzer.get_leaderboard()

# 可选：保存/加载模型
analyzer.save_model('autogluon_models_temp')
analyzer.load_model('autogluon_models_temp')
# 加载后如需刷新重要性与榜单，可再次针对给定数据调用评估（当前实现不含“加载后直接刷新”的便捷方法，可按需扩展）。
```

## 清理模型目录
如不需要复用模型并想释放空间：
```bash
rm -rf autogluon_models_temp
```

## UI 集成建议
- 提供“训练并计算重要性”按钮（触发 `analyze_feature_importance`）。
- 可选提供“清理模型目录”按钮以释放空间。
- 若需要跨会话复用，增加“载入已有模型目录”的入口并在载入后提供重新评估选项。