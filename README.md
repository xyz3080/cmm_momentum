# 深度学习加权动量因子项目

本项目复现并改造论文 **All Days Are Not Created Equal: Understanding Momentum by Learning to Weight Past Returns** 的核心思路：用公司特征学习过去日收益的加权方式，构造深度学习加权动量因子 CMM，并与传统等权动量因子比较。

## 推荐运行顺序

1. `python src/data_clean_pipeline.py`
   - 读取日行情和财务数据。
   - 构造 `t-252` 到 `t-22` 的 231 个日收益窗口。
   - 按 `public_date <= signal_date` 生成 point-in-time 财务特征。
   - 输出模型训练数据到 `output/datasets/`。

2. `python src/train_cmm_model.py`
   - 训练论文式 CMM 模型。
   - 保存模型和预测信号到 `output/models/cmm/`。

3. `notebooks/03_compare_momentum.ipynb`
   - 比较 CMM 和传统动量因子。
   - 输出十分组累计净值图和绩效表到 `output/reports/model_compare/`。

4. `notebooks/04_explain_cmm_improvement.ipynb`
   - 检验 CMM 相对传统动量改进的来源。

5. `notebooks/05_barra_cmm_attribution.ipynb`
   - 参考 Barra 风格模型做收益分解。

6. `python src/validate_outputs.py`
   - 检查关键产物是否存在，并验证时间对齐、防泄露和测试集预测唯一性。

## 目录结构

```text
config.yaml

notebooks/
  02_train_cmm_model.ipynb
  03_compare_momentum.ipynb
  04_explain_cmm_improvement.ipynb
  05_barra_cmm_attribution.ipynb

src/
  data_clean_pipeline.py 数据清洗主逻辑
  train_cmm_model.py    模型训练脚本入口
  backtest.py           涨跌停、调仓、十分组和绩效工具
  validate_outputs.py   结果完整性和防泄露检查

output/
  datasets/              清洗后的训练数据与列清单
  models/cmm/            模型权重、训练历史、预测结果
  reports/model_compare/ 因子对比图表和绩效表
  reports/cmm_explain/   CMM 改进机制检验
  reports/barra_attribution/ Barra 风格收益分解
```

共享原始数据和参考资料位于 yinhua 根目录：

```text
../data/
  daily/                 原始日行情 CSV，按交易日存放
  financial/             原始财务 Feather 数据

../docs/
  paper/                 论文原文
  plan/                  项目计划文档
  reports/               研报和参考材料
```

## 关键产物

- `output/datasets/cmm_model_training_data.parquet`
  - 每行是一个股票-月份样本。
  - `ret_lag_252` 到 `ret_lag_22` 是过去 231 个交易日对数收益。
  - `z_` 开头列是截面标准化后的特征。
  - `target_1m_ret_cs_z` 是训练标签。

- `output/models/cmm/cmm_model.pt`
  - 训练好的 PyTorch 模型。

- `output/models/cmm/cmm_predictions.parquet`
  - 每个股票-月份的 CMM 预测信号。

- `output/reports/model_compare/performance_metrics_test.csv`
  - CMM 与传统动量的测试集多空绩效对比。

## 方法说明

CMM 模型不直接用神经网络预测收益。神经网络只把公司特征 `z_i,t` 映射成一个标量 `z_hat_i,t`，然后用它生成过去 231 个日收益的 softmax 权重：

```text
score_i,t-d = z_hat_i,t * r_i,t-d
w_i,t-d = softmax(score_i,t-d)
CMM_i,t = sum_d w_i,t-d * r_i,t-d
```

训练目标是让 `CMM_i,t` 的月度截面标准化值拟合下个月截面标准化收益。

## 注意事项

- 原始财务数据必须按 `public_date` 做 point-in-time 对齐，不能直接按 `report_date` 使用。
- 当前回测比较是研究验证版，默认等权十分组；正式投资组合还应进一步处理交易成本、容量和组合约束。
- 数据文件较大，`output/datasets/cmm_model_training_data.parquet` 约 1.2G。
- `output/` 只保留当前代码能够重新生成或验证的核心产物；阶段性探索结果不再混放在主输出目录。
