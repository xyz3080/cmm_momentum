# 深度学习加权动量因子

本项目复现并改造论文 **All Days Are Not Created Equal: Understanding Momentum by Learning to Weight Past Returns** 的核心方法，在 A 股市场构建深度学习加权动量因子（CMM），并完成传统动量对照、中性化、IC、分组组合、全截面多空组合及风格暴露检验。

## 目录结构

```text
repository/
  code/backtest/              通用月频信号回测框架
  data/                       共用原始行情与财务数据，不进入 Git
  docs/                       共用论文、研报和研究文档
  cmm_momentum/
    config.yaml               数据、模型和回测参数
    requirements.txt          Python 依赖
    notebooks/
      02_explain_cmm_improvement.ipynb
    src/
      data_clean_pipeline.py  point-in-time 数据清洗与特征构造
      train_cmm_model.py      expanding-window 模型训练
      model_compare_workflow.py 统一因子回测与绩效输出
      style_exposure_workflow.py CMM 中性化因子的风格暴露
      validate_outputs.py     产物、防泄露与版本一致性检查
      run_pipeline.py         流水线统一入口
    tests/                    核心逻辑与回测边界测试
    result/                   模型、数据集和研究结果
```

`code/backtest/` 当前实现“月频信号、下一交易日收盘调仓、日频盯市与绩效计算”。数据适配、权重构造、执行引擎、指标和绘图相互独立，未来增加周频或日频调仓时可以复用执行与指标层。

## 环境与数据

```bash
cd cmm_momentum
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

默认数据位置：

```text
data/daily/                         按交易日存放的 A 股日行情 CSV
data/financial/A_stock_financial.feather  财务数据
```

财务数据必须包含实际公告日。所有财务特征都按 `public_date <= signal_date` 做 point-in-time 对齐，不能用报告期直接代替可得日。

## 运行流程

完整运行：

```bash
python src/run_pipeline.py
```

按需运行部分阶段：

```bash
python src/run_pipeline.py --stages data model backtest style validate
python src/run_pipeline.py --stages backtest style validate
python src/run_pipeline.py --dry-run
```

各阶段依次完成：

1. `data`：构造股票-月份训练样本、231 日收益窗口和 87 个模型特征。
2. `model`：执行 expanding-window 训练，只保存互不重叠的样本外测试预测。
3. `backtest`：统一回测 CMM、传统动量及其中性化版本。
4. `style`：估计市值行业中性化 CMM 的月度风格暴露。
5. `explain`：执行 CMM 改进机制解释 notebook。
6. `validate`：检查文件完整性、特征版本、时间对齐和预测唯一性。

轻量验证：

```bash
pytest -q
python src/validate_outputs.py
```

## 数据与特征

每行训练数据对应一个股票-信号月：

- `ret_lag_252` 至 `ret_lag_22`：过去第 252 至第 22 个交易日的 231 个日对数收益，跳过最近约一个月。
- 20 个量价特征：不同期限历史收益、波动率、换手率、成交额及期限结构。
- 38 个非比值 point-in-time 财务特征。
- 28 个相对基本面残差特征：对财务变量做有符号对数变换后，在当月截面回归并保留残差。
- 1 个公告时距特征：`signal_date - financial_public_date` 的日历天数。
- `target_1m_ret_cs_z`：下一持有期收益的月度截面标准化值，是模型标签。

所有入模特征在每个信号月截面内依次进行极端值处理、缺失值中位数填充和标准化。原始基本面比值不直接入模；有保留价值的信息使用相对基本面残差表达。清洗结果以 `float32`、Zstandard 压缩和 50,000 行 row group 写入 Parquet。

## 模型

神经网络将当月公司特征映射为标量 `z_hat`。对股票 `i` 的历史日收益 `r(i,t-d)`：

```text
score(i,t-d) = z_hat(i,t) * r(i,t-d)
weight(i,t-d) = softmax_d(score(i,t-d))
CMM(i,t) = sum_d weight(i,t-d) * r(i,t-d)
```

网络为三层前馈结构 `128 -> 64 -> 32 -> 1`，隐藏层使用 BatchNorm、GELU 和 Dropout。训练损失是 CMM 月度截面标准化值与下一期截面标准化收益之间的均方误差。时间切分采用 expanding window：训练集持续扩张，验证段只用于早停和模型选择，测试块只生成一次真正样本外预测。

## 回测口径

- 股票池：当月具有因子、目标收益、交易日和退出日信息的样本。
- 调仓：月末生成信号，下一交易日收盘成交；成交后的下一根日线开始计收益。
- 涨跌停：涨停不可买、跌停不可卖；缺失交易状态按不可交易处理。
- 缺失持仓收益：当日按零收益处理，同时输出缺失数量、持仓观测总数和缺失率。
- 十分组：按因子从低到高分为 D1-D10，组内等权。
- 多空组合：对全截面因子去均值后，按绝对值和归一化为总杠杆 1；多头、空头是该组合内部两侧，不是 D10 和 D1。
- 交易成本与单票权重上限由 `config.yaml` 控制，当前默认均不启用。
- IC：每月计算 Pearson IC 和 Spearman Rank IC，最少 50 个有效样本。
- 绩效：使用日收益计算 CAGR、算术年化收益、年化波动、Sharpe 和最大回撤。两种年化收益分列，避免口径混淆。

因子极端值处理属于研究信号预处理，不内置在通用回测引擎中。目前仅在 CMM 信号生成阶段显式执行月度 1%/99% 截面缩尾，再进行市值行业或风格中性化，避免缩尾重新引入已剔除的暴露。

## 主要产物

- `result/datasets/cmm_model_training_data.parquet`：股票-月份训练数据。
- `result/models/cmm/cmm_predictions.parquet`：互不重叠的样本外测试预测。
- `result/reports/model_compare/performance_metrics_test.csv`：各因子全截面多空绩效。
- `result/reports/model_compare/portfolio_leg_performance_test.csv`：多头、空头和多空组合绩效。
- `result/reports/model_compare/factor_ic_summary_test.csv`：IC、Rank IC、ICIR 和 t 值。
- `result/reports/model_compare/annual_performance_test.csv`：年度组合绩效。
- `result/reports/model_compare/decile_nav_*_test.png`：十分组与多空累计净值。
- `result/reports/style_exposure/`：市值行业中性化 CMM 的具体风格暴露。

`result/datasets/`、原始数据和股票级预测明细不纳入 Git；可共享的模型摘要、表格和图片保留在仓库中。
