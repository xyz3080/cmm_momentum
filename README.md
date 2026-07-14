# 中低频因子开发

面向股票中低频因子的数据处理、模型开发、统一回测与研究结果管理仓库。当前核心项目是深度学习加权动量因子，并沉淀可复用的因子回测框架。

```text
code/backtest/        通用因子回测框架
cmm_momentum/         深度学习加权动量因子研究
```

研究项目内部保留各自的 `notebooks/`、`src/`、`result/`、测试和方法说明；经过测试且可跨项目复用的组件放在 `code/`。

原始数据、授权资料和本地研究文档不纳入 Git。当前研究的复现入口、数据要求和方法细节见 [`cmm_momentum/README.md`](cmm_momentum/README.md)。
