# yinhua 研究工作区

这个目录作为多个研究项目的总工作区使用。

```text
data/                 多项目共享的原始行情、财务和其他数据
docs/                 多项目共享的论文、计划、研报和参考资料
code/backtest/        多项目复用的因子回测框架
cmm_momentum/         深度学习加权动量因子项目
```

后续新项目继续在根目录下建立独立项目文件夹，每个项目内部保留自己的 `notebooks/`、`src/`、`result/`、测试和 `README.md`。共享原始数据和参考资料放在 `data/`、`docs/`，可复用且经过测试的通用代码放在 `code/`。

`data/` 和 `docs/` 包含大体积或受授权约束的内容，不纳入 Git。当前可复现研究入口和环境说明见 [`cmm_momentum/README.md`](cmm_momentum/README.md)。
