# Notebook 绘图验证报告

五本均在独立发布副本、全新真实 Jupyter 内核中逐单元执行；禁止读取原 D_HSBM/loop_SBM 目录。

| Notebook | 代码单元 | 新生成 PNG 文件 | 结果 |
| --- | ---: | ---: | --- |
| `BP_growth_validation_multigeneration.ipynb` | 15/15 | 4 | 通过 |
| `main_plot.ipynb` | 11/11 | 5 | 通过 |
| `main_plot_DCHSBM.ipynb` | 9/9 | 5 | 通过 |
| `realdata_plot.ipynb` | 31/31 | 43 | 通过 |
| `transition_detectivity.ipynb` | 8/8 | 4 | 通过 |

合计 74/74 个代码单元通过，61 份 PNG 实际生成；文件数包含各阶数组合的矩阵与软后验图，不代表 61 个独立实验。PNG 已目视检查；保存的 PDF/SVG 文件已记录并核对文件哈希。

验证环境：Python 3.10.19、NumPy 1.23.1、SciPy 1.8.1、Matplotlib 3.10.8、pandas 2.3.3。依赖记录见 requirements-plotting.txt。运行工具的真实内核回归测试 3/3 通过。

## 修复内容

- 补齐绘图所需的原始小汇总、保存拟合、测试折、轨迹与来源哈希；不打包整个约 51.2 GiB 数据树。
- 将输入和输出配置改为仓库内相对路径，修复 DCHSBM 的错漏路径。
- 实证本关闭可选诊断导出时，请求显示的混淆矩阵现在按需生成，避免缺图报错。
- 原科学计算/评分/绘图函数保持原数值逻辑；修改集中在路径、缓存加载及缺图生成。核心 Python/Rust 源码和构建文件逐字节保留。

## BP 样本量与验证边界

BP 默认读取经核验的真实测量缓存重画：单代普通组每组 100,000 roots、路径 A/critical/B 每组 30,000；这不是一次新的 200,000-path 实验。多代使用原 completed-main：M=20,000、L=12、A/B 每 case 10 次独立运行。图中和 Notebook 中均已明确标注，原重跑开关及算法保留。

本检查确认所有默认 Notebook 能从随包数值输入重画图；未重新训练模型、重跑大模拟或重新编译 Rust。可选主图 signal 诊断和汇总重建也另行验证，重建数组与原汇总逐值一致。

在项目根目录安装绘图环境后运行：

```sh
python -m pip install -r requirements-plotting.txt
python tools/verify_notebooks.py --work-dir ../D-HySBM-notebook-check --timeout 600
```

工作目录须位于源码包外且为空。测试生成的执行版 Notebook、图像和详细 JSON 会保存在该目录。
