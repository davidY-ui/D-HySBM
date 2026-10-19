> Historical data note, not an executable quick start for this release. The referenced datasets, old absolute paths, and some helper scripts are not supplied. Use the current root README and docs/DATA.md for the supported entry points.

# Primary School：与 High School 一致的五折数据

共 242 个节点、12704 条唯一超边，来自 106879 次接触事件。
源 JSON 未修改。使用已有 primary_school_meta.csv 节点映射；所有折均保留 0..241 节点。
本目录只有数据与生成脚本，没有训练模型或产生 AUC 结果。

## 文件结构

- primary_school.npz：完整静态网络，字段与 high_school.npz 相同：hyperedges、A、B。
  A 为接触事件次数，B 在超边端点处填 A。
- primary_school_meta.csv：原 metadata 文件的原样副本，可用于事后 NMI 评价。
- fold_00 至 fold_04：每折含 primary_school_train0.npz / primary_school_test0.npz
  （编号随折变化）、train.txt、counts.json。
- orders：与 high_school_cv5 一样的八种保留阶集合文件，每行一个阶数。
- split_assignment.npz：原始唯一超边索引、测试折号、阶数与 N。
- manifest.json：源数据哈希、节点政策、采样种子、计数与处理说明。

train NPZ 字段、dtype 与 High School 对应文件一致：hyperedges、A、B、N、
edge_sizes、original_indices、selected_hye_sizes、fold。
每折的训练 B 是二值 uint8 关联矩阵，与 High School 五折文件一致。
test NPZ 字段：hyperedges、y_true、edge_sizes、N、fold、original_indices、observed_weights。
负边的 y_true=0、original_indices=-1、observed_weights=0。
NPZ 超边为一维 object 数组，加载时使用 allow_pickle=True，仅加载可信文件。

## 划分规则

沿用 High School 的种子 20260916 与 NumPy SeedSequence 规则，按阶数分别打乱并分成五份。
同一节点集合的重复接触先合并为权重，不能跨训练和测试拆开。
每条真实超边恰好测试一次，各折约 80% 训练、20% 测试。
每折、每阶按 1:1 抽取测试负边，负边排除完整观测网络，折内不重复。
不同模型或保留阶组合共用同一份测试样本。
数据为静态聚合，时间戳仍保留在源 JSON 中。

| 折 | 训练正边 | 测试正边 | 测试负边 |
| --- | ---: | ---: | ---: |
| 00 | 10162 | 2542 | 2542 |
| 01 | 10162 | 2542 | 2542 |
| 02 | 10163 | 2541 | 2541 |
| 03 | 10164 | 2540 | 2540 |
| 04 | 10165 | 2539 | 2539 |

## 使用

训练使用 fold_XX/train.txt，并指定 --N 242；不把测试文件传给训练入口。
评估读取对应 primary_school_testX.npz。
现有 High School 专用脚本若写死 high_school_train/test 文件名，需改成 primary_school_train/test；
本目录保留相同内容 schema，但不伪装为 High School 数据。
类别标签仅用于事后分析，没有参与划分或负采样。

生成器拒绝覆盖非空目录；重现时将 --output 指向新目录。
