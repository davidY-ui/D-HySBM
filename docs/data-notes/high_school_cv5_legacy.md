> Historical data note, not an executable quick start for this release. The referenced datasets, old absolute paths, and some helper scripts are not supplied. Use the current root README and docs/DATA.md for the supported entry points.

# High School：5 折训练与测试文件

本目录由 `prepare_high_school_holdout.py` 生成。原始 `high_school.npz` 未修改。
当前只有数据划分和评估工具，尚未运行完整 BP–EM 拟合，也没有新的实证 AUC 结果。

## 1. 哪个文件用于训练，哪个用于测试？

每个 `fold_00` 至 `fold_04` 中有：

| 文件 | 用途 |
| --- | --- |
| `train.txt` | **传给现有训练入口**，一行一个正超边，节点 ID 以单个空格分隔 |
| `train.npz` | 同一训练集的 NumPy 备份，包含原始权重和完整 327 节点的 incidence |
| `test.npz` | **仅用于训练后评分**，内含测试正边、同阶负边及 `y_true` |
| `counts.json` | 本折各阶训练／测试数量 |

根目录的 `orders/*.txt` 用于指定要训练的阶数组合，例如 `orders/2_3_4.txt`。
`manifest.json` 记录原文件 SHA256、生成种子、采样规则与全部计数；
`split_assignment.npz` 记录每条原始超边属于哪一个测试折。

`train.npz` 的字段：`hyperedges`, `A`, `B`, `N`, `edge_sizes`,
`original_indices`, `selected_hye_sizes`, `fold`。
`A` 保留原始接触次数，但当前二值 HySBM 训练只使用超边是否存在。
不要把一条重复接触的节点集合拆成多条事件后分别放入训练和测试。

`test.npz` 的字段：`hyperedges`, `y_true`, `edge_sizes`, `N`, `fold`,
`original_indices`, `observed_weights`。其中 `y_true=1` 是留出的真实超边，
`y_true=0` 是负边；负边的 `original_indices=-1`，权重为零。
在 `.npz` 中读到对象类型超边列表时需要 `allow_pickle=True`，仅加载可信文件。

## 2. “多次拟合”的两种含义

1. **同一折的不同初始化**：使用同一份 `train.txt` 和同一份 `test.npz`，
   只改变推断随机种子，例如 `101, 202, 303`。这检查优化初值的影响。
2. **不同折**：切换到另一个 `fold_XX` 的训练、测试文件，并从头拟合。
   每折约 80% 的正边用于训练、20% 用于测试。这检查数据划分的影响。

推荐先做 **5 折 × 3 次初始化 = 每个模型配置 15 次拟合**。
同一折下不同模型、不同阶数组合也共用这些文件；不要为某个模型重新抽负边。
这里是 5 折交叉验证，不是 5 份独立随机 80/20 划分：测试正边互不重叠，
每条真实边恰好测试一次；训练集相互重叠，不能把 5 折当成 5 个独立网络。

如果要选一个初始化作为该折的模型，应按**训练 free energy**选择，不能按测试 AUC 选择。
若保留每个初始化的 AUC，先在折内汇总或按事先确定的训练准则选模型，再跨折报告；
不要把同一测试折的 3 个初始化当作 3 份独立测试数据。

## 3. 训练命令（order-specific affinity）

以下命令均在 `D:\loop_SBM\cite\hypergraph-message-passing` 目录执行。
例子：第 0 折、阶数 `{2,3,4}`、初始化种子 101。

```powershell
python main_message_passing_loop_realnetwork.py --hye_file data/processed_input/high_school_cv5_seed20260916/fold_00/train.txt --N 327 --K 10 --selected_hye_sizes data/processed_input/high_school_cv5_seed20260916/orders/2_3_4.txt --seed 101 --train_rounds 1 --em_iter 20 --em_thresh 1e-5 --mp_iter 2000 --mp_thresh 1e-5 --mp_patience 50 --dropout 0 --dropout_vertex 0 --dropout_vertex_FVS 0 --dropout_vertex_rest 0 --dropout_FVS 0 --n_jobs 1 --save_dir data/processed_input/high_school_cv5_seed20260916/fold_00/results/orders_2_3_4/seed_101
```

重要事项：

- **删除原命令中的 `--real_dataset high_school`**，否则会读取完整原始网络或触发输入互斥错误。
- `--hye_file` 目前支持 `.txt`/`.pkl`，**不能直接传 `train.npz`**。
- 不传 `--node_assignments`，这里评价超边预测，不需要真实班级标签。
- 保留 `--N 327`，不因训练折出现孤立节点而删点或重编号。
- 这里固定 `K=10`。若需要调 K 或选最优阶数组合，应在训练折内部另划验证集，
  不要根据本目录的外层测试 AUC 反复调参。
- `--train_rounds 1` 用于逐初始化保存；种子 202、303 使用不同的 `--save_dir`，
  例如 `seed_202`、`seed_303`，避免结果被覆盖。
- 也可以 `--train_rounds 3 --seed 101` 一次运行 3 个初始化，但现有入口只保存最低
  训练 free energy 的模型；其内部种子是 `101, 1125, 2149`，不是 `101,202,303`。
- 不要直接复用此前在完整网络上拟合得到的 `p`、`q`、初始化解或 best-K 结果。

第 1 折把输入和输出路径中的 `fold_00` 一起改成 `fold_01`，依此类推。
修改阶数组合时，只改 `--selected_hye_sizes` 与结果目录；不要重做划分。

## 4. 训练后测试

```powershell
python evaluate_high_school_holdout.py --fit data/processed_input/high_school_cv5_seed20260916/fold_00/results/orders_2_3_4/seed_101/inferred_params.npz --test data/processed_input/high_school_cv5_seed20260916/fold_00/test.npz --orders 2 3 4 --output data/processed_input/high_school_cv5_seed20260916/fold_00/results/orders_2_3_4/seed_101/heldout_metrics.json
```

评估工具只读拟合结果，不再训练；输出逐阶 ROC-AUC、macro AUC、按测试正边数加权的 AUC。
它使用正确的升序平均秩处理并列分数，不调用旧 notebook 中排序方向错误的函数。
加权 AUC 不是 pooled/micro AUC。输出文件已存在时拒绝覆盖，可省略 `--output` 只查看结果。

如果比较 `{2}`、`{2,3}`、`{2,3,4}`、`{2,3,4,5}` 对二阶预测的贡献，
所有评估命令都用 **`--orders 2`**，且使用相同测试文件。
不能将 `{2}` 上的 AUC 和 `{2,3,4}` 上的加权 AUC 直接比较并称为阶数选择。
只评价拟合结果中存在 affinity 的阶；评估工具不会凭空补出未训练阶的矩阵。

第五阶总共只有 7 条正边，每个测试折仅 1–2 条，不作为主要模型选择证据。
当前工具报告 ROC-AUC；PR 指标和其他评价如需使用，可另行增加。

## 5. 数据计数和检查

| 折 | 训练正边 | 测试正边 | 测试负边 |
| --- | ---: | ---: | ---: |
| 00 | 6252 | 1566 | 1566 |
| 01 | 6253 | 1565 | 1565 |
| 02 | 6255 | 1563 | 1563 |
| 03 | 6256 | 1562 | 1562 |
| 04 | 6256 | 1562 | 1562 |

每阶均单独打乱再分折，整数取整导致每折比例略有不同。
负边按阶与测试正边等量，排除了**完整原始正边集合**，每折内部没有重复负边。
所有节点保留原始 ID `0..326`。未使用班级标签作划分或推断。
已检查训练／测试正边互斥、测试正边跨折互斥且覆盖全部 7818 条边、负边不与任何原始正边重合。

重现同一划分时，请对新输出目录运行：

```powershell
python prepare_high_school_holdout.py --seed 20260916 --folds 5 --output data/processed_input/high_school_cv5_reproduced
```

生成器拒绝覆盖非空目录。固定种子对应同一套划分，不要为每个 BP 初始化重新运行生成器。
