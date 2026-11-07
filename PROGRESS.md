# 进度对齐（更新于 2026-11-07，周六）

周一 11/9 是 baseline checkpoint。这份文档说明：做到哪了、定了哪些事、代码在哪、每个人接下来做什么。

## 1. checkpoint 三项要求的进度

| 教授的要求 | 状态 | 谁做的 / 谁负责 |
|---|---|---|
| (1) 特征 + 防偷看测试 | ✅ 完成 | Qidi 做，Jin 检查 |
| (2) 一个测试年里 LR 在三种验证方式下的结果 + 真实 AUC 的 bootstrap 区间 | ✅ 完成（2012 年） | 切分和 bootstrap 抽样：Qidi；LR 和评估：Jin |
| (3) 循环 12 年，画关键图第一版 | ⏳ 下一步 | 循环：Jin；关键图：Yixun（Jin 先做第一版，再交给 Yixun） |

## 2. 已经定下来的事（改动前请先在群里说）

| 事项 | 决定 | 原因 |
|---|---|---|
| 特征 | Kaggle 比赛文档的 27 个指标去掉 4 个重复项，**用 23 个** | momentum_10d/20d、roc_5/10 是 return 的精确倍数。论文里要写明，周一也要跟教授提一句（proposal 原文写的是 27 个） |
| 特征来源 | https://www.kaggle.com/competitions/stock-market-signal-predict-next-day-returns | 只借用特征定义，价格数据还是我们的 NASDAQ-100 CSV |
| 样本量 | 491,772 行，上涨占 50.58%（proposal 里写的是 494,075 / 50.4%，需要更新） | 多删了成交量为 0 和价格连续不变的天 |
| LR | 只用 L2；默认 C=1；调参网格 C ∈ {0.001, 0.01, 0.1, 1, 10} | 和 LightGBM 的预算相同，都是 5 组（教授要求） |
| LightGBM 网格 | (num_leaves, learning_rate, n_estimators) = (7,0.05,200), (15,0.05,200), (31,0.05,200), (15,0.1,100), (31,0.1,100) | 现在就定死（教授要求） |
| 随机种子 | **SEED = 0**，全项目统一 | ⚠️ Qidi 的 Colab 里现在是 5420，请改成 0，两边的分折结果才会一致 |
| 估计误差 | 估计值 = 5 折 AUC 的平均；误差 = 估计 − 真实 AUC；用固定的默认配置算，不用调参后的最优配置 | 写在 CLAUDE.md 的规则里 |
| 2024-01 到 2026-02 | 最后只用一次，之前一律不碰 | proposal |

## 3. 代码在哪（仓库：https://github.com/wjin2025/ese5420-finalpaper；每个文件谁写的，完整列表见 README.md）
Qidi 的 Colab 全部内容（Step 0–7）存档在 `notebooks/qidi_colab.ipynb`；Jin 早期的 Colab 存档在 `notebooks/jin_early_colab.ipynb`；proposal 移到了 `docs/`。

| 文件 | 内容 | 作者 |
|---|---|---|
| `analysis.ipynb` | **主 notebook**，从上往下跑：读数据 → 特征 → 防偷看测试 → 切分 → LR 2012 | 大家 |
| `src/features.py` | `make_features`（23 个特征 + 标签） | Qidi（从 Colab 原样搬过来） |
| `src/splits.py` | 外层切分、内层三种切分、bootstrap 抽样 | Qidi（从 Colab 搬过来，只是把全局变量改成函数参数，SEED 改成 0） |
| `src/evaluate.py` | LR Pipeline、三种方式的估计、真实 AUC 和 bootstrap 区间、写结果文件 | Jin |
| `src/config.py` | SEED、模型默认配置、调参网格（事先定死） | Jin |
| `tests/test_no_lookahead.py` | 防偷看测试：把 t 之后的价格全部乱改，t 及之前的 23 个特征必须完全不变；另外确认它能抓到一个故意偷看的特征 | Jin |
| `tests/test_splits.py` | Qidi 在 Colab 里的切分检查，共 54 项，全部通过 | Qidi |
| `results/main.csv` | 每一行：year, model, scheme, estimate, true_auc, true_lo, true_hi, error | 自动生成 |
| `CLAUDE.md` | 项目规则汇总 | — |

数据文件 `data/NASDAQ100_Historical_Data.csv` 不上传到 GitHub，从 Kaggle 下载后放进 `data/` 文件夹。

## 3b. ⚠️ 需要全组决定：Qidi 的 Step 6–7 和 Jin 的版本不一致

Qidi 在 Colab 里新增了 Step 6（定死配置）和 Step 7（LR baseline，2018 年）。Jin 在仓库里也做了 LR baseline（2012 年）。两个版本在下面几处不一致，**要统一成一个版本**：

| 项目 | Jin 的版本（CLAUDE.md / src/config.py） | Qidi 的版本（Colab Step 6） |
|---|---|---|
| LR 预处理 | 标准化 → LR | **先把每个特征裁剪到训练折的 1%–99% 分位数**，再标准化 → LR |
| LightGBM 默认配置 | 库的默认参数 + seed、deterministic | num_leaves=32, n_estimators=300, lr=0.05, min_child_samples=200, subsample=0.8, colsample=0.8 |
| LightGBM 网格（5 组） | (num_leaves, lr, n_estimators) 三个参数一起变 | 只变 num_leaves ∈ {8, 16, 32, 64, 128}，其他参数固定 |
| 种子 | 0 | 5420 |
| 示范年份 | 2012 | 2018 |
| bootstrap AUC 的算法 | sklearn `roc_auc_score` 加权重 | 自己写的向量化算法，速度快，结果和 sklearn 一致（Qidi 加了 assert 检查） |

Qidi 版本 2018 年的结果：shuffled 0.5264 / date 0.5233 / walk-forward 0.5187，真实 AUC 0.5157 [0.4934, 0.5395]，排序和 Jin 的 2012 年一样。

## 4. 第一个结果：LR，测试年 2012

| 验证方式 | 估计（5 折平均） | 真实 AUC | 95% 区间（1,000 次按交易日重抽样） | 误差 |
|---|---|---|---|---|
| shuffled 5 折 | 0.5255 | 0.5089 | [0.4888, 0.5275] | +0.0166 |
| 按日期随机分 5 折 | 0.5221 | 0.5089 | 同上 | +0.0132 |
| walk-forward（最后 5 年） | 0.5148 | 0.5089 | 同上 | +0.0059 |

- 三种方式都高估了 AUC，高估程度 shuffled > 按日期分组 > walk-forward，方向和 proposal 的预期一致。
- 但三个估计值都还在真实 AUC 的 95% 区间内，**只看一年还不能说是偏差**，要看 12 年的结果。
- 真实 AUC 只有 0.509，接近随机猜测，proposal 的风险部分预料过这种情况。

## 5. 接下来每个人做什么

**Qidi**
- [ ] 把 Colab 里的 `SEED` 从 5420 改成 0
- [ ] 打开 `src/features.py` 和 `src/splits.py`，确认和 Colab 里的一致，没有搬错
- [ ] 检查 Jin 的 `src/evaluate.py`：每个折是否只在训练行上学标准化（用的是 Pipeline）

**Yixun**
- [ ] 接受 GitHub 邀请：https://github.com/wjin2025/ese5420-finalpaper/invitations
- [ ] 检查 `src/evaluate.py` 里的 `true_auc_with_interval`（bootstrap 是你负责的部分）：每一组重抽样里，同一天被抽中几次，那一天所有股票的权重就是几
- [ ] 想一下关键图怎么画：真实 AUC 的 bootstrap 区间怎么放进图里（教授要求）。Jin 会先出第一版

**Jin**
- [ ] 跑 12 个测试年（LR），画关键图第一版 → 交给 Yixun
- [ ] 然后做 LightGBM（同一套流程）

## 6. 周一要跟教授说的
1. 特征用 Kaggle 文档的 27 个里的 23 个，以及原因。
2. 样本量更新为 491,772 行。
3. 调参网格已经定死（见第 2 节）。
4. 防偷看测试的做法和结果。
5. 2012 年的结果和 12 年结果的第一版图。
