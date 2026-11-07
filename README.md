# Group 5420：shuffled CV 会把次日股票预测的 AUC 高估多少？

ESE 5420 期末项目。我们预测 NASDAQ-100 股票第二天是否上涨，比较三种验证方式（shuffled 5 折、按日期随机分 5 折、walk-forward）估出来的样本外 AUC 和真实 AUC 差多少，以及这个偏差会不会导致选错模型。研究计划见 [docs/proposal.txt](docs/proposal.txt)，进度见 [PROGRESS.md](PROGRESS.md)。

成员：Qidi（特征、切分）、Jin（模型）、Yixun（误差分析、bootstrap、关键图）。

## 目录（按项目的各个部分）

| 部分 | 文件 | 负责人 | 状态 |
|---|---|---|---|
| **0. 计划** | [docs/proposal.txt](docs/proposal.txt)、[docs/5420_Project_Proposal.pdf](docs/5420_Project_Proposal.pdf) | 全组 | 教授已批准 |
| | [CLAUDE.md](CLAUDE.md)：规则汇总（种子、配置、不能违反的规则） | Jin | |
| **1. 数据读取与检查** | [src/load_data.py](src/load_data.py)：读 CSV、检查重复值 / 缺失值 / 非正价格 | Jin | ✅ |
| | [notebooks/qidi_colab.ipynb](notebooks/qidi_colab.ipynb) Step 1a–1g：详细的数据质量检查（成交量为 0、价格低于 $5、极端涨跌、缺失交易日、每年市场概况） | **Qidi** | ✅ |
| **2. 特征与标签** | [src/features.py](src/features.py)：Kaggle 比赛文档的 27 个指标，去掉 4 个重复项后剩 23 个，再加次日涨跌标签 | **Qidi** | ✅ |
| **3. 防偷看测试** | [tests/test_no_lookahead.py](tests/test_no_lookahead.py)：4 个截止日、全部股票，加上一个故意偷看特征的对照 | Jin | ✅ 通过 |
| | [notebooks/qidi_colab.ipynb](notebooks/qidi_colab.ipynb) Step 3 check：每只股票随机挑 3 天，共 300 次扰动测试 | **Qidi** | ✅ 通过 |
| **4. 切分** | [src/splits.py](src/splits.py)：外层切分（测试年 Y，去掉训练集最后一天）、内层三种切分、bootstrap 抽样 | **Qidi** | ✅ |
| | [tests/test_splits.py](tests/test_splits.py)：54 项切分检查 | **Qidi** | ✅ 通过 |
| **5. 模型与评估** | [src/config.py](src/config.py)：种子、默认配置、调参网格 | Jin | ✅（和 Qidi 的 Step 6 有冲突，见 PROGRESS.md） |
| | [src/evaluate.py](src/evaluate.py)：LR Pipeline、三种方式的估计、真实 AUC 和 bootstrap 区间、写结果文件 | Jin | ✅ 2012 年 |
| | [notebooks/qidi_colab.ipynb](notebooks/qidi_colab.ipynb) Step 6–9：Qidi 版的定死配置、LR baseline（2018 年）、LR 12 年循环、关键图第一版 | **Qidi** | 待和 Jin 的版本合并 |
| **6. 结果与图** | [results/main.csv](results/main.csv)：每行 year, model, scheme, estimate, true_auc, true_lo, true_hi, error | 自动生成 | 2012 年 LR |
| | `figures/`：关键图 | Yixun | ⏳ |
| **主 notebook** | [analysis.ipynb](analysis.ipynb)：调用 `src/` 的代码，从上往下跑完全部步骤，输出可见 | 全组 | Step 0–4 |
| **早期 notebook** | [notebooks/jin_early_colab.ipynb](notebooks/jin_early_colab.ipynb)：Jin 早期的 Colab 版本，已被上面的代码取代，只作存档 | Jin | 存档 |

## 两条约定
1. **`src/` 里的代码是正式版本。** notebooks/ 里的 Colab 文件是存档快照。Qidi 的 Colab 快照日期是 2026-11-07，最新版在 Qidi 的 Colab 里。要改特征或切分，请改 `src/` 里的代码（或者告诉 Jin），改完重跑测试。
2. **从 Colab 搬进来的代码，逻辑一律不改。** `src/features.py` 和 `src/splits.py` 只做了两处机械性改动：全局变量改成函数参数；种子统一成 0。每个文件开头都写了作者。

## 怎么运行（本地）
```bash
conda activate finalpaper          # Python 3.11，库的版本见 requirements.txt
# 把 Kaggle 的 NASDAQ100_Historical_Data.csv 放进 data/（不上传 GitHub）
python -m pytest -v                # 防偷看测试和切分测试
```
然后在 VS Code 或 Jupyter 里打开 `analysis.ipynb`，kernel 选 "Python (finalpaper)"，点 Run All。

在 Colab 里运行：先 `git clone` 本仓库（需要 GitHub token，不要写进 notebook），把 CSV 放进 `data/`，然后就能 `from src.features import make_features` 等。

## 数据
- 价格：Kaggle "NASDAQ100 Historical Data 2000-2026"（jacksaleeby），CC BY-SA 4.0。514,075 行、100 只股票，2000-01-03 到 2026-02-18。
- 特征定义：Kaggle 比赛 "Stock Market Signal: Predict Next-Day Returns" 文档里的指标。
- 建模用的表：491,772 行，上涨占 50.58%。

## AI 使用（proposal 要求记录）
Jin 用 Claude（Anthropic）协助搭建项目结构、搬运代码、编写测试和评估代码、审查是否有数据泄露。每一行代码都由组员检查过，所有数字都用固定种子重跑过。
