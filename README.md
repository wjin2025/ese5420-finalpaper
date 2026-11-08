# Group 5420：shuffled CV 会把次日股票预测的 AUC 高估多少？

**一句话**：我们用过去的价格预测 NASDAQ-100 股票第二天涨不涨，然后比较三种交叉验证（CV）方式估出来的 AUC 和真实 AUC 差多少。重点看最常用的 shuffled（随机打乱）CV 会不会高估模型的表现。

**成员**：Jin、Qidi、Yixun。目前的数据、特征、切分、模型和评估，由 **Jin 和 Qidi 一起完成**；Yixun 负责关键图的完善和误差分析。

---

## 从哪里开始看

| 想做什么 | 打开 |
|---|---|
| 了解研究计划 | [docs/proposal.txt](docs/proposal.txt) |
| 看结果和代码（推荐） | [analysis.ipynb](analysis.ipynb)：主 notebook，第 1 步到第 6 步，每一步都有说明和输出 |
| 看做到哪了、谁接下来做什么 | [PROGRESS.md](PROGRESS.md) |
| 看关键图 | [figures/key_figure_v1_logistic.png](figures/key_figure_v1_logistic.png) |

---

## 项目流程（文件名里的数字就是第几步）

```
原始价格 CSV
   │
   ▼
第 1 步  读数据、检查数据            src/step1_load_data.py
   │
   ▼
第 2 步  造 23 个特征和标签          src/step2_features.py      ← 检查：tests/test_step2_no_lookahead.py（防偷看）
   │
   ▼
第 3 步  切分                        src/step3_splits.py        ← 检查：tests/test_step3_splits.py
         · 每个测试年 Y：用 Y 之前的数据训练，Y 当测试
         · 训练部分里做三种 CV：shuffled / 按日期分组 / walk-forward
         · 测试年的交易日 bootstrap 抽样 1,000 次
   │
   ▼
第 4 步  模型（逻辑回归、LightGBM）   src/step4_models.py
   │
   ▼
第 5 步  每种 CV 的估计值 vs 真实值    src/step5_evaluate.py      ← 检查：tests/test_step5_evaluate.py
         逐年跑 12 个测试年 → results/results.csv
   │
   ▼
第 6 步  12 年汇总 + 关键图           src/step6_summary.py       → figures/key_figure_v1_logistic.png
```

所有事先定死的设置（随机种子、模型参数、调参网格）都集中在 [src/config.py](src/config.py)，其他文件都从这里读。

---

## 每个文件夹是什么

| 文件 / 文件夹 | 是什么 | 需要打开吗 |
|---|---|---|
| `analysis.ipynb` | **主 notebook**，按第 1–6 步调用 `src/` 里的代码，显示全部结果 | ✅ 主要看这个 |
| `src/` | 每一步的代码，文件名带步骤编号 | 想看某一步的具体代码时打开 |
| `tests/` | 自动检查：防偷看、切分正确、评估代码和 sklearn 一致 | 一般不用打开，在 notebook 里运行即可 |
| `results/results.csv` | 结果表，每行一个（模型, 测试年, CV 方式）：cv_auc（估计）、test_auc（真实）、test_lo / test_hi（真实值的 95% 区间） | 想看原始数字时 |
| `figures/` | 图 | ✅ |
| `docs/` | proposal（文本版和 PDF 版） | 了解计划时 |
| `notebooks/colab_notebook.ipynb` | 组里的 Colab notebook 的存档（提交给教授的 Colab 链接就是它） | 存档 |
| `notebooks/early_draft.ipynb` | 最早的数据读取草稿，已被上面的内容取代 | 存档 |
| `PROGRESS.md` | 进度、已定的决定、每个人的待办 | ✅ |
| `CLAUDE.md` | 给 AI 助手（Claude）看的项目规则 | 不用 |
| `requirements.txt`、`pytest.ini` | Python 库的版本、测试设置 | 不用 |
| `data/` | 原始数据，**不上传 GitHub**（自己从 Kaggle 下载放进来） | — |

---

## 目前的主要结果（逻辑回归，12 个测试年 2012–2023）

| CV 方式 | 平均高估了多少 AUC | 12 年里高估的年数 |
|---|---|---|
| shuffled 5 折 | **+0.0116** | 12 / 12 |
| 按日期随机分 5 折 | +0.0081 | 10 / 12 |
| walk-forward | +0.0020（和 0 没有显著差别） | 9 / 12 |

三种方式两两比较，排序 12 年全部一致：shuffled > 按日期分组 > walk-forward。详细数字见 PROGRESS.md 和 notebook 第 6 步。

---

## 怎么运行（本地）
```bash
conda activate finalpaper     # Python 3.11，库的版本见 requirements.txt
# 把 Kaggle 的 NASDAQ100_Historical_Data.csv 放进 data/
python -m pytest -v           # 61 项自动检查
```
然后打开 `analysis.ipynb`，kernel 选 "Python (finalpaper)"，点 Run All（约 4 分钟）。

## 数据
- 价格：Kaggle "NASDAQ100 Historical Data 2000-2026"（jacksaleeby），CC BY-SA 4.0。514,075 行、100 只股票，2000-01-03 到 2026-02-18。
- 特征定义：Kaggle 比赛 "Stock Market Signal: Predict Next-Day Returns" 文档里的指标（27 个，去掉 4 个重复后用 23 个）。
- 建模用的表：491,772 行，上涨占 50.58%。

## AI 使用（proposal 要求记录）
我们用 Claude（Anthropic）协助整理项目结构、把 Colab 代码整理成文件、编写自动检查和交叉验证、审查是否有数据泄露。每一行代码都由组员检查过，所有数字都用固定种子重跑过。
