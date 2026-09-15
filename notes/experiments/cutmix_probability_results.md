# CutMix 使用機率消融：結果與下一步

整理日期：2026-09-14。三個 seeds（42、43、44）均完成 100 epochs、clean test、75 組 corruption 評估與個別圖表。9 月 14 日比較程式已重新驗證 checkpoint、評估檔案、p=0.5 原始碼快照及資料切分。沒有重新訓練。

## 實驗問題與對照

在其他設定固定時，把每個 batch 使用 CutMix 的機率從 1.0 降為 0.5，是否縮小 Gaussian／shot noise 的準確率退步？

模型為 CIFAR-adapted ResNet-18，訓練／validation 為 45,000／5,000 張，split seed=2026，依 validation accuracy 選模型。CutMix alpha=1.0，其餘訓練設定保持一致。Baseline、p=1 使用已完成的對照，p=0.5 使用獨立 experiment `cutmix_p05_resnet18`。

主要 noise 指標只包含 Gaussian noise 與 shot noise，各五個 severity，共十項等權平均；**不包含 impulse noise**。mCA 則包含全部 15 種 corruption × 5 severities。先在每個 seed 內平均，再計算跨 seed 平均值及樣本標準差（ddof=1）。

## 整體結果

| 方法 | Clean accuracy (%) | mCA (%) | Gaussian／shot noise accuracy (%) |
|---|---:|---:|---:|
| Baseline | 94.42 ± 0.39 | 73.65 ± 0.26 | 51.96 ± 2.69 |
| CutMix p=1.0 | 95.77 ± 0.05 | 71.14 ± 0.46 | 35.55 ± 2.50 |
| CutMix p=0.5 | 95.56 ± 0.30 | 73.56 ± 0.67 | 48.76 ± 3.56 |

## 配對差值：p=0.5 減去對照

單位為百分點；標準差來自同 seed 配對後的差值，不是兩組標準差相減。

| 對照 | Clean delta | mCA delta | Noise delta |
|---|---:|---:|---:|
| CutMix p=1.0 | -0.21 ± 0.31 | +2.41 ± 1.03 | +13.21 ± 5.60 |
| Baseline | +1.14 ± 0.17 | -0.10 ± 0.69 | -3.20 ± 2.05 |

| Seed | Noise delta vs p=1 | mCA delta vs p=1 | Clean delta vs p=1 | 實際 CutMix batch 比例 |
|---|---:|---:|---:|---:|
| 42 | +19.01 | +3.58 | -0.29 | 49.97% |
| 43 | +12.79 | +2.00 | -0.47 | 50.22% |
| 44 | +7.85 | +1.66 | +0.13 | 49.93% |

實際比例統計的是觸發 CutMix 的 batch；觸發後裁切面積為零仍計入，不代表每次都有像素改變。

## 可以下的結論

- 三個 seeds 的 noise 與 mCA 都比 p=1 改善，支持「降低使用頻率能緩解本設定下的 robustness 退步」這個探索性假設。
- p=0.5 在三個 seeds 的主要 noise 指標仍低於 Baseline；未完全恢復到 Baseline。
- p=0.5 的平均 mCA 與 Baseline 接近，但單個 seeds 有高有低。接近不等於統計等效。
- Clean 相對 p=1 在兩個 seeds 下降、一個上升，平均下降 0.21 pp；相對 Baseline 則三個 seeds 都提高。

不能據此宣稱 p=0.5 是最佳機率、CutMix 必然傷害 robustness、結果具有正式統計顯著性，或模型已被證實依賴高頻。此實驗由先前看過的 CIFAR-10-C 結果產生，屬同 benchmark 上的探索性消融。

## 細部分析設計（已完成，2026-09-14）

1. 延伸目前比較輸出：分開 Gaussian、shot、impulse noise，並覆蓋其他十二種 corruption；計算每個 seed 的 p=0.5−p=1 與 p=0.5−Baseline 差值，再彙整平均／SD。
2. 畫 Gaussian 與 shot noise severity 1–5 的三方法曲線，以及 15-corruption 配對差值圖。確認改善是否集中在較嚴重的 noise，同時檢查是否有其他 corruption 退步。現有 CSV 足夠，不需重跑模型。
3. 把上述結果連同本頁表格整理成這輪實驗結論，再決定是否值得擴展。不要現在就直接加入更多 probability、延長 epochs 或組合 AugMix，否則會同時引入多個研究問題。
4. 若需要更強的泛化證據，先寫好新 benchmark／模型與評估規則，再執行獨立驗證。沿用已看過的 benchmark 做更多 seeds 能改善變異估計，但不能把它變回未接觸的確認測試。

以上細部分析已實作並執行，不需要重跑 `run_cutmix_p05.sh`。使用 `python3 -m scripts.compare_cutmix_probability` 可重新產生全部比較表與細部圖表。

## 來源與可追溯性

- [每 seed 結果](../../results/analysis/cutmix-probability/per_seed.csv)。
- `results/analysis/cutmix-probability/mean_std.csv`
- `results/analysis/cutmix-probability/paired_deltas.csv`
- `results/analysis/cutmix-probability/paired_mean_std.csv`
- [實作與設計紀錄](cutmix_probability.md)，實作版本 commit `ac1ed9b`；每個新 run 的實際版本以 environment.json、config.yaml、source.zip 及 source_manifest.json 為準。
- 舊對照沒有原始碼快照；不要把後來的 commit 當成其歷史訓練版本。此限制已在實作紀錄說明。

原始 CSV、checkpoint 和既有結果均保留，本次只整理文件。

## 細部分析結果

已新增八份 CSV（三方法準確率、配對差值，含逐 seed 與彙整，分 corruption 及 corruption/severity）、三張圖與自動摘要。輸出於 `results/analysis/cutmix-probability/`。

- 相對 p=1，15 種 corruption 有 9 種平均改善；Gaussian、shot、impulse noise、JPEG、pixelate 五種在三個 seeds 都改善。
- 平均提升最大的三項為 Gaussian +13.47 pp、shot +12.96 pp、impulse +5.00 pp（先平均五個 severity）。
- Snow 平均 -0.97 pp、brightness -0.15 pp，三個 seeds 皆下降；降低使用機率不是全方位改善。
- 相對 Baseline，Gaussian 平均 -2.83 pp、shot -3.56 pp，三個 seeds 皆較差；但 impulse 平均 +3.23 pp，三個 seeds 皆較好。不能把 Gaussian/shot 的結論概括為所有 noise。
- Gaussian 相對 p=1 的平均改善在 severity 2 最大（+18.12 pp），shot 在 severity 3 最大（+17.95 pp）；兩者五個 severity 的平均差值皆為正，但並非隨嚴重度單調增加。這是在不同 severity 間描述結果，並未做正式交互作用檢定。

圖表：`corruption_deltas.png`、`noise_severity.png`、`all_severity.png`。完整數字：`detail_summary.md`、`corruption_delta.csv`、`corruption_severity_delta.csv`。誤差棒為三個 seeds 的樣本 SD，不是信賴區間。

下一步：先把本輪結論整理成報告段落，將「使用頻率敏感性」與「不同 noise 類型效果不同」作為兩個觀察。若要新增實驗，先固定一個未用於設計的模型或 benchmark 驗證此趨勢，並制定相同訓練預算及主要指標；目前不要為尋找最佳分數而直接掃描更多 probability。

程式變更：`scripts/compare_cutmix_probability.py` 呼叫新增的 `scripts/cutmix_detail_analysis.py`。新增缺失配對檢查與統計測試；19 項測試通過。此變更只影響分析輸出，不改訓練程式、checkpoint 或原始 robustness CSV。