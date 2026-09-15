# 資料增強與 corruption robustness：階段研究報告

版本日期：2026-09-14。範圍為已完成的 CIFAR-10 實驗；共 18 組訓練。本文是重現與探索性消融報告，並未提出新 augmentation 方法。

## 摘要

本研究比較五種資料增強策略對影像分類準確率與 corruption robustness 的影響，並進一步檢查 CutMix 使用頻率的敏感性。在 CIFAR-adapted ResNet-18、固定 validation 切分與三個訓練 seeds 下，AugMix Transform 取得最高平均 corruption accuracy，而 CutMix 取得最高 clean accuracy。將 CutMix 的 batch 使用機率由 1.0 降為 0.5 後，Gaussian／shot noise 平均準確率提升 13.21 個百分點，完整 mCA 提升 2.41 個百分點，但兩種 noise 的表現仍低於 Baseline。細部分析顯示，不同 corruption 類型的反應不一致，改善幅度也不隨 severity 單調增加。這些結果支持在本設定下同時評估 clean performance、corruption 類型與 augmentation 使用頻率的必要性，但未建立頻率特徵或其他機制的因果解釋。

## 實驗方法

所有實驗使用 CIFAR-10，將 50,000 張訓練圖片固定分為 45,000 張訓練及 5,000 張 validation（split seed 2026）。三個訓練 seeds 為 42、43、44。模型為 CIFAR-adapted ResNet-18，訓練 100 epochs、batch size 128，採 SGD（learning rate 0.1、momentum 0.9、weight decay 0.0005）及 cosine schedule。依 validation accuracy 選擇 checkpoint，再評估 clean test。這是統一預算下的比較，不代表每種方法均已個別調至最佳超參數。

Baseline 使用 random crop 與 horizontal flip。Mixup 與 CutMix 的 alpha 均為 1.0；第一輪 CutMix 每 batch 都使用。RandAugment 使用 num_ops=2、magnitude=9。AugMix 採 torchvision 0.21 transform，severity=3、mixture_width=3、chain_depth=-1、alpha=1.0，沒有 JSD loss，並使用預設 all_ops=True。因此本文稱其為 AugMix Transform，而非原論文完整訓練重現。操作集合差異詳見[文獻核對](../notes/experiments/literature_audit.md)。

Corruption robustness 以 15 種 CIFAR-10-C corruption、各五個 severity 的 accuracy 等權平均（mCA）衡量。先在每個 seed 內平均，再計算三個 seeds 的平均值及樣本標準差（ddof=1）。mCA 是 accuracy，不是以參考模型正規化的 mCE。

## 五方法比較

| 方法 | Clean accuracy (%) | mCA (%) |
|---|---:|---:|
| Baseline | 94.42 ± 0.39 | 73.65 ± 0.26 |
| Mixup | 95.27 ± 0.15 | 78.54 ± 0.61 |
| CutMix p=1 | 95.77 ± 0.05 | 71.14 ± 0.46 |
| RandAugment | 95.03 ± 0.09 | 81.33 ± 0.34 |
| AugMix Transform | 94.83 ± 0.27 | 85.45 ± 0.44 |

三個 seeds 的 mCA 排名均相同：AugMix Transform、RandAugment、Mixup、Baseline、CutMix。CutMix 的 clean accuracy 在三個 seeds 均最高，顯示在本設定下 clean accuracy 的排序無法直接預測 corruption robustness 的排序。AugMix Transform 相對 Baseline 平均 mCA 提高 11.80 個百分點。

這類整體現象並非首次出現：AugMix 論文 Table 1 的 CIFAR-10-C 比較也呈現 CutMix 高於 Standard 的 corruption error。因此本文將其定位為不同實驗設定下的重現及細部分析，並不宣稱新穎性。[AugMix 論文](https://arxiv.org/pdf/1912.02781)

## CutMix 使用機率消融

由第一輪結果形成的假設是：降低 CutMix 的使用頻率可能緩解 Gaussian／shot noise 上的退步。我們新增 p=0.5 的三個 seeds，其餘條件固定，並沿用 Baseline 與 p=1 對照。主要指標預先限定為 Gaussian 與 shot noise 各五個 severity 的十項平均，不包含 impulse noise。

| 方法 | Clean accuracy (%) | mCA (%) | Gaussian／shot noise (%) |
|---|---:|---:|---:|
| Baseline | 94.42 ± 0.39 | 73.65 ± 0.26 | 51.96 ± 2.69 |
| CutMix p=1 | 95.77 ± 0.05 | 71.14 ± 0.46 | 35.55 ± 2.50 |
| CutMix p=0.5 | 95.56 ± 0.30 | 73.56 ± 0.67 | 48.76 ± 3.56 |

對同 seed 配對計算後，p=0.5 相對 p=1 的 noise 差值為 +13.21 ± 5.60 pp、mCA 差值為 +2.41 ± 1.03 pp，兩者三個 seeds 皆改善。Clean accuracy 差值為 -0.21 ± 0.31 pp，兩個 seeds 下降、一個上升。相對 Baseline，p=0.5 的主要 noise 指標仍為 -3.20 ± 2.05 pp，三個 seeds 皆較低；mCA 平均差僅 -0.10 pp，但不能據此宣稱統計等效。

## Corruption 類型與 severity

降低 CutMix 使用機率後，15 種 corruption 中有九種平均改善，Gaussian、shot、impulse noise、JPEG 與 pixelate 五種在三個 seeds 都改善。其中 Gaussian、shot、impulse 的平均提升依序為 13.47、12.96、5.00 pp。另一方面，snow 與 brightness 分別平均下降 0.97、0.15 pp，且三個 seeds 皆下降。

與 Baseline 比較時，Gaussian 與 shot 在五 severity 平均後仍於三個 seeds 皆較差，但 impulse noise 卻平均高出 3.23 pp，三個 seeds 皆較好。因此不能把主要指標的結論概括為所有 noise。Gaussian 相對 p=1 的平均改善在 severity 2 最大（18.12 pp），shot 在 severity 3 最大（17.95 pp），不支持「改善隨 severity 單調增加」的描述。

![三種 noise 的 severity 曲線](../results/analysis/cutmix-probability/noise_severity.png)

圖一：各點為三個 seeds 的平均 accuracy，誤差棒為樣本 SD，不是信賴區間。

![各 corruption 的配對差值](../results/analysis/cutmix-probability/corruption_deltas.png)

圖二：先在同 seed 配對，再跨 seeds 計算差值平均及樣本 SD。正值表示 p=0.5 高於對照。

## 討論與限制

本結果支持一個有限的判斷：在此模型、訓練預算與資料集下，CutMix 的 corruption 表現對使用頻率敏感，降低機率可以縮小部分退步，但效果取決於 corruption 類型。這不表示 p=0.5 是最佳值，也沒有證明模型對高頻、局部紋理或邊界的依賴。

所有結果僅涵蓋單一架構、一個資料集與三個 seeds。使用機率改變也改變增強抽樣的隨機序列；相同 seed 是配對設計，不表示不同方法見到完全相同的合成樣本。已觀察的 CIFAR-10-C 結果曾用於提出消融問題，因此後續消融屬探索性延伸，不是未接觸測試集上的確認實驗。本文未進行多重比較顯著性檢定。

舊對照保留設定、環境、切分及評估雜湊，但未保存當時所有未提交原始碼的快照。p=0.5 新實驗保存 source.zip 與逐檔案 SHA-256。實作 release 為 ac1ed9b，但不應把它當作舊對照的原始訓練 commit。原始 test-selected 探索結果未納入本報告統計。

## 結論與後續

目前成果是一份包含五方法重現、三 seed 驗證與使用機率消融的實證分析。下一步應檢查觀察能否轉移至另一資料集，而不是僅在同一 benchmark 上持續搜尋更高分。具體草案見[跨資料集驗證設計](../notes/experiments/transfer_validation_plan.md)，尚未實作或執行。

## 數據來源

- [五方法統計](../results/comparisons/multiseed-42-43-44/mean_std.csv)
- [消融配對統計](../results/analysis/cutmix-probability/paired_mean_std.csv)
- [逐 corruption 差值](../results/analysis/cutmix-probability/corruption_delta.csv)
- [逐 severity 差值](../results/analysis/cutmix-probability/corruption_severity_delta.csv)
- [實作與紀錄](../notes/experiments/cutmix_probability.md)
