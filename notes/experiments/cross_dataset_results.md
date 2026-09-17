# 跨資料集整合結果

日期：2026-09-17。CIFAR-10 完成 18 組，CIFAR-100 完成 9 組，共 27 組正式完成實驗；中斷後封存的 run 不計入。兩個資料集分別驗證來源、設定、資料切分與評估完整性後再比較，不合併其絕對 accuracy。

## 主要指標：p=0.5 減 p=1

同 seed 配對差值的平均 ± 樣本標準差（百分點，ddof=1）。Noise 只含 Gaussian 與 shot 各五個 severity。

| 資料集 | Clean delta | mCA delta | Noise delta |
|---|---:|---:|---:|
| CIFAR-10 | -0.21 ± 0.31 | +2.41 ± 1.03 | +13.21 ± 5.60 |
| CIFAR-100 | -0.69 ± 0.09 | +1.32 ± 0.70 | +5.14 ± 2.29 |

兩個資料集各三個 seeds 的 noise 與 mCA 差值皆為正，因此預先指定的改善方向在 CIFAR-100 上再次出現。Clean 在 CIFAR-100 三個 seeds 都下降，CIFAR-10 則兩個下降、一個上升。這顯示 robustness 改善伴隨 clean performance 的取捨，不能稱為全面優於 p=1。

## CIFAR-100 絕對結果

| 方法 | Clean (%) | mCA (%) | Noise (%) |
|---|---:|---:|---:|
| Baseline | 76.31 ± 0.41 | 47.78 ± 0.23 | 26.29 ± 0.59 |
| CutMix p=1 | 79.04 ± 0.09 | 46.26 ± 0.60 | 15.85 ± 1.47 |
| CutMix p=0.5 | 78.35 ± 0.10 | 47.58 ± 0.20 | 20.98 ± 1.00 |

p=0.5 相對 Baseline 的主要 noise 差值為 -5.30 ± 1.44 pp，三個 seeds 都較差；clean 平均提高 2.03 pp，mCA 平均低 0.19 pp。不能把 mCA 均值接近解釋成統計等效。

## 一致與不一致的細部趨勢

- Gaussian、shot、JPEG compression、pixelate 在兩個資料集的全部三個 seeds 都優於 p=1。
- Impulse noise 的平均差值在兩資料集均為正，但 CIFAR-100 有一個 seed 為負（跨 severity 平均的範圍 -0.40 至 +7.37 pp），不能稱為每次都改善。
- Snow 在兩個資料集的三個 seeds 均小幅下降。
- CIFAR-10 的 brightness 在三個 seeds 都下降，但 CIFAR-100 並非三次都下降；zoom blur 在 CIFAR-100 三次皆下降，但 CIFAR-10 有正有負。
- 不把 corruption 或 severity 視為額外獨立訓練 seeds。跨資料集幅度不同也不等於已通過統計交互作用檢定。

## 可放入報告的結論

在 CIFAR-adapted ResNet-18、100-epoch 固定訓練預算下，將 CutMix 的 batch 使用機率由 1.0 降至 0.5，可在 CIFAR-10 與 CIFAR-100 的三個 seeds 中一致改善 Gaussian／shot noise 平均準確率及完整 mCA。然而，主要 noise 表現仍低於各自 Baseline，且 clean accuracy 平均下降。降低使用頻率能緩解部分 corruption 退步，但效果依 corruption 類型而異，亦非所有指標均改善。

這是同一模型、同一 CIFAR 資料家族及相關 corruption 設計下的有限跨任務驗證。CIFAR-100 的 validation 為分層切分、正規化來自訓練子集，與 CIFAR-10 的既有處理不同；因而不把差值幅度差異歸因於類別數本身。未證明最佳機率、真實世界泛化、高頻依賴機制或正式統計顯著性。舊 CIFAR-10 對照缺乏當時完整 source snapshot 的限制仍適用。

## 產物與重現

- `results/analysis/cross_dataset/paired_effects.csv`：兩資料集的整體配對差值。
- `corruption_effects.csv`、`severity_effects.csv`：完整細部數字，不只列有利項目。
- `paired_effects.png`、`noise_effects.png`：平均值與樣本 SD 圖。
- `README.md`：自動摘要。

```bash
python3 -m scripts.compare_transfer
```

此指令先重建並驗證兩份資料集分析，再整合輸出。只分析現有結果；不會訓練或重新推論模型。

## 下一步

目前已有完整的比較、消融、跨資料集驗證鏈，先整理定稿與相關工作定位，不必立即追加訓練。若報告目標要求更廣泛證據，再另行選定第二架構、固定協定與計算預算；如果研究目標改為解釋機制，需設計直接測量或干預，不能只靠增加 accuracy 表格推論原因。這两種方向需先確定研究目標，再投入新實驗。
