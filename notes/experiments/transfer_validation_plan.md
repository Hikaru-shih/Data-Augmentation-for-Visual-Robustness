最新狀態（2026-09-17）：CIFAR-100 九組已完成並通過比較驗證，共 27 組完成實驗。請見[跨資料集整合結果](cross_dataset_results.md)。下方先前日期的執行紀錄保留作歷史紀錄。

# 跨資料集驗證草案：CIFAR-100 / CIFAR-100-C

日期：2026-09-14。狀態：2026-09-15 已完成程式實作、設定與測試；尚未下載資料或訓練。操作與变更見 [cifar100_execution.md](cifar100_execution.md)。這不是事前登錄的宣稱；實作完成後、查看新測試結果前需固定最終協定與 commit。

## 研究問題

在新的 100 類分類任務中，CutMix p=0.5 相對 p=1 能否再次改善 Gaussian／shot noise accuracy？此驗證針對使用機率差值的方向，不要求 CIFAR-100 的絕對 accuracy 接近 CIFAR-10，也不預設 p=1 必然低於 Baseline。

選擇 CIFAR-100 是為了改變分類任務、維持小尺寸影像與既有架構的適用性。官方資料頁說明 CIFAR-100 的 fine-label 分類；benchmark 作者提供 CIFAR-100-C。它仍與 CIFAR-10 同屬 CIFAR 資料家族，且使用相關 corruption 設計，因此只能提供有限的跨資料集證據，不代表真實世界全面泛化。[CIFAR 官方資料](https://www.cs.toronto.edu/~kriz/cifar.html)、[corruption benchmark 作者 repository](https://github.com/hendrycks/robustness)

## 固定比較

- 三組：Baseline、CutMix p=1、CutMix p=0.5；alpha=1。每組 seeds 42、43、44，共九次新訓練，全部從頭初始化。舊 CIFAR-10 checkpoint 不能當作此任務對照。
- CIFAR-adapted ResNet-18，輸出 100 fine classes；100 epochs、batch 128、SGD lr=0.1、momentum=0.9、weight decay=0.0005、cosine schedule。
- 從 CIFAR-100 訓練資料切出每 fine class 50 張 validation，共 5,000 張；split seed=2026，保存索引，三方法共用。其餘 45,000 張訓練。
- 正規化平均／標準差只從這 45,000 張未增強訓練圖片計算，逐 channel、像素加權，使用 population SD；凍結並套用到 validation、clean test 與 corruption。不從測試集估計。
- Validation 選 checkpoint，最後測 clean；CIFAR-100-C 不用於選 epoch 或調參。

## 指標與判讀規則

主要指標沿用 Gaussian、shot noise 各五 severity 的等權平均。每 seed 計算 p=0.5−p=1，报告三個差值、平均與樣本 SD。三個差值皆正才描述為「在三 seeds 重現改善方向」；正負混合則描述為不一致；不可依平均為正就稱統計顯著。

次要指標包括 clean accuracy、完整 15-corruption mCA、全部 corruption 與 severity，以及相對 Baseline 差值。若 noise 改善伴隨 clean 下降，如實報告 trade-off，不事後新增有利門檻。失敗或方向相反也保留。

## 實作前工作

1. 通用化 CIFAR loader 與 corruption loader：100 類標籤驗證、訓練集正規化、分層切分、設定驅動模型輸出及資料路徑。此項已在 2026-09-15 實作；使用獨立 CIFAR-100 設定與批次入口。
2. 增加 100 類與資料來源檢查、切分無重疊、正規化一致、機率分支與配對分析測試。
3. 用合成資料或訓練／validation 小樣本做 smoke test，不先看新 corruption 分數來調參。
4. 建立獨立 experiment 名稱與協定標記、保存來源版本及資料 checksum，再提供批次執行指令。記錄耗時與 GPU 環境；本草案不估計尚未量測的總訓練時間。

這輪只改資料任務與其必要處理；不同时新增 AugMix 組合、其他 probability、其他模型或 200-epoch schedule。若資源不允許九次訓練，可先以現有報告作階段成果，不將尚未完成的驗證寫成結論。
