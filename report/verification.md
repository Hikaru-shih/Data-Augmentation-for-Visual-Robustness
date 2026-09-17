# 報告核對紀錄

日期：2026-09-17。對應 [研究報告](research_findings.md) 定稿候選版。

## 本次實際執行

在專案既有 WSL Python 環境依序執行，兩者皆成功結束：

```bash
python3 -m scripts.summarize_multiseed --check-only
python3 -m scripts.compare_transfer
```

第一項核對原始五方法的 15 組 CIFAR-10 實驗。第二項重新核對兩個資料集的 Baseline、CutMix p=1、p=0.5，並重建統計與圖表；扣除重複對照後合計涵蓋 27 組。這次未重新訓練或執行模型推論。

核對範圍包含結果完整性、evaluation 檔案與 checkpoint 雜湊、checkpoint/config 一致性、訓練歷程、seed、protocol 與切分一致性；比較程式另核對設定及適用的來源紀錄。逐 run 檔案清單與本次 SHA-256 見 [run_inventory.csv](run_inventory.csv)。該清單是本次檔案狀態快照，不代表所有歷史 run 都有完整原始碼快照。

## 表格與結論對照

| 報告內容 | 原始統計來源 |
|---|---|
| CIFAR-10 五方法 mean / sample SD | results/comparisons/multiseed-42-43-44/mean_std.csv；本次 check-only 輸出再次核對 |
| CIFAR-10 三方法與配對差值 | results/analysis/cutmix-probability/mean_std.csv、paired_mean_std.csv |
| CIFAR-100 三方法與配對差值 | results/analysis/cifar100_cutmix-probability/mean_std.csv、paired_mean_std.csv |
| 跨資料集配對結果 | results/analysis/cross_dataset/paired_effects.csv |
| 類型與 severity 結果 | results/analysis/cross_dataset/corruption_effects.csv、severity_effects.csv |

主要 noise 指標僅包含 Gaussian 與 shot，各自五個 severity 等權平均；impulse 另行呈現。所有 ± 均為三個 seeds 的樣本標準差，不是信賴區間。配對差值先於同 seed 內相減，再計算平均與 SD；不同資料集不合併。

重新產生的結果確認：p=0.5 相對 p=1 的 mCA 平均改善在 CIFAR-10 為 2.41 pp，在 CIFAR-100 為 1.32 pp；主要 noise 分別改善 13.21 pp 與 5.14 pp。兩個資料集的主要 noise 仍低於各自 Baseline。Gaussian、shot、JPEG、pixelate 在兩個資料集的全部 seeds 皆改善。

## 版本與限制

- 完成數為 18 組 CIFAR-10 加 9 組 CIFAR-100；中斷封存資料不納入。
- AugMix seed 42 的 run ID 為 validation-v2，但 protocol 仍為 validation-v1。
- 原始五方法的歷史來源紀錄不完整；後來的 Git 版本不能當成它們的確切訓練版本。新增實驗的來源以各 run 的 environment/config/source archive 為準。
- 本次文稿與分析版本和訓練版本分開記錄。資料集、模型 checkpoint 與完整 run 檔案仍須另行備份；Git 版本不等同完整實驗備份。
- 三 seeds、單一模型、CIFAR 家族資料集，不能推出統計顯著性、等效性或機制因果關係。
