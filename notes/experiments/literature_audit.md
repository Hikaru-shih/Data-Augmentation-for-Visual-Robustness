# 文獻核對與下一輪實驗決策

查核日期：2026-09-10。這是針對目前結果的初步查核，不是完整系統性文獻回顧。既有 15 組結果保持不變。

## 已有文獻能支持哪些判斷？

| 來源 | 核對內容 | 對本專案的意義 |
|---|---|---|
| [AugMix 論文，Table 1](https://arxiv.org/pdf/1912.02781) | CIFAR-10-C 的平均分類 error：Standard 29.0%、CutMix 30.3%、AugMix 12.5%，平均涵蓋四種 backbone；CutMix 在四種架構皆比 Standard 高 error。 | CutMix 的 corruption 表現低於 baseline 並非全新現象；本專案提供不同設定下的重現與逐 corruption 分析。這些是 error，不能直接和本專案 accuracy 欄位並列。 |
| [CutMix 官方實作](https://github.com/clovaai/CutMix-PyTorch) | CIFAR-100 範例使用 PyramidNet、300 epochs、cutmix_prob=0.5；ImageNet 範例使用 ResNet-50、300 epochs、prob=1.0。 | 本專案是 CIFAR-10 ResNet-18、100 epochs，且每 batch 都 CutMix（p=1）。不同設定下的結果不能用來宣稱原論文錯誤。p=0.5 是有依據的消融候選，不是 CIFAR-10 的已知最佳值。 |
| [torchvision 0.21 AugMix 文件](https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.AugMix.html) | all_ops 預設 True，包括 brightness、contrast、color、sharpness。 | 本專案沒有顯式傳入 all_ops，因此目前操作集合包含這四項；不應聲稱與作者預設完全相同，或所有測試 corruption 類型皆未在訓練操作中出現。這不表示使用了 CIFAR-10-C 測試圖片。 |
| [AugMix 作者 CIFAR 程式](https://raw.githubusercontent.com/google-research/augmix/master/cifar.py) | all-ops 與 no-jsd 都需以旗標啟用；預設不包含額外四操作，且使用 JSD。 | 本專案是 transform-only、all_ops=True；即使加上 JSD，也須核對操作集合才可談忠實重現。 |
| [IPMix 補充資料，Section F / Table 6](https://proceedings.neurips.cc/paper_files/paper/2023/file/c917d8b9e01427f3184d80ade22f4d1f-Supplemental-Conference.pdf) | 已研究 AugMix、MixUp、CutMix 的順序、隨機選擇與分 epoch 組合，部分組合比單獨方法差。 | 不能把「把 mixing augmentations 組合起來」直接當成新方法。這些多方法組合也不是對本專案精確雙方法設定的定論。 |
| [NoisyMix，AISTATS 2024](https://proceedings.mlr.press/v238/erichson24a.html) | 研究輸入／特徵空間 noisy augmentation 與 robustness。 | 若後續往 noise 與 mixing 組合發展，需要納入相關工作；本次未做其全部實作的逐行核對。 |

## 目前可寫的結論與不能寫的推論

- 可以：在本專案固定訓練協定、三個 seeds 下，CutMix 在 Gaussian/shot noise 皆低於 baseline；AugMix Transform 在 noise 改善明顯。
- 不可以：CutMix 必然傷害 robustness、模型已證實過度依賴高頻、組合 CutMix 與 AugMix 是首次提出。現有 accuracy 結果沒有測量頻率敏感度或證明因果。
- all_ops 差異屬於方法描述與可比較性的限制，並不是要作廢或刪除現有實驗。將既有 AugMix 明確描述為 torchvision 0.21 transform-only / all_ops=True。

## 建議下一輪：先檢查 CutMix 使用頻率

問題：在目前 100-epoch 協定下，把 CutMix 的 batch 使用機率由 1.0 降為 0.5，是否縮小 Gaussian/shot noise 的退步？這是設定敏感度實驗，不是新方法主張。

1. 新增可設定的 CutMix probability，預設 1.0 保持舊設定；p=0.5 時未觸發的 batch 使用原圖與普通 cross-entropy。記錄實際使用比例並補測試。
2. 只新增 p=0.5 的 seeds 42、43、44，共三次訓練，保留其餘設定、validation seed 2026 與切分不變。使用獨立 experiment name；不改寫舊 config 或結果。
3. 事先指定主要指標：每個 seed 內 Gaussian/shot noise 各五個 severity 共十項的等權平均 accuracy；計算相對 p=1.0 的配對差值與樣本 SD。
4. 同時報告 clean accuracy、完整 15-corruption mCA 和其餘 corruption，不能只報改善項目。現有 baseline 和 p=1.0 三組結果可作對照，不需要重跑。
5. 三次都完成後才做整體判讀，不依第一個 seed 的測試結果改參數。若 noise 改善但 clean 下降，記錄 trade-off；若未改善，表示降低使用頻率在此設定下未解決問題，不能反推任何頻率機制。

這個假設由已看過的 CIFAR-10-C 結果產生，所以後續仍是該 benchmark 上的探索性延伸，不能把它當成從未接觸的確認測試。若要主張跨情境泛化，需另設未用於設計的新 benchmark。

## 後續候選，暫不一起執行

- AugMix all_ops=False：固定無 JSD、其他條件不變，隔離操作集合的影響。
- 較長 schedule：若測 200 epochs，baseline 與 CutMix 都需要相同新 schedule 對照，不能只延長其中一個。
- CutMix + AugMix Transform：文獻已涉及相關組合；需明確比較 baseline、CutMix、AugMix、組合四組，評估是否超過單獨 AugMix，而非僅超過 CutMix。

本次只完成文獻與實驗設計；尚未修改 augmentation 行為、建立新訓練設定或啟動訓練。
