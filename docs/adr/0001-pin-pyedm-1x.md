# Pin pyEDM to 1.15.4

演算法的回歸基準是 2330、Horizon 10、2023-07-01 到 2023-10-31 的 Backtest 準確率 0.7710843373493976。這個數字只有在 pyEDM 1.15.4 下才能重現,所以 pyEDM 鎖在 1.15.4,不跟著升到 2.x。

## Considered Options

- **升到 pyEDM 2.x**:實測 2.5.8 的準確率落在 0.64 到 0.65,而且每次執行的數字不同。原因有三個。2.x 的 `EmbedDimension` 對同一份輸入選出不同的嵌入維度。2.x 的 CCM 一律隨機抽樣,沒有對應 1.x `random=False` 的模式,也不吃程式設定的 seed。2.x 的 API 也改了,原程式要先補上 `target` 參數並拿掉 `random` 參數才跑得起來。
- **鎖在 1.15.4**:Python 3.12 加上其餘套件最新版,逐日的預測結果與原始環境完全相同。

## Consequences

- pyEDM 1.x 會 import `pkg_resources`,所以 setuptools 要鎖在 81 以下。
- 升到 2.x 會改變預測結果,屬於演算法變更,要連同新的回歸基準一起做,不能當成一般的套件升級。
