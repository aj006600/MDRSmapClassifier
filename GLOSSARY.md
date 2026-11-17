# MDR S-map Classifier

用 MDR S-map 預測台股個股未來幾個交易日的漲跌方向。這份詞彙表定義專案裡的用詞,文件、API 與介面都用這裡的名字。

## Language

**Direction**:
未來 Horizon 個交易日的平均收盤價是否不低於今天的收盤價。是記為 1,否記為 0。
_Avoid_: 漲跌、label、`y_10` 這類欄位名

**Horizon**:
Direction 往未來看的交易日數。
_Avoid_: tp、TP、預測期

**Forecast**:
對最新一個交易日算出的 Direction,實際結果還沒揭曉。
_Avoid_: realtime、single_pred、即時預測

**Backtest**:
對過去一段期間逐日做 Forecast,再和實際的 Direction 比對算出準確率。
_Avoid_: backtrade

**Search**:
為一支股票與一個 Horizon 找出建模參數的步驟,只用起點之前的資料,找到的參數在之後的交易日沿用。
_Avoid_: first_pred、訓練、調參

**Library**:
用來建模的歷史資料列,沿用 EDM 領域的原名。
_Avoid_: 訓練集
