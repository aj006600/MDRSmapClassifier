# MDR S-map Classifier

用 MDR S-map 預測台股個股的 Direction。專案正從碩士論文的研究程式改寫成可以自己架設的服務。目前在網頁輸入股票代號(上市、上櫃皆可)並選擇 Horizon,服務會向 FinMind 抓這支股票的每日資料、做 Search、逐日預測到最新的交易日,然後顯示 Forecast,以及 Backtest 的準確率、同期間 Direction 為 1 的天數比例與逐日明細。執行過的股票列在網頁下方的清單,點選就能再看它的結果。用詞的定義見 [GLOSSARY.md](GLOSSARY.md)。

## Quick Start

需要 Docker(含 compose plugin)與 make。

```bash
make local       # 前後端起來,前景執行,Ctrl+C 停止
make local-down  # 停止並清理 container 與 network
```

乾淨 clone 不必先建任何檔案,`make local` 會從 `.env.example` 複製出 `.env`。

- 前端 → http://localhost:3000,輸入股票代號、選擇 Horizon,按「執行」
- 後端健康檢查 → http://localhost:8000/health
- API 文件 → http://localhost:8000/docs

兩個埠只發佈到本機(`127.0.0.1`)。每次執行都會重新做 Search,一次要幾分鐘。可以連續發起好幾支股票:同一時間只跑一個,其餘依序排隊;關掉網頁再回來,排隊中與執行中的工作仍然看得到。最新交易日的資料還沒公布完時,執行以前一個資料齊全的交易日為準,結果會標明資料用到哪一天。算好的結果存在 Docker 的 volume 裡,服務重啟後還查得到;重啟當下正在執行的工作會標為失敗,需要重新發起。

## Data Source

每日資料來自 [FinMind](https://finmindtrade.com/) 的免費 API。不填 token 也能使用;想要更高的每小時請求額度,在 FinMind 註冊後把 token 填進 `.env` 的 `FINMIND_TOKEN`。每次執行向 FinMind 發出 4 個請求,額度用盡時,工作會失敗並註明是額度的問題。資料只存在你自己的機器上;API 只回傳預測與統計,不提供原始市場資料的下載。使用時請自行遵守 FinMind 的使用條款。

## Features

演算法使用的特徵全部取自 FinMind 免費層,上市與上櫃的股票用同一組特徵:

| 特徵 | 定義 | FinMind dataset |
|------|------|-----------------|
| 開盤、最高、最低、收盤、成交量 | 當天的價量 | `TaiwanStockPrice` |
| 融資餘額日差 | 當天的融資餘額減前一個交易日的 | `TaiwanStockMarginPurchaseShortSale` |
| 外資持股日差 | 當天的外資持股減前一個交易日的 | `TaiwanStockShareholding` |
| 投信買賣超 | 投信當天買進減賣出的股數 | `TaiwanStockInstitutionalInvestorsBuySell` |
| 自營商買賣超 | 自營商(自行買賣加避險)當天買進減賣出的股數 | `TaiwanStockInstitutionalInvestorsBuySell` |
| EMA9、EMA12、EMA26、MACD、Signal、RSI14 | 技術指標,算法照論文的實作,包括 RSI 對 MACD 而不是收盤價計算 | 由收盤價算出 |

這組特徵與論文有三點不同,所以服務的預測結果與論文不同:

- 論文用到主力買賣超,它需要券商分點資料,免費來源沒有,這一版不使用。
- 論文用投信餘額的日差。公開資料只有每日的投信買賣超,兩者等價,這裡直接使用買賣超。
- 論文用到資金流向,也就是個股成交值除以所屬市場的大盤成交值。FinMind 免費層查得到上市與上櫃大盤的成交值(`TaiwanStockPrice` 的 `TAIEX` 與 `TPEx`),但上市大盤在 2026 年 2 月的每個交易日成交值都是 0,那些日子其實照常開市。執行要抓約四年的資料,之後幾年的上市股票都會碰到這段缺漏,所以這一版不使用資金流向。

籌碼資料缺漏時的處理:

- 籌碼資料比價格晚公布。FinMind 文件記載三大法人約 20:00 更新,融資與外資持股約 21:00 更新。最新幾個交易日的籌碼還沒公布時,執行以最後一個資料齊全的交易日為準。
- 剛上市的股票要過一段時間才有融資與外資持股的資料。在那之前的日子不使用,就像那時候還沒上市。
- 三大法人的資料只列出當天有法人交易的股票。融資或外資持股已公布的日子,三大法人沒有列出就當作買賣超為 0。
- FinMind 偶爾漏掉個別股票某一天的融資餘額或外資持股。只漏一天時沿用前一天的值,所以那天的日差是 0,隔天的日差包含兩天的變化。
- 其他缺漏(連續兩天以上,或缺了買賣超)會讓執行失敗並說明缺的是哪些日子。只有執行實際用到的那段資料會被檢查。

## Docs

| 想做的事 | 看這份 |
|----------|--------|
| 本機開發、測試、lint、清理、worktree | [docs/development.md](docs/development.md) |
| 專案的設計決策 | [docs/adr/](docs/adr/) |

## Structure

```
.
├── backend/          # uv 管理;tests/ 與 uv.lock 也在這裡
│   ├── app/          # FastAPI、資料來源、一次執行的流程、SQLite 儲存
│   └── mdrsmap/      # 論文的演算法,獨立於網頁 API
├── frontend/         # React + Vite,由 nginx 提供靜態檔並把 /api 代理到 backend
├── compose.yaml      # 服務與內部接線
├── compose.local.yaml # 本機開發:發佈 localhost 埠、後端 hot reload
├── Makefile          # make help 列出所有指令
└── data/             # 論文使用的匯出資料
```

## License

[MIT](LICENSE)
