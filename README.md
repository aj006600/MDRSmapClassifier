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

兩個埠只發佈到本機(`127.0.0.1`)。每次執行都會重新做 Search,一次要幾分鐘。算好的結果存在 Docker 的 volume 裡,服務重啟後還查得到。

## Data Source

每日資料來自 [FinMind](https://finmindtrade.com/) 的免費 API。不填 token 也能使用;想要更高的每小時請求額度,在 FinMind 註冊後把 token 填進 `.env` 的 `FINMIND_TOKEN`。資料只存在你自己的機器上;API 只回傳預測與統計,不提供原始市場資料的下載。使用時請自行遵守 FinMind 的使用條款。

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
