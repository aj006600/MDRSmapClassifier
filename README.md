# MDR S-map Classifier

開源、自己架設的台股 Direction 預測服務。在網頁輸入股票代號(上市、上櫃皆可)並選擇 Horizon,服務會向 FinMind 抓這支股票的公開資料、做 Search、逐日預測到最新的交易日,然後顯示兩樣東西:

- **Forecast**:最近 Horizon 個交易日各自的 Direction 預測,這些日子的實際結果還沒揭曉。
- **Backtest**:從起點以來逐日預測的準確率,以及同期間 Direction 為 1 的天數比例,讓你和「每天都猜 1」比較。

同一支股票之後再執行時,服務沿用上次 Search 找到的參數,幾秒內更新到最新的交易日。網頁能做的事都能經由 HTTP API 做。服務跑在你自己的機器上,抓回來的資料也只存在你的機器上。用詞的定義見 [GLOSSARY.md](GLOSSARY.md)。

> 本服務的 Forecast 與 Backtest 只是模型的計算結果,不構成投資建議。

## Direction

某個交易日的 Direction 為 1,表示它之後 Horizon 個交易日的平均收盤價不低於當天的收盤價;低於則為 0。它比的是接下來一段期間的平均價,不是「明天漲或跌」。

Horizon 可以選 5、10 或 20 個交易日,預設為 10。例如 Horizon 為 10 時,某天的 Direction 為 1,表示從隔天起的 10 個交易日,收盤價的平均不低於那天的收盤價。

## Quick Start

需要 Docker(含 compose plugin)。不需要 Python、Node.js,也不需要原始碼。

1. 到 [Releases](https://github.com/aj006600/mdrsmap-classifier/releases) 找要安裝的版本,以下以 `v0.1.0` 為例。開一個資料夾,下載這個版本的 compose 檔與設定檔範本:

   ```bash
   mkdir mdrsmap-classifier && cd mdrsmap-classifier
   curl -fsSLO https://raw.githubusercontent.com/aj006600/mdrsmap-classifier/v0.1.0/compose.yaml
   curl -fsSL https://raw.githubusercontent.com/aj006600/mdrsmap-classifier/v0.1.0/.env.example -o .env
   ```

2. 打開 `.env`,把 `VERSION` 填成同一個版本(`VERSION=v0.1.0`)。有 FinMind token 的話填進 `FINMIND_TOKEN`,見 [FinMind Token](#finmind-token);不填也能用。

3. 啟動。第一次會從 GitHub Container Registry 拉 image:

   ```bash
   docker compose up -d
   ```

啟動之後:

- 網頁 → http://localhost:3000,輸入股票代號、選擇 Horizon,按「執行」
- API 文件 → http://localhost:8000/docs
- 健康檢查 → http://localhost:8000/health

兩個埠只發佈到本機(`127.0.0.1`),同一個網路上的其他機器連不到。埠已經被占用時,在 `.env` 加上 `FRONTEND_PORT=3001` 或 `PORT=8001` 換成別的埠。

第一次執行一支股票要做 Search,需要幾分鐘;之後再執行同一支股票只補算新增的交易日。怎麼執行、怎麼讀結果、什麼時候該重新搜尋,見 [docs/usage.md](docs/usage.md)。

停止服務用 `docker compose down`,算過的結果會保留。算好的結果存在 Docker 的 named volume `<資料夾名稱>_data` 裡,上例是 `mdrsmap-classifier_data`。compose 以資料夾名稱區分不同的安裝,所以重新啟動與升級都要在同一個資料夾執行。要連資料一起刪除,用 `docker compose down --volumes`。

### Upgrade

1. 把 `.env` 的 `VERSION` 改成新的版本,例如 `VERSION=v0.2.0`。`.env` 保留原本的檔案,不要用新的範本覆蓋;新版本的 `.env.example` 如果多了變數,把它們加進自己的 `.env`。
2. 下載新版本的 `compose.yaml`,覆蓋舊的:

   ```bash
   curl -fsSLO https://raw.githubusercontent.com/aj006600/mdrsmap-classifier/v0.2.0/compose.yaml
   ```

3. 拉新的 image 並重建 container:

   ```bash
   docker compose pull
   docker compose up -d
   ```

資料庫留在原本的 volume 裡,升級前算過的結果升級後仍然查得到。新版本若改了資料表的結構,後端啟動時會自動套用。新版本若改了特徵或演算法,已存的參數仍是舊版本找到的,要不要重新搜尋見 [docs/usage.md](docs/usage.md#when-to-re-search)。

`VERSION` 留空時拉的是 `latest`,也就是最近一次發佈的版本。這樣升級只需要第 2、3 步,但每次 `docker compose pull` 都可能換到新版本,所以建議填固定的版本。

### FinMind Token

不填 token 時,服務用 FinMind 給未註冊使用者的請求額度。額度用盡時,工作會失敗並註明是額度的問題,稍後再試即可。想要比較高的每小時額度,在 [FinMind](https://finmindtrade.com/) 註冊,把取得的 API token 填進 `.env`:

```
FINMIND_TOKEN=<你的 token>
```

改完 `.env` 後執行 `docker compose up -d`,compose 會以新的設定重建 backend。每次執行最多向 FinMind 發出 4 個請求;同一天重複執行同一支股票同一個 Horizon 不會發出請求。

## Data Source and Terms

每日資料來自 [FinMind](https://finmindtrade.com/) 的 API,原始資料由臺灣證券交易所與證券櫃檯買賣中心公布。服務只把抓回來的資料存在你自己的機器上;網頁與 API 只提供預測與統計,不提供原始市場資料的下載。

使用前請讀 FinMind 的[免責聲明與資料授權](https://finmind.github.io/Disclaimer/)與[商業使用 Q&A](https://finmind.github.io/CommercialUse/),並自行遵守。其中與本服務最相關的幾條:

- 不得把 FinMind 提供的原始資料對外再散布、轉售,或架設鏡像 API。
- 商業用途需要 FinMind 的付費方案。
- 把資料用在對外的用途時,要標示原資料提供機關。

這個專案的程式碼以 [MIT](LICENSE) 授權釋出。MIT 授權只涵蓋程式碼,不涵蓋經由 FinMind 取得的資料。

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

## Known Limitations

- **後端的 image 只有 amd64**:pyEDM 1.15.4 在 PyPI 上沒有 arm64 的 Linux 版本。Apple Silicon 的 Mac 上,Docker Desktop 會自動模擬 amd64 來跑後端,但 Search 比原生慢。arm64 的 Linux 主機要先讓 Docker 能模擬 amd64(安裝 QEMU 的 binfmt 支援)。前端的 image 有 amd64 與 arm64 兩種。
- **一次只跑一個工作**:其餘依發起的順序排隊。第一次執行一支股票要做 Search,需要幾分鐘。
- **只能手動執行**:沒有排程。要更新到最新的交易日,就再執行一次。
- **資料晚上才齊**:依 FinMind 的文件,當天的籌碼資料約 21:00 才公布完(沒有實測)。在那之前執行,結果以前一個交易日為準;當天稍後再執行,仍會拿到同一份結果。細節見 [docs/usage.md](docs/usage.md#data-publication-time)。
- **參數不會自動更新**:同一支股票一直沿用同一組參數,直到你要求重新搜尋。
- **特徵與論文不同**:見 [Features](#features)。
- **沒有帳號與權限**:一份安裝給一個人或一個小團隊使用,埠只發佈到本機。
- **只有台股的上市與上櫃股票,只用日資料**:沒有盤中即時資料。

## Relation to the Thesis

這個專案源自作者碩士論文的研究程式。演算法搬進 `backend/mdrsmap/` 之後維持與論文相同的做法:目標嵌入維度、CCM 特徵選擇、多視角隨機搜尋、S-map 加權距離與彈性網路 logistic 的參數選擇,亂數種子也固定。回歸測試以論文的 2330 資料與設定跑一次完整的 Search,確認逐日預測與存好的標準答案相同,重構時改壞演算法會立刻被發現。

服務的特徵改用 FinMind 的公開資料,與論文不同(見 [Features](#features)),所以服務的預測結果與論文不同,論文的準確率也不代表服務的準確率。想知道模型在某支股票上準不準,看那支股票的 Backtest。

## Docs

| 想做的事 | 看這份 |
|----------|--------|
| 執行、讀 Backtest 與 Forecast、重新搜尋、資料公布時間、API | [docs/usage.md](docs/usage.md) |
| 從原始碼跑、本機開發、測試、發版 | [docs/development.md](docs/development.md) |
| 專案的設計決策 | [docs/adr/](docs/adr/) |

## Structure

```
.
├── backend/           # uv 管理;tests/ 與 uv.lock 也在這裡
│   ├── app/           # FastAPI、資料來源、一次執行的流程、SQLite 儲存
│   └── mdrsmap/       # 論文的演算法,獨立於網頁 API
├── frontend/          # React + Vite,由 nginx 提供靜態檔並把 /api 代理到 backend
├── compose.yaml       # 自架用:拉 Release 的 image,只對本機發佈埠
├── compose.local.yaml # 本機開發:疊在 compose.yaml 上,改從原始碼 build、後端 hot reload
├── Makefile           # make help 列出所有指令
└── data/              # 論文使用的匯出資料
```

## License

[MIT](LICENSE)
