# Development

本機開發的日常操作。所有指令在 repo 根目錄執行。

## Quick Start

```bash
make local       # 前後端起來,後端 hot reload(前景執行,Ctrl+C 停止)
make local-down  # 停止並清理 container 與 network
```

乾淨 clone 不必先建任何檔案:`make local` 發現缺 `.env` 時會從 `.env.example` 複製一份。

- 前端 → http://localhost:3000
- 後端 → http://localhost:8000/health,API 文件在 http://localhost:8000/docs

兩個埠只發佈到 loopback(`127.0.0.1`),同一個網路上的別台機器連不到。在遠端主機上開發時,用 SSH 或 VS Code 把埠轉接到自己的電腦。

前後端怎麼溝通:瀏覽器只打 frontend,`/api/*` 由 frontend 的 nginx 反向代理到 backend(同源、免 CORS)。

```
瀏覽器 → frontend(nginx:80) ──/api/──▶ backend(uvicorn:8000)
                          └─ 其他路徑 = SPA 靜態檔
```

## Configuration

設定一律走環境變數,本機的來源是 `.env`:

| 檔案 | 進版控 | 用途 |
|------|--------|------|
| `.env.example` | 是 | 範本:列出所有變數,非機密給可用的本機預設值,機密給 placeholder |
| `.env` | 否 | 你這台機器實際讀的值;缺檔時 `make local` 自動從 example 複製 |

新增變數:先加進 `.env.example`,再到自己的 `.env` 填值。

## Runs and storage

一次執行的流程:API 收到發起執行的請求後,把工作連同當天的日期記進資料庫就回應。後端啟動時開一條派工的執行緒(`backend/app/jobs.py` 的 `Worker`),依發起的順序一次取出一個排隊中的工作,用 `spawn` 開一個背景程序去跑,跑完才取下一個。背景程序依序經過三個階段:抓資料、Search、逐日預測;沿用參數時沒有 Search 階段。它把目前的階段、結果與失敗原因寫進資料庫,API 再從資料庫讀出來回給前端。

- 同一支股票同一個 Horizon 已經有排隊中或執行中的工作時,再次發起會回傳那個工作,不新增;要求重新搜尋時也一樣
- 後端停止時會終止執行中的背景程序。下次啟動時,上次還在執行中的工作標為失敗、不重跑;排隊中的工作照原本的順序繼續
- 背景程序沒寫下結果就結束(例如記憶體不足被系統終止)時,工作標為失敗
- 做 Search 時,起點設在最新交易日往前 100 個交易日。最新交易日是最近一個資料齊全的交易日,資料還沒公布完的交易日不算
- Library 取起點之前的 600 個交易日,不足 600 個交易日的股票不執行。這兩個數字設定在 `backend/app/runs.py` 的 `RunConfig`
- 演算法的設定值在 `backend/mdrsmap/config.py`,預設為論文的值

同一支股票同一個 Horizon 的第二次以後的執行:

| 情況 | 行為 |
|------|------|
| 已有結果 | 沿用存下的參數與起點,跳過 Search,只預測上次之後新增的交易日;上次還沒揭曉的 Forecast,揭曉後轉入 Backtest |
| 當天已有成功的工作 | API 直接回傳那個工作(HTTP 200),不排進佇列,也不向資料來源發出請求 |
| 要求重新搜尋(`re_search`) | 不沿用已存的參數,以最新的交易日重設起點並重做 Search,原本的逐日預測整個換掉 |

抓回來的每日資料存進資料庫,同一支股票之後只抓最後一個存下的交易日之後的日子。最後一個資料齊全的交易日之後的日子不存,那是資料來源還沒公布完的日子,下次執行會重新抓。

特徵從起點往前 Library 加上 60 個交易日的那一天算起,只檢查這段期間的資料有沒有缺漏。技術指標的值受資料從哪一天開始影響:固定以起點決定起算日,同一個起點每次算出的特徵才會相同,沿用參數補算的結果也才會和一次從起點算到同一天的相同。

失敗原因是寫給使用者看的。查無股票代號、歷史資料不足、資料來源額度用盡、資料來源連不上或回應異常、資料來源的資料有補不了的缺漏,各有自己的說明;其他例外一律標成程式錯誤並附上例外的內容。對應寫在 `backend/app/jobs.py` 的 `_failure_reason`。

資料庫是 SQLite,檔案在 backend container 的 `/data/mdrsmap.db`,`/data` 掛的是 named volume `data`。存的東西有工作的狀態、每日市場資料,以及每支股票每個 Horizon 的 Search 日期、起點、參數與逐日預測。各種操作對資料的影響:

| 操作 | 資料庫 |
|------|--------|
| `make local-down`、Ctrl+C、重新 `make local` | 保留 |
| `make clean` | 刪除(`down --volumes`) |

資料表的結構變更寫在 `backend/app/store.py` 的 `_MIGRATIONS`,後端啟動時套用還沒套用的項目。只能在最後面新增,不能改動已經有人用過的項目。

## Backend platform

backend 的 image 固定 build 成 `linux/amd64`(`compose.yaml` 的 `platform`)。pyEDM 1.15.4 在 PyPI 上只有 x86_64 的 Linux wheel,沒有 arm64 的 wheel,也沒有 sdist,所以 `uv sync` 在 arm64 的 Linux 裡解不出這個版本。為什麼鎖在 1.15.4 見 [docs/adr/0001-pin-pyedm-1x.md](adr/0001-pin-pyedm-1x.md)。

上游 v1.15.4 的原始碼在 arm64 的 Linux 上編得起來,需要 C++ 編譯器、LAPACK 與 pybind11。這樣裝出來的 pyEDM 沒有經過回歸基準的驗證,所以目前不採用。

在 Apple Silicon 的 Mac 上,Docker Desktop 會模擬 amd64 來跑 backend。第一次 build 比較久,執行速度也比原生慢。frontend 沒有這個限制,照主機的架構 build。

## Dependencies

後端的相依寫在 `backend/pyproject.toml`,實際版本由 `backend/uv.lock` 鎖定。`pyedm`、`setuptools`、`scikit-learn` 三個套件有刻意的版本上限,理由寫在 `pyproject.toml` 的註釋裡,升級前先讀。

改了 `pyproject.toml` 之後,在與 image 相同的環境裡重新產生 lock 檔:

```bash
docker run --rm --platform linux/amd64 -v "$PWD/backend":/app -w /app \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim uv lock
```

`tests/test_environment.py` 在容器裡 import pyEDM 並確認版本,lock 檔被誤升級時這個測試會失敗。

## Hot Reload

`make local` 掛載 `backend/app/`、`backend/mdrsmap/` 與 `backend/tests/`,改 Python 檔立即生效。
生效的方式是重新啟動後端,所以當下執行中的工作會被標為失敗,要重新發起。
**改 `pyproject.toml` / `uv.lock` 要重跑 `make local`**(相依烤在 image 裡,掛載蓋不到)。

前端在 `make local` 下是 build 好的靜態檔,改 jsx 不會即時反映;要前端 hot reload 就直接
`cd frontend && npm install && npm run dev`(vite dev server 會把 `/api` 代理到 localhost:8000)。

## Testing and Linting

```bash
make test                             # 完整套 pytest,跟 CI 一樣
make test ARGS="tests/test_main.py"   # 只跑一個檔
make test ARGS="-k health"            # 只跑名字符合的測試
make test ARGS='-m "not slow"'        # 略過慢速測試(演算法的回歸測試)
make lint                             # ruff check 與 ruff format --check
make format                           # ruff format(就地改檔)
```

都跑在 backend 的 dev stage 容器裡,跟 CI 的 `test-backend` job 同一顆 image,檢查的項目也相同:`make test` 與 `make lint` 在本機過了,CI 的 `test-backend` 就會過。CI 另外驗證前端 build 得過,本機要確認就跑 `docker build frontend`。

### Testing workflow

本機跑一次、CI 再跑一次是刻意的,兩次回答的問題不同:本機答「我剛改的有沒有弄壞」,
CI 在乾淨的機器上對**合併後的結果**跑完整套,答「合進 main 能不能用」——CI 不信任本機的結果,那次不能省。
能省的是本機那次的**範圍**:

1. **開發中**只跑跟改動相關的測試,幾秒一輪。`tests/` 鏡射原始碼的結構,改 `app/x.py` 就跑
   `make test ARGS="tests/test_x.py"`,改 `mdrsmap/x.py` 就跑 `tests/mdrsmap/test_x.py`;修一個紅的測試就 `ARGS="-k <測試名>"`。
2. **commit 前**跑 `make lint`,幾秒。
3. **push、開 PR**,CI 跑完整套,不用等它,繼續做別的;紅了再回來修。

只有改到共用的東西(相依、設定、conftest / 共用 fixture、跨多個模組)或不確定影響範圍時,才在 push 前跑一次完整的 `make test`。

### API and data source tests

API 的測試不連網路:資料來源換成 `tests/fakes.py` 裡的假實作,有的回傳固定資料,有的模擬失敗、一直不回應或讓程序直接結束。演算法的設定在 `tests/conftest.py` 裡調小,每個測試用自己的暫時資料庫。執行一樣在背景程序裡跑,所以假資料來源要放在能被 import 的模組裡,不能寫在 conftest。回傳固定資料的假實作可以把每次查詢的起訖日寫進暫時目錄的檔案,測試讀這個檔案確認背景程序向資料來源請求了哪些日子。

「今天」由 `create_app` 的 `today` 參數決定,測試把它換成指定的日子,模擬隔幾個交易日再執行一次。

FinMind 轉接層的測試以 `tests/data/finmind/` 裡事先錄下的回應當輸入,檔名是 `<dataset>_<股票代號>.json`。上市的 2330、上櫃的 6488 與不存在的代號 99999 各錄一組;額度用盡、連不上這類錯誤的回應不錄,直接寫在測試裡。要重新錄的話,用不帶 token 的請求抓同一段日期,存成同樣的檔名:

```bash
cd backend/tests/data/finmind
base='https://api.finmindtrade.com/api/v4/data'
dates='start_date=2026-10-01&end_date=2026-10-14'
for id in 2330 6488 99999; do
  for dataset in TaiwanStockPrice TaiwanStockMarginPurchaseShortSale \
      TaiwanStockShareholding TaiwanStockInstitutionalInvestorsBuySell; do
    curl -s "$base?dataset=$dataset&data_id=$id&$dates" > "${dataset}_${id}.json"
  done
done
```

2330 與 6488 每個交易日都有完整的籌碼資料。測試要模擬缺資料的日子時,從錄下的回應裡拿掉那一天,不另外錄檔。

### Algorithm regression test

`tests/mdrsmap/` 的回歸測試以論文的 2330 資料與設定(Horizon 10、起點 2023-07-01)跑一次完整的 Search,驗證準確率與逐日預測都和存好的標準答案 `tests/mdrsmap/data/2330_directions.csv` 相同。它標成 `slow`:在 Apple Silicon 上模擬 amd64 要跑好幾分鐘,本機可以用 `-m "not slow"` 略過,CI 每次都跑。

標準答案只在刻意改變演算法時重新產生:在 dev stage 容器裡以 `tests/mdrsmap/conftest.py` 的 `load_paper_features()` 讀出同一份資料、跑 `mdrsmap.search`,把逐日的 Direction 寫回那個 csv,並更新測試裡的準確率。結果只在同一個平台上可重現(pyEDM 的鄰居搜尋遇到距離平手時,順序取決於平台),所以一定要在容器裡產生。

## Inspection

```bash
make ps                                  # 正在跑的 container(名稱 / 狀態 / 埠)
make compose ARGS="logs -f backend"       # 追後端 log
make compose ARGS="exec backend sh"       # 進容器
```

`make compose` 對 `make local` 那組 container 下任何 compose 子命令,帶的是同一組 compose 檔與 image tag。

```bash
PORT=8001 FRONTEND_PORT=3001 make local   # 撞埠時換埠(只從 shell 給,不寫進 .env)
```

## Cleanup

磁碟被 Docker 吃滿時,回收這個專案佔掉的空間(設定檔 `.env` 不會被碰):

```bash
make clean     # make local 的 container / volume / network + 本專案的 image + 沒人在用的 build cache
```

**堆積來源**:

| 堆積物 | 怎麼來的 |
|--------|---------|
| build cache | build 的中間結果(`uv sync`、`npm ci` 裝好的套件)存起來讓下次快,通常是最大宗 |
| 沒名字的舊 image | 重 build 同一個 tag,新 image 拿走名字,舊的變成 `<none>` 卻沒被刪 |
| 沒 down 的 container | `make local` 按 Ctrl+C 只是停下來,container 與 network 都還在 |

**怎麼認出這個專案的東西**:每個 Dockerfile 都有一行專案 label(`org.opencontainers.image.title`),
build 出來的每顆 image 都帶著它,連重 build 後沒名字(`<none>`)的舊 image 也是。`make clean` 只刪帶這個 label 的,
不比對 image 名字——`shop` 與 `shop-admin` 這種名字相近的專案才不會互刪。

**build cache 沒有 label**,沒辦法按專案挑。`make clean` 先把本專案的 image 全刪掉,它的 cache 就變成沒人在用,
最後一步 `docker builder prune` 把沒人在用的 cache 丟掉。別的專案同樣沒人在用的 cache 也會一起丟——
它們下次 build 慢一點,不會壞——這是 `make clean` 唯一不限於本專案的地方。

**什麼時候跑**:磁碟告警的時候,或改完 `Dockerfile` / `uv.lock` 想確認從乾淨狀態 build 得起來。
日常收工用 `make local-down` 就好——它只停 container、留 image,下次 `make local` 才起得快;
`make clean` 之後的第一次 `make local` 要從頭 build,會等好幾分鐘。

## Parallel development (worktree)

一個 issue 一個 worktree,多個 worktree 可以同時開發、同時跑 `make local` / `make test`。指令都從主 checkout 跑:

```bash
make worktree ISSUE=12 BRANCH=feat/login   # 開在 .claude/worktrees/<app>-issue-12,並複製主 checkout 的 .env
cd .claude/worktrees/<app>-issue-12
make test                                  # 與其他 worktree 同時跑也不會互相干擾
PORT=8001 FRONTEND_PORT=3001 make local      # 另一個 make local 已佔用 8000 / 3000 時換埠
cd - && make worktree-rm ISSUE=12          # 收工:有未 commit 或未 push 的東西會拒絕
```

為什麼不會互相干擾:

- **container 與 network**:compose project 名取目錄名,每個 worktree 一套。
- **image**:本機 build 的 tag 帶目錄名(`<app>-backend:local-<目錄名>`),同時 build 不會搶同一個名字。
- **設定**:`.env` 從主 checkout 複製,機密跟得過去。

branch 從 `origin/main` 開且不追蹤它(`--no-track`),不帶參數的 `git push` 不會推進 main;第一次 push 寫明
`git push -u origin <branch>`。origin 上已有同名 branch 時,`make worktree` 接著它開。

`make clean` 以專案 label 清 image,會連其他 worktree build 的一起清;正在跑的刪不掉,會停在那一步。
