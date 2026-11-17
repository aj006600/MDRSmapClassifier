## Commands

下列是本專案的 canonical 入口,測試與檢查一律走這裡(與 CI 同一套):

| 指令 | 用途 |
|---|---|
| `<本機開發,例:make dev / docker compose up / npm run dev>` | 起本機環境 |
| `<測試,例:make test / uv run pytest / npm test>` | 跑測試 |
| `<檢查,例:make lint / ruff check / npm run lint>` / `<格式化>` | lint / format |

<其餘指令:`make help`(或 `npm run`、`just --list`,依專案填)。>

## Development workflow

Trunk-based:`main` 是唯一長期分支,所有改動走短命 feature branch → PR → squash merge,分支命名 `feat/xxx`、`fix/xxx`。**開工前先切分支**——commit 落在 `main` 上不會被攔,但 push 會(`.claude/hooks/block-push-main.sh` 在 python3 可用時硬擋),屆時得把 commit 搬到分支才走得下去。

一段工作完成後可自行 push feature branch,但**開 PR 前要先問過我**——PR 是 CI 的觸發點。<merge 之後的部署節奏:依專案補一句,例如「merge 進 main 部署 dev/qas,打 v* tag 才上 prod」。>

PR 有 conflict 時自行 `git rebase origin/main` 再 force-push 自己的 feature branch,不用問;文件表格類衝突兩邊都留,程式碼衝突解不了就停下來問。

## Conventions

- **Commit 前先跑 lint**(本專案:`<檢查指令>`)。CI 也會把關,本機先跑省一輪紅燈。
- **Commit message**:輕量 Conventional Commits,英文——`<type>: <description>`,type 取 `feat/fix/docs/test/refactor/chore`,scope 選用(`feat(api): ...` 也可)。
- **PR 標題與內文**:英文,**不用 emoji**(commit message 同樣不用)。標題即 Conventional Commit(squash 後成為 main 的 commit)。body 只有兩節:`## Summary`(為什麼+做了什麼,1-3 個 bullet)、`## Testing`(怎麼驗證的)——**十行內**,不重述 diff,審計報告與過程敘事留在對話與 commit message。
- **不留 AI credit**:commit message、PR / issue 的標題內文與留言、文件、程式碼,都不寫 AI 工具的署名(`Co-Authored-By: Claude ...` 這類 trailer、「Generated with Claude Code」這類字樣)。commit / PR / issue 由 `.claude/hooks/ai-credit-guard.sh` 擋,文件與程式碼靠這條。
- **GitLab 操作**:remote 指向 GitLab 時,MR、issue、pipeline 等 API 操作以 remote URL 內嵌的 token 為唯一憑證來源,API base URL 同樣取自該 remote 的 host(self-hosted 常見)——`git remote get-url origin` 現取、存進 shell 變數後當場帶入呼叫。
- **文件語言**:章節標題英文(`## Quick Start`)、內文繁體中文、技術術語保留英文(fixture、endpoint 不翻譯)。
- **文檔同步**:user-facing 變更(指令、架構、安裝步驟)在**同一個 PR** 內同步 README/docs——文檔更新是變更的一部分,不是事後待辦。
- **文件只講自己**:ADR 與文件只描述本專案自己的決策與理由,不記錄本專案不使用的系統的識別名或狀態(別人的 database、service、container);要指涉時用它與本專案的關係描述(「同一台伺服器上既有的那顆 database」)。
- **文件不寫 issue 與 PR 編號**:README 與 `docs/` 下的文件(含 ADR 與 research)不寫 `#N`、`!N` 當出處,理由直接用白話寫出來;出處在 git 歷史裡。`docs/agents/` 是寫給 agent 的操作說明,不在此限。
- **程式碼註釋**:繁體中文(術語英文),不用 emoji。讀者是會這個語言、讀過專案的 domain 文件(<例:CONTEXT.md>),但沒看過 ADR、issue、也沒參與過討論的人——他只讀這一則就要懂。
  - **寫什麼**:檔案 header 寫這個檔負責什麼、為什麼存在;function / class 的 docstring 寫對呼叫者的承諾與陷阱,名字與 signature 已經說完的就不寫;行內註釋只寫約束、陷阱、非顯然的理由,不敘述下一行做什麼。
  - **怎麼寫**:一句只講一件事,主詞動詞齊全;三件以上並列的子句就用清單(名詞列舉不算);不用符號(`→`、`×`、`=`)代替動詞,不用自創縮語;不解釋就能用的詞只有 domain 文件的詞彙與該語言 / library 的標準術語。寧可多一行,不要省掉主詞。docstring 第一句要短(一行放得下,依 formatter 行寬,全形字算兩格),放不下就拆成一句摘要、空一行、再接內文。
  - **不寫什麼**:不寫 issue 編號與 ADR 編號,理由直接用白話寫出來;不寫會過期的事(目前誰在呼叫、把實作步驟再講一遍)。只有理由長到放不下、且這段 code 看起來像寫錯時,才在最後加一行指向 `docs/` 下的完整檔案路徑。
  - 寫給模型看的字(tool function 的 docstring、`Field(description=...)`、prompt 字串)不是註釋,不適用本條;改它們要實測。
- **規則提議**:同類糾正或卡關**第二次**發生(或一次但代價高:不可逆操作、部署事故級),主動提議把教訓寫進來——先展示擬新增的那一行與觸發原因,我同意才寫。歸位優先序:能用工具強制的進工具/CI → 綁定單檔的進該檔註釋 → 概念進 `docs/` → 決策進 `docs/adr/` → 都不是且每個 session 都適用,才進 CLAUDE.md。
- **測試品質**(pytest 範例,依測試框架調整):單元測試目錄鏡射原始碼模組結構;fixture 先用 conftest 既有的,重複自建即違規;測試名稱與 docstring 必須與實際行為一致;每個測試驗證一個真實行為——斷言偶然字串(使用者可見的契約訊息除外)、恆真斷言、與既有測試重複覆蓋的,一律刪除。
- **技術調研**:用到不熟悉或版本敏感的 API/框架時,**先查證再寫**——context7 查 library 文檔、WebSearch/WebFetch 查一手來源、`gh search` 找官方範例與成熟專案;大題目啟用 `/research`(背景 agent、只認一手來源、產出帶引用的 Markdown 進 repo)。外部資訊一律**驗證(實測或對照官方文檔)後才採用**。
