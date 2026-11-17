# MDR S-map Classifier

用 MDR S-map 預測台股個股的 Direction。專案正從碩士論文的研究程式改寫成可以自己架設的服務,目前只有前後端的骨架。用詞的定義見 [GLOSSARY.md](GLOSSARY.md)。

## Quick Start

需要 Docker(含 compose plugin)與 make。

```bash
make local       # 前後端起來,前景執行,Ctrl+C 停止
make local-down  # 停止並清理 container 與 network
```

乾淨 clone 不必先建任何檔案,`make local` 會從 `.env.example` 複製出 `.env`。

- 前端 → http://localhost:3000
- 後端健康檢查 → http://localhost:8000/health

兩個埠只發佈到本機(`127.0.0.1`)。

## Docs

| 想做的事 | 看這份 |
|----------|--------|
| 本機開發、測試、lint、清理、worktree | [docs/development.md](docs/development.md) |
| 專案的設計決策 | [docs/adr/](docs/adr/) |

## Structure

```
.
├── backend/          # FastAPI + uv;tests/ 與 uv.lock 也在這裡
├── frontend/         # React + Vite,由 nginx 提供靜態檔並把 /api 代理到 backend
├── compose.yaml      # 服務與內部接線
├── compose.local.yaml # 本機開發:發佈 localhost 埠、後端 hot reload
├── Makefile          # make help 列出所有指令
├── main/ · notebooks/ # 論文的研究程式,之後會搬進 backend
└── data/             # 論文使用的匯出資料
```

## License

[MIT](LICENSE)
