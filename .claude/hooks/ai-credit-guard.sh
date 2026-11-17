#!/usr/bin/env bash
# Guard hook:擋下 `git commit`、`gh pr|issue create/edit/comment`、`glab mr|issue create/update/note`
# 的指令文字中出現的 AI 工具署名(Co-Authored-By: Claude、Generated with Claude Code 等)。
# Claude Code PreToolUse hook(註冊於 .claude/settings.json),不是 git pre-commit hook。
#
# 為什麼是硬擋:署名一進 git 歷史或 PR 就收不回來,而工具預設會加。user 層的 attribution
# 設定(見 README Prerequisites)能關掉預設行為,本 hook 補沒設定那台機器的洞。
#
# 限制:只掃指令文字,看不到 `-F <file>` / `--body-file <path>` 的檔案內容。

command -v python3 >/dev/null 2>&1 || exit 0

input=$(cat)

# 快速放行:條件是下方精確判斷的超集合,不改變結果,只為省下 python3 啟動成本
case "$input" in
    *commit*|*"gh pr "*|*"gh issue "*|*"glab mr "*|*"glab issue "*) ;;
    *) exit 0 ;;
esac

command=$(echo "$input" | python3 -c 'import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get("tool_input", {}).get("command", ""))
except Exception:
    pass
' 2>/dev/null)

[ -z "$command" ] && exit 0

# 只在 commit / PR / MR / issue 的寫入操作時觸發(word-boundary)
if ! echo "$command" | grep -qE '(^|[^a-zA-Z])(git commit|gh (pr|issue) (create|edit|comment)|glab (mr|issue) (create|update|note))([^a-zA-Z]|$)'; then
    exit 0
fi

findings=$(INPUT_TEXT="$command" python3 <<'PYEOF'
import os, re

text = os.environ.get("INPUT_TEXT", "")
patterns = [
    r"co-authored-by:[^\n]*(claude|anthropic)",
    r"generated with[^\n]*claude",
    r"noreply@anthropic\.com",
]
hits = []
for p in patterns:
    for m in re.finditer(p, text, re.IGNORECASE):
        line = text[text.rfind("\n", 0, m.start()) + 1:].split("\n", 1)[0].strip()
        hits.append(line)
print("\n".join(dict.fromkeys(hits)))
PYEOF
)

if [ -n "$findings" ]; then
    echo "[ai-credit-guard] BLOCKED — 指令文字含 AI 工具署名:" >&2
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        echo "  - $f" >&2
    done <<< "$findings"
    echo "[ai-credit-guard] 本 repo 的 commit、PR、issue 不留 AI 工具署名(見 CLAUDE.md Conventions)。拿掉這幾行後重新執行。" >&2
    # PreToolUse 只有 exit 2 會擋下並把 stderr 交給 Claude;exit 1 是 non-blocking error,指令照樣執行
    exit 2
fi

exit 0
