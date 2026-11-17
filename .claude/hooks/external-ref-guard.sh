#!/usr/bin/env bash
# Guard hook:擋下 `git commit`、`gh pr|issue create/edit/comment` 的指令文字中指向
# 其他 GitHub repo 的引用。
# Claude Code PreToolUse hook(註冊於 .claude/settings.json),不是 git pre-commit hook。
#
# 為什麼是硬擋而非警告:這類引用會在對方 issue 觸發跨 repo "mentioned in" backlink,
# 對方 watcher 立刻收到通知,事後編輯也收不回來。
#
# 限制:只掃指令文字,看不到 `--body-file <path>` 的檔案內容。

command -v python3 >/dev/null 2>&1 || exit 0
command -v git >/dev/null 2>&1 || exit 0

input=$(cat)

# 快速放行:條件是下方精確判斷的超集合,不改變結果,只為省下 python3 啟動成本
case "$input" in
    *commit*|*"gh pr "*|*"gh issue "*) ;;
    *) exit 0 ;;
esac

# 第一行 cwd、其餘是 command(command 可能多行)
parsed=$(echo "$input" | python3 -c 'import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get("cwd", ""))
    print(data.get("tool_input", {}).get("command", ""))
except Exception:
    pass
' 2>/dev/null)
cwd=${parsed%%$'\n'*}
# 先剝 cwd 再剝換行:command 為空時 $(...) 吃掉結尾換行,parsed 只剩 cwd 一行
command=${parsed#"$cwd"}
command=${command#$'\n'}

[ -z "$command" ] && exit 0

# 只在 commit / PR / issue 的寫入操作時觸發(word-boundary)
if ! echo "$command" | grep -qE '(^|[^a-zA-Z])(git commit|gh (pr|issue) (create|edit|comment))([^a-zA-Z]|$)'; then
    exit 0
fi

# 在 worktree 裡 CLAUDE_PROJECT_DIR 仍指向主 checkout,只有 cwd 是 session 實際所在的那個
cd "${cwd:-${CLAUDE_PROJECT_DIR:-.}}" 2>/dev/null || exit 0

origin_url=$(git remote get-url origin 2>/dev/null)
current_owner_repo=$(echo "$origin_url" | python3 -c 'import re, sys
u = sys.stdin.read().strip()
m = re.match(r"(?:https?://github\.com/|git@github\.com:)([^/]+)/([^/]+?)(?:\.git)?/?$", u)
print(f"{m.group(1)}/{m.group(2)}" if m else "")
' 2>/dev/null)

# 沒有 GitHub origin 就無從比對
[ -z "$current_owner_repo" ] && exit 0

findings=$(CURRENT_REPO="$current_owner_repo" INPUT_TEXT="$command" python3 <<'PYEOF'
import os, re

text = os.environ.get("INPUT_TEXT", "")
current = os.environ.get("CURRENT_REPO", "").lower()
BACKTICK = chr(96)  # $(...) 內不能出現字面 backtick,用 chr(96) 迂迴


def in_code_span(s, pos):
    """pos 是否位於該行的 backtick code span 內——GitHub 對 code span 不做 reference
    parsing,包在 backtick 裡的引用不會觸發 backlink,可以放行。"""
    line_start = s.rfind("\n", 0, pos) + 1
    return s[line_start:pos].count(BACKTICK) % 2 == 1


findings = set()

# owner/repo#N 縮寫。用 lookbehind/lookahead 排除路徑類誤判
# (前面不可為英數/./_/-/斜線,後面不可為英數)。
short_re = re.compile(
    r"(?<![A-Za-z0-9_./-])([A-Za-z0-9][A-Za-z0-9_.-]*)/([A-Za-z0-9][A-Za-z0-9_.-]*)#(\d+)(?![A-Za-z0-9])"
)
for m in short_re.finditer(text):
    owner, repo, num = m.group(1), m.group(2), m.group(3)
    if f"{owner}/{repo}".lower() == current:
        continue
    if in_code_span(text, m.start()):
        continue
    findings.add(f"{owner}/{repo}#{num}")

# GitHub issue/PR 完整 URL
url_re = re.compile(
    r"https?://github\.com/([A-Za-z0-9][A-Za-z0-9_.-]*)/([A-Za-z0-9][A-Za-z0-9_.-]*)/(issues|pull)/(\d+)"
)
for m in url_re.finditer(text):
    owner, repo, kind, num = m.group(1), m.group(2), m.group(3), m.group(4)
    if f"{owner}/{repo}".lower() == current:
        continue
    if in_code_span(text, m.start()):
        continue
    findings.add(f"github.com/{owner}/{repo}/{kind}/{num}")

print("\n".join(sorted(findings)))
PYEOF
)

if [ -n "$findings" ]; then
    echo "[external-ref-guard] BLOCKED — 偵測到指向外部 GitHub repo 的引用:" >&2
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        echo "  - $f" >&2
    done <<< "$findings"
    echo "[external-ref-guard] 這會建立不可逆的跨 repo 'mentioned in' backlink,並推播給對方 issue 的 watchers。本 repo:$current_owner_repo" >&2
    echo "[external-ref-guard] 安全寫法:用 backtick 包住(\`owner/repo#N\`)或改寫(例:「上游 issue 在 owner/repo,編號 N」)。" >&2
    # PreToolUse 只有 exit 2 會擋下並把 stderr 交給 Claude;exit 1 是 non-blocking error,指令照樣執行
    exit 2
fi

exit 0
