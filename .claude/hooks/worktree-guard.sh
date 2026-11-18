#!/usr/bin/env bash
# Guard hook:worktree 一律由 `make worktree` 開,擋下另外兩條會開出新 worktree 的路:
# Bash 的 `git worktree add`,以及沒帶 `path` 的 EnterWorktree(它會自己建 worktree)。
# Claude Code PreToolUse hook(註冊於 .claude/settings.json,matcher 是 Bash|EnterWorktree)。
#
# 為什麼是硬擋:`make worktree` 開的 branch 不追蹤 origin/main、會複製主 checkout 的 .env,
# 目錄名也固定,`make worktree-rm` 才找得到它並檢查有沒有未 push 的東西。另外兩條路這些都不做,
# 而 agent 就算讀過 Makefile 也可能直接選內建工具,只寫在 CLAUDE.md 擋不住。
#
# 放行:帶 `path` 的 EnterWorktree(進入 `make worktree` 建好的目錄),以及 subagent 的
# worktree isolation(不經過這兩個工具)。

# 缺依賴時 fail open:本 hook 防的是誤操作,不該讓環境問題卡死 session
command -v python3 >/dev/null 2>&1 || exit 0

input=$(cat)

# 快速放行:條件是下方精確判斷的超集合,不改變結果,只為省下 python3 啟動成本
case "$input" in
    *orktree*) ;;
    *) exit 0 ;;
esac

verdict=$(echo "$input" | python3 -c '
import json
import shlex
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    print("allow")
    raise SystemExit

tool = data.get("tool_name", "")
tool_input = data.get("tool_input") or {}

if tool == "EnterWorktree":
    # 只有帶 path 是進入既有的 worktree;帶 name 或兩者都不帶,都會新建一個
    print("allow" if tool_input.get("path") else "block")
    raise SystemExit

command = tool_input.get("command", "")
try:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    tokens = list(lexer)
except ValueError:
    # 解析不了(heredoc、引號不平衡等):fail open
    print("allow")
    raise SystemExit

SEPARATORS = {"&&", "||", ";", "|", "&", "(", ")"}
# git 的全域選項裡會吃掉下一個參數的那幾個(`git -C <dir> worktree add` 這種寫法)
GLOBAL_OPTS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"}


def subcommand_args(args):
    """回傳 git 子命令與其後的參數;遇到 shell 分隔符就停。"""
    skip_next = False
    for i, tok in enumerate(args):
        if tok in SEPARATORS:
            return None, []
        if skip_next:
            skip_next = False
            continue
        if tok in GLOBAL_OPTS_WITH_VALUE:
            skip_next = True
            continue
        if tok.startswith("-"):
            continue
        return tok, args[i + 1:]
    return None, []


blocked = False
for i, tok in enumerate(tokens):
    if tok == "git" or tok.endswith("/git"):
        sub, rest = subcommand_args(tokens[i + 1:])
        if sub == "worktree" and rest and rest[0] == "add":
            blocked = True
            break

print("block" if blocked else "allow")
' 2>/dev/null)

if [ "$verdict" = "block" ]; then
    echo "[worktree-guard] BLOCKED — 本專案的 worktree 一律由 make worktree 開。" >&2
    echo "[worktree-guard] make worktree 開的 branch 不追蹤 origin/main、會複製 .env,目錄也開在 make worktree-rm 收得到的位置(見 CLAUDE.md Development workflow)。" >&2
    echo "[worktree-guard] 正確做法:在主 checkout 跑 make worktree ISSUE=<n> BRANCH=<feat/slug>,再用 EnterWorktree 的 path 進它建好的目錄。" >&2
    echo "[worktree-guard] 手上沒有 issue 編號,或這個專案沒有 make worktree 時,先停下來問使用者。" >&2
    # PreToolUse 只有 exit 2 會擋下並把 stderr 交給 Claude;exit 1 是 non-blocking error,指令照樣執行
    exit 2
fi

exit 0
