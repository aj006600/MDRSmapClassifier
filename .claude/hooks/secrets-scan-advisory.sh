#!/usr/bin/env bash
# Advisory hook:`git commit` / `git add` 時,掃描 staged 檔案與內容是否有
# 疑似密鑰(可疑檔名、已知 token pattern),有就警告。
# Claude Code PreToolUse hook(註冊於 .claude/settings.json),不是 git pre-commit hook。
#
# 兩個依 repo 調整的點:改成強制擋下(檔尾警告分支的 exit 0 改 exit 2——PreToolUse 只認 exit 2 為擋下);以及豁免清單——
# 預設不豁免任何檔,repo 若有刻意進版控的非機密 env 檔(例:env/.env.dev),在下方
# 「檢查 1」的 case 加一條 continue。

command -v python3 >/dev/null 2>&1 || exit 0
command -v git >/dev/null 2>&1 || exit 0

input=$(cat)

# 快速放行:條件是下方精確判斷的超集合,不改變結果,只為省下 python3 啟動成本
case "$input" in
    *commit*|*add*) ;;
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

# 只在 git commit / git add 時觸發(word-boundary 避免誤判)
if ! echo "$command" | grep -qE '(^|[^a-zA-Z])git (commit|add)([^a-zA-Z]|$)'; then
    exit 0
fi

# 在 worktree 裡 CLAUDE_PROJECT_DIR 仍指向主 checkout,只有 cwd 是 session 實際所在的那個
cd "${cwd:-${CLAUDE_PROJECT_DIR:-.}}" 2>/dev/null || exit 0

# 收集 staged 檔案。純 `git add` 執行前還沒東西 staged → 掃不到就掃不到(advisory,不擋)
staged=$(git diff --cached --name-only 2>/dev/null)
[ -z "$staged" ] && exit 0

warnings=()

# 檢查 1:可疑檔名(env 檔、私鑰、憑證)。
# .example / .template / .sample 後綴放行(慣例上就是要進版控的範本)。
while IFS= read -r file; do
    [ -z "$file" ] && continue
    base=$(basename "$file")
    case "$base" in
        *.example|*.template|*.sample)
            continue
            ;;
    esac
    case "$base" in
        .env|.env.*)
            warnings+=("staged env file: $file")
            ;;
        *.pem|*.key|*.p12|*.pfx|id_rsa|id_ed25519|id_ecdsa|id_dsa)
            warnings+=("staged credential/key file: $file")
            ;;
        credentials.json|service-account*.json|*secrets.json|*secrets.yaml|*secrets.yml)
            warnings+=("staged secrets file: $file")
            ;;
    esac
done <<< "$staged"

# 檢查 2:staged diff 內的已知 token pattern。
# 選低誤判率的 pattern(不透明 token 格式,不是泛用字眼)。
diff=$(git diff --cached 2>/dev/null)
if [ -n "$diff" ]; then
    patterns=(
        'AKIA[0-9A-Z]{16}'                                  # AWS Access Key ID
        'ASIA[0-9A-Z]{16}'                                  # AWS 臨時 session key
        'ghp_[A-Za-z0-9]{36,}'                              # GitHub personal access token
        'gho_[A-Za-z0-9]{36,}'                              # GitHub OAuth token
        'github_pat_[A-Za-z0-9_]{82,}'                      # GitHub fine-grained PAT
        'sk-[A-Za-z0-9]{48,}'                               # OpenAI API key
        'xox[baprs]-[A-Za-z0-9-]{10,}'                      # Slack token
        '-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----'             # PEM 私鑰標頭(RSA/EC/DSA/OPENSSH/ENCRYPTED/PKCS#8 全型式)
    )
    for p in "${patterns[@]}"; do
        # -e 防止以 - 開頭的 pattern(如私鑰標頭)被 grep 當成選項
        if echo "$diff" | grep -qE -e "$p"; then
            warnings+=("potential secret pattern matched: /$p/")
        fi
    done
fi

if [ ${#warnings[@]} -gt 0 ]; then
    echo "[secrets-scan-advisory] commit 前請檢查 staged 內容:" >&2
    for w in "${warnings[@]}"; do
        echo "  - $w" >&2
    done
    echo "[secrets-scan-advisory] 僅警告、不擋。若為真密鑰:unstage 並加入 .gitignore。" >&2
fi

exit 0
