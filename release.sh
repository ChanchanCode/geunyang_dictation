#!/bin/zsh
# 새 버전 릴리스 — 버전 올리고, 페이로드·설치기 굽고, GitHub Release 로 올린다.
#   ./release.sh 1.1.0 "바뀐 점 한 줄"
# 앱은 릴리스 자산 payload.tgz 를 그 이름 그대로 받아 자동 업데이트한다. 이름 바꾸지 말 것.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
V="$1"; NOTES="${2:-새 버전}"
[ -n "$V" ] || { echo "사용: ./release.sh <버전> [릴리스 노트]"; exit 1; }
echo "$V" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$' || { echo "버전은 1.2.3 형식."; exit 1; }

sed -i '' "s/^VERSION = \".*\"/VERSION = \"$V\"/" engine.py
grep -q "^VERSION = \"$V\"$" engine.py || { echo "engine.py VERSION 치환 실패"; exit 1; }

T=$(mktemp -d)
tar czf "$T/payload.tgz" app.py engine.py viewer.html requirements.txt make_app.sh README.md assets
./build_installer.sh

git add -A
git commit -q -m "v$V — $NOTES" || echo "→ 커밋할 변경 없음"
git tag -f "v$V" >/dev/null
git push -q origin main
git push -qf origin "v$V"

gh release delete "v$V" -y >/dev/null 2>&1 || true
gh release create "v$V" "$T/payload.tgz" "dist/그냥받아쓰기-설치.command" \
  --title "v$V" --notes "$NOTES"
rm -rf "$T"
echo "완료: v$V — 친구 앱의 설정(⚙)에 업데이트 버튼이 뜬다."
