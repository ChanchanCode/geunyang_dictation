#!/bin/zsh
# 원파일 설치기 생성 → dist/그냥받아쓰기-설치.command
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/dist/그냥받아쓰기-설치.command"
mkdir -p "$ROOT/dist"
T=$(mktemp -d)
tar czf "$T/payload.tgz" -C "$ROOT" app.py engine.py viewer.html requirements.txt make_app.sh README.md assets
cat > "$OUT" << 'HEAD'
#!/bin/zsh
# 그냥 받아쓰기 — 원파일 설치기 (더블클릭 실행)
set -e
clear 2>/dev/null || true
echo "🐈 그냥 받아쓰기 설치를 시작해요."
[ "$(uname -m)" = "arm64" ] || { echo "Apple Silicon(M칩) 맥 전용이에요."; exit 1; }
DIR="$HOME/.geunyang_dictation"
mkdir -p "$DIR"
echo "→ 앱 파일 설치: $DIR"
sed -n '/^__PAYLOAD__$/,$p' "$0" | tail -n +2 | base64 -d | tar xzf - -C "$DIR"
cd "$DIR"
if ! command -v ffmpeg >/dev/null 2>&1 && [ ! -x /opt/homebrew/bin/ffmpeg ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "→ ffmpeg 설치 (Homebrew)"
    brew install ffmpeg
  else
    echo "→ Homebrew가 필요해요. 지금 설치할게요 (맥 암호를 물어볼 수 있어요)."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
    brew install ffmpeg
  fi
fi
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "→ uv(파이썬 관리자) 설치"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
echo "→ 파이썬 환경 구성 (수 분 걸려요)"
uv venv --python 3.10 -q
uv pip install -q -r requirements.txt
chmod +x make_app.sh
./make_app.sh
echo
echo "✅ 설치 완료!"
echo "· Spotlight(⌘Space)에서 '그냥 받아쓰기'를 실행하세요"
echo "· 첫 녹음 시작 때 음성 모델(~3.2GB)을 받아서 몇 분 걸려요"
echo "· 마이크 권한 창이 뜨면 '허용'"
echo "· (선택) 한국어 번역: Antigravity CLI(agy) 설치·로그인 시 자동으로 켜져요"
open "$HOME/Applications/그냥 받아쓰기.app" 2>/dev/null || true
exit 0
__PAYLOAD__
HEAD
base64 -i "$T/payload.tgz" >> "$OUT"
chmod +x "$OUT"
rm -rf "$T"
echo "생성: $OUT ($(du -h "$OUT" | cut -f1))"
