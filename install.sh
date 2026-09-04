#!/bin/zsh
# 그냥 받아쓰기 설치 — Apple Silicon macOS 전용
set -e
cd "$(dirname "$0")"
[ "$(uname -m)" = "arm64" ] || { echo "Apple Silicon(M칩) Mac 전용이다."; exit 1; }
if ! command -v ffmpeg >/dev/null; then
  command -v brew >/dev/null || { echo "Homebrew가 필요하다: https://brew.sh"; exit 1; }
  echo "ffmpeg 설치 중..."; brew install ffmpeg
fi
if ! command -v uv >/dev/null; then
  echo "uv 설치 중..."; curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "파이썬 환경 구성 중..."
uv venv --python 3.10 -q
uv pip install -q -r requirements.txt
./make_app.sh
cat << 'MSG'

설치 완료.
- 실행: Spotlight에서 '그냥 받아쓰기' (또는 ~/Applications/그냥 받아쓰기.app)
- 메뉴바의 '냥'을 눌러 녹음 시작/종료. 뷰어: http://127.0.0.1:8765
- 첫 녹음 시작 시 모델 다운로드(~3.2GB)로 수 분 걸린다.
- 마이크 권한 창이 뜨면 허용.
- (선택) 한국어 번역: Antigravity CLI(agy) 설치·로그인 시 자동 활성화.
  터미널에서 agy 한 번 실행해 구글 계정으로 로그인해두면 된다.
MSG
