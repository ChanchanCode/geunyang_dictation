#!/bin/zsh
# '그냥 받아쓰기.app' 번들 생성 → ~/Applications
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$HOME/Applications/그냥 받아쓰기.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>그냥 받아쓰기</string>
  <key>CFBundleDisplayName</key><string>그냥 받아쓰기</string>
  <key>CFBundleIdentifier</key><string>com.geunyang.dictation</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleExecutable</key><string>run</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>수업을 녹음해 받아쓰기 위해 마이크를 사용한다.</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
cat > "$APP/Contents/MacOS/run" << RUN
#!/bin/zsh
# 런처: 클릭 = 무조건 뷰어 오픈. 엔진이 없으면 백그라운드로 시작.
export PATH="/opt/homebrew/bin:/usr/local/bin:\$HOME/.local/bin:\$PATH"
URL="http://127.0.0.1:8765"
if curl -s -m 1 "\$URL/api/state" > /dev/null 2>&1; then
  open "\$URL"; exit 0
fi
nohup "$ROOT/.venv/bin/python" -u "$ROOT/app.py" >> "$ROOT/.app.log" 2>&1 &
exit 0
RUN
chmod +x "$APP/Contents/MacOS/run"
# 아이콘(.icns) — 렌더러가 있으면 만들고, 없으면 조용히 생략
if command -v qlmanage >/dev/null; then
  T=$(mktemp -d)
  qlmanage -t -s 1024 -o "$T" "$ROOT/assets/icon/gyang_dictation.svg" >/dev/null 2>&1 || true
  PNG=$(ls "$T"/*.png 2>/dev/null | head -1)
  if [ -n "$PNG" ]; then
    mkdir -p "$T/icon.iconset"
    for s in 16 32 128 256 512; do
      sips -z $s $s "$PNG" --out "$T/icon.iconset/icon_${s}x${s}.png" >/dev/null
      sips -z $((s*2)) $((s*2)) "$PNG" --out "$T/icon.iconset/icon_${s}x${s}@2x.png" >/dev/null
    done
    iconutil -c icns "$T/icon.iconset" -o "$APP/Contents/Resources/icon.icns" 2>/dev/null || true
  fi
  rm -rf "$T"
fi
codesign --force -s - "$APP" 2>/dev/null || true  # ad-hoc 서명 — 마이크 권한 재프롬프트 방지
echo "생성: $APP"
