#!/usr/bin/env python3
"""lecture-scribe — 강의 실시간 녹음 + 로컬 Whisper(MLX) 전사 + agy(Antigravity) 한국어 번역 + 라이브 뷰어.

사용:  .venv/bin/python scribe.py                # 시작, Ctrl+C로 종료
       .venv/bin/python scribe.py --lang en      # 언어 고정 (기본: 청크별 한/영 자동 감지)
       .venv/bin/python scribe.py --no-translate # 번역 끄기 (전사만)
       .venv/bin/python scribe.py --list-devices # 오디오 장치 목록

번역 백엔드: agy 사이드카 (galpi PLAN-AI.md 실측 방식).
- 미니멀 에이전트 `scribe`(send_message만) 자동 생성: ~/.gemini/config/agents/scribe/agent.md
- 이력 재전송 비용 때문에 12턴마다 프로세스 재기동
- agy 미로그인/부재 시 번역만 꺼지고 전사는 계속
"""
import argparse, datetime, json, pathlib, queue, re, shutil, signal, subprocess, threading, time, wave, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"
ROOT = pathlib.Path(__file__).resolve().parent
RMS_GATE = 0.003        # 전체 스케일 대비 RMS. 이하면 무음으로 보고 전사 생략
AGY_MODEL = "gemini-3.7-flash-low"
AGY_TURNS_MAX = 12      # 이력 누적 방지: 이 턴 수마다 재기동
JUNK = {"thank you.", "thanks for watching.", "thanks for watching!", "you",
        "시청해주셔서 감사합니다.", "구독과 좋아요 부탁드립니다.", "감사합니다."}

AGENT_MD = """---
name: scribe
description: Lecture transcript translator
tools:
    - send_message
hidden: true
---

# Agent System Instructions

You are a translation engine. You receive numbered chunks of a live lecture transcript
(finance/statistics/CS graduate level, English or mixed English-Korean).
Translate each chunk into natural Korean. Keep technical terms in English in parentheses
when helpful. Output exactly one line per input item, in the form "N. 번역문".
No other text. Never call tools other than send_message.
"""

HTML = """<!doctype html>
<meta charset="utf-8">
<title>lecture-scribe</title>
<style>
body{background:#111;color:#ddd;font:17px/1.6 -apple-system,system-ui,sans-serif;
     max-width:780px;margin:0 auto;padding:0 24px 40vh}
header{position:sticky;top:0;background:#111;padding:12px 0;border-bottom:1px solid #333;
       display:flex;gap:12px;align-items:baseline}
#st{font-size:13px;color:#888}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#4c4;margin-right:6px}
.dead{background:#c44}
p{margin:12px 0}
time{color:#777;font-size:12px;margin-right:8px;font-variant-numeric:tabular-nums}
.ko{display:block;color:#7fa8c9;font-size:15px;margin-top:2px}
p:last-of-type{color:#fff}
</style>
<header><b>lecture-scribe</b><span id="st"></span></header>
<div id="log"></div>
<script>
let rev=-1;
const esc=t=>t.replace(/&/g,'&amp;').replace(/</g,'&lt;');
async function tick(){
 try{
  const s=await (await fetch('/t')).json();
  document.getElementById('st').innerHTML=
    '<span class="dot'+(s.status==='recording'?'':' dead')+'"></span>'+s.status+
    ' · 시작 '+s.start+' · '+s.lines.length+'청크 · 번역 '+s.trans;
  if(s.rev!==rev){
    const atBottom=window.innerHeight+window.scrollY>=document.body.scrollHeight-120;
    document.getElementById('log').innerHTML=
      s.lines.map(l=>'<p><time>'+l.t+'</time>'+esc(l.text)+
        (l.ko?'<span class="ko">'+esc(l.ko)+'</span>':'')+'</p>').join('');
    rev=s.rev;
    if(atBottom)window.scrollTo(0,document.body.scrollHeight);
  }
 }catch(e){}
 setTimeout(tick,2000);
}
tick();
</script>
"""

state = {"status": "starting", "start": "", "trans": "준비", "rev": 0, "lines": []}
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/t":
            with lock:
                body = json.dumps(state, ensure_ascii=False).encode()
            ct = "application/json; charset=utf-8"
        else:
            body = HTML.encode()
            ct = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def seg_rms(path):
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data.astype(np.float32) ** 2))) / 32768.0


def hangul_dominant(text):
    h = sum(1 for c in text if "가" <= c <= "힣")
    a = sum(1 for c in text if c.isascii() and c.isalpha())
    return h > a


# ---------- agy 사이드카 ----------

class Agy:
    """stream-json 1턴 왕복. 이벤트 형태 {"event":"x","x":{...}} (galpi 실측)."""

    def __init__(self, binary, workdir):
        self.bin, self.workdir = binary, workdir
        self.proc, self.q, self.turns = None, None, 0

    def _reader(self, proc, q):
        for line in proc.stdout:
            try:
                j = json.loads(line)
            except ValueError:
                continue
            ev = j.get("event")
            q.put((ev, j.get(ev) or {}))
        q.put(("eof", {}))

    def start(self):
        self.q = queue.Queue()
        self.turns = 0
        self.proc = subprocess.Popen(
            [self.bin, "--print=", "--agent", "scribe", "--model", AGY_MODEL,
             "--input-format", "stream-json", "--output-format", "stream-json"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, cwd=self.workdir)
        threading.Thread(target=self._reader, args=(self.proc, self.q), daemon=True).start()
        deadline = time.time() + 40
        while time.time() < deadline:  # init까지 대기 (콜드 ~6초)
            try:
                ev, _ = self.q.get(timeout=deadline - time.time())
            except queue.Empty:
                break
            if ev == "init":
                return True
            if ev == "eof":
                break
        self.stop()
        return False

    def turn(self, prompt, timeout=120):
        if self.proc is None or self.proc.poll() is not None or self.turns >= AGY_TURNS_MAX:
            self.stop()
            if not self.start():
                return None
        self.turns += 1
        try:
            self.proc.stdin.write(json.dumps(
                {"event": "user", "message": {"content": prompt}}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self.stop()
            return None
        text, deadline = "", time.time() + timeout
        while True:
            try:
                ev, body = self.q.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                self.stop()
                return None
            if ev == "step_update" and body.get("step_type") == "agent_response" \
                    and body.get("text_delta"):
                text += body["text_delta"]
            elif ev == "result":
                if body.get("status") == "SUCCESS":
                    return text or body.get("response") or ""
                return None
            elif ev == "eof":
                self.stop()
                return None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None


def ensure_agent():
    d = pathlib.Path.home() / ".gemini/config/agents/scribe"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "agent.md"
    if not f.exists() or f.read_text() != AGENT_MD:
        f.write_text(AGENT_MD)


def translator_loop(trans_stop, flush_evt, sess):
    binary = shutil.which("agy") or str(pathlib.Path.home() / ".local/bin/agy")
    if not pathlib.Path(binary).exists():
        with lock:
            state["trans"] = "꺼짐(agy 없음)"
        return
    ensure_agent()
    work = sess / ".agy"
    work.mkdir(exist_ok=True)
    agy = Agy(binary, str(work))
    fails = 0
    with lock:
        state["trans"] = "대기"
    while not trans_stop.is_set() and fails < 3:
        with lock:
            pend = [l for l in state["lines"] if l["ko"] is None]
        if not pend:
            trans_stop.wait(2)
            continue
        # 배치 3개가 모이거나 25초 넘게 기다린 게 있어야 턴을 쓴다 (턴당 기본입력 4k 절약)
        if len(pend) < 3 and time.time() - pend[0]["at"] < 25 and not flush_evt.is_set():
            trans_stop.wait(2)
            continue
        batch = pend[:6]
        prompt = "Translate:\n" + "\n".join(f"{k+1}. {l['text']}" for k, l in enumerate(batch))
        out = agy.turn(prompt)
        if out is None:
            fails += 1
            print(f"번역 턴 실패 ({fails}/3) — agy 로그인 상태 확인 (터미널에서 agy 실행)")
            trans_stop.wait(5)
            continue
        got = {}
        for line in out.splitlines():
            m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
            if m:
                got[int(m.group(1))] = m.group(2).strip()
        if not got:
            fails += 1
            trans_stop.wait(5)
            continue
        fails = 0
        with lock:
            for k, l in enumerate(batch):
                l["ko"] = got.get(k + 1, "")
            state["rev"] += 1
            state["trans"] = "작동"
    agy.stop()
    if fails >= 3:
        with lock:
            state["trans"] = "중단(3연속 실패)"
        print("번역 3연속 실패 — 번역 비활성화. 전사는 계속된다.")


# ---------- 메인 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=":0", help="avfoundation 오디오 장치 (기본 :0 = 내장 마이크)")
    ap.add_argument("--seg", type=int, default=15, help="청크 길이(초)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lang", default=None, help="언어 고정(en/ko). 기본: 청크별 자동 감지")
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="브라우저 자동 열기 생략")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation",
                        "-list_devices", "true", "-i", ""])
        return

    start = datetime.datetime.now()
    sess = ROOT / "transcripts" / start.strftime("%Y-%m-%d_%H%M")
    sess.mkdir(parents=True, exist_ok=True)
    md = sess / "transcript.md"
    md.write_text(f"# 강의 전사 {start:%Y-%m-%d %H:%M}\n\n")
    state["start"] = start.strftime("%H:%M:%S")

    print("모델 로드 중 (첫 실행이면 ~1.6GB 다운로드)...")
    mlx_whisper.transcribe(np.zeros(16000, np.float32), path_or_hf_repo=MODEL, language="en")
    print("모델 준비 완료.")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    trans_stop, flush_evt = threading.Event(), threading.Event()
    if args.no_translate:
        state["trans"] = "꺼짐"
        trans_thread = None
    else:
        trans_thread = threading.Thread(target=translator_loop,
                                        args=(trans_stop, flush_evt, sess), daemon=True)
        trans_thread.start()

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "avfoundation", "-i", args.device,
         "-ac", "1", "-ar", "16000",
         "-f", "segment", "-segment_time", str(args.seg),
         "-reset_timestamps", "1", str(sess / "seg_%04d.wav")],
        stdin=subprocess.DEVNULL)

    url = f"http://127.0.0.1:{args.port}"
    print(f"녹음 시작. 뷰어: {url}  (Ctrl+C로 종료)")
    print(f"기록: {md}")
    if not args.no_open:
        webbrowser.open(url)
    with lock:
        state["status"] = "recording"
        state["rev"] += 1

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    done = set()

    def process(p, idx):
        ts = (start + datetime.timedelta(seconds=idx * args.seg)).strftime("%H:%M:%S")
        done.add(p.name)
        try:
            if seg_rms(p) < RMS_GATE:
                return
            r = mlx_whisper.transcribe(str(p), path_or_hf_repo=MODEL,
                                       language=args.lang,
                                       condition_on_previous_text=False)
            if args.lang is None and r["language"] not in ("en", "ko"):
                r = mlx_whisper.transcribe(str(p), path_or_hf_repo=MODEL,
                                           language="en",
                                           condition_on_previous_text=False)
        except Exception as e:
            print(f"[{ts}] 전사 실패: {e}")
            return
        text = " ".join(
            s["text"].strip() for s in r["segments"]
            if s.get("no_speech_prob", 0) < 0.6
            and s.get("compression_ratio", 0) < 2.4       # 반복 환각 차단
            and s["text"].strip())
        if not text or text.lower() in JUNK:
            return
        # 한국어가 지배적인 청크는 번역 불필요("") — 판정은 whisper lang이 아니라 글자 비율
        ko = "" if hangul_dominant(text) else None
        with lock:
            state["lines"].append({"t": ts, "text": text, "ko": ko, "at": time.time()})
            state["rev"] += 1
        with md.open("a") as f:
            f.write(f"**{ts}** {text}\n\n")

    def sweep(final):
        segs = sorted(sess.glob("seg_*.wav"))
        for i, p in enumerate(segs):
            if p.name in done:
                continue
            if final or i + 1 < len(segs):  # 다음 청크가 생겨야 이 청크가 완결
                process(p, i)

    while not stop.is_set():
        if ffmpeg.poll() is not None:
            print("ffmpeg 종료됨 — 마이크 권한/장치를 확인 (--list-devices).")
            with lock:
                state["status"] = "error"
                state["rev"] += 1
            break
        sweep(final=False)
        stop.wait(0.5)

    with lock:
        state["status"] = "finishing"
        state["rev"] += 1
    if ffmpeg.poll() is None:
        ffmpeg.terminate()
        ffmpeg.wait()
    sweep(final=True)

    # 남은 번역을 최대 90초 기다린다
    flush_evt.set()
    if trans_thread is not None:
        deadline = time.time() + 90
        while time.time() < deadline and trans_thread.is_alive():
            with lock:
                if all(l["ko"] is not None for l in state["lines"]):
                    break
            time.sleep(1)
    trans_stop.set()

    # 최종 md 재작성 (원문 + 번역)
    with lock:
        lines = [dict(l) for l in state["lines"]]
        state["status"] = "done"
        state["rev"] += 1
    with md.open("w") as f:
        f.write(f"# 강의 전사 {start:%Y-%m-%d %H:%M}\n\n")
        for l in lines:
            f.write(f"**{l['t']}** {l['text']}\n")
            if l["ko"]:
                f.write(f"> {l['ko']}\n")
            f.write("\n")
    print(f"종료. {len(lines)}개 청크 기록 → {md}")
    time.sleep(2)  # 뷰어가 마지막 상태를 받아가게 잠깐 유지
    server.shutdown()


if __name__ == "__main__":
    main()
