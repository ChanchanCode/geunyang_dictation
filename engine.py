#!/usr/bin/env python3
"""그냥 받아쓰기 — 엔진.

오디오(ffmpeg 마이크, s16le 16k mono)
 ├─ [라이브]  parakeet-mlx 스트리밍(tdt-0.6b-v3) → 1~3초 지연 회색 표시(초안, 확정 아님)
 └─ [확정]   VAD 경계 청크(침묵에서 자름, 고정 컷 없음) → 워커 프로세스의
             mlx-whisper large-v3-turbo, 언어 자동(en/ko) → 검정 확정 라인
확정 라인만 agy(Antigravity) 번역: 실시간 배치 + 종료 시 전체 문맥 교정 패스.
세션은 transcripts/<YYYY-MM-DD_HHMM>/ 에 transcript.md·lines.json·meta.json·audio.m4a.

워커 모드: `python engine.py --worker` (stdin: 헤더 JSON 줄 + float32 raw, stdout: 결과 JSON 줄)
"""
import asyncio, datetime, json, logging, os, pathlib, queue, re, shutil, socket, struct, subprocess, sys, threading, time
import numpy as np

logging.basicConfig(level=logging.ERROR)
for _n in ("whisperlivekit",):
    logging.getLogger(_n).setLevel(logging.ERROR)

ROOT = pathlib.Path(__file__).resolve().parent
TR = ROOT / "transcripts"
CONFIG = ROOT / "config.json"
VERSION = "1.2.1"          # release.sh 가 여기를 올린다
UPDATE_REPO = "ChanchanCode/geunyang_dictation"
MODEL = "mlx-community/whisper-large-v3-turbo"
SR = 16000
FFMPEG = shutil.which("ffmpeg") or next(
    (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
     if pathlib.Path(p).exists()), "ffmpeg")
FFPROBE = shutil.which("ffprobe") or str(pathlib.Path(FFMPEG).with_name("ffprobe"))
AUTOSAVE_SEC = 20        # 녹음 중 lines.json 스냅샷 주기 — 크래시가 나도 이만큼만 잃는다
MEDIA_EXT = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4",
             ".mov", ".mkv", ".m4v", ".caf", ".aiff", ".aif", ".wma"}
AGY_MODEL = "gemini-3.8-flash-low"          # 실시간 번역 — 짧은 배치, 속도·쿼터 우선
AGY_MODEL_HEAVY = "gemini-3.8-flash-medium" # 다듬기·요약·자료 변환 — 긴 문맥, 정확도 우선
AGY_TURNS_MAX = 12
AGY_RETRY_WAIT = (10, 30, 60)  # 다듬기·요약: 인터넷은 되는데 답이 없을 때 다시 시도 간격(초)
NET_WAIT_MAX = 1800     # 인터넷이 끊기면 이만큼(초) 기다렸다 이어서 한다. 넘으면 멈추고 '다시 시도'
JOB_RESUME_MAX = 2      # 앱이 꺼져 끊긴 다듬기·요약을 다음 실행 때 자동으로 이어 하는 최대 횟수
CTX_MAX = 1200           # 세션 맥락(meta.context) 최대 길이
CTX_FILE_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".txt", ".md"}
CTX_FILES_MAX_MB = 60
CTX_TEXT_DIR = "ctx_text"   # 자료(PDF·이미지) → 텍스트 변환본. 다듬기·요약이 참고한다
MAT_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic"}  # 자료 폴더로 복사하는 종류
DOW = "월화수목금토일"
JUNK = {"thank you.", "thanks for watching.", "thanks for watching!", "you", "bye.",
        "시청해주셔서 감사합니다.", "구독과 좋아요 부탁드립니다.", "감사합니다."}

# ---------- 사용 언어 ----------
# Whisper 는 99개 언어를 자동감지한다. 후보를 넓게 두면 영어 강의 중 튀어나온 한국어
# 한두 문장이 일본어·중국어로 찍히고, 그 뒤 전사가 통째로 어긋난다. 그래서 감지 결과를
# 버리는 대신 후보 자체를 사용자가 고른 언어로 좁힌다.
MAJOR_LANGS = [("en", "영어"), ("ko", "한국어"), ("ja", "일본어"), ("zh", "중국어"),
               ("es", "스페인어"), ("fr", "프랑스어"), ("de", "독일어"), ("ru", "러시아어"),
               ("pt", "포르투갈어"), ("it", "이탈리아어"), ("hi", "힌디어"), ("ar", "아랍어"),
               ("vi", "베트남어"), ("id", "인도네시아어")]
LANG_CODES = [c for c, _ in MAJOR_LANGS]
DEFAULT_LANGS = ["en", "ko"]

def norm_langs(val):
    """허용 코드만 남기고 MAJOR_LANGS 순서로 정렬·중복 제거. 비면 None."""
    if not isinstance(val, (list, tuple)):
        return None
    out = [c for c in LANG_CODES if c in val]
    return out or None


# ---------- 워커 프로세스 (확정 전사 전용 — MLX 컨텍스트 격리) ----------

def pick_lang(audio, langs):
    """허용 언어 안에서만 언어를 고른다. transcribe 가 이미 로드한 모델 인스턴스를
    재사용하므로 추가 메모리는 없다(인코더 패스 1회 비용만 든다)."""
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder
    from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES
    model = ModelHolder.get_model(MODEL, mx.float16)
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
    mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
    _, probs = model.detect_language(mel)
    return max(langs, key=lambda c: probs.get(c, 0.0))


def worker_main():
    import mlx_whisper
    mlx_whisper.transcribe(np.zeros(SR, np.float32), path_or_hf_repo=MODEL, language="en")  # warmup
    sys.stdout.write(json.dumps({"ready": True}) + "\n"); sys.stdout.flush()
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return
        hdr = json.loads(line)
        n = hdr["n"]
        prompt = hdr.get("prompt") or None
        langs = norm_langs(hdr.get("langs")) or DEFAULT_LANGS
        audio = np.frombuffer(sys.stdin.buffer.read(n * 4), dtype=np.float32)
        out = {"text": "", "lang": langs[0]}
        try:
            def tr(lang):
                return mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, language=lang,
                                              condition_on_previous_text=False,
                                              initial_prompt=prompt)
            if len(langs) == 1:
                r = tr(langs[0])          # 단일 선택 — 감지 자체가 필요 없다
            else:
                r = tr(None)
                if r["language"] not in langs:   # 후보 밖으로 샜다 → 허용 언어 안에서 재선택
                    r = tr(pick_lang(audio, langs))
            text = " ".join(s["text"].strip() for s in r["segments"]
                            if s.get("no_speech_prob", 0) < 0.6
                            and s.get("compression_ratio", 0) < 2.4 and s["text"].strip())
            if text.lower().strip() in JUNK:
                text = ""
            out = {"text": text, "lang": r["language"]}
        except Exception as e:
            out["error"] = str(e)[:200]
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n"); sys.stdout.flush()


class FixWorker:
    """확정 전사 워커 프로세스 프록시. 직렬 호출."""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()

    def start(self):
        self.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "engine.py"), "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=str(ROOT))
        line = self.proc.stdout.readline()  # ready 대기 (워밍업 포함)
        return bool(line) and json.loads(line).get("ready")

    def transcribe(self, audio_f32, prompt="", langs=None):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                if not self.start():
                    return None
            try:
                b = audio_f32.astype(np.float32).tobytes()
                self.proc.stdin.write((json.dumps(
                    {"n": len(b) // 4, "prompt": prompt,
                     "langs": langs or DEFAULT_LANGS}) + "\n").encode())
                self.proc.stdin.write(b); self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                if not line:
                    raise IOError("worker died")
                return json.loads(line)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.proc = None
                return None

    def stop(self):
        with self.lock:   # 전사 중인 호출(복구 스레드 등)이 끝난 뒤에 내린다
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
            self.proc = None


# ---------- agy 사이드카 (galpi PLAN-AI 실측 프로토콜) ----------

def acct_home(alias):
    """gemini-acct 와 동일: main=실제 홈, 그 외=gemini-accounts/<alias> 가짜 홈."""
    if alias in ("main", "", None):
        return pathlib.Path.home()
    return ACCT_STORE / alias


def _ensure_keychain(h):
    """계정 키체인: unlock 불가(login.keychain-db 이름 특별취급) → 부팅마다 신선 재생성.
    갓 만든 키체인은 그 부팅 세션 동안 저장 가능. 내용물은 agy 캐시일 뿐(토큰 정본은 파일)."""
    kc = h / "Library/Keychains/login.keychain-db"
    stamp = h / ".keychain-boot"
    try:
        boot = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True,
                              text=True, timeout=5).stdout.splitlines()[0]
    except Exception:
        return
    try:
        if kc.exists() and stamp.read_text() == boot:
            return
    except Exception:
        pass
    try:
        kc.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["security", "delete-keychain", str(kc)], capture_output=True, timeout=10)
        kc.unlink(missing_ok=True)
        r = subprocess.run(["security", "create-keychain", "-p", "", str(kc)],
                           capture_output=True, timeout=10)
        if r.returncode != 0:
            return
        subprocess.run(["security", "set-keychain-settings", str(kc)],
                       capture_output=True, timeout=10)
        stamp.write_text(boot)
        # create-keychain 이 실계정 검색 목록에 자기를 등록하는 부작용 제거(다른 항목 보존)
        r = subprocess.run(["security", "list-keychains", "-d", "user"],
                           capture_output=True, text=True, timeout=10)
        keep = [x.strip().strip('"') for x in r.stdout.splitlines()
                if x.strip() and "gemini-accounts" not in x]
        if keep:
            subprocess.run(["security", "list-keychains", "-d", "user", "-s", *keep],
                           capture_output=True, timeout=10)
    except Exception:
        pass


def _acct_env(alias):
    """alias 계정으로 agy 를 띄울 env. main 이면 None(상속)."""
    if alias in ("main", "", None):
        return None
    h = acct_home(alias)
    if not h.is_dir():
        return None
    _ensure_keychain(h)
    return {**os.environ, "HOME": str(h)}


def agy_accounts():
    """계정 목록: main + gemini-accounts/*. 로그인 여부는 oauth 토큰 존재로 판정."""
    cur = load_cfg().get("agy_account", "main")
    homes = [("main", pathlib.Path.home())]
    if ACCT_STORE.is_dir():
        homes += [(d.name, d) for d in sorted(ACCT_STORE.iterdir()) if d.is_dir()]
    out = []
    for a, h in homes:
        try:
            email = (h / ".gemini/.acct-email").read_text().strip() or "-"
        except Exception:
            email = "-"
        tok = h / ".gemini/antigravity-cli/antigravity-oauth-token"
        # main 은 토큰이 macOS 키체인에 있어 파일로 판정 불가 → None(미상)
        logged = None if a == "main" else (tok.exists() and tok.stat().st_size > 0)
        out.append({"alias": a, "email": email, "logged_in": logged,
                    "current": a == cur})
    return out


_net = {"t": 0.0, "ok": True, "seen": False}

def online():
    """인터넷 연결 여부 (구글 443 접속, 5초 캐시). 이 프로세스에서 한 번도 접속된 적이 없으면
    확인 방법 자체가 막힌 환경일 수 있으니 판단하지 않고 True — 기다리기·중단을 걸지 않는다."""
    if time.time() - _net["t"] >= 5:
        try:
            socket.create_connection(("www.google.com", 443), timeout=3).close()
            _net.update(ok=True, seen=True)
        except OSError:
            _net["ok"] = False
        _net["t"] = time.time()
    return _net["ok"] or not _net["seen"]

def offline():
    return not online()


class Agy:
    """agy 사이드카. agent=None 이면 기본 에이전트(요약·자료 변환처럼 자유 형식 답이 필요할 때).
    add_dir 를 주면 그 폴더의 파일을 Gemini 가 view_file 로 직접 읽는다(PDF·이미지)."""

    def __init__(self, workdir, model=AGY_MODEL, agent="scribe", add_dir=None, print_timeout="5m"):
        self.bin = shutil.which("agy") or str(pathlib.Path.home() / ".local/bin/agy")
        self.workdir = workdir
        self.model, self.agent, self.add_dir, self.print_timeout = model, agent, add_dir, print_timeout
        self.proc, self.q, self.turns = None, None, 0
        self.account = None

    def available(self):
        return pathlib.Path(self.bin).exists()

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
        if self.agent:
            ensure_agent()
        self.q = queue.Queue(); self.turns = 0
        self.account = load_cfg().get("agy_account", "main")
        args = [self.bin, "--print=", "--model", self.model,
                "--input-format", "stream-json", "--output-format", "stream-json",
                "--print-timeout", self.print_timeout]
        if self.agent:
            args += ["--agent", self.agent]
        if self.add_dir:
            args += ["--add-dir", str(self.add_dir), "--dangerously-skip-permissions"]
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, cwd=self.workdir, env=_acct_env(self.account))
        threading.Thread(target=self._reader, args=(self.proc, self.q), daemon=True).start()
        deadline = time.time() + 40
        while time.time() < deadline:
            try:
                ev, _ = self.q.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                break
            if ev == "init":
                return True
            if ev == "eof":
                break
        self.stop(); return False

    def turn(self, prompt, timeout=180, preamble="", abort=None):
        """preamble 은 프로세스가 새로 뜬 첫 턴에만 앞에 붙는다(긴 참고자료를 매 턴 다시 보내지 않게).
        abort() 가 15초 간격으로 세 번 연속 참이면(인터넷 끊김 등) timeout 까지 기다리지 않고 포기한다."""
        if self.proc is None or self.proc.poll() is not None or self.turns >= AGY_TURNS_MAX:
            self.stop()
            if not self.start():
                return None
        if preamble and self.turns == 0:
            prompt = preamble + prompt
        self.turns += 1
        try:
            self.proc.stdin.write(json.dumps(
                {"event": "user", "message": {"content": prompt}}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self.stop(); return None
        text, deadline, bad = "", time.time() + timeout, 0
        while True:
            left = deadline - time.time()
            try:
                ev, body = self.q.get(timeout=max(0.1, min(left, 15) if abort else left))
            except queue.Empty:
                if abort and time.time() < deadline:
                    bad = bad + 1 if abort() else 0
                    if bad < 3:
                        continue
                self.stop(); return None
            if ev == "step_update" and body.get("step_type") == "agent_response" \
                    and body.get("text_delta"):
                text += body["text_delta"]
            elif ev == "result":
                return (text or body.get("response") or "") if body.get("status") == "SUCCESS" else None
            elif ev == "eof":
                self.stop(); return None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate(); self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None


AGENT_MD = """---
name: scribe
description: Lecture transcript translator
tools:
    - send_message
hidden: true
---

# Agent System Instructions

You are a translation engine. You receive numbered chunks of a live lecture transcript
(graduate level, English or mixed English-Korean).
Translate each chunk into natural Korean. Keep technical terms in English in parentheses
when helpful. Output exactly one line per input item, in the form "N. 번역문".
No other text. Never call tools other than send_message.
"""

def ensure_agent():
    homes = [pathlib.Path.home()]
    alias = load_cfg().get("agy_account", "main")
    if alias != "main" and acct_home(alias).is_dir():
        homes.append(acct_home(alias))
    for h in homes:
        d = h / ".gemini/config/agents/scribe"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "agent.md"
        if not f.exists() or f.read_text() != AGENT_MD:
            f.write_text(AGENT_MD)


DIGEST_PROMPT = """You prepare a short ENGLISH context note for a speech recognizer (Whisper) and a
translator working on a live university lecture (graduate level). The materials below may be in Korean;
translate them. Output ONLY the note, English only, plain text, no markdown, max 900 characters:
Line 1: course / lecture topic in one sentence.
Line 2: "Terms: " then the key technical terms, proper nouns, acronyms (expanded once) and names
likely to be spoken, comma-separated, most important first (up to ~40 items).
Line 3 (optional): "Notes: " one sentence on the lecture agenda or structure.
"""

def digest_context(sess, text=""):
    """맥락 자료(한국어 메모·PDF·이미지) → agy(Gemini)로 영어 맥락 노트. (note, err) 반환.
    파일은 sess/ctx/ 에 있고 --add-dir 로 넘겨 Gemini 가 view_file 로 직접 읽는다(PDF·이미지 지원 실측)."""
    work = sess / "ctx"
    files = sorted(p.name for p in work.iterdir() if p.is_file() and not p.name.startswith(".")) \
        if work.is_dir() else []
    text = (text or "").strip()
    if not files and not text:
        return None, "정리할 자료가 없어요"
    binary = shutil.which("agy") or str(pathlib.Path.home() / ".local/bin/agy")
    if not pathlib.Path(binary).exists():
        return None, "agy(Antigravity CLI)가 없어 Gemini 정리를 못 해요"
    prompt = DIGEST_PROMPT
    if files:
        prompt += ("\nFiles in the workspace directory (read EVERY one with view_file; PDFs may have many "
                   "pages, skim all of them): " + ", ".join(files) + "\n")
    if text:
        prompt += "\nUser note:\n" + text[:4000] + "\n"
    args = [binary, "-p", prompt, "--model", AGY_MODEL, "--print-timeout", "4m"]
    if files:
        args += ["--add-dir", str(work), "--dangerously-skip-permissions"]
    cwd = ROOT / ".agywork"
    cwd.mkdir(exist_ok=True)
    alias = load_cfg().get("agy_account", "main")
    try:
        r = subprocess.run(args, capture_output=True, text=True, cwd=str(cwd),
                           env=_acct_env(alias), timeout=250)
    except subprocess.TimeoutExpired:
        return None, "Gemini 응답 시간 초과"
    out = re.sub(r"^```.*?\n|\n```$", "", (r.stdout or "").strip(), flags=re.S).strip()
    if r.returncode != 0 or not out:
        err = (r.stderr or "").strip().splitlines()
        return None, ("Gemini 실패: " + err[-1][:120]) if err else "Gemini 실패(agy 로그인 확인)"
    return out[:CTX_MAX], ""


# ---------- 자료(PDF·이미지) → 텍스트 변환 ----------
# 업로드 직후 백그라운드로 돌고, 다듬기·요약이 용어 표기·페이지 대응에 쓴다. ctx_text/<파일명>.md

CONVERT_PROMPT = """Transcribe the document "{name}" in the workspace directory into plain Markdown text,
completely and faithfully, page by page. Read it with view_file (every page).
Rules:
- Before each page's content write one line exactly: --- p.N ---   (N = physical page number, from 1)
- Copy all text on the page in reading order: titles, bullets, tables (as Markdown tables),
  equations (as LaTeX in $...$), captions. Describe each figure/chart in one line: [Figure: ...]
- Do not summarize, skip, translate, or comment. Keep the original language.
- If the document has more than 120 pages, transcribe the first 120 and end with: --- truncated ---
Output only the transcription, no preamble, no code fence.
"""

_conv = {}                       # sid -> {"busy","note","error"}
_conv_lock = threading.Lock()

def conv_status(sid):
    return dict(_conv.get(sid) or {})

def conv_busy():
    return any(v.get("busy") for v in list(_conv.values()))

def ctx_text_files(sess):
    d = sess / CTX_TEXT_DIR
    return sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix == ".md") \
        if d.is_dir() else []

def ctx_pending(sess):
    """아직 텍스트로 변환되지 않은 자료 파일 이름."""
    cd = sess / "ctx"
    if not cd.is_dir():
        return []
    return [p.name for p in sorted(cd.iterdir())
            if p.is_file() and not p.name.startswith(".")
            and not (sess / CTX_TEXT_DIR / (p.name + ".md")).exists()]

def _pdftotext(path):
    """poppler 가 있으면 페이지 단위 로컬 추출(Gemini 실패 시 대체)."""
    if not shutil.which("pdftotext"):
        return ""
    try:
        r = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                           capture_output=True, text=True, timeout=120)
    except Exception:
        return ""
    if r.returncode != 0:
        return ""
    out = []
    for k, pg in enumerate(r.stdout.split("\f"), 1):
        pg = re.sub(r"[ \t]+\n", "\n", pg).strip()
        if pg:
            out.append(f"--- p.{k} ---\n{pg}")
    return "\n\n".join(out)

def _strip_fence(t):
    t = (t or "").strip()
    m = re.match(r"^```[a-zA-Z]*\n(.*?)\n```$", t, re.S)
    return m.group(1).strip() if m else t

def convert_one(sess, name):
    """ctx/<name> → ctx_text/<name>.md. 텍스트 파일은 복사, PDF·이미지는 Gemini(view_file)로
    페이지별 전사. 결과 텍스트(실패 시 '')."""
    src, dst = sess / "ctx" / name, sess / CTX_TEXT_DIR / (name + ".md")
    dst.parent.mkdir(exist_ok=True)
    text = ""
    if src.suffix.lower() in (".txt", ".md"):
        text = src.read_text(errors="replace")
    else:
        agy = Agy(str(ROOT / ".agywork"), model=AGY_MODEL_HEAVY, agent=None,
                  add_dir=sess / "ctx", print_timeout="12m")
        if agy.available():
            out = _strip_fence(agy.turn(CONVERT_PROMPT.format(name=name), timeout=720))
            agy.stop()
            if out and "--- p." in out and len(out) >= 300:
                text = out
        if not text and src.suffix.lower() == ".pdf":
            text = _pdftotext(src)
    if not text.strip():
        return ""
    dst.write_text(text)
    return text

def convert_ctx(sess, note=None):
    """미변환 자료를 순차 변환. note(msg) 로 진행 표시. 실패한 이름 목록 반환.
    다른 스레드가 이미 변환 중이면 곧바로 [] (호출자는 wait_conv 로 기다린다)."""
    sid = sess.name
    pend = ctx_pending(sess)
    if not pend:
        return []
    with _conv_lock:
        if _conv.get(sid, {}).get("busy"):
            return []
        _conv[sid] = {"busy": True, "note": "강의자료 변환 준비 중…", "error": ""}
    failed = []
    try:
        for k, name in enumerate(pend):
            msg = f"강의자료 변환 중… ({k + 1}/{len(pend)}) {name}"
            _conv[sid]["note"] = msg
            if note:
                note(msg)
            try:
                if not convert_one(sess, name):
                    failed.append(name)
            except Exception:
                logging.exception("자료 변환 실패")
                failed.append(name)
    finally:
        _conv[sid] = {"busy": False, "note": "",
                      "error": (f"자료 변환 실패: {', '.join(failed)}"[:120] if failed else "")}
    return failed

def wait_conv(sid, timeout=900):
    t0 = time.time()
    while _conv.get(sid, {}).get("busy") and time.time() - t0 < timeout:
        time.sleep(1)

def ctx_text_all(sess, budget=30000):
    """변환된 자료를 파일별 머리말과 함께 합친다. budget 자 초과분은 자른다."""
    parts = []
    for n in ctx_text_files(sess):
        t = (sess / CTX_TEXT_DIR / n).read_text(errors="replace").strip()
        if t:
            parts.append(f"=== {n[:-3]} ===\n{t}")
    out = "\n\n".join(parts)
    if len(out) > budget:
        out = out[:budget] + "\n… (이하 생략)"
    return out


# ---------- 과목 · 주차 · 제목 추천 · 자료 폴더 ----------

def subject_of(meta):
    """meta.subject 가 있으면 그것, 없으면 제목의 첫 토큰('재무론2 2주차_1' → '재무론2')."""
    s = (meta.get("subject") or "").strip()
    if s:
        return s
    t = (meta.get("title") or "").strip()
    if not t:
        return ""
    return re.sub(r"\(.*$", "", re.split(r"[\s_]+", t)[0]).strip()

def _norm(s):
    return re.sub(r"[\s_\-·.]+", "", s or "").lower()

def _sid_date(sid):
    try:
        return datetime.date.fromisoformat((sid or "")[:10])
    except ValueError:
        return None

def _monday(d):
    return d - datetime.timedelta(days=d.weekday())

def subjects_info(sid=None):
    """과목 목록(최근 사용순) + sid 세션 날짜의 주차·요일. 주차는 가장 최근 'N주차' 제목이
    함의하는 1주차 월요일 기준(사용자 번호 매김을 따른다), 없으면 첫 세션 주를 1주차로."""
    sessions = list_sessions()               # 최신순
    subs, w1, earliest = {}, None, None
    for m in sessions:
        d = _sid_date(m.get("id"))
        if d is None:
            continue
        earliest = d if earliest is None or d < earliest else earliest
        if m.get("id") == sid:
            continue
        s = subject_of(m)
        if s:   # '머신러닝 2' 와 '머신러닝2' 는 한 과목 — 표시는 가장 최근 표기
            e = subs.setdefault(_norm(s), {"name": s, "count": 0, "last": m["id"]})
            e["count"] += 1
        if w1 is None:
            mm = re.search(r"(\d+)\s*주차", m.get("title") or "")
            if mm:
                w1 = _monday(d) - datetime.timedelta(days=7 * (int(mm.group(1)) - 1))
    d = _sid_date(sid) or datetime.date.today()
    if w1 is None:
        w1 = _monday(earliest or d)
    week = max(1, (_monday(d) - w1).days // 7 + 1)
    return {"subjects": list(subs.values()), "week": week, "dow": DOW[d.weekday()]}

def suggest_title(sid, subject):
    """'재무론2_2주차(화)'. 같은 제목이 이미 있으면 _2, _3…"""
    info = subjects_info(sid)
    base = f"{subject}_{info['week']}주차({info['dow']})"
    taken = {m.get("title") for m in list_sessions() if m.get("id") != sid}
    t, k = base, 2
    while t in taken:
        t = f"{base}_{k}"; k += 1
    return t

def guess_subject(name, exclude_sid=None):
    """자료 파일명으로 과목 추정. 파일명에 과목명이 들어 있거나, 이전 세션 자료와
    (숫자를 뺀) 이름 앞부분이 길게 같으면 그 과목. 확신 없으면 ''."""
    nn = _norm(pathlib.Path(name).stem)
    strip = lambda s: re.sub(r"\d+", "", _norm(pathlib.Path(s).stem))
    best = ("", 0)
    for m in list_sessions():
        if m.get("id") == exclude_sid:
            continue
        s = subject_of(m)
        if not s:
            continue
        if _norm(s) and _norm(s) in nn:
            return s
        cd = TR / m["id"] / "ctx"
        if cd.is_dir():
            for p in cd.iterdir():
                a, b = strip(name), strip(p.name)
                k = len(os.path.commonprefix([a, b]))
                if k >= 10 and k >= 0.6 * min(len(a), len(b)) and k > best[1]:
                    best = (s, k)
    return best[0]

def file_materials(sess, meta, names=None):
    """과목이 정해진 세션의 자료(PDF·이미지)를 설정한 자료 폴더/<과목>/ 로 복사.
    과목 폴더는 공백·대소문자 무시로 찾고 없으면 만든다. 같은 이름이 있으면 건너뛴다.
    자료 폴더 미설정·과목 미지정이면 아무것도 안 한다. 복사한 이름 목록 반환."""
    root, subj = load_cfg().get("mat_dir") or "", subject_of(meta)
    if not root or not subj:
        return []
    root = pathlib.Path(root).expanduser()
    if not root.is_dir():
        return []
    key = _norm(subj)
    dst = next((d for d in sorted(root.iterdir()) if d.is_dir() and _norm(d.name) == key),
               None) or (root / subj)
    cd, done = sess / "ctx", []
    if not cd.is_dir():
        return []
    for p in sorted(cd.iterdir()):
        if not p.is_file() or p.name.startswith(".") or p.suffix.lower() not in MAT_EXT:
            continue
        if names is not None and p.name not in names:
            continue
        t = dst / p.name
        if t.exists():
            continue
        try:
            dst.mkdir(exist_ok=True)
            shutil.copy2(p, t); done.append(p.name)
        except OSError:
            logging.exception("자료 폴더 복사 실패")
    return done


# ---------- 과목 용어집 (지난 세션의 맥락 Terms 줄 + 요약의 핵심 용어) ----------

def _summary_terms(md):
    """summary.md '## 핵심 용어' 의 '- **term (한국어)** — 설명' → ['term', ...]."""
    out, on = [], False
    for line in (md or "").splitlines():
        s = line.strip()
        if s.startswith("## "):
            on = "용어" in s
            continue
        m = on and re.match(r"^[-*]\s+\*\*(.+?)\*\*", s)
        if m:
            t = re.sub(r"\s*\([^)]*[가-힣][^)]*\)\s*", " ", m.group(1)).strip()
            if t:
                out.append(t)
    return out

def subject_terms(subject, exclude_sid=None, limit=80):
    """같은 과목 지난 세션들에서 모은 용어. 여러 강의에 반복된 용어가 앞, 같으면 최근 것이 앞."""
    key = _norm(subject)
    if not key:
        return []
    count, order, name = {}, [], {}
    for m in list_sessions():                 # 최신순
        if m.get("id") == exclude_sid or _norm(subject_of(m)) != key:
            continue
        src = []
        mm = re.search(r"^Terms:\s*(.+)$", m.get("context") or "", re.M | re.I)
        if mm:
            src += mm.group(1).split(",")
        sf = TR / m["id"] / "summary.md"
        if sf.exists():
            try:
                src += _summary_terms(sf.read_text())
            except OSError:
                pass
        seen = set()
        for t in src:
            t = t.strip().strip(".;:")
            k = t.lower()
            if not (2 <= len(t) <= 60) or k in seen:
                continue
            seen.add(k)
            if k not in count:
                count[k] = 0; order.append(k); name[k] = t
            count[k] += 1
    ranked = sorted(order, key=lambda k: (-count[k], order.index(k)))
    return [name[k] for k in ranked[:limit]]

def subject_terms_text(meta, sid, budget):
    terms = subject_terms(subject_of(meta), sid)
    s = ""
    for t in terms:
        if len(s) + len(t) + 2 > budget:
            break
        s += (", " if s else "") + t
    return s


# ---------- 본문 검색 ----------

_search_cache = {}   # path -> (mtime, data)

def _cached(path, loader):
    try:
        mt = path.stat().st_mtime
    except OSError:
        return None
    c = _search_cache.get(str(path))
    if c and c[0] == mt:
        return c[1]
    try:
        data = loader(path)
    except Exception:
        data = None
    _search_cache[str(path)] = (mt, data)
    return data

def _snip(text, toks, width=46):
    low = text.lower()
    at = min((low.find(t) for t in toks if low.find(t) >= 0), default=0)
    a, b = max(0, at - width), min(len(text), at + width + 40)
    return ("…" if a else "") + text[a:b].strip() + ("…" if b < len(text) else "")

def search_sessions(q, per=6, max_sessions=40):
    """제목·전사(원문·번역)·요약에서 찾는다. 공백으로 나눈 낱말이 모두 들어간 줄만."""
    toks = [t for t in (q or "").lower().split() if t]
    if not toks:
        return []
    hit = lambda s: all(t in s.lower() for t in toks)
    out = []
    for m in list_sessions():
        sid = m.get("id")
        if not sid:
            continue
        sess = TR / sid
        hits, n = [], 0
        lines = _cached(sess / "lines.json", lambda p: json.loads(p.read_text())) or []
        for k, l in enumerate(lines):
            for f in ("text", "ko"):
                v = l.get(f) or ""
                if v and hit(v):
                    n += 1
                    if len(hits) < per:
                        hits.append({"i": l.get("i", k), "t": l.get("t", ""), "f": f, "s": _snip(v, toks)})
                    break
        md = _cached(sess / "summary.md", lambda p: p.read_text()) or ""
        sec = -1
        for line in md.splitlines():
            if line.startswith("## "):
                sec += 1
            s = re.sub(r"[*`#>]|^\s*[-•]\s*", "", line).strip()
            if s and not s.startswith("[줄") and hit(s):
                n += 1
                if len(hits) < per:
                    hits.append({"f": "sum", "k": sec, "s": _snip(s, toks)})
        title = m.get("title") or ""
        if n or hit(title + " " + sid):
            out.append({"id": sid, "title": title, "subject": subject_of(m), "n": n, "hits": hits})
            if len(out) >= max_sessions:
                break
    return out


# ---------- 과목 정리본 (주차별 요약 → 한 과목 노트) ----------

COURSE_DIR = TR / ".course"

def course_key(subject):
    return _norm(subject) or "_"

def course_sessions(subject):
    """같은 과목의 전사가 있는 세션, 오래된 순."""
    key = course_key(subject)
    return [m for m in reversed(list_sessions())
            if _norm(subject_of(m)) == key and (TR / m["id"] / "lines.json").exists()]

def course_info(subject):
    key = course_key(subject)
    md_p, meta_p = COURSE_DIR / f"{key}.md", COURSE_DIR / f"{key}.json"
    sess = course_sessions(subject)
    rows = [{"id": m["id"], "title": m.get("title") or m["id"],
             "summary": (TR / m["id"] / "summary.md").exists(),
             "summary_at": m.get("summary_at", "")} for m in sess]
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:
        meta = {}
    md = md_p.read_text() if md_p.exists() else ""
    used = meta.get("used") or {}
    stale = bool(md) and any(r["id"] not in used or (r["summary"] and used[r["id"]] != r["summary_at"])
                             for r in rows)
    return {"subject": subject, "md": md, "at": meta.get("at", ""), "sessions": rows,
            "stale": stale, "outline": parse_outline(md)}


POLISH_HEAD = """다음은 대학원 강의를 자동 받아쓰기(ASR)한 결과의 일부다. 줄마다 원문을 교정한다.

원문 교정 (영어, 가끔 한국어 섞임) — 번역문만큼 원문도 매끄럽게 읽히도록 정리한다:
- 잘못 들린 단어·구절을 앞뒤 문맥과 강의자료에 맞게 바로잡는다: 소리는 비슷한데 뜻이 안 통하는 단어,
  깨진 전문용어·고유명사·수식 표현, 잘못 붙거나 끊긴 어절, 어긋난 시제·수 일치, 빠지거나 엉뚱한 문장부호.
- 받아쓰기 잡음을 걷어낸다: 말더듬과 같은 말 반복("the the", "I think I think"), 뜻 없는 간투사
  (um, uh, you know, like, sort of — 뜻이 있으면 남긴다), 시작했다 만 어구.
- 전문용어·이름·기호는 강의자료 표기를 따른다.
- 화자가 한 말·순서·상세도는 지킨다. 요약·축약·확장·의역·내용 추가 금지.
- 줄을 합치거나 나누지 마라. 입력이 N줄이면 출력도 N줄, 번호 그대로.
- 복구 불가능하게 뭉개진 줄은 그대로 둔다.
"""
POLISH_KO = """
번역:
- 전체 문맥과 용어 일관성을 지킨 자연스러운 한국어. 전문용어는 영어 그대로 둬도 된다.

출력 형식 — 각 줄을 정확히 이 꼴로, 다른 말은 쓰지 마라:
N. <교정한 원문> ||| <한국어 번역>
"""
POLISH_EN = """
출력 형식 — 각 줄을 정확히 이 꼴로, 다른 말은 쓰지 마라:
N. <교정한 원문>
"""

def polish_preamble(ctx, slides, terms=""):
    """새 사이드카 프로세스의 첫 턴에만 붙는 참고자료(맥락 노트 + 과목 용어 + 강의자료 본문)."""
    s = f"수업 맥락: {ctx}\n\n" if ctx else ""
    if terms:
        s += f"이 과목 지난 강의에서 쓰인 용어 (소리가 비슷하게 잘못 들린 단어는 이 표기로): {terms}\n\n"
    if slides:
        s += ("강의자료 본문 (참고용 — 용어·표기·수식은 이걸 따른다. 자료에만 있고 화자가 말하지 않은 "
              "내용을 끌어오지는 마라):\n" + slides + "\n\n")
    return s


SUMMARY_PROMPT = """너는 대학원 강의 노트를 만드는 조교다. 아래 강의 전사(자동 받아쓰기를 교정한 것)와
강의자료를 읽고, 한국어로 '강의 요약·정리본'을 Markdown 으로 쓴다.

원칙:
- 담백하게. 미사여구·감상·자기평가 없이 내용만. 대학원 수준 독자라 기초 개념 설명은 생략한다.
- 전문용어는 영어 그대로 쓰고 필요하면 괄호로 한국어를 붙인다. 수식은 $...$ (LaTeX).
- 강의에서 실제로 말한 내용만 쓴다. 강의자료에만 있고 언급되지 않은 내용은 넣지 않는다.
- 교수가 강조한 점, 직관적 설명·예시, 시험·과제·공지는 놓치지 마라.
- 강의자료가 있으면 어느 페이지를 다뤘는지 대응시킨다 (자료의 '--- p.N ---' 표시가 페이지다).

출력 형식 — 정확히 이 구조로. 다른 말은 쓰지 마라:

# <강의 제목 한 줄>
오늘 다룬 슬라이드: p.<시작>–<끝>   (자료가 없거나 대응이 안 되면 "자료 없음")

## 1. <섹션 제목>
[줄 <시작번호>–<끝번호> | 슬라이드 p.<x>–<y>]
- 핵심 내용을 불릿으로. 정의·주장·모형·수식·예시·교수 코멘트. 필요한 만큼 (3~10개)

## 2. <섹션 제목>
[줄 …]
- …
(강의 흐름을 따라 5~12개 섹션. 전사 순서대로, 줄 번호 구간은 겹치지 않게)

## 핵심 용어
- **term** — 한 줄 설명

## 공지·과제·시험
- (있을 때만. 없으면 이 섹션을 통째로 생략)

## 한눈에
전체 흐름을 3~5문장으로.

규칙: 번호 섹션 제목 바로 다음 줄의 대괄호 표시는 반드시 쓴다. 줄 번호는 아래 전사의 [번호]를 그대로 쓴다.
슬라이드 대응이 없으면 "[줄 12–34]" 처럼 슬라이드 부분을 뺀다.
"""

COURSE_PROMPT = """너는 대학원 강의 노트를 만드는 조교다. 아래는 한 과목의 강의별 요약이다(오래된 순).
이것만 근거로 한국어 '과목 정리본'을 Markdown 으로 쓴다.

원칙:
- 담백하게. 미사여구·감상 없이 내용만. 대학원 수준 독자라 기초 개념 설명은 생략한다.
- 전문용어는 영어 그대로, 필요하면 괄호로 한국어. 수식은 $...$ (LaTeX).
- 요약에 없는 내용은 넣지 않는다. 강의끼리 이어지는 개념은 한곳에 묶고, 여러 번 강조된 점은 놓치지 마라.

출력 형식 — 정확히 이 구조로. 다른 말은 쓰지 마라:

# <과목명> 정리본

## 강의 흐름
- **<강의 제목>** — 한 줄 요지   (강의마다 한 줄, 순서대로)

## 1. <주제>
- 여러 강의에 걸친 내용을 주제별로 정리: 정의·주장·모형·수식·직관·예시·교수 강조점
- 불릿 끝에 출처 강의를 (강의 제목) 으로
(주제 4~10개, 강의 순서를 대체로 따른다)

## 핵심 용어
- **term** — 한 줄 설명

## 공지·과제·시험
- (있을 때만, 출처 강의 표시. 없으면 섹션 생략)
"""


def parse_outline(md):
    """summary.md → [{title, start, end, pages}]. start 는 전사 라인의 i (없으면 None)."""
    secs, cur = [], None
    for raw in (md or "").splitlines():
        line = raw.strip()
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            cur = {"title": m.group(1), "start": None, "end": None, "pages": ""}
            secs.append(cur); continue
        if cur is not None and cur["start"] is None and line.startswith("["):
            m = re.match(r"^\[\s*줄\s*(\d+)(?:\s*[–\-~]\s*(\d+))?\s*(?:\|\s*슬라이드\s*([^\]]*))?\]", line)
            if m:
                cur["start"] = int(m.group(1))
                cur["end"] = int(m.group(2)) if m.group(2) else None
                cur["pages"] = (m.group(3) or "").strip()
    return secs

def parse_polish(out, want_ko):
    """'N. 원문 ||| 번역' 파싱 → {번호: (원문, 번역 or None)}."""
    got = {}
    for line in (out or "").splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
        if not m:
            continue
        body = m.group(2).strip()
        if want_ko and "|||" in body:
            en, ko = body.split("|||", 1)
            got[int(m.group(1))] = (en.strip(), ko.strip())
        else:
            got[int(m.group(1))] = (body, None)
    return got


def parse_numbered(out):
    got = {}
    for line in (out or "").splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
        if m:
            got[int(m.group(1))] = m.group(2).strip()
    return got


def hangul_dominant(text):
    h = sum(1 for c in text if "가" <= c <= "힣")
    a = sum(1 for c in text if c.isascii() and c.isalpha())
    return h > a


def whisper_prompt(ctx, limit=260, extra=""):
    """맥락 텍스트 → Whisper initial_prompt. 긴 한국어 프롬프트(슬라이드 요약 등)는
    디코더를 망가뜨려 빈 결과·환각만 낸다(실측). 영문 용어만 추려 200자 이내로.
    extra(과목 누적 용어)는 세션 용어 뒤 남는 자리만 채운다."""
    m = re.search(r"^Terms:\s*(.+)$", ctx or "", re.M | re.I)  # Gemini 정리 노트면 용어 줄만
    src = (m.group(1) if m else (ctx or "")) + ", " + (extra or "")
    out, seen = [], set()
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9\-]*(?:[ /][A-Za-z][A-Za-z0-9\-]*)*", src):
        t = m.group(0).strip()
        if len(t) < 3 or t.lower() in seen:
            continue
        seen.add(t.lower()); out.append(t)
    s = ""
    for t in out:
        if len(s) + len(t) + 2 > limit:
            break
        s += (", " if s else "") + t
    return s


def live_cut(text, last_line, lo=0):
    """마지막 확정 라인의 끝 3~4단어를 라이브 텍스트 text[lo:] 뒤쪽(최대 8000자)에서 찾아
    그 직후 문자 오프셋을 돌려준다. 못 찾으면 None. (parakeet 스트림 타임스탬프는 창 기준
    상대값이라 시간으로는 정렬 불가 → 단어 매칭.)"""
    lw = re.findall(r"\w+", last_line.lower())[-8:]
    base = max(lo, len(text) - 8000)
    toks = [(base + m.end(), m.group(0).lower()) for m in re.finditer(r"\w+", text[base:])]
    words = [w for _, w in toks]
    for n in (4, 3):
        for i in range(len(lw) - n, -1, -1):
            seq = lw[i:i + n]
            for j in range(len(words) - n, -1, -1):
                if words[j:j + n] == seq:
                    return toks[j + n - 1][0]
    return None


# ---------- 설정 ----------

DEFAULT_CFG = {"translate": "live", "theme": "auto", "layout": "inline",
               "ko_width": 360, "ko_font": 14, "line_h": 1.65, "toc_w": 200,
               "agy_account": "main", "port": 8765, "live": "on",
               "langs": list(DEFAULT_LANGS), "input": ""}
ACCT_STORE = pathlib.Path.home() / ".claude/.state/gemini-accounts"
CFG_ALLOWED = {"translate": ("live", "after", "off"),
               "theme": ("auto", "light", "dark", "term", "term-light"),
               "layout": ("inline", "side"),
               "live": ("on", "off")}   # live=off: parakeet 미로드 (RAM ~2.3GB 절약)
CFG_NUM = {"ko_width": (200, 600), "ko_font": (11, 20), "line_h": (1.2, 2.4), "toc_w": (140, 420)}

def load_cfg():
    try:
        return {**DEFAULT_CFG, **json.loads(CONFIG.read_text())}
    except Exception:
        return dict(DEFAULT_CFG)

def save_cfg(cfg):
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))


# ---------- VAD 세그먼터 (에너지 기반, 침묵 경계 컷) ----------

class Segmenter:
    FRAME = int(SR * 0.05)          # 50ms
    GATE = 0.0035                   # 발화 시작 게이트
    SIL_GATE = 0.0022               # 침묵 판정 게이트(히스테리시스 — 발화 중 순간 저음을 침묵으로 오인 방지)
    SIL_SHORT = 20                  # 1.0초 침묵: 발화가 충분히 길 때만 컷
    SIL_LONG = 40                   # 2.0초 침묵: 무조건 컷
    MIN_CUT = 6.0                   # 짧은 침묵 컷의 최소 발화 길이 — 구 단위 파편화 방지
    MIN_UTT = 1.0                   # 이보다 짧으면 버림(잡음)
    MAX_UTT = 25.0                  # 강제 컷
    PRE_ROLL = 4                    # 발화 시작 전 0.2초 포함

    def __init__(self, on_utterance):
        self.on_utt = on_utterance
        self.buf = np.zeros(0, dtype=np.float32)
        self.pre = []
        self.in_speech = False
        self.sil = 0
        self.pos = 0.0              # 전체 오디오 진행(초)
        self.utt_start = 0.0

    def feed(self, pcm_f32):
        self.buf = np.concatenate([self.buf, pcm_f32])
        while len(self.buf) >= self.FRAME:
            frame, self.buf = self.buf[:self.FRAME], self.buf[self.FRAME:]
            self._frame(frame)

    def _frame(self, frame):
        rms = float(np.sqrt(np.mean(frame ** 2)))
        self.pos += 0.05
        if not self.in_speech:
            self.pre.append(frame)
            if len(self.pre) > self.PRE_ROLL:
                self.pre.pop(0)
            if rms >= self.GATE:
                self.in_speech = True
                self.sil = 0
                self.cur = list(self.pre)
                self.utt_start = self.pos - 0.05 * len(self.pre)
                self.pre = []
        else:
            self.cur.append(frame)
            self.sil = self.sil + 1 if rms < self.SIL_GATE else 0
            dur = self.pos - self.utt_start
            if (self.sil >= self.SIL_LONG
                    or (self.sil >= self.SIL_SHORT and dur - self.sil * 0.05 >= self.MIN_CUT)
                    or dur >= self.MAX_UTT):
                self._cut(trim=self.sil if self.sil >= self.SIL_SHORT else 0)

    def _cut(self, trim=0):
        audio = np.concatenate(self.cur)
        if trim > 2:                       # 끝의 침묵은 0.1초만 남기고 잘라냄
            audio = audio[:len(audio) - (trim - 2) * self.FRAME]
        dur = len(audio) / SR
        if dur >= self.MIN_UTT:
            self.on_utt(audio, self.utt_start, dur)
        self.in_speech = False
        self.sil = 0
        self.cur = []

    def flush(self):
        if self.in_speech and self.cur:
            self._cut()


# ---------- 엔진 ----------

class Engine:
    def __init__(self):
        self.cfg = load_cfg()
        self.lock = threading.Lock()
        self.state = "idle"      # idle|starting|recording|paused|finishing
        self.error = ""
        self.rev = 0
        self.sid = None
        self.lines = []          # [{i,t,a,text,lang,ko}]
        self.live = []           # [(start,end,text)]
        self.live_buffer = ""
        self._live_text = ""     # parakeet 스트림 원문 전체
        self._live_base = 0      # 이 세션에서 새로 만들어지는 라인의 첫 i (이전 세션 라인에 앵커 금지)
        self._live_cut = (0, 0)  # (확정 라인 수, 그 라인 끝에 대응하는 라이브 텍스트 오프셋)
        self.fixed_end = 0.0
        self.trans_note = ""
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True).start()
        self.fix = FixWorker()
        self.fixq = queue.Queue()
        threading.Thread(target=self._fix_worker, daemon=True).start()
        threading.Thread(target=self._translator, daemon=True).start()
        self._pk = None   # parakeet-mlx 라이브 모델 (프로세스당 1회 로드)
        self._liveq = None       # 녹음 중일 때만: 라이브 층에 오디오를 넘기는 큐
        self._live_stop = None
        self._session_task = None
        self._stop_evt = None
        self._pause_evt = None
        self._abort = False
        self._ctx = ""
        self._edited = False     # 사용자가 라인을 지우거나 고쳤나 (전부 지운 세션 저장 판단용)
        self.polish = {"sid": None, "busy": False, "note": "", "error": ""}
        self.summary = {"sid": None, "busy": False, "note": "", "error": ""}
        self.course = {"subject": "", "busy": False, "note": "", "error": ""}
        self._flush_translate = threading.Event()
        self._src = None         # 파일 가져오기 중이면 그 경로 (마이크 대신)
        self._src_tmp = False    # 업로드로 받은 임시 파일이면 끝난 뒤 지운다
        self._imp = None         # {"name","dur","done"} 가져오기 진행
        self._part = 0           # 이번 녹음이 쓰는 오디오 파트 번호
        self._saved_at = 0.0
        self.recover = ""        # 복구 중인 세션 id
        self._rec_lock = threading.Lock()
        self._live_sids = set()  # 이 프로세스에서 녹음을 시작한 세션
        threading.Thread(target=self._recover_all, daemon=True).start()

    # -- 스레드/루프 유틸
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def bump(self):
        with self.lock:
            self.rev += 1

    # -- 컨트롤 (외부 스레드에서 호출)
    def start(self, sid=None, src=None, tmp=False):
        if self.state != "idle":
            return False
        self._src, self._src_tmp = (str(src) if src else None), bool(src and tmp)
        self.state = "starting"; self.bump()
        self._call(self._session(sid))
        return True

    def pause(self):
        if self._src:        # 파일 가져오기는 일시정지가 없다 (다시 열면 처음부터 읽는다)
            return
        if self.state == "recording" and self._pause_evt:
            self.loop.call_soon_threadsafe(self._pause_evt.set)

    def resume(self):
        if self.state == "paused" and self._pause_evt:
            self.loop.call_soon_threadsafe(self._pause_evt.clear)
            with self.lock:
                self.state = "recording"
            self.bump()

    def stop(self):
        if self.state in ("recording", "paused", "starting") and self._stop_evt:
            self.loop.call_soon_threadsafe(self._stop_evt.set)

    def abort(self):
        """저장·교정 없이 즉시 중단 (세션 삭제용)."""
        if self.state in ("recording", "paused", "starting") and self._stop_evt:
            self._abort = True
            self.loop.call_soon_threadsafe(self._stop_evt.set)

    # -- 세션 본체 (엔진 루프 안)
    async def _session(self, sid=None):
        start_wall = datetime.datetime.now()
        if not sid:
            sid = start_wall.strftime("%Y-%m-%d_%H%M")
        sess = TR / sid
        sess.mkdir(parents=True, exist_ok=True)
        if self._needs_recover(sess):   # 지난번 크래시로 남은 녹음이 있으면 먼저 살린다
            await asyncio.get_event_loop().run_in_executor(None, self._recover, sess)
        self._live_sids.add(sid)
        prev = []
        lf = sess / "lines.json"
        if lf.exists():  # 이어하기 — 기존 라인·번역 로드
            try:
                prev = json.loads(lf.read_text())
                for l in prev:
                    l.setdefault("at", time.time())
            except Exception:
                prev = []
        elif (sess / "transcript.md").exists():  # 구형 세션 — md 에서 복원
            try:
                md_txt = (sess / "transcript.md").read_text()
                prev = [{"i": k, "t": m.group(1), "a": 0.0, "text": m.group(2),
                         "lang": "en", "ko": (m.group(3) or "").strip(),
                         "at": time.time()}
                        for k, m in enumerate(re.finditer(
                            r"\*\*(\d\d:\d\d:\d\d)\*\* (.+)(?:\n> (.+))?", md_txt))]
            except Exception:
                prev = []
        with self.lock:
            self.sid = sid; self.lines = prev; self.live = []; self.live_buffer = ""
            self._live_text = ""; self._live_cut = (0, 0)
            self._live_base = max((l.get("i", -1) for l in prev), default=-1) + 1
            self._edited = False
            self.fixed_end = 0.0; self.error = ""; self.trans_note = ""
        self.bump()
        meta_p = sess / "meta.json"
        meta = {"id": sid, "started": start_wall.isoformat()}
        try:
            meta.update(json.loads(meta_p.read_text()))
        except Exception:
            pass
        meta["state"] = "recording"
        src = self._src
        if src:
            meta["source"] = pathlib.Path(src).name
            if not meta.get("title"):
                meta["title"] = pathlib.Path(src).stem[:60]; meta["title_auto"] = False
        meta_p.write_text(json.dumps(meta, ensure_ascii=False))
        self._ctx = (meta.get("context") or "").strip()[:CTX_MAX]
        terms = subject_terms_text(meta, sid, 900)   # 같은 과목 지난 강의 용어
        # Whisper 에는 영문 용어만 (한국어 장문은 금지). 세션 용어가 먼저, 남는 자리를 과목 용어로
        self._wprompt = whisper_prompt(self._ctx, extra=terms)
        if terms:
            self._ctx = (self._ctx + "\nCourse terms: " + terms[:400]).strip()
        self._part = next_part(sess)
        self._saved_at = time.time()
        self._imp = None
        if src:
            self._imp = {"name": pathlib.Path(src).name, "dur": 0.0, "done": 0.0}
            dur = await asyncio.get_event_loop().run_in_executor(None, probe_duration, src)
            self._imp["dur"] = round(dur, 1)

        # 워커·라이브 엔진 로드 (첫 실행은 모델 다운로드로 오래 걸림)
        ok = await asyncio.get_event_loop().run_in_executor(None, lambda: self.fix.proc or self.fix.start())
        if not ok:
            with self.lock:
                self.state = "idle"; self.error = "확정 전사 워커 시작 실패"
                self._src = None; self._src_tmp = False; self._imp = None
            self.bump(); return
        liveq = queue.Queue()
        live_stop = threading.Event()
        self._liveq, self._live_stop = liveq, live_stop
        if self.cfg.get("live") != "off" and not src:   # off·파일 가져오기면 parakeet 로드·급전 전부 생략
            threading.Thread(target=self._live_loop, args=(liveq, live_stop),
                             daemon=True).start()
        else:
            live_stop.set()   # 소비자가 없으니 급전도 막는다 (큐 무한 적재 방지)

        stop = asyncio.Event(); pause = asyncio.Event()
        self._stop_evt, self._pause_evt = stop, pause
        self._abort = False
        pcm_path = sess / "audio.pcm"
        pcm_f = open(pcm_path, "ab")
        part = self._part

        def on_utt(audio, at, dur):
            if src:   # 파일은 파일 안의 위치를 시각으로
                wall = hms(at)
            else:
                wall = (datetime.datetime.now()
                        - datetime.timedelta(seconds=max(0.0, seg.pos - at))).strftime("%H:%M:%S")
            self.fixq.put((sid, sess, audio, at, dur, wall, self._wprompt, part))
        seg = Segmenter(on_utt)

        with self.lock:
            self.state = "recording"
        self.bump()
        try:   # 녹음·가져오기 동안 잠자기 방지 (앱이 죽으면 같이 풀린다)
            caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            caff = None
        eof = False

        try:
            while not stop.is_set():
                if pause.is_set():
                    with self.lock:
                        self.state = "paused"
                    self.bump()
                    seg.flush()
                    self._autosave(sess, force=True)
                    while pause.is_set() and not stop.is_set():
                        await asyncio.sleep(0.2)
                    continue
                if src:
                    inp = ["-i", src, "-vn"]
                elif os.environ.get("GY_INPUT"):
                    inp = ["-re", "-i", os.environ["GY_INPUT"]]
                else:
                    arg = await asyncio.get_event_loop().run_in_executor(
                        None, input_arg, self.cfg.get("input", ""))
                    inp = ["-f", "avfoundation", "-i", arg]
                proc = await asyncio.create_subprocess_exec(
                    FFMPEG, "-hide_banner", "-loglevel", "error", *inp,
                    "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                    stdin=asyncio.subprocess.DEVNULL)
                try:
                    while not stop.is_set() and not pause.is_set():
                        if src and self.fixq.qsize() > 6:   # 전사가 따라올 때까지 읽기를 멈춘다 (메모리)
                            await asyncio.sleep(0.2); continue
                        try:
                            chunk = await asyncio.wait_for(proc.stdout.read(SR // 5 * 2), timeout=2)
                        except asyncio.TimeoutError:
                            if proc.returncode is not None:
                                raise IOError("ffmpeg 종료(마이크 권한?)")
                            continue
                        if not chunk:
                            if src:
                                eof = True; stop.set(); break
                            raise IOError("ffmpeg 종료(마이크 권한?)")
                        pcm_f.write(chunk)
                        chunk_f = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        if not live_stop.is_set():
                            liveq.put(chunk_f)
                        seg.feed(chunk_f)
                except IOError as e:
                    with self.lock:
                        self.error = str(e)
                    stop.set()
                finally:
                    if proc.returncode is None:
                        proc.terminate()
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=3)
                        except asyncio.TimeoutError:
                            proc.kill()
        finally:
            with self.lock:
                self.state = "finishing"
            self.bump()
            seg.flush()
            pcm_f.close()
            live_stop.set()
            self._liveq = None
            if caff:
                caff.terminate()
            if src and not eof and not self._abort:
                self._drop_fixq()   # 가져오기 중단 — 이미 받아쓴 데까지만 남긴다
            if self._abort:
                self._drop_fixq()   # 확정 대기열 폐기 — 교정·저장 전부 생략
                with self.lock:
                    self.sid = None; self.lines = []
            else:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, self._finalize, sid, sess, start_wall)
                except Exception as ex:
                    logging.exception("finalize 실패")
                    with self.lock:
                        self.error = f"저장 실패: {ex}"[:120]
            if src and self._src_tmp:   # 업로드 사본 — 오디오는 m4a 로 남았다
                try:
                    pathlib.Path(src).unlink()
                    pathlib.Path(src).parent.rmdir()
                except OSError:
                    pass
            with self.lock:
                self.state = "idle"; self.live = []; self.live_buffer = ""; self._live_text = ""
                self._src = None; self._src_tmp = False; self._imp = None
            self.bump()

    # -- 라이브 미리보기 (parakeet-mlx 스트리밍, 스레드) — 확정 층과 독립
    def _live_loop(self, q, stop_evt):
        _br = re.compile(r"\[\s*[A-Za-z_ ]{0,24}(\]|$)")
        try:
            import mlx.core as mx
            if self._pk is None:
                from parakeet_mlx import from_pretrained
                self._pk = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
        except Exception:
            logging.exception("라이브 모델 로드 실패 — 미리보기 없이 진행")
            return
        buf = np.zeros(0, dtype=np.float32)
        try:
            with self._pk.transcribe_stream(context_size=(256, 256)) as tr:
                while not stop_evt.is_set():
                    try:
                        c = q.get(timeout=0.3)
                    except queue.Empty:
                        continue
                    if q.qsize() > 80:  # 16초 이상 밀리면 사이를 버리고 현재로 점프
                        try:
                            while q.qsize() > 10:
                                q.get_nowait()
                        except queue.Empty:
                            pass
                        buf = np.zeros(0, dtype=np.float32)
                        continue
                    buf = np.concatenate([buf, c])
                    if len(buf) < SR:  # 1초 단위 공급 (호출 오버헤드 절감)
                        continue
                    tr.add_audio(mx.array(buf))
                    buf = np.zeros(0, dtype=np.float32)
                    text = _br.sub(" ", tr.result.text or "").strip()
                    with self.lock:
                        self._live_text = text
                        self.rev += 1
        except Exception:
            logging.exception("라이브 스트림 종료")

    def _drop_fixq(self):
        try:
            while True:
                self.fixq.get_nowait(); self.fixq.task_done()
        except queue.Empty:
            pass

    # -- 확정 전사 소비자 (스레드)
    def _fix_worker(self):
        while True:
            sid, sess, audio, at, dur, wall, ctx, part = self.fixq.get()
            try:
                self._fix_one(sid, sess, audio, at, dur, wall, ctx, part)
            finally:
                imp = self._imp
                if imp is not None and sid == self.sid:
                    imp["done"] = round(at + dur, 1)
                self.fixq.task_done()

    def _autosave(self, sess, force=False):
        """녹음 중 lines.json 스냅샷 (원자적 쓰기). 크래시 복구의 기준점이 된다."""
        if self.state not in ("recording", "paused") or self.sid != sess.name:
            return
        if not force and time.time() - self._saved_at < AUTOSAVE_SEC:
            return
        self._saved_at = time.time()
        with self.lock:
            snap = [dict(l) for l in self.lines]
        try:
            write_json_atomic(sess / "lines.json", snap)
        except OSError:
            pass

    def _fix_one(self, sid, sess, audio, at, dur, wall, ctx, part=0):
            r = self.fix.transcribe(audio, ctx, self.cfg.get("langs") or DEFAULT_LANGS)
            if r is None:
                with self.lock:
                    self.error = "확정 전사 워커 응답 없음(재시작 시도 중)"; self.rev += 1
                return
            if not r.get("text"):
                return
            text = r["text"]
            line = {"t": wall, "a": round(at, 2), "d": round(dur, 2), "p": part, "text": text,
                    "lang": r.get("lang", "en"),
                    "ko": "" if hangul_dominant(text) else None, "at": time.time()}
            with self.lock:
                active = sid == self.sid
                if active:
                    line["i"] = max((l.get("i", -1) for l in self.lines), default=-1) + 1
                    self.lines.append(line)
                    self.fixed_end = at + dur
                    if self.error.startswith("확정 전사"):
                        self.error = ""
                    self.rev += 1
            if active:
                try:
                    with (sess / "transcript.md").open("a") as f:  # 크래시 대비 즉시 append
                        f.write(f"**{wall}** {text}\n\n")
                except OSError:
                    pass
                self._autosave(sess)

    # -- 번역 (스레드): 실시간 배치
    def _translator(self):
        agy = Agy(str(ROOT / ".agywork"))
        (ROOT / ".agywork").mkdir(exist_ok=True)
        fails = 0
        while True:
            if self.cfg.get("translate") != "live" or not agy.available():
                time.sleep(2); continue
            with self.lock:
                pend = [l for l in self.lines if l["ko"] is None]
            if not pend:
                time.sleep(2); continue
            if len(pend) < 3 and time.time() - pend[0]["at"] < 25 \
                    and not self._flush_translate.is_set():
                time.sleep(2); continue
            batch = pend[:6]
            if agy.account is not None and agy.account != self.cfg.get("agy_account", "main"):
                agy.stop()  # 계정 전환 → 다음 turn 에서 새 HOME 으로 스폰
            ctx = getattr(self, "_ctx", "")
            out = agy.turn((f"수업 맥락: {ctx}\n" if ctx else "") + "Translate:\n" + "\n".join(
                f"{k+1}. {l['text']}" for k, l in enumerate(batch)))
            got = parse_numbered(out)
            if not got:
                fails += 1
                with self.lock:
                    self.trans_note = f"번역 실패 {fails}회 (agy 로그인 확인)"
                time.sleep(10 if fails < 5 else 60)
                continue
            fails = 0
            with self.lock:
                for k, l in enumerate(batch):
                    if l["ko"] is None:
                        l["ko"] = got.get(k + 1, "")
                self.trans_note = ""; self.rev += 1
                sid = self.sid
            if sid:
                self._autosave(TR / sid)

    # -- 종료 처리: 번역 마무리/교정 → 저장 → 오디오 압축
    def _finalize(self, sid, sess, start_wall):
        if not sess.exists():  # 마무리 중 세션이 삭제된 경우 (UI 삭제 등)
            return
        # 확정 큐 소진 대기 — 처리 중인 마지막 발화까지 (empty() 는 꺼낸 즉시 참이 되어 유실됨)
        t0 = time.time()   # 파일 가져오기는 남은 전사를 끝까지 기다린다
        while self.fixq.unfinished_tasks and (self._src or time.time() - t0 < 120):
            time.sleep(0.5)
        # 종료는 빨라야 한다 — 전체 문맥 다듬기(agy)는 여기서 하지 않고
        # 뷰어의 '다듬기' 버튼(polish)으로 따로 돌린다.
        with self.lock:
            lines = [dict(l) for l in self.lines]
        # 저장 — 라인이 비었는데 기존 md 에 내용이 있으면 보존(덮어쓰기 금지)
        md = sess / "transcript.md"
        if not lines and not self._edited and md.exists() and md.stat().st_size > 300:
            return
        with md.open("w") as f:
            f.write(f"# 그냥 받아쓰기 — {start_wall:%Y-%m-%d %H:%M}\n\n")
            for l in lines:
                f.write(f"**{l['t']}** {l['text']}\n")
                if l["ko"]:
                    f.write(f"> {l['ko']}\n")
                f.write("\n")
        write_json_atomic(sess / "lines.json", lines)
        ended = datetime.datetime.now()
        pcm_to_m4a(sess)   # 오디오 압축 (pcm → 이번 파트 m4a)
        meta = {"id": sid, "started": start_wall.isoformat()}
        try:  # 이름·과목·최초 시작시각·맥락·요약 시각 등 기존 값 보존
            om = json.loads((sess / "meta.json").read_text())
            meta.update({k: v for k, v in om.items() if v not in (None, "")})
        except Exception:
            pass
        meta.update({"ended": ended.isoformat(), "n_lines": len(lines), "state": "done",
                     "polished": False,   # 새 라인이 생겼으니 다듬기는 다시 필요
                     "preview": (lines[0]["text"][:80] if lines else "")})
        write_json_atomic(sess / "meta.json", meta)
        self.fix.stop()  # 워커 메모리 반납 — 다음 세션 시작 때 다시 띄운다

    # -- 전체 문맥 다듬기 (원문 교정 + 번역). 종료 후 버튼으로 실행.
    def start_polish(self, sid, tries=0):
        """tries>0 은 앱이 꺼져 끊긴 작업을 다음 실행 때 자동으로 이어 하는 경우."""
        sess = TR / sid
        if not (sess / "lines.json").exists():
            return False, "다듬을 내용이 없어요"
        if self.state != "idle":
            return False, "녹음이 끝난 뒤에 다듬을 수 있어요"
        if self.polish["busy"] or self.summary["busy"] or self.course["busy"]:
            return False, "이미 작업 중이에요"
        if not Agy(str(ROOT / ".agywork")).available():
            return False, "agy(Antigravity CLI)가 없어 다듬기를 못 해요"
        self.polish = {"sid": sid, "busy": True, "error": "",
                       "note": "끊긴 다듬기를 이어서 하는 중…" if tries else "준비 중…"}
        update_meta(sess, job="polish", job_tries=tries)   # 앱이 꺼져도 다음 실행 때 이어 하도록 표시
        self.bump()
        threading.Thread(target=self._polish_job, args=(sid, sess), daemon=True).start()
        return True, ""

    def _polish_job(self, sid, sess):
        def note(t):
            self.polish = {"sid": sid, "busy": True, "note": t, "error": ""}
            self.bump()
        try:
            skipped = self._polish_core(sid, sess, note)
            self.polish = {"sid": sid, "busy": False, "error": "",
                           "note": "다듬기 완료" + (f" ({skipped}구간은 응답이 없어 원문 유지)" if skipped else "")}
        except Exception as e:
            logging.exception("다듬기 실패")
            self.polish = {"sid": sid, "busy": False, "note": "", "error": f"다듬기 실패: {e}"[:140]}
        update_meta(sess, job=None, job_tries=None)   # 실패는 자동으로 다시 하지 않는다 — 버튼으로 이어서
        self.bump()

    def _polish_core(self, sid, sess, note):
        """전체 문맥 다듬기 본체. 자료가 아직 텍스트로 안 바뀌었으면 먼저 변환해 참고한다.
        청크마다 저장하고 진행(meta.polish_done/polish_n)을 남겨, 중단되면 다음에 거기서 이어 한다.
        완료 시 meta.polished=True, 응답이 끝내 없어 원문으로 둔 청크 수를 돌려준다. 실패는 예외로."""
        lines = json.loads((sess / "lines.json").read_text())
        for k, l in enumerate(lines):
            l.setdefault("i", k)
        if not lines:
            raise ValueError("다듬을 라인이 없어요")
        self._wait_online(note, "다듬기")   # 끊긴 채 자료 변환을 하면 품질 낮은 대체 변환이 굳는다
        if ctx_pending(sess):
            if conv_status(sid).get("busy"):
                note("강의자료 변환을 기다리는 중…"); wait_conv(sid)
            else:
                convert_ctx(sess, note)
        try:
            meta = json.loads((sess / "meta.json").read_text())
        except Exception:
            meta = {}
        ctx = meta.get("context", "")
        want_ko = self.cfg.get("translate") != "off"
        head = POLISH_HEAD + (POLISH_KO if want_ko else POLISH_EN)
        pre = polish_preamble(ctx, ctx_text_all(sess, 24000), subject_terms_text(meta, sid, 1500))
        agy = Agy(str(ROOT / ".agywork"), model=AGY_MODEL_HEAVY, print_timeout="8m")
        (ROOT / ".agywork").mkdir(exist_ok=True)
        STEP, n = 25, len(lines)
        # 끊긴 다듬기는 끝난 청크 다음부터. 그사이 라인 수가 바뀌었으면(이어 녹음·병합·삭제) 처음부터
        done = meta.get("polish_done", 0) if meta.get("polish_n") == n else 0
        done = n if done >= n else done - done % STEP
        update_meta(sess, polish_done=done, polish_n=n)
        skipped = good = 0
        try:
            for i in range(done, n, STEP):
                chunk = lines[i:i + STEP]
                label = f"다듬는 중… {i}/{n}줄" + (" (이어서)" if done else "")
                note(label)
                out = self._ask(agy, head + "\n" + "\n".join(
                    f"{k+1}. {l['text']}" for k, l in enumerate(chunk)),
                    lambda o: bool(parse_polish(o, want_ko)), note, label, timeout=420, preamble=pre)
                if out is None:
                    # 연결은 되는데 이 청크만 답이 안 오면 원문으로 두고 넘어간다. 첫 청크부터 안 되면 로그인 문제로 본다
                    if not good or skipped >= 2:
                        raise RuntimeError("Gemini 응답을 받지 못했어요 (agy 로그인 확인)")
                    skipped += 1
                    logging.error("다듬기 청크 건너뜀: %s %d", sid, i)
                    update_meta(sess, polish_done=min(i + STEP, n), polish_n=n)
                    continue
                good += 1
                self._apply_polish(chunk, parse_polish(out, want_ko))
                write_session(sess, lines)          # 청크마다 저장 — 중단돼도 진행분은 남는다
                update_meta(sess, polish_done=min(i + STEP, n), polish_n=n)
        finally:
            agy.stop()
        update_meta(sess, polished=True, polish_done=None, polish_n=None)
        return skipped

    @staticmethod
    def _apply_polish(chunk, got):
        for k, l in enumerate(chunk):
            g = got.get(k + 1)
            if not g:
                continue
            en, ko = g
            o = l["text"]
            # 길이가 크게 어긋나면 환각으로 보고 원문 유지 (간투사 제거로 조금 짧아지는 건 허용)
            if en and 0.4 * len(o) <= len(en) <= 2 * len(o) + 20:
                if en != o and not l.get("text0"):
                    l["text0"] = o          # 원본 보존 (되돌리기용)
                l["text"] = en
            if ko and l.get("ko") != "":
                l["ko"] = ko

    def _wait_online(self, note, label):
        """인터넷이 끊겼으면 돌아올 때까지 기다린다. NET_WAIT_MAX 를 넘으면 예외(진행분은 남아 있다)."""
        if online():
            return
        note(f"{label} — 인터넷 연결이 끊겼어요. 연결되면 이어서 해요")
        t0 = time.time()
        while offline():
            if time.time() - t0 > NET_WAIT_MAX:
                raise RuntimeError(f"인터넷 연결이 {NET_WAIT_MAX // 60}분 넘게 끊겨 멈췄어요")
            time.sleep(5)
        note(label)

    def _ask(self, agy, prompt, ok, note, label, timeout, preamble="", tries=len(AGY_RETRY_WAIT)):
        """agy 한 턴을 ok(답) 이 참일 때까지 다시 묻는다. 인터넷이 끊기면 돌아올 때까지 기다리고
        (재시도 횟수에 안 센다), 연결은 되는데 답이 계속 안 오면 tries 번 뒤 None."""
        n = 0
        while True:
            out = agy.turn(prompt, timeout=timeout, preamble=preamble, abort=offline)
            if ok(out):
                return out
            agy.stop()      # 실패한 대화를 이어 쓰지 않는다 — 새 프로세스로 (preamble 도 다시 붙는다)
            if offline():
                self._wait_online(note, label)
                continue
            n += 1
            if n > tries:
                return None
            wait = AGY_RETRY_WAIT[min(n, len(AGY_RETRY_WAIT)) - 1]
            note(f"{label} — 응답이 없어 {wait}초 뒤 다시 시도해요 ({n}/{tries})")
            time.sleep(wait)
            note(label)

    # -- 강의 요약·정리본 (자료 변환 → 필요하면 다듬기 → Gemini 한 턴). 종료 후 버튼으로 실행.
    def start_summary(self, sid, tries=0):
        sess = TR / sid
        if not (sess / "lines.json").exists():
            return False, "요약할 내용이 없어요"
        if self.state != "idle":
            return False, "녹음이 끝난 뒤에 요약할 수 있어요"
        if self.polish["busy"] or self.summary["busy"] or self.course["busy"]:
            return False, "이미 작업 중이에요"
        if not Agy(str(ROOT / ".agywork")).available():
            return False, "agy(Antigravity CLI)가 없어 요약을 못 해요"
        self.summary = {"sid": sid, "busy": True, "error": "",
                        "note": "끊긴 요약을 이어서 하는 중…" if tries else "준비 중…"}
        update_meta(sess, job="summary", job_tries=tries)
        self.bump()
        threading.Thread(target=self._summary_job, args=(sid, sess), daemon=True).start()
        return True, ""

    def _summary_job(self, sid, sess):
        def note(t):
            self.summary = {"sid": sid, "busy": True, "note": t, "error": ""}
            self.bump()
        try:
            self._summary_core(sid, sess, note)
            self.summary = {"sid": sid, "busy": False, "note": "요약 완료", "error": ""}
        except Exception as e:
            logging.exception("요약 실패")
            self.summary = {"sid": sid, "busy": False, "note": "", "error": f"요약 실패: {e}"[:140]}
        update_meta(sess, job=None, job_tries=None)
        self.bump()

    def _summary_core(self, sid, sess, note):
        """자료 변환 → (필요하면) 다듬기 → 요약. 실패는 예외로."""
        self._wait_online(note, "요약")
        if ctx_pending(sess):
            if conv_status(sid).get("busy"):
                note("강의자료 변환을 기다리는 중…"); wait_conv(sid)
            else:
                convert_ctx(sess, note)
        mp = sess / "meta.json"
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {"id": sid}
        if not meta.get("polished"):
            self._polish_core(sid, sess, note)
            meta = json.loads(mp.read_text())
        lines = json.loads((sess / "lines.json").read_text())
        for k, l in enumerate(lines):
            l.setdefault("i", k)
        if not lines:
            raise ValueError("요약할 라인이 없어요")
        ctx = meta.get("context", "")
        slides = ctx_text_all(sess, 40000)
        terms = subject_terms_text(meta, sid, 1500)
        prompt = SUMMARY_PROMPT + (f"\n수업 맥락: {ctx}\n" if ctx else "")
        if terms:
            prompt += f"\n이 과목 지난 강의 용어 (표기 참고용): {terms}\n"
        if slides:
            prompt += "\n=== 강의자료 ===\n" + slides + "\n"
        prompt += "\n=== 강의 전사 ===\n" + "\n".join(f"[{l['i']}] {l['text']}" for l in lines)
        label = "요약 만드는 중… (몇 분 걸려요)"
        note(label)
        agy = Agy(str(ROOT / ".agywork"), model=AGY_MODEL_HEAVY, agent=None, print_timeout="12m")
        try:
            out = self._ask(agy, prompt, lambda o: "## " in _strip_fence(o), note, label,
                            timeout=720, tries=2)
        finally:
            agy.stop()
        if out is None:
            raise RuntimeError("Gemini 응답을 받지 못했어요 (agy 로그인 확인)")
        write_text_atomic(sess / "summary.md", _strip_fence(out))   # 쓰다 꺼져도 이전 요약이 깨지지 않게
        # 다듬기 등이 그사이 쓴 값을 덮지 않게 다시 읽어서 반영
        update_meta(sess, summary_at=datetime.datetime.now().isoformat(timespec="seconds"))

    # -- 과목 정리본: 요약이 없는 강의는 먼저 요약하고, 강의별 요약을 모아 한 번에 정리
    def start_course(self, subject):
        subject = " ".join(str(subject or "").split())[:40]
        if not subject:
            return False, "과목이 없어요"
        if self.polish["busy"] or self.summary["busy"] or self.course["busy"]:
            return False, "이미 작업 중이에요"
        if not course_sessions(subject):
            return False, "전사가 있는 강의가 없어요"
        if not Agy(str(ROOT / ".agywork")).available():
            return False, "agy(Antigravity CLI)가 없어 정리본을 못 만들어요"
        self.course = {"subject": subject, "busy": True, "note": "준비 중…", "error": ""}
        self.bump()
        threading.Thread(target=self._course_job, args=(subject,), daemon=True).start()
        return True, ""

    def _course_job(self, subject):
        def note(t):
            self.course = {"subject": subject, "busy": True, "note": t, "error": ""}
            self.bump()
        try:
            sess_list = course_sessions(subject)
            todo = [m for m in sess_list if not (TR / m["id"] / "summary.md").exists()]
            for k, m in enumerate(todo):
                if self.state != "idle" and self.sid == m["id"]:
                    continue   # 녹음 중인 강의는 건너뛴다
                tag = f"요약 {k + 1}/{len(todo)} · {m.get('title') or m['id']} — "
                self._summary_core(m["id"], TR / m["id"], lambda t: note(tag + t))
            parts, used = [], {}
            for m in course_sessions(subject):
                sf = TR / m["id"] / "summary.md"
                if not sf.exists():
                    continue
                parts.append(f"=== {m.get('title') or m['id']} ({m['id'][:10]}) ===\n{sf.read_text().strip()}")
                used[m["id"]] = m.get("summary_at", "")
            if not parts:
                raise ValueError("요약이 있는 강의가 없어요")
            terms = ", ".join(subject_terms(subject)[:80])
            prompt = COURSE_PROMPT + f"\n과목명: {subject}\n"
            if terms:
                prompt += f"누적 용어: {terms}\n"
            prompt += "\n" + "\n\n".join(parts)
            label = f"정리본 만드는 중… (강의 {len(parts)}개)"
            note(label)
            agy = Agy(str(ROOT / ".agywork"), model=AGY_MODEL_HEAVY, agent=None, print_timeout="12m")
            try:
                out = self._ask(agy, prompt, lambda o: "## " in _strip_fence(o), note, label,
                                timeout=720, tries=2)
            finally:
                agy.stop()
            if out is None:
                raise RuntimeError("Gemini 응답을 받지 못했어요 (agy 로그인 확인)")
            COURSE_DIR.mkdir(parents=True, exist_ok=True)
            key = course_key(subject)
            write_text_atomic(COURSE_DIR / f"{key}.md", _strip_fence(out))
            write_json_atomic(COURSE_DIR / f"{key}.json", {
                "subject": subject, "used": used,
                "at": datetime.datetime.now().isoformat(timespec="seconds")})
            self.course = {"subject": subject, "busy": False, "note": "", "error": ""}
        except Exception as e:
            logging.exception("정리본 실패")
            self.course = {"subject": subject, "busy": False, "note": "", "error": f"정리본 실패: {e}"[:140]}
        self.bump()

    # -- 라인 편집 (HTTP용). 활성 세션은 메모리, 아니면 디스크.
    def edit_lines(self, sid, action, ids=None, text=None, ko=None):
        sess = TR / sid
        if not sess.is_dir():
            return False, "세션이 없어요"
        if self.sid == sid and self.state != "idle":
            with self.lock:
                out, err = apply_line_ops(self.lines, action, ids, text, ko)
                if out is None:
                    return False, err
                self.lines = out
                self._edited = True
                self.rev += 1
                snap = [dict(l) for l in out]
            try:  # 크래시 대비 스냅샷 — transcript.md 는 종료 시 재작성된다
                (sess / "lines.json").write_text(json.dumps(snap, ensure_ascii=False))
            except OSError:
                pass
            return True, ""
        lf = sess / "lines.json"
        try:
            lines = json.loads(lf.read_text())
        except Exception:
            return False, "저장된 라인이 없어 수정할 수 없어요"
        for k, l in enumerate(lines):
            l.setdefault("i", k)
        out, err = apply_line_ops(lines, action, ids, text, ko)
        if out is None:
            return False, err
        write_session(sess, out)
        return True, ""

    # -- 조회 (HTTP용)
    def snapshot(self):
        with self.lock:
            return {"state": self.state, "sid": self.sid, "rev": self.rev,
                    "error": self.error, "trans": self.trans_note,
                    "translate_mode": self.cfg.get("translate"),
                    "live_mode": self.cfg.get("live", "on"),
                    "langs": self.cfg.get("langs") or DEFAULT_LANGS,
                    "lines": self.lines,
                    "live": [{"s": s, "e": e, "text": t} for s, e, t in self.live],
                    "live_buffer": self._live_tail(), "fixed_end": self.fixed_end,
                    "polish": dict(self.polish), "summary": dict(self.summary),
                    "course": dict(self.course), "recover": self.recover,
                    "import": dict(self._imp) if self._imp else None}

    # -- 크래시 복구: 녹음 도중 앱이 죽으면 audio.pcm 과 state=recording 이 남는다.
    #    자동 저장된 lines.json 을 기준으로, 그 뒤 구간만 pcm 에서 다시 받아쓰고 m4a 로 굳힌다.
    def _needs_recover(self, sess):
        if (sess / "audio.pcm").exists():
            return True
        try:
            return json.loads((sess / "meta.json").read_text()).get("state") in ("recording", "finishing")
        except Exception:
            return False

    def _recover_all(self):
        time.sleep(3)   # 서버·뷰어가 먼저 뜨게
        if not TR.is_dir():
            return
        for d in sorted(TR.iterdir()):
            if not d.is_dir() or d.name.startswith(".") or d.name in self._live_sids:
                continue
            if self._needs_recover(d):
                try:
                    self._recover(d)
                except Exception:
                    logging.exception("복구 실패: %s", d.name)
        if self.state == "idle":
            self.fix.stop()
        self._resume_jobs()

    def _resume_jobs(self):
        """앱이 꺼져 끊긴 다듬기·요약(meta.job)을 이어서 한다. 같은 작업이 JOB_RESUME_MAX 번 넘게
        끊겼으면 자동으로 하지 않고 오류로 알린다 (버튼으로 이어서 할 수 있다)."""
        if self.state != "idle" or self.polish["busy"] or self.summary["busy"] or self.course["busy"]:
            return
        found = []
        for d in sorted(TR.iterdir(), reverse=True):
            if d.is_dir() and not d.name.startswith("."):
                try:
                    meta = json.loads((d / "meta.json").read_text())
                except Exception:
                    continue
                if meta.get("job") in ("polish", "summary"):
                    found.append((d, meta["job"], meta.get("job_tries", 0) + 1))
        for k, (d, kind, tries) in enumerate(found):
            name = "다듬기" if kind == "polish" else "요약"
            if k == 0 and tries <= JOB_RESUME_MAX:
                ok, err = (self.start_polish if kind == "polish" else self.start_summary)(d.name, tries=tries)
                if ok:
                    continue
                logging.error("%s 이어 하기 실패: %s %s", name, d.name, err)
            elif k == 0:
                setattr(self, kind, {"sid": d.name, "busy": False, "note": "",
                                     "error": f"{name} 실패: 여러 번 중단돼 멈췄어요 — 다시 시도해 주세요"})
                self.bump()
            update_meta(d, job=None, job_tries=None)

    def _recover(self, sess):
        with self._rec_lock:   # 이 프로세스에서 시작한 세션은 크래시 잔재가 아니다
            if sess.name in self._live_sids or not self._needs_recover(sess):
                return
            self.recover = sess.name; self.bump()
            try:
                self._recover_core(sess)
            finally:
                self.recover = ""; self.bump()

    def _recover_core(self, sess):
        mp = sess / "meta.json"
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {"id": sess.name}
        try:
            lines = json.loads((sess / "lines.json").read_text())
        except Exception:
            lines = []
        for k, l in enumerate(lines):
            l.setdefault("i", k)
        pcm = sess / "audio.pcm"
        part = next_part(sess)
        if pcm.exists() and pcm.stat().st_size > SR:
            mine = [l for l in fill_parts(lines) if l.get("p") == part and isinstance(l.get("a"), (int, float))]
            start = max((l["a"] + (l.get("d") or 1.0) for l in mine), default=0.0)
            total = pcm.stat().st_size / 2 / SR
            end_wall = datetime.datetime.fromtimestamp(pcm.stat().st_mtime)
            ctx = (meta.get("context") or "").strip()[:CTX_MAX]
            prompt = whisper_prompt(ctx, extra=subject_terms_text(meta, sess.name, 900))
            langs = self.cfg.get("langs") or DEFAULT_LANGS
            new = []

            def emit(audio, at, dur):
                at += start
                r = self.fix.transcribe(audio, prompt, langs)
                if r and r.get("text"):
                    wall = end_wall - datetime.timedelta(seconds=max(0.0, total - at))
                    new.append({"t": wall.strftime("%H:%M:%S"), "a": round(at, 2), "d": round(dur, 2),
                                "p": part, "text": r["text"], "lang": r.get("lang", "en"),
                                "ko": "" if hangul_dominant(r["text"]) else None, "at": time.time()})
            seg = Segmenter(emit)
            with pcm.open("rb") as f:
                f.seek(int(start * SR) * 2)
                while True:
                    b = f.read(SR * 2 * 10)
                    if len(b) < 2:
                        break
                    seg.feed(np.frombuffer(b[:len(b) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0)
                seg.flush()
            base = max((l.get("i", -1) for l in lines), default=-1) + 1
            for k, l in enumerate(new):
                l["i"] = base + k
            lines += new
            ended = end_wall
        else:
            ended = datetime.datetime.now()
        if lines:
            write_session(sess, lines)
        pcm_to_m4a(sess)
        try:
            meta = json.loads(mp.read_text())   # write_session 이 갱신한 n_lines·preview 포함
        except Exception:
            pass
        meta.update({"state": "done", "polished": False, "ended": ended.isoformat()})
        write_json_atomic(mp, meta)

    def _live_tail(self):
        """라이브 박스에 보일 텍스트 = 마지막 확정 라인 이후. lock 안에서 호출.
        매칭 실패(초안 구간 재디코딩으로 단어가 바뀜)하면 마지막 성공 컷을 유지한다."""
        n, text = len(self.lines), self._live_text
        if not n or self.lines[-1].get("i", -1) < self._live_base:
            return text[-2000:]
        idx, cut = self._live_cut
        c = live_cut(text, self.lines[-1]["text"], cut if idx < n else 0)
        if c is not None:
            self._live_cut = (n, c); cut = c
        elif idx == 0:
            return ("…" + text[-600:]) if len(text) > 600 else text
        return text[cut:].lstrip(" ,.;:")

    def set_cfg(self, key, val):
        if key == "langs":
            v = norm_langs(val)
            if v is None:      # 전부 해제는 막는다 — 최소 하나는 남아야 전사가 된다
                return
            self.cfg[key] = v
        elif val in CFG_ALLOWED.get(key, ()):
            self.cfg[key] = val
        elif key == "agy_account":
            if val == "main" or (isinstance(val, str) and "/" not in val
                                 and (ACCT_STORE / val).is_dir()):
                self.cfg[key] = val
            else:
                return
        elif key == "input":            # 입력 장치 이름 — 빈 문자열이면 기본
            self.cfg[key] = str(val or "").strip()[:120]
        elif key == "mat_dir":          # 자료 자동 정리 폴더 — 빈 문자열이면 끔
            v = str(val or "").strip()
            if v and not pathlib.Path(v).expanduser().is_dir():
                return
            self.cfg[key] = v
        elif key in CFG_NUM:
            try:
                v = float(val)
            except (TypeError, ValueError):
                return
            lo, hi = CFG_NUM[key]
            if not lo <= v <= hi:
                return
            self.cfg[key] = round(v, 2) if key == "line_h" else int(v)
        else:
            return
        save_cfg(self.cfg)
        if key == "live":
            self._apply_live(val)
        self.bump()

    def _apply_live(self, val):
        """라이브 층 즉시 켜기/끄기. 끄면 스트림을 닫고 parakeet 가중치(~2.3GB)를 놓는다."""
        if val == "off":
            if self._live_stop is not None:
                self._live_stop.set()
            with self.lock:
                self._live_text = ""; self.live_buffer = ""; self.rev += 1
            self._pk = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
        elif self._liveq is not None and self._live_stop is not None \
                and self._live_stop.is_set() and self.state in ("recording", "paused"):
            self._live_stop = threading.Event()   # 녹음 중이면 그 자리에서 다시 띄운다
            threading.Thread(target=self._live_loop,
                             args=(self._liveq, self._live_stop), daemon=True).start()

    def set_translate(self, mode):
        self.set_cfg("translate", mode)


# ---------- 메모리 예상치 ----------
# 실측(M4 Pro, 이 저장소 .venv): 확정 워커 RSS 1.83GB(디코드 순간 피크 2.48),
# 라이브 parakeet 스트림 RSS 1.59GB(피크 2.62), 메인 프로세스 base ~0.2GB + ffmpeg.
MEM_PARTS = [("기본", 0.25), ("확정 전사", 1.85), ("라이브 미리보기", 1.60)]

def total_ram():
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                  capture_output=True, text=True).stdout.strip()) / 1e9
    except Exception:
        try:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        except Exception:
            return 0.0

def mem_estimate(cfg=None):
    """녹음 중 예상 램 사용량(GB). 실측 상수 기반 추정이지 현재 사용량이 아니다."""
    cfg = cfg or load_cfg()
    live_on = cfg.get("live", "on") != "off"
    parts = [{"name": n, "gb": g} for n, g in MEM_PARTS
             if live_on or n != "라이브 미리보기"]
    est = sum(p["gb"] for p in parts)
    off = sum(g for n, g in MEM_PARTS if n != "라이브 미리보기")
    on = sum(g for _, g in MEM_PARTS)
    total = total_ram()
    pct = (lambda v: round(v / total * 100, 1) if total else 0.0)
    return {"total": round(total, 1), "est": round(est, 2), "pct": pct(est),
            "est_off": round(off, 2), "pct_off": pct(off),
            "est_on": round(on, 2), "pct_on": pct(on),
            "live": "on" if live_on else "off", "parts": parts}


# ---------- 자동 업데이트 (GitHub Releases) ----------
INSTALL_DIR = pathlib.Path.home() / ".geunyang_dictation"
PAYLOAD_MEMBERS = {"app.py", "engine.py", "viewer.html", "requirements.txt", "make_app.sh"}
_upd = {"at": 0.0, "data": None}

def _vtuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or "0")) or (0,)

def check_update(force=False):
    """최신 릴리스 조회(공개 저장소라 토큰 불필요). 1시간 캐시.
    can_update 는 설치본(~/.geunyang_dictation)에서 돌 때만 참 — 개발 저장소를
    릴리스본으로 덮어쓰는 사고를 막는다."""
    can = ROOT == INSTALL_DIR
    now = time.time()
    if not force and _upd["data"] and now - _upd["at"] < 3600:
        return {**_upd["data"], "can_update": can}
    out = {"current": VERSION, "latest": "", "url": "", "notes": "",
           "available": False, "error": ""}
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://api.github.com/repos/{UPDATE_REPO}/releases/latest",
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "geunyang-dictation"})
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read().decode())
        out["latest"] = (d.get("tag_name") or "").lstrip("v")
        out["notes"] = (d.get("body") or "").strip()[:300]
        out["url"] = next((a["browser_download_url"] for a in d.get("assets", [])
                           if a.get("name") == "payload.tgz"), "")
        out["available"] = bool(out["url"]) and _vtuple(out["latest"]) > _vtuple(VERSION)
    except Exception as e:
        out["error"] = f"확인 실패: {str(e)[:70]}"
    _upd.update(at=now, data=out)
    return {**out, "can_update": can}

def apply_update():
    """payload.tgz 를 받아 설치 폴더에 덮어쓴다. transcripts·config.json 은 페이로드에
    없으니 그대로 남는다. (성공, 새 버전 또는 오류메시지) 반환."""
    info = check_update(force=True)
    if not info["can_update"]:
        return False, "설치본이 아니어서 업데이트하지 않아요"
    if not info["available"]:
        return False, info["error"] or "이미 최신이에요"
    import tarfile, tempfile, urllib.request
    try:
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            tgz = td / "payload.tgz"
            urllib.request.urlretrieve(info["url"], tgz)
            with tarfile.open(tgz) as tf:
                names = tf.getnames()
                if any(n.startswith("/") or ".." in pathlib.PurePosixPath(n).parts
                       for n in names):
                    return False, "받은 파일 경로가 이상해요"
                if not PAYLOAD_MEMBERS <= set(names):
                    return False, "받은 파일에 앱 구성요소가 빠졌어요"
                tf.extractall(td / "x")
            stage = td / "x"
            req_old = (ROOT / "requirements.txt").read_text()
            for item in stage.iterdir():
                dst = ROOT / item.name
                if item.is_dir():
                    shutil.rmtree(dst, ignore_errors=True)
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
            if (ROOT / "requirements.txt").read_text() != req_old:
                uv = shutil.which("uv") or str(pathlib.Path.home() / ".local/bin/uv")
                subprocess.run([uv, "pip", "install", "-q", "-r", "requirements.txt"],
                               cwd=str(ROOT), timeout=900)
            os.chmod(ROOT / "make_app.sh", 0o755)
            subprocess.run(["/bin/zsh", str(ROOT / "make_app.sh")],
                           cwd=str(ROOT), timeout=300)
    except Exception as e:
        return False, f"업데이트 실패: {str(e)[:110]}"
    _upd.update(at=0.0, data=None)
    return True, info["latest"]


_usage_cache = {}

def agy_usage():
    """agy /usage 조회(무과금). 계정별 5분 캐시 — 설정창을 열 때만 호출된다."""
    alias = load_cfg().get("agy_account", "main")
    now = time.time()
    c = _usage_cache.get(alias)
    if c and now - c["t"] < 300:
        return c["data"]
    binary = shutil.which("agy") or str(pathlib.Path.home() / ".local/bin/agy")
    if not pathlib.Path(binary).exists():
        data = {"status": "missing"}
    else:
        work = ROOT / ".agywork"
        work.mkdir(exist_ok=True)
        try:
            r = subprocess.run([binary, "-p", "/usage", "--output-format", "json"],
                               input="", capture_output=True, text=True,
                               timeout=30, cwd=str(work), env=_acct_env(alias))
            if r.returncode != 0:
                data = {"status": "auth" if "auth" in (r.stderr or "").lower() else "error"}
            else:
                j = json.loads(r.stdout)
                data = {"status": "ok"}
                for g in j.get("command", {}).get("data", {}).get("groups", []):
                    for b in g.get("buckets", []):
                        if b.get("id") == "gemini-5h":
                            data["h5"] = b.get("remaining_fraction")
                        elif b.get("id") == "gemini-weekly":
                            data["wk"] = b.get("remaining_fraction")
        except Exception:
            data = {"status": "error"}
    _usage_cache[alias] = {"t": now, "data": data}
    return data


# ---------- 오디오 파트 · 입력 장치 ----------
# 녹음을 이어 붙이면 실행마다 audio.m4a, audio-2.m4a, audio-3.m4a … 가 생긴다.
# 라인의 p 는 파트 번호(0부터), a 는 그 파트 안의 시작 초, d 는 길이.

def part_name(k):
    return "audio.m4a" if k == 0 else f"audio-{k + 1}.m4a"

def next_part(sess):
    return len(list(sess.glob("audio*.m4a")))

def write_json_atomic(path, obj):
    write_text_atomic(path, json.dumps(obj, ensure_ascii=False))

def write_text_atomic(path, text):
    """임시 파일에 쓰고 바꿔 끼운다 — 쓰는 도중 앱이 꺼져도 원래 파일이 반쯤 잘리지 않는다."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)

def update_meta(sess, **kw):
    """meta.json 을 다시 읽어 kw 를 반영(값이 None 이면 키 삭제)하고 원자적으로 쓴다."""
    mp = sess / "meta.json"
    try:
        meta = json.loads(mp.read_text())
    except Exception:
        meta = {"id": sess.name}
    for k, v in kw.items():
        if v is None:
            meta.pop(k, None)
        else:
            meta[k] = v
    write_json_atomic(mp, meta)
    return meta

def pcm_to_m4a(sess):
    """audio.pcm → 다음 파트 m4a. 성공하면 pcm 을 지운다."""
    pcm = sess / "audio.pcm"
    if not pcm.exists():
        return True
    if pcm.stat().st_size < SR * 2 // 10:     # 0.1초도 안 되면 버린다
        pcm.unlink(); return True
    out = sess / part_name(next_part(sess))
    r = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", str(pcm),
                        "-c:a", "aac", "-b:a", "48k", "-movflags", "+faststart", str(out)])
    if r.returncode == 0:
        pcm.unlink(); return True
    out.unlink(missing_ok=True)
    return False

def fill_parts(lines):
    """p 가 없는 옛 라인에 파트 번호를 채운다 — 이어 녹음하면 a 가 0 근처로 되돌아가는 점을 쓴다."""
    cur, prev = 0, -1.0
    for l in lines:
        a = l.get("a")
        if isinstance(l.get("p"), int):
            cur = l["p"]
        elif isinstance(a, (int, float)):
            if a < prev - 1.0:
                cur += 1
            l["p"] = cur
        if isinstance(a, (int, float)):
            prev = a
    return lines

def hms(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600:02d}:{sec // 60 % 60:02d}:{sec % 60:02d}"

def probe_duration(path):
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        pass
    try:
        r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        return 0.0

def list_input_devices():
    """avfoundation 오디오 입력 장치 이름 (인덱스 순)."""
    try:
        r = subprocess.run([FFMPEG, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                           capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    out, on = [], False
    for line in r.stderr.splitlines():
        if "audio devices" in line:
            on = True; continue
        if on:
            m = re.search(r"\]\s*\[(\d+)\]\s*(.+)$", line)
            if not m:
                break
            out.append(m.group(2).strip())
    return out

def input_arg(name):
    """설정한 장치 이름 → ffmpeg avfoundation 입력. 없거나 빠졌으면 기존 기본값(:0)."""
    if name:
        devs = list_input_devices()
        if name in devs:
            return f":{devs.index(name)}"
    return ":0"


def write_session(sess, lines):
    """세션 라인을 디스크에 반영 — lines.json · transcript.md · meta(n_lines/preview)."""
    write_json_atomic(sess / "lines.json", lines)
    md = sess / "transcript.md"
    head = ""
    try:
        first = md.read_text().split("\n", 1)[0]
        if first.startswith("# "):
            head = first + "\n\n"
    except Exception:
        pass
    with md.open("w") as f:
        f.write(head or f"# 그냥 받아쓰기 — {sess.name}\n\n")
        for l in lines:
            f.write(f"**{l['t']}** {l['text']}\n")
            if l.get("ko"):
                f.write(f"> {l['ko']}\n")
            f.write("\n")
    mp = sess / "meta.json"
    try:
        meta = json.loads(mp.read_text())
    except Exception:
        meta = {"id": sess.name}
    meta["n_lines"] = len(lines)
    meta["preview"] = lines[0]["text"][:80] if lines else ""
    write_json_atomic(mp, meta)


def apply_line_ops(lines, action, ids, text, ko):
    """라인 목록에 편집 적용. (새 목록 or None, 에러) — i 는 안정 키라 재번호하지 않는다."""
    ids = set(ids or [])
    if action == "delete":
        keep = [l for l in lines if l.get("i") not in ids]
        return (keep, "") if len(keep) != len(lines) else (None, "대상 라인이 없어요")
    if action == "edit":
        tgt = next((l for l in lines if l.get("i") in ids), None)
        if tgt is None:
            return None, "대상 라인이 없어요"
        if text is not None:
            t = " ".join(str(text).split())
            if not t:
                return None, "빈 내용으로는 못 바꿔요 (삭제를 쓰세요)"
            tgt["text"] = t[:2000]
        if ko is not None:
            tgt["ko"] = " ".join(str(ko).split())[:2000]
        return lines, ""
    return None, "알 수 없는 동작"


def list_sessions():
    out = []
    if TR.exists():
        for d in sorted(TR.iterdir(), reverse=True):
            m = d / "meta.json"
            if m.exists():
                try:
                    out.append(json.loads(m.read_text()))
                except Exception:
                    pass
            elif (d / "transcript.md").exists():
                out.append({"id": d.name, "state": "done", "n_lines": None})
    return out


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker_main()
