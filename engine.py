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
import asyncio, datetime, json, logging, os, pathlib, queue, re, shutil, struct, subprocess, sys, threading, time
import numpy as np

logging.basicConfig(level=logging.ERROR)
for _n in ("whisperlivekit",):
    logging.getLogger(_n).setLevel(logging.ERROR)

ROOT = pathlib.Path(__file__).resolve().parent
TR = ROOT / "transcripts"
CONFIG = ROOT / "config.json"
VERSION = "1.2.0"          # release.sh 가 여기를 올린다
UPDATE_REPO = "ChanchanCode/geunyang_dictation"
MODEL = "mlx-community/whisper-large-v3-turbo"
SR = 16000
FFMPEG = shutil.which("ffmpeg") or next(
    (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
     if pathlib.Path(p).exists()), "ffmpeg")
AGY_MODEL = "gemini-3.8-flash-low"          # 실시간 번역 — 짧은 배치, 속도·쿼터 우선
AGY_MODEL_HEAVY = "gemini-3.8-flash-medium" # 다듬기·요약·자료 변환 — 긴 문맥, 정확도 우선
AGY_TURNS_MAX = 12
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

    def turn(self, prompt, timeout=180, preamble=""):
        """preamble 은 프로세스가 새로 뜬 첫 턴에만 앞에 붙는다(긴 참고자료를 매 턴 다시 보내지 않게)."""
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
        text, deadline = "", time.time() + timeout
        while True:
            try:
                ev, body = self.q.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
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
        if s:
            e = subs.setdefault(s, {"name": s, "count": 0, "last": m["id"]})
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

def polish_preamble(ctx, slides):
    """새 사이드카 프로세스의 첫 턴에만 붙는 참고자료(맥락 노트 + 강의자료 본문)."""
    s = f"수업 맥락: {ctx}\n\n" if ctx else ""
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


def whisper_prompt(ctx, limit=260):
    """맥락 텍스트 → Whisper initial_prompt. 긴 한국어 프롬프트(슬라이드 요약 등)는
    디코더를 망가뜨려 빈 결과·환각만 낸다(실측). 영문 용어만 추려 200자 이내로."""
    m = re.search(r"^Terms:\s*(.+)$", ctx or "", re.M | re.I)  # Gemini 정리 노트면 용어 줄만
    src = m.group(1) if m else (ctx or "")
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
               "ko_width": 360, "ko_font": 14, "line_h": 1.65,
               "agy_account": "main", "port": 8765, "live": "on",
               "langs": list(DEFAULT_LANGS)}
ACCT_STORE = pathlib.Path.home() / ".claude/.state/gemini-accounts"
CFG_ALLOWED = {"translate": ("live", "after", "off"),
               "theme": ("auto", "light", "dark", "term", "term-light"),
               "layout": ("inline", "side"),
               "live": ("on", "off")}   # live=off: parakeet 미로드 (RAM ~2.3GB 절약)
CFG_NUM = {"ko_width": (200, 600), "ko_font": (11, 20), "line_h": (1.2, 2.4)}

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
        self._flush_translate = threading.Event()

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
    def start(self, sid=None):
        if self.state != "idle":
            return
        self.state = "starting"; self.bump()
        self._call(self._session(sid))

    def pause(self):
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
        meta_p.write_text(json.dumps(meta, ensure_ascii=False))
        self._ctx = (meta.get("context") or "").strip()[:CTX_MAX]
        self._wprompt = whisper_prompt(self._ctx)  # Whisper 에는 영문 용어만 (한국어 장문은 금지)

        # 워커·라이브 엔진 로드 (첫 실행은 모델 다운로드로 오래 걸림)
        ok = await asyncio.get_event_loop().run_in_executor(None, lambda: self.fix.proc or self.fix.start())
        if not ok:
            with self.lock:
                self.state = "idle"; self.error = "확정 전사 워커 시작 실패"
            self.bump(); return
        liveq = queue.Queue()
        live_stop = threading.Event()
        self._liveq, self._live_stop = liveq, live_stop
        if self.cfg.get("live") != "off":   # off 면 parakeet 로드·급전 전부 생략
            threading.Thread(target=self._live_loop, args=(liveq, live_stop),
                             daemon=True).start()
        else:
            live_stop.set()   # 소비자가 없으니 급전도 막는다 (큐 무한 적재 방지)

        stop = asyncio.Event(); pause = asyncio.Event()
        self._stop_evt, self._pause_evt = stop, pause
        self._abort = False
        pcm_path = sess / "audio.pcm"
        pcm_f = open(pcm_path, "ab")

        def on_utt(audio, at, dur):
            wall = (datetime.datetime.now()
                    - datetime.timedelta(seconds=max(0.0, seg.pos - at))).strftime("%H:%M:%S")
            self.fixq.put((sid, sess, audio, at, dur, wall, self._wprompt))
        seg = Segmenter(on_utt)

        with self.lock:
            self.state = "recording"
        self.bump()

        try:
            while not stop.is_set():
                if pause.is_set():
                    with self.lock:
                        self.state = "paused"
                    self.bump()
                    seg.flush()
                    while pause.is_set() and not stop.is_set():
                        await asyncio.sleep(0.2)
                    continue
                import os
                src = (["-re", "-i", os.environ["GY_INPUT"]]
                       if os.environ.get("GY_INPUT") else ["-f", "avfoundation", "-i", ":0"])
                proc = await asyncio.create_subprocess_exec(
                    FFMPEG, "-hide_banner", "-loglevel", "error", *src,
                    "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                    stdin=asyncio.subprocess.DEVNULL)
                try:
                    while not stop.is_set() and not pause.is_set():
                        try:
                            chunk = await asyncio.wait_for(proc.stdout.read(SR // 5 * 2), timeout=2)
                        except asyncio.TimeoutError:
                            if proc.returncode is not None:
                                raise IOError("ffmpeg 종료(마이크 권한?)")
                            continue
                        if not chunk:
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
            if self._abort:
                try:  # 확정 대기열 폐기 — 교정·저장 전부 생략
                    while True:
                        self.fixq.get_nowait(); self.fixq.task_done()
                except queue.Empty:
                    pass
                with self.lock:
                    self.sid = None; self.lines = []
            else:
                try:
                    await asyncio.get_event_loop().run_in_executor(None, self._finalize, sid, sess, start_wall)
                except Exception as ex:
                    logging.exception("finalize 실패")
                    with self.lock:
                        self.error = f"저장 실패: {ex}"[:120]
            with self.lock:
                self.state = "idle"; self.live = []; self.live_buffer = ""; self._live_text = ""
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

    # -- 확정 전사 소비자 (스레드)
    def _fix_worker(self):
        while True:
            sid, sess, audio, at, dur, wall, ctx = self.fixq.get()
            try:
                self._fix_one(sid, sess, audio, at, dur, wall, ctx)
            finally:
                self.fixq.task_done()

    def _fix_one(self, sid, sess, audio, at, dur, wall, ctx):
            r = self.fix.transcribe(audio, ctx, self.cfg.get("langs") or DEFAULT_LANGS)
            if r is None:
                with self.lock:
                    self.error = "확정 전사 워커 응답 없음(재시작 시도 중)"; self.rev += 1
                return
            if not r.get("text"):
                return
            text = r["text"]
            line = {"t": wall, "a": round(at, 2), "text": text,
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

    # -- 종료 처리: 번역 마무리/교정 → 저장 → 오디오 압축
    def _finalize(self, sid, sess, start_wall):
        if not sess.exists():  # 마무리 중 세션이 삭제된 경우 (UI 삭제 등)
            return
        # 확정 큐 소진 대기 — 처리 중인 마지막 발화까지 (empty() 는 꺼낸 즉시 참이 되어 유실됨)
        t0 = time.time()
        while self.fixq.unfinished_tasks and time.time() - t0 < 120:
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
        (sess / "lines.json").write_text(json.dumps(lines, ensure_ascii=False))
        ended = datetime.datetime.now()
        # 오디오 압축 (pcm → m4a)
        pcm = sess / "audio.pcm"
        if pcm.exists():
            parts = sorted(sess.glob("audio*.m4a"))
            out = sess / ("audio.m4a" if not parts else f"audio-{len(parts)+1}.m4a")
            r = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                                "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", str(pcm),
                                "-c:a", "aac", "-b:a", "48k", str(out)])
            if r.returncode == 0:
                pcm.unlink()
        meta = {"id": sid, "started": start_wall.isoformat()}
        try:  # 이름·최초 시작시각·맥락 보존
            om = json.loads((sess / "meta.json").read_text())
            for k in ("title", "started", "context", "context_src", "archived"):
                if om.get(k):
                    meta[k] = om[k]
        except Exception:
            pass
        meta.update({"ended": ended.isoformat(), "n_lines": len(lines), "state": "done",
                     "polished": False,   # 새 라인이 생겼으니 다듬기는 다시 필요
                     "preview": (lines[0]["text"][:80] if lines else "")})
        (sess / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
        self.fix.stop()  # 워커 메모리 반납 — 다음 세션 시작 때 다시 띄운다

    # -- 전체 문맥 다듬기 (원문 교정 + 번역). 종료 후 버튼으로 실행.
    def start_polish(self, sid):
        sess = TR / sid
        if not (sess / "lines.json").exists():
            return False, "다듬을 내용이 없어요"
        if self.state != "idle":
            return False, "녹음이 끝난 뒤에 다듬을 수 있어요"
        if self.polish["busy"]:
            return False, "이미 다듬는 중이에요"
        if not Agy(str(ROOT / ".agywork")).available():
            return False, "agy(Antigravity CLI)가 없어 다듬기를 못 해요"
        self.polish = {"sid": sid, "busy": True, "note": "준비 중…", "error": ""}
        self.bump()
        threading.Thread(target=self._polish_job, args=(sid, sess), daemon=True).start()
        return True, ""

    def _polish_job(self, sid, sess):
        def note(t):
            self.polish = {"sid": sid, "busy": True, "note": t, "error": ""}
            self.bump()
        try:
            self._polish_core(sid, sess, note)
            self.polish = {"sid": sid, "busy": False, "note": "다듬기 완료", "error": ""}
        except Exception as e:
            self.polish = {"sid": sid, "busy": False, "note": "", "error": f"다듬기 실패: {e}"[:140]}
        self.bump()

    def _polish_core(self, sid, sess, note):
        """전체 문맥 다듬기 본체. 자료가 아직 텍스트로 안 바뀌었으면 먼저 변환해 참고한다.
        완료 시 meta.polished=True. 실패는 예외로."""
        lines = json.loads((sess / "lines.json").read_text())
        for k, l in enumerate(lines):
            l.setdefault("i", k)
        if not lines:
            raise ValueError("다듬을 라인이 없어요")
        if ctx_pending(sess):
            if conv_status(sid).get("busy"):
                note("강의자료 변환을 기다리는 중…"); wait_conv(sid)
            else:
                convert_ctx(sess, note)
        try:
            ctx = json.loads((sess / "meta.json").read_text()).get("context", "")
        except Exception:
            ctx = ""
        want_ko = self.cfg.get("translate") != "off"
        head = POLISH_HEAD + (POLISH_KO if want_ko else POLISH_EN)
        pre = polish_preamble(ctx, ctx_text_all(sess, 24000))
        agy = Agy(str(ROOT / ".agywork"), model=AGY_MODEL_HEAVY, print_timeout="8m")
        (ROOT / ".agywork").mkdir(exist_ok=True)
        STEP, fails = 25, 0
        for i in range(0, len(lines), STEP):
            chunk = lines[i:i + STEP]
            note(f"다듬는 중… {i}/{len(lines)}줄")
            out = agy.turn(head + "\n" + "\n".join(
                f"{k+1}. {l['text']}" for k, l in enumerate(chunk)), timeout=420, preamble=pre)
            got = parse_polish(out, want_ko)
            if not got:
                fails += 1
                if fails >= 3:
                    agy.stop()
                    raise RuntimeError("Gemini 응답을 받지 못했어요 (agy 로그인 확인)")
                continue
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
            write_session(sess, lines)          # 청크마다 저장 — 중단돼도 진행분은 남는다
        agy.stop()
        mp = sess / "meta.json"
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            meta = {"id": sid}
        meta["polished"] = True
        mp.write_text(json.dumps(meta, ensure_ascii=False))
        return lines

    # -- 강의 요약·정리본 (자료 변환 → 필요하면 다듬기 → Gemini 한 턴). 종료 후 버튼으로 실행.
    def start_summary(self, sid):
        sess = TR / sid
        if not (sess / "lines.json").exists():
            return False, "요약할 내용이 없어요"
        if self.state != "idle":
            return False, "녹음이 끝난 뒤에 요약할 수 있어요"
        if self.polish["busy"] or self.summary["busy"]:
            return False, "이미 작업 중이에요"
        if not Agy(str(ROOT / ".agywork")).available():
            return False, "agy(Antigravity CLI)가 없어 요약을 못 해요"
        self.summary = {"sid": sid, "busy": True, "note": "준비 중…", "error": ""}
        self.bump()
        threading.Thread(target=self._summary_job, args=(sid, sess), daemon=True).start()
        return True, ""

    def _summary_job(self, sid, sess):
        def note(t):
            self.summary = {"sid": sid, "busy": True, "note": t, "error": ""}
            self.bump()
        try:
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
            prompt = SUMMARY_PROMPT + (f"\n수업 맥락: {ctx}\n" if ctx else "")
            if slides:
                prompt += "\n=== 강의자료 ===\n" + slides + "\n"
            prompt += "\n=== 강의 전사 ===\n" + "\n".join(f"[{l['i']}] {l['text']}" for l in lines)
            note("요약 만드는 중… (몇 분 걸려요)")
            agy = Agy(str(ROOT / ".agywork"), model=AGY_MODEL_HEAVY, agent=None, print_timeout="12m")
            out = _strip_fence(agy.turn(prompt, timeout=720))
            agy.stop()
            if not out or "## " not in out:
                raise RuntimeError("Gemini 응답을 받지 못했어요 (agy 로그인 확인)")
            (sess / "summary.md").write_text(out)
            meta["summary_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            mp.write_text(json.dumps(meta, ensure_ascii=False))
            self.summary = {"sid": sid, "busy": False, "note": "요약 완료", "error": ""}
        except Exception as e:
            logging.exception("요약 실패")
            self.summary = {"sid": sid, "busy": False, "note": "", "error": f"요약 실패: {e}"[:140]}
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
                    "polish": dict(self.polish), "summary": dict(self.summary)}

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


def write_session(sess, lines):
    """세션 라인을 디스크에 반영 — lines.json · transcript.md · meta(n_lines/preview)."""
    (sess / "lines.json").write_text(json.dumps(lines, ensure_ascii=False))
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
    mp.write_text(json.dumps(meta, ensure_ascii=False))


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
