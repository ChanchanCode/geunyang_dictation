#!/usr/bin/env python3
"""그냥 받아쓰기 — 메뉴바 앱 (rumps) + 로컬 뷰어 서버.

메뉴바 '냥' 아이콘: 대기 냥 · 녹음 냥● · 일시정지 냥‖
뷰어: http://127.0.0.1:<port>  (config.json)
"""
import json, pathlib, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rumps

import engine as eng

ROOT = pathlib.Path(__file__).resolve().parent
ENGINE = eng.Engine()
VIEWER = ROOT / "viewer.html"


def ctx_files(sid):
    cd = eng.TR / sid / "ctx"
    return sorted(p.name for p in cd.iterdir() if p.is_file() and not p.name.startswith(".")) \
        if cd.is_dir() else []


def ctx_info(sid):
    """자료 목록 + 크기 + 텍스트 변환 여부 (자료 패널용)."""
    cd, td = eng.TR / sid / "ctx", eng.TR / sid / eng.CTX_TEXT_DIR
    out = []
    for n in ctx_files(sid):
        t = td / (n + ".md")
        out.append({"name": n, "size": (cd / n).stat().st_size, "converted": t.exists(),
                    "chars": t.stat().st_size if t.exists() else 0})
    return out


def read_meta(sid):
    try:
        return json.loads((eng.TR / sid / "meta.json").read_text())
    except Exception:
        return {"id": sid}


def write_meta(sid, meta):
    (eng.TR / sid / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))


def set_subject(sid, subject, title=None):
    """과목 지정(+제목). 제목을 안 주면 제목이 비었거나 자동 제목일 때만 추천 제목으로 바꾼다.
    과목이 정해졌으니 이미 올린 자료를 자료 폴더로 복사한다."""
    meta = read_meta(sid)
    meta["subject"] = subject
    if title:
        meta["title"] = title; meta["title_auto"] = False
    elif not meta.get("title") or meta.get("title_auto"):
        meta["title"] = eng.suggest_title(sid, subject); meta["title_auto"] = True
    write_meta(sid, meta)
    return meta, eng.file_materials(eng.TR / sid, meta)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, body, ct="application/json; charset=utf-8", code=200, dl=None):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        if dl:
            self.send_header("Content-Disposition", f'attachment; filename="{dl}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/" or p == "/index.html":
            self._send(VIEWER.read_bytes(), "text/html; charset=utf-8")
        elif p == "/api/state":
            self._send(json.dumps(ENGINE.snapshot(), ensure_ascii=False).encode())
        elif p == "/api/sessions":
            self._send(json.dumps(eng.list_sessions(), ensure_ascii=False).encode())
        elif p.startswith("/api/session/"):
            sid = pathlib.Path(p.rsplit("/", 1)[1]).name
            f = eng.TR / sid / "lines.json"
            lines = json.loads(f.read_text()) if f.exists() else []
            if not lines and (eng.TR / sid / "transcript.md").exists():
                # 진행 중 크래시 등으로 lines.json 이 없으면 md 를 원문만이라도 보여준다
                import re
                md = (eng.TR / sid / "transcript.md").read_text()
                lines = [{"t": m.group(1), "text": m.group(2),
                          "ko": (m.group(3) or "").strip()}
                         for m in re.finditer(
                             r"\*\*(\d\d:\d\d:\d\d)\*\* (.+)(?:\n> (.+))?", md)]
            for k, l in enumerate(lines):
                l.setdefault("i", k)   # 라인 편집용 안정 키
            meta = read_meta(sid)
            sf = eng.TR / sid / "summary.md"
            summary = sf.read_text() if sf.exists() else ""
            self._send(json.dumps({
                "lines": lines, "context": meta.get("context", ""),
                "polished": bool(meta.get("polished")), "ctx_files": ctx_files(sid),
                "ctx_info": ctx_info(sid),
                "ctx_text": eng.ctx_text_files(eng.TR / sid), "conv": eng.conv_status(sid),
                "subject": eng.subject_of(meta), "title": meta.get("title", ""),
                "title_auto": bool(meta.get("title_auto")),
                "summary": summary, "summary_at": meta.get("summary_at", ""),
                "outline": eng.parse_outline(summary)}, ensure_ascii=False).encode())
        elif p == "/api/subjects":
            # 과목 캡슐용: 과목 목록(최근순) + 이 세션 날짜의 주차·요일
            q = dict(x.split("=", 1) for x in self.path.split("?", 1)[1].split("&")
                     if "=" in x) if "?" in self.path else {}
            sid = pathlib.Path(q.get("sid", "")).name
            self._send(json.dumps(eng.subjects_info(sid or None), ensure_ascii=False).encode())
        elif p == "/api/config":
            self._send(json.dumps({**ENGINE.cfg, "lang_options": eng.MAJOR_LANGS},
                                  ensure_ascii=False).encode())
        elif p == "/api/update":
            self._send(json.dumps(eng.check_update("force=1" in self.path),
                                  ensure_ascii=False).encode())
        elif p == "/api/mem":
            self._send(json.dumps(eng.mem_estimate(ENGINE.cfg), ensure_ascii=False).encode())
        elif p == "/api/agy":
            self._send(json.dumps(eng.agy_usage()).encode())
        elif p == "/api/agy_accounts":
            self._send(json.dumps(eng.agy_accounts(), ensure_ascii=False).encode())
        elif p.startswith("/download/"):
            sid = pathlib.Path(p.rsplit("/", 1)[1]).name
            f = eng.TR / sid / "transcript.md"
            if f.exists():
                self._send(f.read_bytes(), "text/markdown; charset=utf-8",
                           dl=f"{sid}.md")
            else:
                self._send(b"not found", "text/plain", 404)
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        if self.path == "/api/session/new":
            import datetime, shutil
            if eng.TR.exists():  # 안 쓴 빈 세션 정리
                for d in eng.TR.iterdir():
                    try:
                        md = json.loads((d / "meta.json").read_text())
                        if md.get("state") == "new" and not (d / "lines.json").exists()                                 and not (d / "transcript.md").exists():
                            shutil.rmtree(d)
                    except Exception:
                        pass
            base = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
            sid, k = base, 2
            while (eng.TR / sid).exists():
                sid = f"{base}-{k}"; k += 1
            (eng.TR / sid).mkdir(parents=True)
            (eng.TR / sid / "meta.json").write_text(json.dumps(
                {"id": sid, "state": "new",
                 "started": datetime.datetime.now().isoformat()}, ensure_ascii=False))
            self._send(json.dumps({"id": sid}).encode())
            return
        if self.path == "/api/control":
            a = body.get("action")
            if a == "quit":
                self._send(b'{"ok":true}')
                def _bye():
                    import os, time
                    if ENGINE.state in ("recording", "paused"):
                        ENGINE.stop()
                    t0 = time.time()
                    while ENGINE.state != "idle" and time.time() - t0 < 300:
                        time.sleep(1)
                    os._exit(0)
                threading.Thread(target=_bye, daemon=True).start()
                return
            if a == "start":
                ENGINE.start(body.get("sid"))
            else:
                {"pause": ENGINE.pause, "resume": ENGINE.resume,
                 "stop": ENGINE.stop}.get(a, lambda: None)()
            self._send(b'{"ok":true}')
        elif self.path == "/api/session/rename":
            sid = pathlib.Path(str(body.get("id", ""))).name
            title = str(body.get("title", "")).strip()[:60]
            d = eng.TR / sid
            if sid and title and d.is_dir():
                mf = d / "meta.json"
                try:
                    meta = json.loads(mf.read_text()) if mf.exists() else {"id": sid, "state": "done"}
                except ValueError:
                    meta = {"id": sid, "state": "done"}
                meta["title"] = title
                meta["title_auto"] = False        # 사용자가 직접 정한 제목은 추천이 덮지 않는다
                mf.write_text(json.dumps(meta, ensure_ascii=False))
                self._send(b'{"ok":true}')
            else:
                self._send(b'{"ok":false}', code=400)
        elif self.path == "/api/session/subject":
            # 과목 지정: {id, subject, title?} → meta.subject(+추천 제목), 자료 폴더 복사
            sid = pathlib.Path(str(body.get("id", ""))).name
            subject = " ".join(str(body.get("subject", "")).split())[:40]
            title = str(body.get("title", "")).strip()[:60] or None
            if not sid or not subject or not (eng.TR / sid).is_dir():
                self._send(b'{"ok":false,"error":"session"}', code=400); return
            meta, copied = set_subject(sid, subject, title)
            self._send(json.dumps({"ok": True, "subject": subject, "title": meta.get("title", ""),
                                   "copied": copied}, ensure_ascii=False).encode())
        elif self.path == "/api/session/context":
            sid = pathlib.Path(str(body.get("id", ""))).name
            m = eng.TR / sid / "meta.json"
            if sid and m.exists():
                try:
                    meta = json.loads(m.read_text())
                except ValueError:
                    meta = {"id": sid}
                meta["context"] = str(body.get("context", ""))[:eng.CTX_MAX]
                m.write_text(json.dumps(meta, ensure_ascii=False))
            self._send(b'{"ok":true}')
        elif self.path == "/api/session/ctx_files":
            # 맥락 자료 업로드: {id, files:[{name, b64}]} → transcripts/<sid>/ctx/
            import base64
            sid = pathlib.Path(str(body.get("id", ""))).name
            d = eng.TR / sid
            if not sid or not d.is_dir():
                self._send(b'{"ok":false,"error":"session"}', code=400); return
            cd = d / "ctx"; cd.mkdir(exist_ok=True)
            used = sum(p.stat().st_size for p in cd.iterdir() if p.is_file())
            saved, skipped = [], []
            for f in body.get("files", [])[:30]:
                name = pathlib.Path(str(f.get("name", "file"))).name.replace("\x00", "")
                if pathlib.Path(name).suffix.lower() not in eng.CTX_FILE_EXT:
                    skipped.append(name); continue
                try:
                    data = base64.b64decode(str(f.get("b64", "")).split(",")[-1])
                except Exception:
                    skipped.append(name); continue
                if used + len(data) > eng.CTX_FILES_MAX_MB * 1024 * 1024:
                    skipped.append(name); continue
                (cd / name).write_bytes(data); used += len(data); saved.append(name)
            meta = read_meta(sid)
            guess, copied = "", []
            if saved and not eng.subject_of(meta):      # 과목이 아직 없으면 자료 이름으로 추정
                guess = next((g for g in (eng.guess_subject(n, sid) for n in saved) if g), "")
                if guess:
                    meta, copied = set_subject(sid, guess)
            elif saved:
                copied = eng.file_materials(d, meta, saved)
            conv = eng.conv_status(sid)
            if saved:   # 상세 텍스트 변환은 오래 걸리니 백그라운드(빠른 맥락 정리와 병렬) — 다듬기·요약이 기다렸다 쓴다
                threading.Thread(target=eng.convert_ctx, args=(d,), daemon=True).start()
                if not conv.get("busy"):
                    conv = {"busy": True, "note": "강의자료 변환 준비 중…", "error": ""}
            self._send(json.dumps({"ok": True, "saved": saved, "skipped": skipped,
                                   "files": ctx_files(sid), "info": ctx_info(sid), "conv": conv,
                                   "guess": guess,
                                   "subject": eng.subject_of(meta), "title": meta.get("title", ""),
                                   "copied": copied}, ensure_ascii=False).encode())
        elif self.path == "/api/session/ctx_remove":
            # 자료 하나 빼기: ctx/<name> 과 변환본. 자료 폴더에 복사한 파일은 그대로 둔다
            sid = pathlib.Path(str(body.get("id", ""))).name
            name = pathlib.Path(str(body.get("name", ""))).name
            d = eng.TR / sid
            if sid and name and (d / "ctx" / name).is_file():
                (d / "ctx" / name).unlink()
                (d / eng.CTX_TEXT_DIR / (name + ".md")).unlink(missing_ok=True)
            self._send(json.dumps({"ok": True, "files": ctx_files(sid), "info": ctx_info(sid)},
                                  ensure_ascii=False).encode())
        elif self.path == "/api/session/ctx_clear":
            import shutil
            sid = pathlib.Path(str(body.get("id", ""))).name
            cd = eng.TR / sid / "ctx"
            if sid and cd.is_dir():
                shutil.rmtree(cd, ignore_errors=True)
                shutil.rmtree(eng.TR / sid / eng.CTX_TEXT_DIR, ignore_errors=True)
            self._send(b'{"ok":true}')
        elif self.path == "/api/session/digest":
            # 자료(ctx/ 파일 + 메모) → Gemini 영어 맥락. 동기 처리(수십 초) — 스레드 서버라 다른 요청은 안 막힘
            sid = pathlib.Path(str(body.get("id", ""))).name
            d = eng.TR / sid
            if not sid or not d.is_dir():
                self._send(b'{"ok":false,"error":"session"}', code=400); return
            m = d / "meta.json"
            try:
                meta = json.loads(m.read_text())
            except Exception:
                meta = {"id": sid}
            src = str(body.get("text", "")).strip()
            # 메모가 비었으면(이미 영어로 정리된 뒤 자료만 추가한 경우) 원래 메모를 다시 넣어 내용 유실 방지
            note, err = eng.digest_context(d, src or meta.get("context_src", ""))
            if note is None:
                self._send(json.dumps({"ok": False, "error": err}, ensure_ascii=False).encode()); return
            meta["context"] = note
            if src:
                meta["context_src"] = src[:4000]  # 원래 메모(한국어) 보존
            m.write_text(json.dumps(meta, ensure_ascii=False))
            self._send(json.dumps({"ok": True, "context": note}, ensure_ascii=False).encode())
        elif self.path == "/api/session/polish":
            sid = pathlib.Path(str(body.get("id", ""))).name
            ok, err = ENGINE.start_polish(sid)
            self._send(json.dumps({"ok": ok, "error": err}, ensure_ascii=False).encode())
        elif self.path == "/api/session/summary":
            sid = pathlib.Path(str(body.get("id", ""))).name
            ok, err = ENGINE.start_summary(sid)
            self._send(json.dumps({"ok": ok, "error": err}, ensure_ascii=False).encode())
        elif self.path == "/api/pick_folder":
            # 자료 폴더 선택 — macOS 폴더 선택창(osascript). 취소하면 path 빈 문자열
            import subprocess
            try:
                r = subprocess.run(
                    ["osascript", "-e", "activate",
                     "-e", 'POSIX path of (choose folder with prompt "강의자료를 정리할 폴더를 고르세요")'],
                    capture_output=True, text=True, timeout=180)
                path = r.stdout.strip().rstrip("/") if r.returncode == 0 else ""
            except Exception:
                path = ""
            if path:
                ENGINE.set_cfg("mat_dir", path)
            self._send(json.dumps({"ok": bool(path), "path": path}, ensure_ascii=False).encode())
        elif self.path == "/api/session/lines":
            # 라인 편집: {id, action:"delete"|"edit", ids:[i...], text?, ko?}
            sid = pathlib.Path(str(body.get("id", ""))).name
            ids = [i for i in body.get("ids", []) if isinstance(i, int)][:2000]
            ok, err = ENGINE.edit_lines(
                sid, str(body.get("action", "")), ids,
                body.get("text") if isinstance(body.get("text"), str) else None,
                body.get("ko") if isinstance(body.get("ko"), str) else None)
            self._send(json.dumps({"ok": ok, "error": err}, ensure_ascii=False).encode())
        elif self.path == "/api/session/archive":
            sid = pathlib.Path(str(body.get("id", ""))).name
            m = eng.TR / sid / "meta.json"
            if sid and m.exists():
                try:
                    meta = json.loads(m.read_text())
                except ValueError:
                    meta = {"id": sid}
                meta["archived"] = bool(body.get("archived"))
                m.write_text(json.dumps(meta, ensure_ascii=False))
            self._send(b'{"ok":true}')
        elif self.path == "/api/session/delete":
            import shutil, time
            sid = pathlib.Path(str(body.get("id", ""))).name
            d = eng.TR / sid
            if sid and ENGINE.sid == sid and ENGINE.state in ("recording", "paused", "starting"):
                ENGINE.abort()  # 저장·교정 없이 즉시 중단
                t0 = time.time()
                while ENGINE.state != "idle" and time.time() - t0 < 30:
                    time.sleep(0.3)
            if sid and d.is_dir() and not (ENGINE.state != "idle" and ENGINE.sid == sid):
                trash = eng.TR / ".trash"
                trash.mkdir(exist_ok=True)
                shutil.move(str(d), str(trash / f"{sid}-{int(time.time())}"))
            self._send(b'{"ok":true}')
        elif self.path == "/api/agy_login":
            import re as _re, os, subprocess
            alias = _re.sub(r"[^a-zA-Z0-9_-]", "", str(body.get("alias", "")))[:20]
            if alias and alias != "main":
                work = ROOT / ".agywork"
                work.mkdir(exist_ok=True)
                sc = work / "login.command"
                acct_tool = pathlib.Path.home() / ".claude/bin/gemini-acct"
                if acct_tool.exists():
                    cmd = f"\"$HOME/.claude/bin/gemini-acct\" login {alias}"
                else:  # 다중 계정 도구가 없는 컴퓨터 — 기본 계정 로그인
                    cmd = "agy || \"$HOME/.local/bin/agy\""
                sc.write_text(
                    "#!/bin/zsh\n"
                    f"echo '[그냥 받아쓰기] agy 로그인을 시작해요.'\n"
                    "echo '브라우저가 열리면 구글 계정으로 로그인하세요.'\n"
                    f"{cmd}\n"
                    "echo; echo '끝났어요 — 이 창을 닫고 앱 설정을 다시 열면 반영돼요.'\n")
                os.chmod(sc, 0o755)
                subprocess.Popen(["open", "-a", "Terminal", str(sc)])
                self._send(b'{"ok":true}')
            else:
                self._send(b'{"ok":false}', code=400)
        elif self.path == "/api/update":
            if ENGINE.state != "idle":
                self._send(json.dumps({"ok": False, "msg": "녹음을 끝낸 뒤에 업데이트할 수 있어요"},
                                      ensure_ascii=False).encode())
                return
            ok, msg = eng.apply_update()
            self._send(json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False).encode())
            if ok:
                def _restart():   # 응답이 브라우저에 닿은 뒤 새 코드로 다시 뜬다
                    import os, subprocess, time
                    time.sleep(1.0)
                    subprocess.Popen(
                        ["/bin/zsh", "-c",
                         'sleep 2; open "$HOME/Applications/그냥 받아쓰기.app"'],
                        start_new_session=True)
                    os._exit(0)
                threading.Thread(target=_restart, daemon=True).start()
        elif self.path == "/api/config":
            for k in ("translate", "theme", "layout", "ko_width", "ko_font", "line_h",
                      "agy_account", "live", "langs", "mat_dir", "toc_w"):
                if k in body:
                    ENGINE.set_cfg(k, body[k])
            self._send(b'{"ok":true}')
        else:
            self._send(b"not found", "text/plain", 404)


class App(rumps.App):
    def __init__(self):
        super().__init__("그냥 받아쓰기", title="냥", quit_button=None)
        self.m_rec = rumps.MenuItem("녹음 시작", callback=self.on_rec)
        self.m_pause = rumps.MenuItem("일시정지", callback=self.on_pause)
        self.m_stop = rumps.MenuItem("녹음 종료", callback=self.on_stop)
        self.m_view = rumps.MenuItem("뷰어 열기", callback=self.on_view)
        self.m_tr = rumps.MenuItem("번역")
        self.tr_items = {}
        for key, label in [("live", "실시간"), ("after", "종료 후"), ("off", "끄기")]:
            it = rumps.MenuItem(label, callback=self.mk_tr(key))
            self.tr_items[key] = it
            self.m_tr.add(it)
        self.m_live = rumps.MenuItem("라이브 미리보기", callback=self.on_live)
        self.menu = [self.m_rec, self.m_pause, self.m_stop, None,
                     self.m_view, self.m_tr, self.m_live, None,
                     rumps.MenuItem("종료", callback=self.on_quit)]
        self.timer = rumps.Timer(self.on_tick, 1)
        self.timer.start()

    def mk_tr(self, key):
        def cb(_):
            ENGINE.set_translate(key)
        return cb

    def on_live(self, _):
        ENGINE.set_cfg("live", "off" if ENGINE.cfg.get("live", "on") != "off" else "on")

    def on_rec(self, _):
        ENGINE.start()
        webbrowser.open(f"http://127.0.0.1:{ENGINE.cfg['port']}")

    def on_pause(self, _):
        if ENGINE.state == "paused":
            ENGINE.resume()
        else:
            ENGINE.pause()

    def on_stop(self, _):
        ENGINE.stop()

    def on_view(self, _):
        webbrowser.open(f"http://127.0.0.1:{ENGINE.cfg['port']}")

    def on_quit(self, _):
        if ENGINE.state in ("recording", "paused"):
            ENGINE.stop()
            rumps.notification("그냥 받아쓰기", "", "저장 후 종료합니다.")
            for _ in range(600):
                if ENGINE.state == "idle":
                    break
                import time
                time.sleep(0.5)
        rumps.quit_application()

    def on_tick(self, _):
        st = ENGINE.state
        self.title = {"idle": "냥", "starting": "냥…", "recording": "냥●",
                      "paused": "냥‖", "finishing": "냥…"}.get(st, "냥")
        self.m_rec.set_callback(self.on_rec if st == "idle" else None)
        self.m_pause.title = "재개" if st == "paused" else "일시정지"
        self.m_pause.set_callback(self.on_pause if st in ("recording", "paused") else None)
        self.m_stop.set_callback(self.on_stop if st in ("recording", "paused") else None)
        mode = ENGINE.cfg.get("translate")
        for k, it in self.tr_items.items():
            it.state = 1 if k == mode else 0
        self.m_live.state = 0 if ENGINE.cfg.get("live", "on") == "off" else 1


def main():
    server = ThreadingHTTPServer(("127.0.0.1", ENGINE.cfg["port"]), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    webbrowser.open(f"http://127.0.0.1:{ENGINE.cfg['port']}")  # 앱 실행 = 뷰어 오픈
    import traceback
    try:
        App().run()
    except Exception:
        traceback.print_exc()
        import time
        while True:  # 메뉴바가 죽어도 서버는 유지
            time.sleep(60)


if __name__ == "__main__":
    main()
