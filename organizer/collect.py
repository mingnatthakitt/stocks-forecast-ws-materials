#!/usr/bin/env python3
"""
collect.py — ORGANISER TOOL. LAN submission collector + live scoreboard.

Run this on the presenter's laptop during the workshop; teams upload their
submission zip from any browser on the same Wi-Fi (or a phone hotspot):

    python collect.py --pin 4242
    # → serves on http://0.0.0.0:8000 — share http://<your-LAN-IP>:8000

Why LAN-local instead of Streamlit Cloud / Vercel (the three hard reasons):
  1. The private 2025 labels must NEVER leave the organiser machine.
  2. Submissions contain executable code (predict.py) — scoring them on a public
     host is RCE-as-a-service; here they run only inside evaluate.py's sandboxed,
     timed subprocesses on YOUR machine.
  3. Serverless limits (Vercel 250 MB / 60 s) cannot carry a torch scoring stack.

What it does
------------
* POST /upload   — validates (pin, size, zip, contains predict.py) and saves to
                   submissions/round<N>/submission_<Team>_round<N>.zip
                   (re-uploads overwrite — the last upload before the deadline wins).
* auto-score     — after each upload it re-scores that round via evaluate.py in a
                   background worker (disable with --no-score).
* GET /status    — JSON of received submissions (filename, size, sha256, time).
* GET /healthz   — no-PIN liveness probe, so a team whose browser cannot load the
                   page can tell "server down" from "this Wi-Fi blocks my laptop".
* GET /board     — live leaderboard page (pin-protected): submissions + the latest
                   scores from results/round<N>.json. Share it — or keep it on the
                   projector only, your call.

Security posture: a shared room PIN gates every route; upload size is capped; zip
contents are inspected (predict.py must be present); filenames/team names are
sanitised; nothing is executed here — execution happens only inside evaluate.py.
Fallback: if the Wi-Fi dies, the old flow (AirDrop/USB/email of zips into
submissions/round<N>/) still works unchanged.

Network reality check: university and eduroam Wi-Fi often enables AP/client
isolation, which blocks laptop-to-laptop traffic even though both devices are on
the "same" SSID and both passed the portal login. Portal login is authentication
to the network, not permission to reach other clients. Always smoke-test
/healthz from a second device the day before; if it is blocked, run the hotspot
fallback in organizer/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
MAX_ZIP_MB = 25.0
# A real submission is a model + a features file: well under 10 MB unpacked.
# The cap is on the UNCOMPRESSED total, because that is what evaluate.py writes
# into results/_sandbox/ — a 25 MB zip can otherwise unpack to tens of gigabytes
# and fill the organiser's disk in the middle of a deadline burst.
MAX_UNPACKED_MB = 200.0

app = FastAPI(title="Can AI Predict the Market? — submission collector")
_state = {"pin": "", "auto_score": True, "scoring": False, "last_score": {},
          "last_error": ""}
_score_lock = threading.Lock()
_score_dirty: set[int] = set()          # rounds the worker still owes a run
_score_wake = threading.Event()
SCORE_DEBOUNCE_S = 4.0                  # let a burst of deadline uploads land first


# ----------------------------------------------------------------- helpers
def _lan_ips() -> list[str]:
    """This machine's non-loopback IPv4 addresses, best guess first.

    The UDP connect to 8.8.8.8 sends no packets; it just asks the routing table
    which local address the OS would use to reach the internet — that is the one
    other devices on the same LAN will use too.
    """
    ips: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            ips = [s.getsockname()[0]]
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if info[0] == socket.AF_INET and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


def _qr(url: str) -> str:
    """ASCII QR of the upload URL so phones can join by camera. Optional dep."""
    try:
        import qrcode  # noqa: PLC0415
    except ImportError:
        return "   (no QR: pip install qrcode in the stocks-ml env)"
    try:
        buf = io.StringIO()
        code = qrcode.QRCode(border=1)
        code.add_data(url)
        code.make(fit=True)
        code.print_ascii(out=buf, invert=True)
        return "\n".join("   " + line for line in buf.getvalue().splitlines())
    except Exception as exc:  # noqa: BLE001
        # A QR is a nicety. Never let it stop the portal from starting.
        return f"   (QR unavailable: {type(exc).__name__}: {exc} — the URL above still works)"


def _check_port(host: str, port: int) -> int:
    """Return a bindable port, walking up if the chosen one is taken.

    A busy port is the most common day-of failure (another dev server, a stale
    collect.py from a previous round) and uvicorn's error is easy to miss while
    the room is walking in. Try the requested port, then the next 20.

    Both the LAN bind and the loopback bind are probed: a service holding only
    127.0.0.1 can coexist with our 0.0.0.0 bind, and then localhost on the
    presenter's own machine answers from the *wrong* app while every other
    device in the room works fine. That failure is very confusing, so skip it.
    """
    for candidate in range(port, port + 20):
        if all(_port_free(host, candidate, addr) for addr in ("", "127.0.0.1")):
            return candidate
    sys.exit(f"ERROR: no free port in {port}-{port + 19}. Stop the other process "
             f"(lsof -nP -iTCP:{port} -sTCP:LISTEN) and retry.")


def _port_free(host: str, port: int, addr: str) -> bool:
    """Can we bind this (address, port) right now?"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host if (addr and host != "0.0.0.0") else addr, port))
        except OSError:
            return False
    return True


def _slug(name: str) -> str:
    """Team name → safe filename fragment."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    if not slug:
        raise HTTPException(400, "team name must contain letters or digits")
    return slug[:40]


def _team_from_stem(stem: str) -> str:
    """Recover a team name from an export filename.

    The "_round<N>" suffix must be matched at the END of the name, never split
    on its first occurrence: a team called "Alpha_Round_Trio" slugifies to
    Alpha_Round_Trio, and a naive split("_round")[0] would file them under the
    name "Alpha" — two different teams showing one row on the board, and a
    collision guard that then rejects a legitimate name.
    """
    name = stem.removeprefix("submission_")
    return re.sub(r"_round\d+$", "", name, flags=re.IGNORECASE) or name


def _guard_collision(team: str, rnd: int, slug: str) -> None:
    """Refuse a name that would land on ANOTHER team's file.

    "Team A/B" and "Team A_B" both slug to Team_A_B — and macOS filesystems are
    case-insensitive, so "alpha" and "Alpha" collide too. Without this, team 2's
    upload silently replaces team 1's submission and team 1 just vanishes from
    the leaderboard. Re-uploading under your OWN previous name is fine.
    """
    receipt = HERE / "submissions" / f"round{rnd}" / f"receipt_{slug}_round{rnd}.json"
    try:
        previous = json.loads(receipt.read_text()).get("team", "")
    except (OSError, json.JSONDecodeError):
        return
    if previous != team:
        raise HTTPException(
            409, f"'{previous}' already submitted under that name — your team name must "
                 f"match exactly. Check the spelling with the facilitator.")


def _check_pin(pin: str) -> None:
    if not _state["pin"] or pin != _state["pin"]:
        raise HTTPException(401, "wrong room PIN")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _save_zip(team: str, rnd: int, data: bytes) -> Path:
    """Validate and persist one submission zip. Raises HTTPException on bad input."""
    if len(data) > MAX_ZIP_MB * 1_000_000:
        raise HTTPException(413, f"zip larger than {MAX_ZIP_MB:.0f} MB — did you export correctly?")
    if not data.startswith(b"PK"):
        raise HTTPException(400, "that is not a zip file — hand in the export cell's zip")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = [n for n in zf.namelist() if not n.endswith("/")]
        names = [n for n in names if not n.startswith(("__MACOSX/", "."))]
        roots = {n.split("/")[0] for n in names}
        has_predict = "predict.py" in roots or (
            len(roots) == 1 and f"{next(iter(roots))}/predict.py" in names)
        unpacked = sum(i.file_size for i in zf.infolist())
        broken = zf.testzip()  # CRC check: a half-uploaded zip fails here, not mid-scoring
    except zipfile.BadZipFile:
        raise HTTPException(400, "corrupt zip — re-run the export cell")
    if unpacked > MAX_UNPACKED_MB * 1_000_000:
        raise HTTPException(
            413, f"unzips to {unpacked / 1e6:.0f} MB — over the {MAX_UNPACKED_MB:.0f} MB "
                 f"limit. That is not a model; check what the export cell packed.")
    if broken is not None:
        raise HTTPException(400, f"corrupt zip ({broken} failed its checksum) — "
                                 "the upload probably dropped; try again")
    if not has_predict:
        raise HTTPException(400, "zip does not contain predict.py at its root — "
                                 "re-run the export cell (it builds the zip for you)")

    out_dir = HERE / "submissions" / f"round{rnd}"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(team)
    _guard_collision(team, rnd, slug)
    dest = out_dir / f"submission_{slug}_round{rnd}.zip"
    tmp = dest.with_suffix(".part")
    try:
        tmp.write_bytes(data)
        tmp.replace(dest)  # atomic on the same volume
        (out_dir / f"receipt_{slug}_round{rnd}.json").write_text(json.dumps({
            "team": team, "round": rnd, "bytes": len(data), "sha256_16": _sha256(data),
            "received": datetime.now().isoformat(timespec="seconds"),
        }))
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"could not save the upload ({exc.strerror or exc}) — "
                                 "ask the facilitator, your file may not have arrived")
    return dest


def _received() -> list[dict]:
    out = []
    for round_dir in sorted((HERE / "submissions").glob("round*")):
        for receipt in sorted(round_dir.glob("receipt_*.json")):
            try:
                out.append(json.loads(receipt.read_text()))
            except json.JSONDecodeError:
                continue
    return sorted(out, key=lambda r: r["received"], reverse=True)


def _score_round(rnd: int) -> dict:
    """Run evaluate.py for one round (blocking). Returns the results summary."""
    with _score_lock:
        _state["scoring"] = True
        _state["last_error"] = ""
        try:
            proc = subprocess.run(
                [sys.executable, str(HERE / "evaluate.py"), "--round", str(rnd)],
                cwd=HERE, capture_output=True, text=True, timeout=900)
            res_file = RESULTS_DIR / f"round{rnd}.json"
            ok = proc.returncode == 0
            summary = {"ok": ok, "round": rnd,
                       "log": (proc.stdout + proc.stderr)[-4000:]}
            if not ok:
                # evaluate.py only writes results at the very end, so a non-zero
                # exit means results/round<N>.json on disk is from an EARLIER run.
                # Surfacing it as if it were fresh is worse than showing nothing:
                # the room would read stale MAEs as this round's answer.
                tail = [ln for ln in summary["log"].splitlines() if ln.strip()][-3:]
                summary["stale"] = res_file.exists()
                _state["last_error"] = (f"round {rnd} scoring failed — {' / '.join(tail)}")
            if res_file.exists():
                data = json.loads(res_file.read_text())
                summary["teams"] = [
                    {"team": t["team"], "arch": t.get("arch", "?"), "status": t["status"],
                     **({"mae": t["mae"], "dir_acc": t["dir_acc"]} if t["status"] == "ok" else {})}
                    for t in data["teams"]]
                summary["baseline_zero_mae"] = data.get("baseline_zero_mae")
                if ok:
                    _state["last_score"][str(rnd)] = datetime.now().isoformat(timespec="seconds")
            return summary
        finally:
            _state["scoring"] = False


def _score_worker() -> None:
    """Single debounced scoring loop.

    Every upload used to start its own thread, each re-scoring the WHOLE round.
    Fifteen teams uploading in the minute before the deadline therefore queued
    fifteen full scoring runs — the board would show numbers from the first run
    while the queue drained for minutes, and the last team would not see its score
    until the dust settled. One worker that coalesces a burst into a single run
    fixes that: uploads mark the round dirty, the worker settles for a moment,
    scores once, and repeats only if something arrived while it was working.
    """
    while True:
        _score_wake.wait()
        _score_wake.clear()
        time.sleep(SCORE_DEBOUNCE_S)  # let a burst of uploads land first
        while True:
            pending = sorted(_score_dirty)
            _score_dirty.clear()
            for rnd in pending:
                print(f"  ⟳ auto-scoring round {rnd} …", flush=True)
                _score_round(rnd)
            if not _score_dirty:
                break


def _queue_score(rnd: int) -> None:
    """Ask the worker to (re)score a round. Returns immediately."""
    _score_dirty.add(rnd)
    _score_wake.set()


# ----------------------------------------------------------------- routes
@app.get("/healthz")
def healthz():
    """No-PIN liveness probe. Purpose: let a team on a locked-down network tell
    'the server is down' apart from 'this Wi-Fi blocks client-to-client traffic'
    before they give up and hunt for the organiser."""
    return {"ok": True, "service": "workshop-collector", "host": socket.gethostname(),
            "lan_ips": _lan_ips(), "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE_UPLOAD.replace("__PIN_HINT__",
                               "your room PIN" if _state["pin"] else "(no PIN set — start collect.py with --pin)")


@app.post("/upload")
async def upload(team: str = Form(...), rnd: int = Form(...), pin: str = Form(""),
                 file: UploadFile = File(...)):
    _check_pin(pin)
    if rnd not in (1, 2):
        raise HTTPException(400, "round must be 1 or 2")
    if len(team) > 60:
        raise HTTPException(400, "team name too long")
    data = await file.read()
    dest = _save_zip(team, rnd, data)
    receipt = {"team": team, "round": rnd, "file": dest.name,
               "bytes": len(data), "sha256_16": _sha256(data),
               "received": datetime.now().isoformat(timespec="seconds")}
    if _state["auto_score"]:
        _queue_score(rnd)
        receipt["scoring"] = "queued — the board refreshes itself in a few seconds"
    return JSONResponse({"ok": True, **receipt})


@app.get("/status")
def status(pin: str = ""):
    _check_pin(pin)
    return {"received": _received(), "scoring": _state["scoring"] or bool(_score_dirty),
            "last_score": _state["last_score"]}


@app.post("/score")
def score(rnd: int = Form(...), pin: str = Form("")):
    _check_pin(pin)
    return _score_round(rnd)


@app.get("/board", response_class=HTMLResponse)
def board(pin: str = ""):
    return PAGE_BOARD  # the page reads the PIN from ?pin= / localStorage itself


@app.get("/board-data")
def board_data(pin: str = ""):
    _check_pin(pin)
    rounds = {}
    for rnd in (1, 2):
        f = RESULTS_DIR / f"round{rnd}.json"
        if f.exists():
            data = json.loads(f.read_text())
            rounds[str(rnd)] = {
                "baseline": data.get("baseline_zero_mae"),
                "teams": [{"team": t["team"], "arch": t.get("arch", "?"),
                           "status": t["status"],
                           "mae": t.get("mae"), "dir_acc": t.get("dir_acc")}
                          for t in data["teams"]],
            }
    return {"rounds": rounds, "received": _received(),
            "scoring": _state["scoring"] or bool(_score_dirty),
            "last_score": _state["last_score"], "last_error": _state["last_error"]}


# ----------------------------------------------------------------- pages
PAGE_UPLOAD = """<!DOCTYPE html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Submit your model</title><style>
:root{--bg:#0b0f14;--panel:#11161d;--line:#1e2630;--text:#d7e2ee;--dim:#7d8b9b;
--green:#2dd4a7;--gold:#f5c451;--red:#ff5d73}
*{box-sizing:border-box;margin:0} body{background:var(--bg);color:var(--text);
font:15px/1.5 ui-monospace,Menlo,Consolas,monospace;padding:28px;max-width:640px;margin:auto}
h1{font-size:20px;letter-spacing:1px;color:var(--gold);margin-bottom:4px}
p.sub{color:var(--dim);font-size:12.5px;margin-bottom:22px}
label{display:block;color:var(--dim);font-size:11px;text-transform:uppercase;
letter-spacing:1px;margin:14px 0 4px}
input,select{width:100%;background:var(--panel);border:1px solid var(--line);
border-radius:8px;color:var(--text);padding:10px 12px;font:inherit}
button{width:100%;margin-top:18px;background:var(--green);color:#04120d;border:0;
border-radius:10px;padding:13px;font:700 15px ui-monospace,Menlo,monospace;
letter-spacing:1px;cursor:pointer}
button:disabled{opacity:.5}
.msg{margin-top:16px;padding:12px 14px;border-radius:8px;border:1px solid var(--line);
display:none;white-space:pre-wrap;font-size:13px}
.ok{border-color:var(--green);color:var(--green)} .err{border-color:var(--red);color:var(--red)}
.hint{color:var(--dim);font-size:12px;margin-top:18px}
a{color:var(--gold)}
</style></head><body>
<h1>📦 SUBMIT YOUR MODEL</h1>
<p class='sub'>export cell ran → hand in the zip it built · re-upload anytime before the deadline — the last one counts</p>
<form id='f'>
<label>Room PIN — __PIN_HINT__</label><input id='pin' autocomplete='off'>
<label>Team name (exactly as agreed)</label><input id='team' autocomplete='off'>
<label>Round</label><select id='rnd'><option value='1'>Round 1 — human only</option><option value='2'>Round 2 — human + AI</option></select>
<label>Submission zip</label><input id='file' type='file' accept='.zip'>
<button id='b'>UPLOAD ▸</button>
<div class='msg' id='m'></div>
</form>
<p class='hint'>After uploading, watch the scoreboard: <a href='/board?pin='>the live leaderboard</a>
(enter the same PIN). Re-uploading replaces your previous submission.<br><br>
Page not loading at all? You are probably on a Wi-Fi that blocks device-to-device
traffic — join the hotspot the facilitator announces, or hand the zip to them on a
USB stick. You can also test the line with <a href='/healthz'>/healthz</a>.</p>
<script>
const f=document.getElementById('f'),m=document.getElementById('m'),b=document.getElementById('b');
const saved=localStorage.getItem('pin'); if(saved) document.getElementById('pin').value=saved;
f.onsubmit=async e=>{e.preventDefault();b.disabled=true;b.textContent='UPLOADING…';
const fd=new FormData();fd.append('team',team.value);fd.append('rnd',rnd.value);
fd.append('pin',pin.value);fd.append('file',file.files[0]);
try{const r=await fetch('/upload',{method:'POST',body:fd});const j=await r.json();
if(!r.ok){m.className='msg err';m.textContent='✕ '+(j.detail||'upload failed');}
else{m.className='msg ok';localStorage.setItem('pin',pin.value);
m.textContent='✓ received '+j.file+' ('+(j.bytes/1024).toFixed(0)+' KB, sha '+j.sha256_16+')\\n'+(j.scoring||'');
f.reset();}}catch(err){m.className='msg err';m.textContent='✕ cannot reach the server at '+location.host+'.\n'+
 'Either the collector laptop is down, or this Wi-Fi blocks device-to-device traffic.\n'+
 'Ask the facilitator to announce the hotspot, or hand in the zip on a USB stick.';}
b.disabled=false;b.textContent='UPLOAD ▸';};
</script></body></html>"""

PAGE_BOARD = """<!DOCTYPE html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Live scoreboard</title><style>
:root{--bg:#0b0f14;--panel:#11161d;--line:#1e2630;--text:#d7e2ee;--dim:#7d8b9b;
--green:#2dd4a7;--gold:#f5c451;--red:#ff5d73}
*{box-sizing:border-box;margin:0} body{background:var(--bg);color:var(--text);
font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;padding:26px;max-width:900px;margin:auto}
h1{font-size:19px;color:var(--gold);letter-spacing:1px;margin-bottom:2px}
p.sub{color:var(--dim);font-size:12px;margin-bottom:18px}
table{width:100%;border-collapse:collapse;margin-bottom:20px}
th{text-align:left;color:var(--dim);font-size:11px;text-transform:uppercase;
letter-spacing:1px;padding:7px 9px;border-bottom:1px solid var(--line)}
td{padding:8px 9px;border-bottom:1px solid var(--line)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.g{color:var(--green)} .r{color:var(--red)} .y{color:var(--gold)} .dim{color:var(--dim)}
.sec{color:var(--green);font-size:13px;letter-spacing:1px;margin:16px 0 8px;text-transform:uppercase}
input{background:var(--panel);border:1px solid var(--line);border-radius:8px;
color:var(--text);padding:9px 11px;font:inherit;width:220px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--dim);margin-right:6px}
.live{background:var(--green)}
.warn{border:1px solid var(--red);color:var(--red);border-radius:8px;padding:10px 12px;
margin-bottom:16px;font-size:13px}
</style></head><body>
<h1>📊 LIVE SCOREBOARD</h1>
<p class='sub'><span class='dot' id='dot'></span><span id='state'>connecting…</span> ·
MAE pooled over all 5 tickers · lower is better · <span id='ls'></span></p>
<div id='pinbox'><label class='dim'>room PIN</label><br><input id='pin' autocomplete='off'>
<button onclick='save()' style='margin:8px 0;padding:8px 14px;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:8px;cursor:pointer'>GO</button></div>
<div id='content'></div>
<script>
// 5 decimals, not 4: every MAE here is ~0.015x, so the whole competition is decided in
// the fourth decimal and 4dp renders genuinely different models as identical.
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pinEl=document.getElementById('pin');
pinEl.value=localStorage.getItem('pin')||'';
function save(){localStorage.setItem('pin',pinEl.value);tick();}
async function tick(){
 const pin=localStorage.getItem('pin')||'';
 try{const r=await fetch('/board-data?pin='+encodeURIComponent(pin));
 if(r.status===401){document.getElementById('state').textContent='wrong or missing PIN';
 document.getElementById('pinbox').style.display='block';return;}
 const j=await r.json();
 document.getElementById('pinbox').style.display='none';
 document.getElementById('dot').className='dot live';
 document.getElementById('state').textContent=j.scoring?'SCORING…':'online';
 document.getElementById('ls').textContent=Object.entries(j.last_score).map(([k,v])=>'round '+k+' scored '+v).join(' · ');
 let h='';
 if(j.last_error) h+=`<div class='warn'>⚠ ${esc(j.last_error)} — numbers below may be from the previous run.</div>`;
 for(const rnd of ['1','2']){const d=j.rounds[rnd];if(!d)continue;
  h+=`<div class='sec'>Round ${rnd} — ${rnd==='1'?'human only':'human + AI'} · baseline ${d.baseline?d.baseline.toFixed(5):'?'}</div>`;
  h+="<table><tr><th>#</th><th>Team</th><th>Model</th><th class='num'>MAE</th><th class='num'>DirAcc</th></tr>";
  let i=1;for(const t of d.teams){
   h+=t.status==='ok'
    ?`<tr><td class='dim'>${i++}</td><td>${esc(t.team)}</td><td class='dim'>${esc(t.arch)}</td><td class='num g'>${t.mae.toFixed(5)}</td><td class='num'>${(t.dir_acc*100).toFixed(1)}%</td></tr>`
    :`<tr><td class='dim'>—</td><td>${esc(t.team)}</td><td class='dim'>${esc(t.arch)}</td><td class='num r' colspan='2'>${esc(String(t.status).toUpperCase())}</td></tr>`;}
  h+='</table>';}
 h+="<div class='sec'>received submissions</div><table><tr><th>Team</th><th>Round</th><th class='num'>Size</th><th>Received</th></tr>";
 for(const r of j.received) h+=`<tr><td>${esc(r.team)}</td><td class='dim'>round ${esc(r.round)}</td><td class='num'>${(r.bytes/1024).toFixed(0)} KB</td><td class='dim'>${esc(r.received)}</td></tr>`;
 h+='</table>';
 document.getElementById('content').innerHTML=h;
 }catch(e){document.getElementById('state').textContent='offline — is collect.py running?';}
}
tick();setInterval(tick,3000);
</script></body></html>"""


def _import(path: Path, team: str | None, rnd: int | None) -> None:
    """USB-stick hand-in: file a zip exactly as an upload would, then exit.

    The portal's file picker runs on your own machine, so walking round with a
    stick means selecting each zip yourself. This does the same thing without the
    form — same validation, same receipt, same team name on the board.
    """
    if not path.is_file():
        sys.exit(f"ERROR: no such file: {path}")
    data = path.read_bytes()
    stem = path.stem
    if rnd is None:
        m = re.search(r"_round(\d+)$", stem, re.I)
        rnd = int(m.group(1)) if m else 1
    if team is None:  # prefer the team's own claim, then the export filename
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                team = json.loads(zf.read("config.json")).get("team")
        except Exception:  # noqa: BLE001
            team = None
        team = team or _team_from_stem(stem)
    _state["pin"] = ""
    try:
        dest = _save_zip(team, rnd, data)
    except HTTPException as exc:
        sys.exit(f"REJECTED ({exc.status_code}): {exc.detail}")
    print(f"✅ filed {dest.name}  |  team='{team}' round={rnd}  "
          f"({len(data) / 1024:.0f} KB, sha {_sha256(data)})")
    print("   scored? → python evaluate.py --round "
          f"{rnd}   (or just let the portal's auto-score pick it up)")


def main() -> None:
    ap = argparse.ArgumentParser(description="LAN submission collector + live scoreboard.")
    ap.add_argument("--pin", default="", help="room PIN teams must type (recommended)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-score", action="store_true",
                    help="only collect; scoring stays a manual evaluate.py step")
    ap.add_argument("--qr", action="store_true",
                    help="print a scannable QR code of the upload URL")
    ap.add_argument("--import", dest="import_path", metavar="ZIP", default=None,
                    help="file one submission zip from disk (USB-stick hand-in) "
                         "and exit — no server, same checks as an upload")
    ap.add_argument("--team", default=None, help="team name for --import "
                    "(default: the name inside the zip)")
    ap.add_argument("--round", type=int, choices=[1, 2], default=None,
                    help="round for --import (default: read from the filename, else 1)")
    args = ap.parse_args()

    if args.import_path:
        _import(Path(args.import_path).expanduser(), args.team, args.round)
        return

    port = _check_port(args.host, args.port)
    if port != args.port:
        print(f"port {args.port} is busy (another process is listening) — using {port} instead.")
    args.port = port

    _state["pin"] = args.pin.strip()
    _state["auto_score"] = not args.no_score
    RESULTS_DIR.mkdir(exist_ok=True)

    ips = _lan_ips()
    share = ips[0] if ips else "<your-LAN-IP>"
    url = f"http://{share}:{args.port}/"

    print("=" * 62)
    print("  📦  SUBMISSION COLLECTOR")
    print("=" * 62)
    print(f"  GIVE TEAMS THIS :  {url}")
    print(f"  scoreboard      :  http://{share}:{args.port}/board")
    print(f"  reachability    :  http://{share}:{args.port}/healthz  (no PIN, plain test)")
    print(f"  your LAN IPs    :  {', '.join(ips) or '?? (no network interface found)'}")
    print(f"  room PIN        :  {_state['pin'] or '(NONE — pass --pin!)'}")
    print(f"  auto-score      :  {'on — every upload re-scores that round' if _state['auto_score'] else 'off'}")
    if args.qr:
        print()
        print(_qr(url))
    print("-" * 62)
    print("  Before the room fills up, open the URL on a PHONE and confirm it")
    print("  loads. If it does not, the venue Wi-Fi is blocking client-to-client")
    print("  traffic (common on university/eduroam networks) — see the 'Network'")
    print("  section of organizer/README.md for the hotspot fallback.")
    print("  Deadline chaos fallback: teams hand you the zip on a USB stick, you")
    print(f"  drop it in submissions/round<N>/ and run evaluate.py as usual.")
    print("=" * 62)

    if _state["auto_score"]:
        threading.Thread(target=_score_worker, daemon=True, name="score-worker").start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
