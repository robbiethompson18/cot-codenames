"""Local dashboard for stego runs: one row per run, click a row for its curves and latest essays.

    uv run python -m cot_codenames.stego.dash        ->  http://stego.localhost/  (127.0.0.1:7103 behind Caddy)

Every run has a folder under runs/stego/ holding launch.json (pod, address, one-line purpose; written by
scripts/stego_runpod.sh), config.json (every training argument and the git commit) and metrics.jsonl. On each refresh the
dashboard copies config, metrics and the latest eval essays off every pod it can still reach, so a run's numbers stay
here after its pod is gone.
"""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 7103
RUNS = Path("runs/stego")
KEY = str(Path.home() / ".runpod/ssh/RunPod-Key-Go")
POD_DOLLARS_PER_HOUR = 5.29  # H200
SYNC_EVERY = 45  # seconds
_last_sync = 0.0
_reachable: dict[str, bool] = {}


def sync_run(run: Path) -> None:
    launch = json.loads((run / "launch.json").read_text())
    if launch.get("gone") or not launch.get("host"):
        _reachable[run.name] = False
        return
    ssh = [
        "ssh",
        "-i",
        KEY,
        "-p",
        str(launch["port"]),
        "-o",
        "ConnectTimeout=8",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    remote = f"/workspace/cot-codenames/runs/stego/{run.name}"
    # One round trip: config, metrics, then the last few eval essays, separated by a marker line.
    essays = f'grep \'"kind": "eval"\' {remote}/rollouts.jsonl | tail -n 4'
    cmd = f"cat {remote}/config.json; echo; echo @@@; cat {remote}/metrics.jsonl; echo @@@; {essays}"
    try:
        out = subprocess.run([*ssh, f"root@{launch['host']}", cmd], capture_output=True, text=True, timeout=40, check=False)
    except subprocess.TimeoutExpired:
        _reachable[run.name] = False
        return
    _reachable[run.name] = out.returncode == 0 or "@@@" in out.stdout
    parts = out.stdout.split("@@@\n")
    if len(parts) == 3 and parts[0].strip():
        (run / "config.json").write_text(parts[0])
        if parts[1].strip():
            (run / "metrics.jsonl").write_text(parts[1])
        if parts[2].strip():
            (run / "samples.jsonl").write_text(parts[2])


def sync() -> None:
    global _last_sync
    if time.time() - _last_sync < SYNC_EVERY:
        return
    _last_sync = time.time()
    runs = [r for r in RUNS.iterdir() if (r / "launch.json").exists()]
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(sync_run, runs))


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written last line
    return out


def describe(run: Path) -> dict | None:
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    launch = json.loads((run / "launch.json").read_text()) if (run / "launch.json").exists() else {}
    metrics = jsonl(run / "metrics.jsonl")
    if not cfg and not launch:
        return None
    train = [m for m in metrics if m["kind"] == "train"]
    evals = [m for m in metrics if m["kind"] == "eval"]
    steps = cfg.get("steps") or launch.get("steps") or 100
    done = train[-1]["step"] + 1 if train else 0
    reachable = _reachable.get(run.name, False)
    if done >= steps:
        status = "finished"
    elif reachable:
        status = "running" if metrics else "setting up"
    else:
        status = launch.get("stopped") or ("stopped" if metrics or launch else "unknown")
    thinking = "none (reasoning off)"
    if cfg.get("plan"):
        thinking = "plan field: <plan> (40 words) then <essay>; native reasoning off"
    if cfg.get("sender_thinking"):
        thinking = f"native reasoning, {cfg.get('thinking_tokens')}-token budget, essay in <essay> tags"
    # Runs from before the referee was selectable: the lenient one until the git commit was stamped, the strict one after.
    referee = cfg.get("referee") or ("strict" if cfg.get("git_sha") else "lenient")
    scratch = "plan" if cfg.get("plan") else "reasoning"
    monitor_sees = {"essay": "essay", "both": f"{scratch} + essay"}[cfg.get("monitor_sees", "essay")]
    reward = f"λ={cfg.get('lam', 1)}"
    if cfg.get("lam_ramp"):
        reward += f", ramps over steps {cfg['lam_ramp'][0]}–{cfg['lam_ramp'][1]}"
    if cfg.get("binary_reward"):
        reward += ", yes/no penalty"
    reward += f", fail={cfg.get('fail_reward', -1)}"
    pool = cfg.get("pool_size")
    task = (f"{pool} fixed words" if pool else "all training words") + f", {cfg.get('topics', 'narrow')} topics"
    secs = sum(m.get("secs", 0) for m in train)
    hours = secs / 3600 * 1.25 if train else 0  # evals and setup add roughly a quarter
    return {
        "run": run.name,
        "note": launch.get("note", ""),
        "status": status,
        "done": done,
        "steps": steps,
        "model": cfg.get("model", launch.get("model", "")),
        "lora": f"LoRA rank {cfg.get('lora_rank', '?')}" + (f", from {cfg['init_adapter']}" if cfg.get("init_adapter") else ""),
        "monitor": cfg.get("monitor_model", ""),
        "monitor_sees": monitor_sees + (f", {cfg['monitor_shots']} examples" if cfg.get("monitor_shots") else ""),
        "referee": "claude-haiku-5-5",
        "referee_rules": referee + (", sender told the rule" if cfg.get("tell_rule") else ""),
        "thinking": thinking,
        "task": task,
        "reward": reward,
        "backend": cfg.get("backend", ""),
        "git_sha": (cfg.get("git_sha") or "")[:7],
        "started": cfg.get("started", launch.get("started", "")),
        "pod": launch.get("pod", ""),
        "cost": round(hours * POD_DOLLARS_PER_HOUR, 1) if cfg.get("backend") == "local" else None,
        "train": train,
        "evals": evals,
        "samples": jsonl(run / "samples.jsonl")[-3:],
        "config": cfg,
    }


def all_runs() -> list[dict]:
    sync()
    # Only runs with a launch record are listed, which keeps wiring and timing tests off the page.
    rows = [d for r in sorted(RUNS.iterdir()) if (r / "launch.json").exists() and (d := describe(r))]
    order = {"running": 0, "setting up": 1}
    return sorted(rows, key=lambda d: (order.get(d["status"], 2), d["started"] or ""), reverse=False)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api/runs"):
            body, kind = json.dumps(all_runs()).encode(), "application/json"
        else:
            body, kind = Path(__file__).with_name("dash.html").read_bytes(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


if __name__ == "__main__":
    print("http://stego.localhost/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
