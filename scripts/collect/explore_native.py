"""Explore native apps in the background while the user keeps working: no focus, no mouse, no keys.

Allowed apps (explore.APPS minus Ghostty, whose Claude Code window we never touch) are driven through
AXPress only, so a press reaches the control without bringing the app forward. If a press does pull
the app to the front, the app the user was in is reactivated right away. Every window of the app is
used, on any Space, picked at random each step. Links are never pressed (they open Arc tabs), and on
top of rlca.explore.allowed a stricter deny list keeps away from anything that schedules, forwards,
marks or changes settings, since the user is around and may not notice.

Other running apps are look-only: each window is captured every few rounds and nothing is pressed.
Apps that aren't running are skipped, never launched.

Writes the same run layout as explore.py (states.jsonl, transitions.jsonl, shots/), so
scripts/optbench/replay_sources.py and label_prep.py work on it.

  .venv/bin/python scripts/collect/explore_native.py --hours 3
"""
import argparse
import random
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from AppKit import NSApplicationActivateIgnoringOtherApps, NSWorkspace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore as E  # noqa: E402
from rlca import explore as X  # noqa: E402
from rlca.ax import attr, frame_of  # noqa: E402
from rlca.screen import all_windows  # noqa: E402

DRIVE = {name for key, name in E.APPS.items() if key != "ghostty"}
# Never captured: our own terminal, Arc (has its own explorer), and the explorer's helpers.
SKIP = {"Ghostty", "Arc", "Python", "python3"}
EXTRA_DENY = re.compile(
    r"\b(forward|remind|later|snooze|schedule|assign|status|away|notif\w*|dnd|disturb|bookmark|favou?rite|"
    r"unread|read|resolve|tag|label|priority|switch|feedback|use|set|change|convert|upgrade|try|"
    r"template|lock|unlock|permission|members?|people|workspace|preferences|settings)\b", re.I)
LOOK_EVERY = 10


def front():
    return NSWorkspace.sharedWorkspace().frontmostApplication()


def windows(pid):
    out = []
    for w in all_windows(pid):
        f = frame_of(w)
        if f and f[2] > 100 and f[3] > 100 and not attr(w, "AXMinimized"):
            out.append(w)
    return out


def regular_apps():
    return {a.localizedName(): a for a in NSWorkspace.sharedWorkspace().runningApplications()
            if a.activationPolicy() == 0 and a.localizedName() not in SKIP}


def press_safely(h, app_name):
    """AXPress, then give focus back if the press pulled the app forward. -> (pressed, refocused)."""
    before = front()
    ok = X.press(h)
    time.sleep(1.2)
    now = front()
    refocused = False
    if before is not None and now is not None and now.localizedName() == app_name \
            and before.localizedName() != app_name:
        before.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        refocused = True
    return ok, refocused


def drive_app(run, app, steps):
    name, pid = app.localizedName(), app.processIdentifier()
    for _ in range(steps):
        if X.mouse_in_corner():
            return "mouse in corner"
        wins = windows(pid)
        if not wins:
            return "no windows"
        win = random.choice(wins)
        got = run.capture(name.lower(), name, window=win)
        if got is None:
            return "no capture"
        sid, elements, handles, sig, _, _ = got
        cands = [(e, h) for e, h in E.candidates(elements, handles, name, "", None)
                 if e.role != "AXLink" and not EXTRA_DENY.search(e.label) and X.has_press(h)]
        if not cands:
            continue
        e, h = E.pick(run, name, cands)
        t0 = time.time()
        ok, refocused = press_safely(h, name)
        after = run.capture(name.lower(), name, window=win)    # None if the press closed the window
        run.log(before=sid, after=after[0] if after else None, target=name.lower(), app=name, element=asdict(e),
                how="press" if ok else None, changed=bool(after and after[3] != sig), refocused=refocused,
                secs=round(time.time() - t0, 2))
    return "steps done"


def look_app(run, app):
    n = 0
    for w in windows(app.processIdentifier()):
        if run.capture(app.localizedName().lower(), app.localizedName(), window=w):
            n += 1
    return f"looked at {n} windows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=15, help="presses per driven app per round")
    ap.add_argument("--out", default=str(ROOT / "data/native" / datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()
    run = E.Run(Path(args.out), dry=False)
    deadline = time.time() + args.hours * 3600
    rnd = 0
    print(f"out {args.out}; driving {sorted(DRIVE)}", flush=True)
    while time.time() < deadline:
        if X.screen_locked():
            time.sleep(60)
            continue
        apps = regular_apps()
        for name, app in sorted(apps.items()):
            if name in DRIVE:
                why = drive_app(run, app, args.steps)
            elif rnd % LOOK_EVERY == 0:
                why = look_app(run, app)
            else:
                continue
            print(f"{datetime.now():%H:%M} round {rnd} {name}: {why} (states {run.n})", flush=True)
            if why == "mouse in corner":
                return
        rnd += 1


if __name__ == "__main__":
    main()
