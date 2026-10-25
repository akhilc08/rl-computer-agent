"""Unattended exploration of the user's allowed apps and sites, saving every screen state.

Round-robin over targets; each round presses up to --steps safe elements (see rlca.explore.allowed),
saving before each press: the window screenshot, the on-screen AX window elements, and (once per
app) the menu bar. Each press is logged as a transition with whether the screen changed. OCR and
the detector are NOT run here: they run offline on the saved screenshots, so their latency can be
measured with nothing else competing.

Sites run in our own Arc window without focus or the mouse (see rlca.explore.ArcWindow), so they
go on while the user works. Native apps need focus, so they (and sites that autoplay sound) run
only once the user has been idle 5 minutes, and stop as soon as the mouse moves. Stops at --hours,
when the screen locks, or when the mouse is pushed into the top-left corner.

  .venv/bin/python scripts/data/explore.py --hours 7 [--targets gmail notion ...] [--arc_window ID] [--dry]
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import Quartz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from AppKit import NSApplicationActivateIgnoringOtherApps, NSWorkspace  # noqa: E402
from ApplicationServices import AXUIElementCreateApplication  # noqa: E402

from rlca import explore as X  # noqa: E402
from rlca.ax import app_menus, attr, frame_of, walk  # noqa: E402
from rlca.screen import ax_window, capture_window  # noqa: E402

SITES = {
    "gmail": "https://mail.google.com/mail/u/0/#inbox", "tiktok": "https://www.tiktok.com/",
"google": "https://www.google.com/search?q=weather+ithaca",
    "youtube": "https://www.youtube.com/", "docs": "https://docs.google.com/document/u/0/",
    "claude": "https://claude.ai/recents", "github": "https://github.com/", "canvas": "https://canvas.cornell.edu/",
    "neetcode": "https://neetcode.io/practice", "gcal": "https://calendar.google.com/calendar/u/0/r/week",
    "amazon": "https://www.amazon.com/",
}
# LinkedIn removed (2026-10-25): visiting profiles shows up in those people's "who viewed you".
APPS = {"notion": "Notion", "spotify": "Spotify", "settings": "System Settings", "zed": "Zed",
        "ghostty": "Ghostty", "notes": "Notes", "slack": "Slack"}
ARC = "Arc"


def domain(url):
    h = urlparse(url).hostname or ""
    return ".".join(h.split(".")[-2:])


def running(name):
    for a in NSWorkspace.sharedWorkspace().runningApplications():
        if a.localizedName() == name:
            return a
    return None


def activate(name):
    app = running(name)
    if app is None:
        return None       # never launch apps the user hasn't opened
    app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
    time.sleep(1.0)
    return app


def frontmost():
    return NSWorkspace.sharedWorkspace().frontmostApplication().localizedName()


def mouse():
    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return round(loc.x), round(loc.y)


def find_role(root, role, limit=20000):
    stack, n = [root], 0
    while stack and n < limit:
        el = stack.pop()
        n += 1
        if attr(el, "AXRole") == role:
            return el
        stack.extend(attr(el, "AXChildren") or [])
    return None


def frontmost_regular():
    """Frontmost app that has normal windows; system helpers like UserNotificationCenter flash to
    the front around a press without taking it from the app we are exploring."""
    a = NSWorkspace.sharedWorkspace().frontmostApplication()
    return a.localizedName() if a.activationPolicy() == 0 else None


class Run:
    def __init__(self, out, dry):
        self.out, self.dry = out, dry
        (out / "shots").mkdir(parents=True, exist_ok=True)
        self.states = open(out / "states.jsonl", "a")
        self.trans = open(out / "transitions.jsonl", "a")
        self.menus_done, self.seen, self.counts = set(), {}, Counter()
        self.n = 0
        self.home_mouse = mouse()

    def capture(self, target, app_name, window=None, url=""):
        """Save the current state; -> (state id, AX elements with live handles, signature) or None."""
        if app_name == ARC and window is None:
            raise RuntimeError("Arc capture without our window would read the user's window")
        app = running(app_name)
        if app is None:
            return None
        pid = app.processIdentifier()
        t0 = time.perf_counter()
        c = capture_window(pid, window=window)
        ms_capture = round((time.perf_counter() - t0) * 1000, 1)
        if not c or c["image"] is None:
            return None
        win, frame = c["window"], c["frame"]
        t0 = time.perf_counter()
        elements, handles = walk_with_handles(win, frame)
        ms_ax = round((time.perf_counter() - t0) * 1000, 1)
        web = None
        if app_name == ARC and (wa := find_role(win, "AXWebArea")) is not None:
            web = frame_of(wa)
        sig = hashlib.sha1(json.dumps([app_name, url.split("#")[0], sorted(e.label for e in elements)]).encode()).hexdigest()[:16]
        sid = f"{self.n:06d}"
        self.n += 1
        shot = self.seen.get(sig)
        if shot is None:
            shot = f"shots/{sid}.jpg"
            c["image"].save(self.out / shot, quality=88)
            self.seen[sig] = shot
        rec = {"id": sid, "t": time.time(), "target": target, "app": app_name, "url": url,
               "title": attr(win, "AXTitle"), "frame": frame, "scale": c["scale"], "web_frame": web,
               "shot": shot, "sig": sig, "ms": {"capture": ms_capture, "ax_window": ms_ax},
               "elements": [asdict(e) for e in elements]}
        if app_name not in self.menus_done:
            t0 = time.perf_counter()
            menubar = attr(AXUIElementCreateApplication(pid), "AXMenuBar")
            rec["menubar"] = [asdict(e) for m in (app_menus(menubar) if menubar else []) for e in walk(m, "menubar")]
            rec["ms"]["ax_menubar"] = round((time.perf_counter() - t0) * 1000, 1)
            self.menus_done.add(app_name)
        self.states.write(json.dumps(rec) + "\n")
        self.states.flush()
        return sid, elements, handles, sig, url, web

    def log(self, **rec):
        self.trans.write(json.dumps(rec) + "\n")
        self.trans.flush()

    def should_stop(self):
        if X.screen_locked():
            return "screen locked"
        if X.mouse_in_corner():
            return "mouse in corner"
        return None


def walk_with_handles(win, frame):
    """On-screen labelled AX elements plus their live AX handles (same order)."""
    handles = []
    elements = walk(win, "window", skip_context={attr(win, "AXTitle")}, bounds=frame, handles=handles)
    return elements, handles


def candidates(elements, handles, app_name, url, web, start_domain=""):
    out = []
    for e, h in zip(elements, handles):
        ok, _ = X.allowed(e, app_name, url)
        if not ok:
            continue
        if app_name == ARC and e.role == "AXLink" and not X.link_target_ok(h, start_domain):
            continue
        if app_name == ARC and X.DENY_PATH.search(urlparse(url).path):
            continue   # already on an edit/settings page: only leave, never act
        if app_name == ARC and not (web and e.frame and inside(e.frame, web)):
            continue   # Arc's sidebar holds the user's tabs; only the page itself, and nothing if we can't find it
        out.append((e, h))
    return out


def inside(f, r):
    cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
    return r[0] <= cx <= r[0] + r[2] and r[1] <= cy <= r[1] + r[3]


def settle(app_or_arc, timeout=6.0):
    time.sleep(1.2)
    if isinstance(app_or_arc, X.ArcWindow):
        t0 = time.time()
        while app_or_arc.loading() and time.time() - t0 < timeout:
            time.sleep(0.4)
        time.sleep(0.5)


def run_site(run, arc, target, steps):
    """One site in our own background Arc window: no focus changes, no mouse."""
    start = SITES[target]
    if not run.dry:
        arc.open(start)
    settle(arc)
    for step in range(steps):
        if (why := run.should_stop()):
            return why
        if not arc.alive():
            return "arc window gone"
        url = arc.url()
        if not url or arc.active_tab() != arc.tab:
            arc.open(start)       # our tab was closed or another became active: reopen and retry
            settle(arc)
            continue
        got = run.capture(target, ARC, window=arc.ax, url=url)
        if got is None:
            return "no capture"
        sid, elements, handles, sig, url, web = got
        if domain(url) != domain(start):
            arc.open(start)       # wandered off the site: start over in the same tab
            settle(arc)
            continue
        cands = [(e, h) for e, h in candidates(elements, handles, ARC, url, web, domain(start))
                 if e.role != "AXLink" or (X.SAFE_URL.fullmatch(X.link_url(h) or "")
                                           and not PROFILE.search(urlparse(X.link_url(h)).path))]
        if not cands or run.dry:
            return "dry run" if run.dry else "nothing safe to press"
        e, h = pick(run, target, cands)
        t0 = time.time()
        tabs_before = arc.tab_ids()
        if e.role == "AXLink":
            arc.open(X.link_url(h))
            how = "follow"
        else:
            how = "press" if X.has_press(h) and X.press(h) else None
        if how is None:
            continue
        settle(arc)
        closed = arc.keep_one_tab(tabs_before)
        after = run.capture(target, ARC, window=arc.ax, url=arc.url())
        run.log(before=sid, after=after[0] if after else None, target=target, app=ARC, element=asdict(e),
                how=how, changed=bool(after and after[3] != sig), new_tab_closed=closed, left_app=None,
                secs=round(time.time() - t0, 2))
    return "steps done"


def run_app(run, target, steps):
    """One native app. Needs focus (and the mouse for controls without AXPress), so only while idle."""
    app_name = APPS[target]
    if activate(app_name) is None:
        return "no app"
    if target == "ghostty" and not run.dry:
        X.key(45, cmd=True)       # Cmd-N: a fresh window, never the one running Claude Code
        time.sleep(1.0)
    win, minimized = ax_window(running(app_name).processIdentifier())
    if win is not None and minimized:
        X.unminimize(win)
        time.sleep(1.0)
    run.home_mouse = mouse()
    for step in range(steps):
        if (why := run.should_stop()):
            return why
        if mouse() != run.home_mouse or X.user_idle() < run.idle_for:
            return "user active"
        if frontmost_regular() not in (app_name, None):
            activate(app_name)
        got = run.capture(target, app_name)
        if got is None:
            return "no capture"
        sid, elements, handles, sig, url, web = got
        cands = candidates(elements, handles, app_name, url, web)
        if not cands or run.dry:
            return "dry run" if run.dry else "nothing safe to press"
        e, h = pick(run, target, cands)
        t0 = time.time()
        tabs_before = X.arc_tabs()
        how = X.activate_element(h, e.frame)
        run.home_mouse = mouse()          # our own click moved it; only the user's moves stop the run
        settle(app_name)
        closed = X.close_new_arc_tabs(tabs_before)
        other = frontmost_regular()
        left = other if other not in (app_name, None) else None
        after = None if left else run.capture(target, app_name)
        run.log(before=sid, after=after[0] if after else None, target=target, app=app_name, element=asdict(e),
                how=how, changed=bool(after and after[3] != sig), left_app=left, arc_tabs_closed=closed,
                secs=round(time.time() - t0, 2))
        if left:
            activate(app_name)
    return "steps done"


def pick(run, target, cands):
    weights = [1 / (1 + run.counts[(target, e.label)]) for e, _ in cands]
    e, h = random.choices(cands, weights=weights)[0]
    run.counts[(target, e.label)] += 1
    return e, h


# TikTok keeps a profile-view history its users can see; stay off profile pages (/@name).
PROFILE = re.compile(r"^/@[^/]+/?$")
# Sites that autoplay sound; with the user at the Mac we skip them rather than mute their audio.
LOUD = {"tiktok", "youtube"}
IDLE_FOR_APPS = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--steps", type=int, default=30, help="presses per target per round")
    ap.add_argument("--targets", nargs="*", default=list(SITES) + list(APPS))
    ap.add_argument("--arc_window", default=None, help="reuse this Arc window id instead of making one")
    ap.add_argument("--idle_for", type=float, default=IDLE_FOR_APPS,
                    help="seconds the user must be idle before native apps and loud sites run")
    ap.add_argument("--dry", action="store_true", help="capture one state per target, press nothing")
    ap.add_argument("--out", default=str(ROOT / "data/explore" / datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()
    bad = [t for t in args.targets if t not in SITES and t not in APPS]
    if bad:
        sys.exit(f"unknown targets {bad}")
    run = Run(Path(args.out), args.dry)
    run.idle_for = args.idle_for
    deadline = time.time() + args.hours * 3600
    arc = None
    if any(t in SITES for t in args.targets):
        arc = X.ArcWindow(running(ARC).processIdentifier(), args.arc_window)
        print(f"arc window {arc.id}", flush=True)
    muted_by_us, was_muted = False, None
    try:
        rnd = 0
        while time.time() < deadline:
            for target in args.targets:
                if time.time() >= deadline:
                    break
                idle = X.user_idle() >= args.idle_for
                if idle and not muted_by_us and not args.dry:
                    was_muted, muted_by_us = X.set_muted(True), True
                elif not idle and muted_by_us:
                    X.set_muted(was_muted)
                    muted_by_us = False
                if target in APPS and not idle and not args.dry:
                    why = "skipped, user active"
                elif target in LOUD and not idle and not args.dry:
                    why = "skipped, user active (sound)"
                elif target in SITES:
                    why = run_site(run, arc, target, args.steps)
                else:
                    why = run_app(run, target, args.steps)
                print(f"{datetime.now():%H:%M} round {rnd} {target}: {why} (states {run.n})", flush=True)
                if why in {"screen locked", "mouse in corner", "arc window gone"}:
                    return
            rnd += 1
            if args.dry:
                break
            if not any(t in SITES and t not in LOUD for t in args.targets) and X.user_idle() < args.idle_for:
                time.sleep(30)
    finally:
        if muted_by_us:
            X.set_muted(was_muted)
        if arc is not None and args.arc_window is None and arc.alive():
            arc.close()


if __name__ == "__main__":
    main()
