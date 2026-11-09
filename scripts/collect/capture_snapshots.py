"""Save an accessibility snapshot of each running app to data/snapshots/.

Snapshots contain visible on-screen text, so data/ is git-ignored.
Usage: python scripts/collect/capture_snapshots.py [AppName ...]
"""
import json
import sys
import time
from pathlib import Path

from AppKit import NSWorkspace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rlca.ax import snapshot  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "data" / "snapshots"


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    apps = [a for a in NSWorkspace.sharedWorkspace().runningApplications() if a.activationPolicy() == 0]
    if names:
        apps = [a for a in apps if a.localizedName() in names]
    for app in apps:
        snap = snapshot(app.processIdentifier(), app.localizedName())
        n_win = sum(e["source"] == "window" for e in snap["elements"])
        n_menu = len(snap["elements"]) - n_win
        path = OUT / f"{stamp}-{app.localizedName().replace(' ', '_')}.json"
        path.write_text(json.dumps(snap, indent=1, ensure_ascii=False))
        print(f"{app.localizedName():16s} window={n_win:4d} menubar={n_menu:4d} -> {path.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
