"""Measure how much of each running app's UI the macOS accessibility tree exposes.

For each app: walk the focused window's tree and the menu bar, count actionable
elements and how many have a usable label, and time the walk. Then force-enable
accessibility (AXManualAccessibility for Electron, AXEnhancedUserInterface for
Chromium/CEF), measure again, and restore the original values.

Needs the terminal to have Accessibility permission.
Usage: python scripts/probes/ax_coverage.py [AppName ...]
"""
import sys
import time

from AppKit import NSWorkspace
from ApplicationServices import (
    AXIsProcessTrusted,
    AXUIElementCopyAttributeValue,
    AXUIElementCreateApplication,
    AXUIElementSetAttributeValue,
)

ACTIONABLE = {
    "AXButton", "AXMenuItem", "AXMenuBarItem", "AXTextField", "AXTextArea",
    "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXComboBox", "AXLink",
    "AXSlider", "AXTab", "AXRow", "AXCell", "AXDisclosureTriangle",
    "AXSearchField", "AXMenuButton", "AXIncrementor", "AXSwitch", "AXToggle",
}
LABEL_ATTRS = ("AXTitle", "AXDescription", "AXValue", "AXHelp", "AXPlaceholderValue")
MAX_NODES = 20000
MAX_DEPTH = 60


def attr(el, name):
    err, val = AXUIElementCopyAttributeValue(el, name, None)
    return val if err == 0 else None


def label_of(el):
    for name in LABEL_ATTRS:
        v = attr(el, name)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Fall back to a titled child (e.g. a button wrapping a static text)
    for child in attr(el, "AXChildren") or []:
        v = attr(child, "AXValue") or attr(child, "AXTitle")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def walk(root):
    stats = {"nodes": 0, "actionable": 0, "labeled": 0, "roles": {}}
    stack = [(root, 0)]
    while stack and stats["nodes"] < MAX_NODES:
        el, depth = stack.pop()
        stats["nodes"] += 1
        role = attr(el, "AXRole") or "?"
        if role in ACTIONABLE:
            stats["actionable"] += 1
            stats["roles"][role] = stats["roles"].get(role, 0) + 1
            if label_of(el):
                stats["labeled"] += 1
        if depth < MAX_DEPTH:
            for child in attr(el, "AXChildren") or []:
                stack.append((child, depth + 1))
    return stats


def measure(app_el):
    out = {}
    for name, root in (("window", attr(app_el, "AXFocusedWindow") or attr(app_el, "AXMainWindow")),
                       ("menubar", attr(app_el, "AXMenuBar"))):
        if root is None:
            out[name] = None
            continue
        t0 = time.perf_counter()
        s = walk(root)
        s["ms"] = round((time.perf_counter() - t0) * 1000)
        out[name] = s
    return out


def fmt(s):
    if s is None:
        return "none"
    pct = 100 * s["labeled"] // max(s["actionable"], 1)
    top = sorted(s["roles"].items(), key=lambda kv: -kv[1])[:4]
    return (f"{s['nodes']} nodes, {s['actionable']} actionable, {pct}% labeled, "
            f"{s['ms']}ms  {top}")


def probe(app):
    pid = app.processIdentifier()
    el = AXUIElementCreateApplication(pid)
    print(f"\n## {app.localizedName()} ({app.bundleIdentifier()})")
    before = measure(el)
    print(f"  window  : {fmt(before['window'])}")
    print(f"  menubar : {fmt(before['menubar'])}")

    originals = {k: attr(el, k) for k in ("AXManualAccessibility", "AXEnhancedUserInterface")}
    forced = [k for k in originals if AXUIElementSetAttributeValue(el, k, True) == 0]
    if not forced:
        print("  forcing : not supported")
        return
    time.sleep(2.0)  # Chromium builds its tree lazily after the flag flips
    after = measure(el)
    print(f"  forced {forced}")
    print(f"  window  : {fmt(after['window'])}")
    for k in forced:
        AXUIElementSetAttributeValue(el, k, bool(originals[k]))


def main(names):
    if not AXIsProcessTrusted():
        sys.exit("Terminal lacks Accessibility permission (System Settings > Privacy & Security > Accessibility).")
    apps = [a for a in NSWorkspace.sharedWorkspace().runningApplications() if a.activationPolicy() == 0]
    if names:
        apps = [a for a in apps if a.localizedName() in names]
    for app in apps:
        probe(app)


if __name__ == "__main__":
    main(sys.argv[1:])
