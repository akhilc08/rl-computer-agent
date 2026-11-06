"""Fill a sandbox Mac with made-up but realistic content: notes in folders, calendars with events,
contacts in groups, reminder lists, and a tree of documents, images, PDFs and spreadsheets.

Everything is invented. Randomised per machine (--seed, default from the hostname) so parallel
sandboxes don't all show the same screens. Tops up only: each app is seeded when it holds fewer
items than its target, since the collector deletes things and calls this again every few rounds.
Refuses to run unless RLCA_SANDBOX=1.

  RLCA_SANDBOX=1 python scripts/vm/seed_content.py
"""
import argparse
import csv
import datetime as dt
import os
import random
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

FIRST = ["Sam", "Priya", "Jordan", "Alex", "Maria", "Wei", "Fatima", "Diego", "Hannah", "Kofi", "Yuki", "Liam",
         "Aisha", "Mateo", "Elena", "Noah", "Zara", "Omar", "Grace", "Ravi", "Chloe", "Ivan", "Leila", "Ben",
         "Sofia", "Tariq", "Mia", "Lucas", "Nina", "Arjun", "Olivia", "Kenji", "Ama", "Felix", "Isla", "Mohammed"]
LAST = ["Rivera", "Natarajan", "Lee", "Chen", "Gomez", "Zhang", "Khan", "Alvarez", "Muller", "Mensah", "Tanaka",
        "OBrien", "Bello", "Rossi", "Petrova", "Smith", "Ahmed", "Kowalski", "Park", "Iyer", "Dubois", "Novak",
        "Haddad", "Cohen", "Silva", "Nakamura", "Owusu", "Berg", "Fischer", "Costa"]
COMPANIES = ["Northwind Labs", "Bluefin Analytics", "Cedar & Pine Design", "Orbit Freight", "Lumen Health",
             "Maple Street Bakery", "Harbor Legal", "Quill Publishing", "Summit Robotics", "Tidewater Energy"]
STREETS = ["Oak St", "Maple Ave", "Elm Rd", "Lakeview Dr", "Hillcrest Ln", "River Rd", "Park Pl", "Cherry St"]
CITIES = [("Ithaca", "NY", "14850"), ("Portland", "OR", "97205"), ("Austin", "TX", "78701"), ("Denver", "CO", "80202"),
          ("Madison", "WI", "53703"), ("Raleigh", "NC", "27601"), ("Boston", "MA", "02116")]
NOTE_FOLDERS = {
    "Work": ["Q3 planning", "1:1 with {p}", "Launch checklist", "Interview notes - {p}", "Vendor comparison",
             "Retro notes", "OKRs draft", "Budget questions", "Onboarding plan", "Design review feedback"],
    "Personal": ["Packing list - {c}", "Gift ideas", "Books to read", "Apartment search", "Workout plan",
                 "Movies to watch", "Car maintenance", "Passwords to reset (no actual passwords)", "Birthday party", "Garden plan"],
    "Recipes": ["Shakshuka", "Miso soup", "Banana bread", "Chana masala", "Pad thai", "Lemon pasta", "Chili",
                "Overnight oats", "Focaccia", "Tomato soup"],
    "School": ["Linear algebra review", "History essay outline", "Lab report - titration", "Spanish vocab week 6",
               "CS project ideas", "Exam schedule", "Reading notes ch. 4", "Group project roles", "Office hours Qs", "Thesis topics"],
}
WORDS = ("plan review draft budget schedule team update follow call meeting notes idea list check buy book send "
         "finish start research compare call email order fix clean water read write test ship").split()
EVENTS = ["Team standup", "Dentist", "Lunch with {p}", "Gym", "Project review", "Call with {p}", "Yoga", "Study group",
          "Flight to {c}", "Coffee with {p}", "Sprint planning", "Doctor appointment", "Book club", "Haircut",
          "Parent-teacher meeting", "1:1 with {p}", "Farmers market", "Concert", "Soccer practice", "Date night"]
REMINDER_LISTS = {
    "Groceries": ["Eggs", "Oat milk", "Spinach", "Rice", "Lemons", "Coffee beans", "Bread", "Apples", "Olive oil", "Yogurt"],
    "Errands": ["Pick up dry cleaning", "Return library books", "Renew passport", "Mail package to {p}", "Get car washed",
                "Buy birthday gift for {p}", "Pay water bill", "Book vet appointment"],
    "Work": ["Send slides to {p}", "Review PR #{n}", "Update roadmap", "Expense report", "Prepare demo",
             "Reply to {p} about contract", "Schedule offsite", "Write weekly update"],
}


def osa(script, timeout=300):
    """Run AppleScript; launch the target app first and retry once if it quit mid-script (-609),
    which happens when the collector closes apps between jobs."""
    m = re.search(r'tell application "([^"]+)"', script)
    for attempt in range(2):
        if m:
            subprocess.run(["open", "-g", "-a", m.group(1)], capture_output=True)
            time.sleep(2)
        r = subprocess.run(["osascript", "-"], input=script, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip()
        print("osascript error:", r.stderr.strip()[:300], flush=True)
        if "-609" not in r.stderr and "-600" not in r.stderr:
            break
    return r.stdout.strip()


def q(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def count(script):
    try:
        return int(osa(script, timeout=60) or 0)
    except ValueError:
        return 0


def person(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def fill(rng, s):
    return s.format(p=person(rng).split()[0], c=rng.choice(CITIES)[0], n=rng.randint(100, 999))


def note_body(rng, title):
    lines = [f"<h1>{q(title)}</h1>"]
    kind = rng.random()
    if kind < 0.35:
        lines.append("<ul>" + "".join(f"<li>{' '.join(rng.sample(WORDS, 3)).capitalize()}</li>" for _ in range(rng.randint(3, 8))) + "</ul>")
    elif kind < 0.55:
        rows = "".join(f"<tr><td>{rng.choice(WORDS)}</td><td>{rng.randint(1, 99)}</td><td>{rng.choice(['yes', 'no', 'maybe'])}</td></tr>"
                       for _ in range(rng.randint(3, 6)))
        lines.append(f"<table><tr><th>Item</th><th>Qty</th><th>Done</th></tr>{rows}</table>")
    for _ in range(rng.randint(1, 4)):
        lines.append("<div>" + " ".join(rng.choices(WORDS, k=rng.randint(8, 25))).capitalize() + ".</div>")
    return "".join(lines)


def seed_notes(rng, target=40):
    if count('tell application "Notes" to count notes of account "On My Mac"') >= target:
        return "notes ok"
    parts = ['tell application "Notes"', 'set acc to account "On My Mac"']
    for folder, titles in NOTE_FOLDERS.items():
        parts.append(f'if not (exists folder "{folder}" of acc) then make new folder at acc with properties {{name:"{folder}"}}')
        for t in titles:
            title = fill(rng, t)
            parts.append(f'make new note at folder "{folder}" of acc with properties {{body:"{q(note_body(rng, title))}"}}')
    parts.append("end tell")
    osa("\n".join(parts))
    return "notes seeded"


def seed_calendar(rng, target=40):
    if count('tell application "Calendar"\nset n to 0\nrepeat with nm in {"Work", "Personal", "Family"}\nif exists calendar nm then set n to n + (count events of calendar nm)\n'
             'end repeat\nreturn n\nend tell') >= target:
        return "calendar ok"
    parts = ['tell application "Calendar"']
    for cal in ("Work", "Personal", "Family"):
        parts.append(f'if not (exists calendar "{cal}") then make new calendar with properties {{name:"{cal}"}}')
    parts.append("set base to (current date)\nset hours of base to 0\nset minutes of base to 0\nset seconds of base to 0")
    for _ in range(60):
        cal = rng.choice(["Work", "Personal", "Family"])
        day = rng.randint(-21, 21)
        title = q(fill(rng, rng.choice(EVENTS)))
        loc = q(f"{rng.randint(10, 999)} {rng.choice(STREETS)}, {rng.choice(CITIES)[0]}") if rng.random() < 0.5 else ""
        if rng.random() < 0.12:
            parts.append(f'make new event at end of events of calendar "{cal}" with properties {{summary:"{title}", '
                         f'start date:base + ({day} * days), end date:base + ({day + 1} * days), allday event:true}}')
        else:
            start = rng.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]) * 60 + rng.choice([0, 30])
            dur = rng.choice([30, 45, 60, 90, 120])
            parts.append(f'make new event at end of events of calendar "{cal}" with properties {{summary:"{title}", '
                         f'location:"{loc}", start date:base + ({day} * days) + ({start} * minutes), '
                         f'end date:base + ({day} * days) + ({start + dur} * minutes)}}')
    parts.append("end tell")
    osa("\n".join(parts))
    return "calendar seeded"


def seed_contacts(rng, target=30):
    if count('tell application "Contacts" to count people') >= target:
        return "contacts ok"
    parts = ['tell application "Contacts"']
    for g in ("Family", "Work", "Book Club"):
        parts.append(f'if not (exists group "{g}") then make new group with properties {{name:"{g}"}}')
    for i in range(40):
        f, l = rng.choice(FIRST), rng.choice(LAST)
        co = rng.choice(COMPANIES)
        city, st, zipc = rng.choice(CITIES)
        parts.append(f'set p to make new person with properties {{first name:"{f}", last name:"{l}", organization:"{co}"}}')
        parts.append(f'make new email at end of emails of p with properties {{label:"work", value:"{f.lower()}.{l.lower()}@example.com"}}')
        parts.append(f'make new phone at end of phones of p with properties {{label:"mobile", value:"(555) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"}}')
        if rng.random() < 0.6:
            parts.append(f'make new address at end of addresses of p with properties {{label:"home", street:"{rng.randint(1, 999)} {rng.choice(STREETS)}", '
                         f'city:"{city}", state:"{st}", zip:"{zipc}"}}')
        if rng.random() < 0.5:
            parts.append(f'add p to group "{rng.choice(["Family", "Work", "Book Club"])}"')
    parts += ["save", "end tell"]
    osa("\n".join(parts))
    return "contacts seeded"


def seed_reminders(rng, target=20):
    if count('tell application "Reminders" to count reminders') >= target:
        return "reminders ok"
    parts = ['tell application "Reminders"', "set base to (current date)"]
    for lst, items in REMINDER_LISTS.items():
        parts.append(f'if not (exists list "{lst}") then make new list with properties {{name:"{lst}"}}')
        for it in items:
            props = f'name:"{q(fill(rng, it))}"'
            if rng.random() < 0.4:
                props += f", due date:base + ({rng.randint(-3, 14)} * days)"
            if rng.random() < 0.2:
                props += ", completed:true"
            if rng.random() < 0.2:
                props += ", priority:1"
            parts.append(f'make new reminder at end of reminders of list "{lst}" with properties {{{props}}}')
    parts.append("end tell")
    osa("\n".join(parts))
    return "reminders seeded"


def seed_files(rng, target=40):
    from PIL import Image, ImageDraw
    home = Path.home()
    if sum(1 for _ in (home / "Documents").rglob("*") if _.is_file()) >= target:
        return "files ok"
    dirs = ["Documents/Work/Reports", "Documents/Work/Invoices", "Documents/School/Essays", "Documents/School/Labs",
            "Documents/Taxes/2025", "Documents/Recipes", "Desktop/Screenshots", "Downloads", "Pictures/Trips"]
    for d in dirs:
        (home / d).mkdir(parents=True, exist_ok=True)
    for i in range(12):
        d = home / rng.choice(dirs[:6])
        title = fill(rng, rng.choice(sum(NOTE_FOLDERS.values(), [])))
        (d / f"{title.replace('/', '-')}.txt").write_text(" ".join(rng.choices(WORDS, k=120)))
        (d / f"{title.replace('/', '-')} notes.md").write_text(f"# {title}\n\n" + "\n".join(f"- {' '.join(rng.sample(WORDS, 4))}" for _ in range(8)))
        rtf = "{\\rtf1\\ansi{\\fonttbl\\f0 Helvetica;}\\f0\\fs24 " + f"{title}\\par " + " ".join(rng.choices(WORDS, k=80)) + "}"
        (d / f"{title.replace('/', '-')}.rtf").write_text(rtf)
    for i in range(6):
        with open(home / rng.choice(["Documents/Work/Reports", "Documents/Taxes/2025", "Downloads"]) / f"{rng.choice(['budget', 'sales', 'expenses', 'grades', 'inventory', 'timesheet'])}-{i}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "item", "amount", "category"])
            for _ in range(rng.randint(10, 40)):
                w.writerow([(dt.date.today() - dt.timedelta(days=rng.randint(0, 200))).isoformat(), rng.choice(WORDS),
                            round(rng.uniform(3, 900), 2), rng.choice(["food", "travel", "office", "rent", "misc"])])
    colors = [(230, 120, 90), (90, 140, 220), (120, 200, 140), (240, 200, 80), (170, 110, 210), (60, 60, 70)]
    for i in range(14):
        img = Image.new("RGB", (rng.choice([800, 1200, 1600]), rng.choice([600, 900])), rng.choice(colors))
        dr = ImageDraw.Draw(img)
        for _ in range(rng.randint(3, 9)):
            x, y = rng.randint(0, img.width - 50), rng.randint(0, img.height - 50)
            dr.ellipse([x, y, x + rng.randint(30, 300), y + rng.randint(30, 300)], fill=rng.choice(colors))
        dr.text((30, 30), fill(rng, rng.choice(["Trip to {c}", "Whiteboard", "Receipt", "Diagram", "Poster draft"])), fill=(255, 255, 255))
        dest = home / rng.choice(["Pictures/Trips", "Desktop/Screenshots", "Downloads", "Documents/School/Labs"])
        name = f"IMG_{rng.randint(1000, 9999)}"
        img.save(dest / f"{name}.png") if rng.random() < 0.5 else img.save(dest / f"{name}.jpg", quality=85)
        if i % 3 == 0:
            img.save(home / "Documents/Work/Invoices" / f"invoice-{rng.randint(1000, 9999)}.pdf")
    return "files seeded"


def main():
    if os.environ.get("RLCA_SANDBOX") != "1":
        sys.exit("refusing: set RLCA_SANDBOX=1 on a sandbox Mac only")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else sum(map(ord, socket.gethostname())) + dt.date.today().toordinal()
    rng = random.Random(seed)
    for step in (seed_files, seed_notes, seed_calendar, seed_contacts, seed_reminders):
        try:
            print(step(rng), flush=True)
        except Exception as e:          # one app failing shouldn't stop the rest
            print(f"{step.__name__} failed: {e}", flush=True)


if __name__ == "__main__":
    main()
