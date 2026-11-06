#!/bin/zsh
# Fill a throwaway Mac (Tart VM or EC2 Mac) with made-up content so explored screens can be published:
# notes, calendar events, contacts, reminders, and a small folder tree. Nothing here is real.
# Refuses to run unless RLCA_SANDBOX=1, so it never touches the owner's Mac.
#   RLCA_SANDBOX=1 scripts/vm/seed_content.sh
set -e
[[ "$RLCA_SANDBOX" == 1 ]] || { echo "refusing: set RLCA_SANDBOX=1 on a sandbox Mac only"; exit 1; }

NOTES=("Grocery list|eggs, oat milk, spinach, rice, lemons" "Trip to Lisbon|flight Fri 9am, hotel near Alfama, tram 28"
       "Book club|next pick: The Remains of the Day, meet the 14th" "Project ideas|habit tracker, plant watering bot"
       "Meeting notes 3/2|Q2 roadmap, hire 2 designers, review budget" "Recipes|shakshuka, miso soup, banana bread")
for n in "${NOTES[@]}"; do
  osascript -e "tell application \"Notes\" to make new note at folder \"Notes\" of account \"On My Mac\" with properties {name:\"${n%%|*}\", body:\"${n#*|}\"}"
done

osascript <<'EOF'
tell application "Calendar"
  if not (exists calendar "Personal") then make new calendar with properties {name:"Personal"}
  set c to calendar "Personal"
  set base to (current date)
  set hours of base to 9
  set minutes of base to 0
  set titles to {"Dentist", "Team standup", "Lunch with Sam", "Gym", "Project review", "Call with Priya", "Yoga", "Study group"}
  repeat with i from 1 to count of titles
    set s to base + ((i - 3) * days) + ((i mod 4) * hours)
    make new event at end of events of c with properties {summary:(item i of titles), start date:s, end date:s + 1 * hours}
  end repeat
end tell
EOF

PEOPLE=("Sam|Rivera|sam.rivera@example.com" "Priya|Natarajan|priya.n@example.com" "Jordan|Lee|jordan.lee@example.com"
        "Alex|Chen|alex.chen@example.com" "Maria|Gomez|maria.gomez@example.com")
for p in "${PEOPLE[@]}"; do
  IFS='|' read -r f l e <<< "$p"
  osascript -e "tell application \"Contacts\"
    set x to make new person with properties {first name:\"$f\", last name:\"$l\"}
    make new email at end of emails of x with properties {label:\"work\", value:\"$e\"}
    save
  end tell"
done

for r in "Pay rent" "Return library books" "Buy birthday gift for Alex" "Renew passport" "Water plants"; do
  osascript -e "tell application \"Reminders\" to make new reminder with properties {name:\"$r\"}"
done

mkdir -p ~/Documents/{Work,School,Taxes} ~/Desktop/Screenshots
for f in Work/q2-roadmap.txt Work/budget-draft.txt School/cs-notes.txt School/essay-outline.txt Taxes/2025-checklist.txt; do
  echo "placeholder for $f" > ~/Documents/$f
done
echo "seeded"
