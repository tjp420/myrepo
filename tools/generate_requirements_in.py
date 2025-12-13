import re
import sys
from pathlib import Path
from subprocess import run

# Re-generate dep_tree.txt using the venv python to ensure UTF-8 encoding
res = run([sys.executable, "-m", "pipdeptree"], capture_output=True, text=True)
res.check_returncode()
dep_txt = res.stdout
p = Path("dep_tree.txt")
p.write_text(dep_txt, encoding="utf-8")
lines = dep_txt.splitlines()
names = []
for line in lines:
    if re.match(r"^[A-Za-z0-9_.+-]+==", line):
        name = line.split("==", 1)[0].strip()
        names.append(name)
# unique preserve order
seen = set()
uniq = []
for n in names:
    nl = n.lower()
    if nl not in seen:
        seen.add(nl)
        uniq.append(n)
# keep editable/git lines from requirements.txt
reqp = Path("requirements.txt")
editable = []
if reqp.exists():
    for l in reqp.read_text().splitlines():
        if l.strip().startswith("-e") or l.strip().startswith("git+"):
            editable.append(l.strip())
# Write requirements.in
outp = Path("requirements.in")
with outp.open("w", encoding="utf-8") as f:
    for e in editable:
        f.write(e + "\n")
    for n in uniq:
        f.write(n + "\n")
print("wrote", outp, "with", len(uniq), "entries")
