import re
from pathlib import Path

p = Path("dep_tree.txt")
text = p.read_text()
lines = text.splitlines()
print("total lines", len(lines))
for i, l in enumerate(lines[:20], 1):
    print(i, repr(l))
matches = [l for l in lines if re.match(r"^[A-Za-z0-9_.+-]+==", l)]
print("matches", len(matches))
for m in matches[:40]:
    print(m)
