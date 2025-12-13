import re
from collections import Counter

fname = "pylint_latest.txt"
text = open(fname, "r", encoding="utf-8", errors="ignore").read()

# patterns to capture class/instance names and variable/function names
p_e1101 = re.compile(r"Instance of '([^']+)' has no '([^']+)' member")
p_e0602 = re.compile(r"Undefined variable '([^']+)'")
p_e0606 = re.compile(r"Possibly using variable '([^']+)' before assignment")
p_e0611 = re.compile(r"No name '([^']+)' in module '([^']+)'")
p_e1128 = re.compile(
    r"Assigning result of a function call, where the function returns None \(assignment-from-none\)"
)

counter = Counter()

for m in p_e1101.finditer(text):
    cls, member = m.groups()
    counter[cls] += 1
    counter[f"{cls}.{member}"] += 1

for m in p_e0611.finditer(text):
    name, mod = m.groups()
    counter[name] += 1

for m in p_e0602.finditer(text):
    name = m.group(1)
    counter[name] += 1

for m in p_e0606.finditer(text):
    name = m.group(1)
    counter[name] += 1

# Also catch some common 'not-async-context-manager' or 'function-redefined' as names
for line in text.splitlines():
    if "function already defined" in line or "class already defined" in line:
        # attempt to capture the symbol from surrounding text
        pass

# produce top 40
most = counter.most_common(40)
print("# Top linter symbols (count, symbol)")
for c, name in most:
    print(f"{c:4d}  {name}")

# Also print plain counts separately for frequent members (like get_reasoning_explanation)
print("\n# Top member-qualified symbols (class.member)")
for name, c in counter.most_common(40):
    if "." in name:
        print(f"{c:4d}  {name}")
