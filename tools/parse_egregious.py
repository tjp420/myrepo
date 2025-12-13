import re

pylint_file = "pylint_latest.txt"
src_file = "agi_chatbot/api_server.py"
text = open(pylint_file, "r", encoding="utf-8", errors="ignore").read()

# Patterns to capture issues
p_e0102 = re.compile(
    r"^(.*):(?P<line>\d+):\d+: E0102: (function already defined|class already defined) line (?P<orig>\d+) \(function-redefined\)",
    re.MULTILINE,
)
p_e1121 = re.compile(r"^(.*):(?P<line>\d+):\d+: E1121: (?P<msg>.+)$", re.MULTILINE)
p_e1123 = re.compile(r"^(.*):(?P<line>\d+):\d+: E1123: (?P<msg>.+)$", re.MULTILINE)

issues = []
for m in p_e0102.finditer(text):
    line = int(m.group("line"))
    orig = int(m.group("orig"))
    issues.append(("E0102", line, f"defined previously at {orig}"))
for m in p_e1121.finditer(text):
    line = int(m.group("line"))
    msg = m.group("msg")
    if "Too many positional arguments" in msg:
        issues.append(("E1121", line, msg.strip()))
for m in p_e1123.finditer(text):
    line = int(m.group("line"))
    msg = m.group("msg")
    issues.append(("E1123", line, msg.strip()))

# Deduplicate and sort by occurrence (line order)
issues_sorted = sorted(issues, key=lambda x: x[1])

# Read source and produce context
src_lines = open(src_file, "r", encoding="utf-8", errors="ignore").read().splitlines()


def context_at(line, before=3, after=3):
    start = max(0, line - 1 - before)
    end = min(len(src_lines), line - 1 + after + 1)
    return "\n".join(f"{i+1:5d}: {src_lines[i]}" for i in range(start, end))


# Print top 20 issues
print("# Top E0102/E1121/E1123 issues (up to 20)")
for i, (code, line, msg) in enumerate(issues_sorted[:20], 1):
    print(f"---\n{i}. {code} at line {line}: {msg}\n")
    print(context_at(line))

# Save concise list for patch planning
with open("tools/egregious_sites.txt", "w", encoding="utf-8") as fh:
    for code, line, msg in issues_sorted:
        fh.write(f"{code}\t{line}\t{msg}\n")

print("\nSaved tools/egregious_sites.txt")
