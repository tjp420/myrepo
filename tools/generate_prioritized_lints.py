import os
import re
import subprocess
from collections import Counter

venv_py = os.path.join(".", ".venv", "Scripts", "python.exe")
cmd = [
    venv_py,
    "-m",
    "pylint",
    "-E",
    "--rcfile=dev_shims.pylintrc",
    "agi_chatbot/api_server.py",
]
print("Running:", " ".join(cmd))
# Ensure env
env = os.environ.copy()
env["AGI_DEV"] = "1"

proc = subprocess.run(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True
)
text = proc.stdout
open("pylint_latest.txt", "w", encoding="utf-8").write(text)

# Patterns
p_e1101 = re.compile(r"Instance of '([^']+)' has no '([^']+)' member")
p_e0611 = re.compile(r"No name '([^']+)' in module '([^']+)'")
p_e0602 = re.compile(r"Undefined variable '([^']+)'")
p_e0606 = re.compile(r"Possibly using variable '([^']+)' before assignment")

counter = Counter()
for m in p_e1101.finditer(text):
    cls, member = m.groups()
    counter[cls] += 1
    counter[f"{cls}.{member}"] += 1
for m in p_e0611.finditer(text):
    name, _ = m.groups()
    counter[name] += 1
for m in p_e0602.finditer(text):
    name = m.group(1)
    counter[name] += 1
for m in p_e0606.finditer(text):
    name = m.group(1)
    counter[name] += 1

most = counter.most_common(60)
print("\nTop linter symbols (count, symbol):")
for name, c in most:
    print(f"{c:4d}  {name}")

# Save a simple list file for later patching
with open("tools/top_lints.txt", "w", encoding="utf-8") as fh:
    for name, c in most:
        fh.write(f"{name}\n")

print("\nSaved tools/top_lints.txt")
print("Exit code:", proc.returncode)
