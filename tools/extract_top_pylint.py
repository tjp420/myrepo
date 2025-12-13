import json

with open("pylint_errors.json", "r", encoding="utf-8") as f:
    data = json.load(f)

prio = ["E0401", "E0611", "E0402", "E0602", "E0601", "E0102"]
errors = [e for e in data if e.get("type") == "error"]


def score(e):
    try:
        return prio.index(e.get("message-id"))
    except ValueError:
        return len(prio)


errors_sorted = sorted(errors, key=score)

for i, e in enumerate(errors_sorted[:20], 1):
    path = e.get("path")
    line = e.get("line")
    mid = e.get("message-id")
    msg = e.get("message")
    print(f"{i}. {path}:{line} {mid} {msg}")
