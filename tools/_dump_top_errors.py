import json

with open("pylint_errors.json", "r", encoding="utf-8") as f:
    data = json.load(f)

errors = [e for e in data if e.get("type") == "error"]
with open("tools/top_errors.txt", "w", encoding="utf-8") as out:
    out.write(f"TOTAL_ERRORS {len(errors)}\n")
    for i, e in enumerate(errors[:200], 1):
        out.write(
            f"{i}. {e.get('path')}:{e.get('line')} {e.get('message-id')} {e.get('message')}\n"
        )
print("WROTE tools/top_errors.txt")
