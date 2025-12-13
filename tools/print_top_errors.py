"""Debug: print top error entries from pylint_errors.json."""

import json


def main():
    with open("pylint_errors.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    errors = [e for e in data if e.get("type") == "error"]
    print("TOTAL_ERRORS", len(errors))
    for i, e in enumerate(errors[:50], 1):
        print(
            f"{i}. {e.get('path')}:{e.get('line')} {e.get('message-id')} {e.get('message')}"
        )


if __name__ == "__main__":
    main()
