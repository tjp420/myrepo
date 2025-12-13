import importlib
import sys
import traceback

try:
    m = importlib.import_module("agi_chatbot.api_server")
    with open("E:/Ai/AGI Chatbot/tools/import_ok_try.txt", "w", encoding="utf-8") as f:
        f.write(f'OK: {getattr(m, "__file__", None)}\n')
    print("WROTE import_ok_try.txt")
except Exception:
    tb = traceback.format_exc()
    with open(
        "E:/Ai/AGI Chatbot/tools/import_error_try.txt", "w", encoding="utf-8"
    ) as f:
        f.write(tb)
    print("WROTE import_error_try.txt")
    sys.exit(1)
