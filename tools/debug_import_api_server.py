import importlib
import os
import sys
import traceback

try:
    importlib.import_module("agi_chatbot.api_server")
    print("IMPORT_OK")
except Exception:
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    with open(
        os.path.join(os.path.dirname(__file__), "import_error.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        traceback.print_exc(file=f)
    print("WROTE_ERROR")
    sys.exit(1)
