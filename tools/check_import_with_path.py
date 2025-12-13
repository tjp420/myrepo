import importlib
import os
import sys
import traceback

# Ensure project root on sys.path same as testclient_smoke
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    m = importlib.import_module("agi_chatbot.core.chatbot")
    print("Imported OK:", getattr(m, "__file__", None))
except Exception:
    traceback.print_exc()
