import importlib
import traceback

try:
    m = importlib.import_module("agi_chatbot.core.chatbot")
    print("Imported agi_chatbot.core.chatbot OK:", getattr(m, "__file__", None))
except Exception:
    traceback.print_exc()
