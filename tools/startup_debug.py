import asyncio
import importlib
import os
import sys
import traceback

# Ensure workspace root is on sys.path
sys.path.insert(0, os.getcwd())

# Force non-dev mode
os.environ["AGI_DEV"] = "0"

print("Starting non-dev startup debug runner", file=sys.stderr)
try:
    m = importlib.import_module("agi_chatbot.api_server")
    print("Imported module:", getattr(m, "__file__", None), file=sys.stderr)
    # Inspect key globals to verify which implementations were imported
    for name in (
        "query_preprocessor",
        "response_cache",
        "lazy_model_loader",
        "chatbot",
        "oracle",
    ):
        obj = getattr(m, name, None)
        print(f"{name}: type={type(obj)}", file=sys.stderr)
        if obj is not None:
            try:
                attrs = sorted([a for a in dir(obj) if not a.startswith("_")])[:40]
                print(f"  attrs sample: {attrs}", file=sys.stderr)
            except Exception:
                pass

    # Ensure we can call the async startup
    async def run_startup():
        try:
            if hasattr(m, "_on_startup"):
                print("Calling _on_startup()", file=sys.stderr)
                await m._on_startup()
            else:
                print("No _on_startup found", file=sys.stderr)
        except Exception:
            print("Exception during _on_startup:", file=sys.stderr)
            traceback.print_exc()

    asyncio.run(run_startup())

except Exception:
    print("Exception during import/startup:", file=sys.stderr)
    traceback.print_exc()
