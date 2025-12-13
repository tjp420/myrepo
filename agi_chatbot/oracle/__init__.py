"""Oracle package shim to make agi_chatbot.oracle importable during linting/tests.

Expose a tiny `TextToSpeech` class used by api_server imports.
"""

__all__ = ["TextToSpeech"]


class TextToSpeech:
    def __init__(self, *args, **kwargs):
        pass

    def speak(self, text: str):
        return None

    def save_to_file(self, text: str, filename: str):
        # Minimal shim: write text to file if possible in test/dev environments.
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(str(text))
        except Exception:
            # Best-effort, but avoid raising during lint/test runs
            return False
        return True
