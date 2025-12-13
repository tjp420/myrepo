import importlib
import os
import sys
import types

from fastapi.testclient import TestClient

# Ensure dev mode for deterministic shims
os.environ.setdefault("AGI_DEV", "1")

# Ensure project root is first on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Lightweight torch shim to avoid importing real torch/site-packages
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.__version__ = "0.0.0"
    _torch.nn = types.SimpleNamespace()
    _torch.Tensor = object
    _torch.load = lambda *a, **k: None
    _torch.save = lambda *a, **k: None
    _torch.__config__ = types.SimpleNamespace(show=lambda: "")

    class _IInfo:
        def __init__(self, typ):
            self.max = 2**63 - 1

    _torch.iinfo = lambda t: _IInfo(t)
    sys.modules["torch"] = _torch

# Lightweight torch_geometric shim
if "torch_geometric" not in sys.modules:
    _tg = types.ModuleType("torch_geometric")
    _tg.data = types.ModuleType("torch_geometric.data")
    _tg.typing = types.ModuleType("torch_geometric.typing")
    _tg.typing.WITH_PT20 = False
    _tg.typing.MAX_INT64 = 2**63 - 1
    _tg.typing.NO_MKL = False
    sys.modules["torch_geometric"] = _tg
    sys.modules["torch_geometric.data"] = _tg.data
    sys.modules["torch_geometric.typing"] = _tg.typing


# Provide a minimal shim for `agi_chatbot.parallel_processor` which some
# startup modules attempt to import during app initialization.
if "agi_chatbot.parallel_processor" not in sys.modules:
    _pp = types.ModuleType("agi_chatbot.parallel_processor")
    _pp.ParallelProcessor = type(
        "ParallelProcessor", (), {"__init__": lambda self, *a, **k: None}
    )
    sys.modules["agi_chatbot.parallel_processor"] = _pp

# Import the API app under dev-mode shims
api_mod = importlib.import_module("agi_chatbot.api_server")
client = TestClient(api_mod.app)


def test_cache_stats_structure():
    r = client.get("/cache/stats")
    assert r.status_code == 200
    j = r.json()
    assert "data" in j
    data = j["data"]
    assert isinstance(data.get("entries"), int)
    assert isinstance(data.get("keys"), list)
    assert isinstance(data.get("top_entries"), list)
