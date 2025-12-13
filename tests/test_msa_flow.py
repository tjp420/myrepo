import importlib
import os
import sys

from fastapi.testclient import TestClient

# Force dev mode for deterministic shims
os.environ["AGI_DEV"] = "1"

# Ensure project root is first on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


api_mod = importlib.import_module("agi_chatbot.api_server")
client = TestClient(api_mod.app)


def _headers():
    headers = {}
    api_key = os.getenv("AGI_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def test_msa_optimize_validate_predict_flow():
    code = "def foo():\n\treturn 1  \n"

    r_opt = client.post(
        "/api/optimize",
        json={
            "code": code,
            "language": "python",
            "context": "File: foo.py",
            "session_id": "session_opt",
        },
        headers=_headers(),
    )
    assert r_opt.status_code == 200
    j_opt = r_opt.json()
    assert j_opt.get("success") is True
    assert j_opt.get("session_id") == "session_opt"
    assert isinstance(j_opt.get("optimized_code"), str)
    assert j_opt["optimized_code"].endswith("\n")
    assert "\t" not in j_opt["optimized_code"]

    optimized_code = j_opt["optimized_code"]

    r_val = client.post(
        "/api/validate",
        json={
            "code": optimized_code,
            "filename": "foo.py",
            "language": "python",
            "session_id": "session_val",
        },
        headers=_headers(),
    )
    assert r_val.status_code == 200
    j_val = r_val.json()
    assert j_val.get("success") is True
    assert j_val.get("session_id") == "session_val"
    assert isinstance(j_val.get("validation"), dict)
    assert isinstance(j_val["validation"].get("syntax_ok"), bool)
    assert isinstance(j_val["validation"].get("security_issues"), list)

    r_pred = client.post(
        "/api/predict",
        json={
            "code": optimized_code,
            "context": "File: foo.py",
            "session_id": "session_pred",
        },
        headers=_headers(),
    )
    assert r_pred.status_code == 200
    j_pred = r_pred.json()
    assert j_pred.get("success") is True
    assert j_pred.get("session_id") == "session_pred"
    assert isinstance(j_pred.get("prediction"), dict)
    assert isinstance(j_pred["prediction"].get("temporal_score"), int)
