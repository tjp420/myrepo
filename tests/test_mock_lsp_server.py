import pytest

from tools.mock_lsp_server import MockLSPServer


@pytest.fixture()
def server():
    return MockLSPServer()


def test_is_valid_message(server):
    assert server.is_valid_message({"jsonrpc": "2.0"})
    assert not server.is_valid_message({"jsonrpc": "1.0"})
    assert not server.is_valid_message("not-a-dict")


def test_initialize(server):
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    resp = server.initialize(msg, None)
    assert resp["id"] == 1
    assert "result" in resp and isinstance(resp["result"], dict)


def test_completion(server):
    msg = {"jsonrpc": "2.0", "id": 2, "method": "textDocument/completion"}
    resp = server.completion(msg, None)
    assert resp["id"] == 2
    assert resp["result"] == []


def test_attachment_put_and_get(server):
    put_msg = {"jsonrpc": "2.0", "id": 3, "method": "workspace/attachmentPut"}
    params = {"id": "att-1", "content": "hello-world"}
    resp_put = server.attachment_put(put_msg, params)
    assert resp_put["id"] == 3

    get_msg = {"jsonrpc": "2.0", "id": 4, "method": "workspace/attachmentGet"}
    resp_get = server.attachment_get(get_msg, {"id": "att-1"})
    assert resp_get["id"] == 4
    assert resp_get["result"]["content"] == "hello-world"

    # not found case
    resp_not_found = server.attachment_get(
        {"jsonrpc": "2.0", "id": 5}, {"id": "missing"}
    )
    assert resp_not_found.get("error") is not None


def test_diagnostic_request(server):
    msg = {"jsonrpc": "2.0", "id": 6, "method": "workspace/diagnosticRequest"}
    resp = server.diagnostic_request(msg, None)
    assert resp["id"] == 6
    assert "diagnostics" in resp["result"]


import pytest


def make_msg(method, id=None, params=None):
    m = {"jsonrpc": "2.0"}
    if method is not None:
        m["method"] = method
    if id is not None:
        m["id"] = id
    if params is not None:
        m["params"] = params
    return m


def test_initialize():
    srv = MockLSPServer()
    msg = make_msg("initialize", id=1, params={})
    resp = srv.initialize(msg, {})
    assert isinstance(resp, dict)
    assert resp.get("jsonrpc") == "2.0"
    assert resp.get("id") == 1
    assert "result" in resp


def test_completion():
    srv = MockLSPServer()
    msg = make_msg("textDocument/completion", id=2, params={})
    resp = srv.completion(msg, {})
    assert resp["id"] == 2
    assert resp["result"] == []


def test_attachment_put_get():
    srv = MockLSPServer()
    put_msg = make_msg(
        "workspace/attachmentPut", id=3, params={"id": "a1", "content": "hello"}
    )
    put_resp = srv.attachment_put(put_msg, put_msg["params"])
    assert put_resp["id"] == 3
    # now get
    get_msg = make_msg("workspace/attachmentGet", id=4, params={"id": "a1"})
    get_resp = srv.attachment_get(get_msg, get_msg["params"])
    assert get_resp["id"] == 4
    assert get_resp["result"]["content"] == "hello"


def test_attachment_get_not_found():
    srv = MockLSPServer()
    get_msg = make_msg("workspace/attachmentGet", id=5, params={"id": "missing"})
    resp = srv.attachment_get(get_msg, get_msg["params"])
    assert "error" in resp
    assert resp["error"]["message"] == "attachment not found"


def test_attachment_put_missing_params():
    srv = MockLSPServer()
    put_msg = make_msg("workspace/attachmentPut", id=6, params={})
    with pytest.raises(KeyError):
        srv.attachment_put(put_msg, put_msg["params"])


def test_diagnostic_request():
    srv = MockLSPServer()
    msg = make_msg("workspace/diagnosticRequest", id=7, params={})
    resp = srv.diagnostic_request(msg, {})
    assert resp["id"] == 7
    assert "diagnostics" in resp["result"]
