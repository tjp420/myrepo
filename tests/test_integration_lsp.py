import json
import socket
import threading
import time

import tools.mock_lsp_server as mock_lsp_server


def _send_lsp(host, port, obj, timeout=2.0):
    data = json.dumps(obj)
    header = f"Content-Length: {len(data.encode('utf-8'))}\r\n\r\n"
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(header.encode("utf-8") + data.encode("utf-8"))
        # read header
        hdr = b""
        while b"\r\n\r\n" not in hdr:
            ch = s.recv(1)
            if not ch:
                break
            hdr += ch
        if not hdr:
            return None
        header_str, _, _ = hdr.partition(b"\r\n\r\n")
        headers = header_str.decode("utf-8").split("\r\n")
        content_length = 0
        for h in headers:
            if h.lower().startswith("content-length:"):
                content_length = int(h.split(":", 1)[1].strip())
        body = b""
        while len(body) < content_length:
            chunk = s.recv(content_length - len(body))
            if not chunk:
                break
            body += chunk
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return None


def test_attachment_put_get_integration():
    # start server in background thread (daemon so tests can exit cleanly)
    t = threading.Thread(target=mock_lsp_server.run, daemon=True)
    t.start()
    # give server time to bind
    time.sleep(0.2)

    host = mock_lsp_server.HOST
    port = mock_lsp_server.PORT

    # attachmentPut
    put_msg = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "workspace/attachmentPut",
        "params": {"id": "int-1", "content": "integration"},
    }
    resp_put = _send_lsp(host, port, put_msg)
    assert resp_put is not None
    assert resp_put.get("id") == 101

    # attachmentGet
    get_msg = {
        "jsonrpc": "2.0",
        "id": 102,
        "method": "workspace/attachmentGet",
        "params": {"id": "int-1"},
    }
    resp_get = _send_lsp(host, port, get_msg)
    assert resp_get is not None
    assert resp_get.get("id") == 102
    assert resp_get.get("result", {}).get("content") == "integration"


def send_lsp_request(msg):
    data = json.dumps(msg)
    header = f'Content-Length: {len(data.encode("utf-8"))}\r\n\r\n'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((mock_lsp_server.HOST, mock_lsp_server.PORT))
        s.sendall(header.encode("utf-8") + data.encode("utf-8"))
        # read header
        hdr = b""
        while b"\r\n\r\n" not in hdr:
            ch = s.recv(1)
            if not ch:
                break
            hdr += ch
        if not hdr:
            return None
        header_str, _, _ = hdr.partition(b"\r\n\r\n")
        headers = header_str.decode("utf-8").split("\r\n")
        content_length = 0
        for h in headers:
            if h.lower().startswith("content-length:"):
                content_length = int(h.split(":", 1)[1].strip())
        body = b""
        while len(body) < content_length:
            chunk = s.recv(content_length - len(body))
            if not chunk:
                break
            body += chunk
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return None


def test_server_integration_attachment_put_get():
    # start server in background
    thr = threading.Thread(target=mock_lsp_server.run, daemon=True)
    thr.start()
    # give server a moment to bind
    time.sleep(0.2)

    put_msg = {
        "jsonrpc": "2.0",
        "id": 100,
        "method": "workspace/attachmentPut",
        "params": {"id": "int-1", "content": "integrate"},
    }
    resp_put = send_lsp_request(put_msg)
    assert resp_put is not None
    assert resp_put.get("id") == 100

    get_msg = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "workspace/attachmentGet",
        "params": {"id": "int-1"},
    }
    resp_get = send_lsp_request(get_msg)
    assert resp_get is not None
    assert resp_get.get("id") == 101
    assert resp_get.get("result", {}).get("content") == "integrate"
