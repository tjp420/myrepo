"""
Simple test client for the mock LSP server.
Usage:
    python tools\mock_lsp_client.py --method workspace/attachmentPut --id att-1 --content hello

This script sends a single LSP request over TCP and prints the raw response.
"""

import argparse
import json
import socket

HOST = "127.0.0.1"
PORT = 2087


def send_message(msg):
    data = json.dumps(msg)
    header = f'Content-Length: {len(data.encode("utf-8"))}\r\n\r\n'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(header.encode("utf-8") + data.encode("utf-8"))
        # read header
        hdr = b""
        while b"\r\n\r\n" not in hdr:
            ch = s.recv(1)
            if not ch:
                break
            hdr += ch
        if not hdr:
            print("no response")
            return
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
            obj = json.loads(body.decode("utf-8"))
            print("response:", json.dumps(obj, indent=2))
        except Exception:
            print("raw body:", body)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="workspace/attachmentPut")
    p.add_argument("--id", default=1)
    p.add_argument("--content", default="hello")
    args = p.parse_args()
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": args.method,
        "params": {"id": str(args.id), "content": args.content},
    }
    send_message(msg)
"""Small test client for the mock LSP server.

Usage: run the server, then run this script to exercise attachmentPut/get.
"""

HOST = "127.0.0.1"
PORT = 2087


def send_msg(sock, obj):
    data = json.dumps(obj)
    header = f"Content-Length: {len(data.encode('utf-8'))}\r\n\r\n"
    sock.sendall(header.encode("utf-8") + data.encode("utf-8"))


def read_resp(sock):
    # read header
    header = b""
    while b"\r\n\r\n" not in header:
        ch = sock.recv(1)
        if not ch:
            return None
        header += ch
    header_str, _, _ = header.partition(b"\r\n\r\n")
    headers = header_str.decode("utf-8").split("\r\n")
    content_length = 0
    for h in headers:
        if h.lower().startswith("content-length:"):
            content_length = int(h.split(":", 1)[1].strip())
    body = b""
    while len(body) < content_length:
        chunk = sock.recv(content_length - len(body))
        if not chunk:
            return None
        body += chunk
    return json.loads(body.decode("utf-8"))


if __name__ == "__main__":
    s = socket.create_connection((HOST, PORT))
    try:
        put = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "workspace/attachmentPut",
            "params": {"id": "test1", "content": "hi"},
        }
        send_msg(s, put)
        print("sent put")
        resp = read_resp(s)
        print("resp:", resp)

        get = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "workspace/attachmentGet",
            "params": {"id": "test1"},
        }
        send_msg(s, get)
        print("sent get")
        resp = read_resp(s)
        print("resp:", resp)
    finally:
        s.close()
