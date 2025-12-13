import http.client
import json
import time

HOST = "127.0.0.1"
PORT = 8000


def wait_for_server(timeout=25):
    start = time.time()
    while time.time() - start < timeout:
        try:
            conn = http.client.HTTPConnection(HOST, PORT, timeout=3)
            conn.request("GET", "/health")
            r = conn.getresponse()
            data = r.read().decode(errors="ignore")
            conn.close()
            print("server_ready:", r.status)
            return True
        except Exception as e:
            print("waiting for server...", str(e))
            time.sleep(1)
    return False


def do_get(path):
    try:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=10)
        conn.request("GET", path)
        r = conn.getresponse()
        body = r.read().decode(errors="ignore")
        print(f"GET {path} ->", r.status, body[:1000])
        conn.close()
    except Exception as e:
        print(f"GET {path} failed:", e)


def do_post(path, payload):
    try:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=10)
        body = json.dumps(payload)
        headers = {"Content-Type": "application/json"}
        conn.request("POST", path, body=body, headers=headers)
        r = conn.getresponse()
        resp = r.read().decode(errors="ignore")
        print(f"POST {path} ->", r.status, resp[:2000])
        conn.close()
    except Exception as e:
        print(f"POST {path} failed:", e)


if __name__ == "__main__":
    ok = wait_for_server(timeout=30)
    if not ok:
        print("Server did not become ready in time; aborting smoke tests.")
        raise SystemExit(2)

    # /health
    do_get("/health")

    # /chat (example payload) -- adjust keys to match API if needed
    chat_payload = {
        "user": "test",
        "message": "hello",
    }
    do_post("/chat", chat_payload)

    # /cache/stats
    do_get("/cache/stats")

    print("Smoke tests completed")
