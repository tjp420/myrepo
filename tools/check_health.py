import http.client
import sys
import time

HOST = "127.0.0.1"
PORT = 8000

ok = False
for i in range(10):
    try:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=3)
        conn.request("GET", "/health")
        r = conn.getresponse()
        print("status", r.status)
        print(r.read().decode())
        conn.close()
        ok = True
        break
    except Exception as e:
        print("try", i, "failed", e)
        time.sleep(1)
if not ok:
    sys.exit(2)
