"""Send one robot.run block straight to the Gateway (bench test without the agent).

  python dev/gw_run.py <port> "<python code>"      (code may also be read from stdin with -)
"""
import json
import sys
import time

import httpx


def main() -> int:
    port, code = sys.argv[1], sys.argv[2]
    if code == "-":
        code = sys.stdin.read()
    base = f"http://127.0.0.1:{port}"
    with httpx.Client(base_url=base, timeout=10.0, trust_env=False) as c:
        r = c.post("/tools/robot.run:invoke", json={"arguments": {"code": code}, "timeout_ms": 600000})
        r.raise_for_status()
        inv = r.json()
        iid = (inv.get("data") or inv).get("invocation_id")
        t0 = time.time()
        while True:
            r = c.get(f"/invocations/{iid}/result")
            if r.status_code == 200:
                print(json.dumps(r.json(), indent=1, ensure_ascii=False)[:6000])
                print(f"-- {time.time() - t0:.1f} s")
                return 0
            time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
