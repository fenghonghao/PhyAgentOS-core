"""Call one piper-vla tool straight through the Gateway (bench test without the agent).

  python dev/gw_call.py <port> vla.policies
  python dev/gw_call.py <port> vla.run '{"policy": "sim_move", "seconds": 3}'
"""
import json
import sys
import time

import httpx


def main() -> int:
    port, tool = sys.argv[1], sys.argv[2]
    args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0, trust_env=False) as c:
        spec = c.get(f"/tools/{tool}").json().get("data") or {}
        if spec.get("semantics") == "query":   # queries go to the bound endpoint operation
            r = c.post(f"/tools/{spec['endpoint_id']}/{spec['operation']}:invoke", json={"arguments": args})
        else:
            r = c.post(f"/tools/{tool}:invoke", json={"arguments": args})
        r.raise_for_status()
        body = r.json()
        data = body.get("data") or body
        iid = data.get("invocation_id")
        if not iid or data.get("result") is not None:
            print(json.dumps(body, indent=1, ensure_ascii=False)[:4000])
            return 0
        t0 = time.time()
        while True:
            r = c.get(f"/invocations/{iid}/result")
            if r.status_code == 200:
                print(json.dumps(r.json(), indent=1, ensure_ascii=False)[:4000])
                print(f"-- {time.time() - t0:.1f} s")
                return 0
            time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main())
