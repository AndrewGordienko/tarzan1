import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from osc.web import Handler


def server():
    s = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def get(s, path):
    return urllib.request.urlopen(f"http://127.0.0.1:{s.server_port}{path}").read()


def post(s, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{s.server_port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def test_customer_app_status_artifacts_and_page():
    s = server()
    assert b"Teach once" in get(s, "/")
    assert json.loads(get(s, "/api/status"))["status"] == "ready"
    assert all(x["exists"] for x in json.loads(get(s, "/api/artifacts"))["artifacts"])
    s.shutdown()


def test_inference_ood_execution_and_path_traversal():
    s = server()
    assert post(s, "/api/demo/infer", {"demo": "heavy"})["accepted"]
    ood = post(s, "/api/demo/infer", {"demo": "ood"})
    assert not ood["accepted"] and ood["program"]["policy_name"] == "unknown_or_unexplained"
    run = post(s, "/api/demo/execute", {"demo": "heavy"})
    assert run["status"] == "complete" and run["trace"]["actions"]
    blocked = post(s, "/api/demo/execute", {"demo": "ood"})
    assert blocked["status"] == "abstained"
    try:
        get(s, "/artifacts/../pyproject.toml")
        assert False
    except Exception:
        pass
    s.shutdown()
