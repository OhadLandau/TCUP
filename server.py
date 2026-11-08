"""TCUP web app server: static pages plus a small JSON API.

`route()` holds all the logic. It is served by the standard library here (`python server.py`) and by gunicorn through `app.py`.
"""
import json
import mimetypes
import os
import sys
import threading
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from engine import ROOT, Engine

WEB = os.path.join(ROOT, "assets")
MAX_UPLOAD = 200 * 1024 * 1024
MAX_BATCHES = 20
PORT = int(os.environ.get("PORT", "8050"))

ENGINE = Engine()
BATCHES = OrderedDict()
LOCK = threading.Lock()


def public_config():
    cfg = ENGINE.config
    return {"model": cfg["model"], "n_features": len(cfg["features"]), "classes": cfg["classes"], "test_metrics": cfg["test_metrics"],
            "reference_samples": cfg["reference_samples"], "n_significant_overall": len(ENGINE.genes["overall"]["significant"])}


def _json(code, obj):
    return code, "application/json", json.dumps(obj).encode("utf-8"), {}


def route(method, raw_path, body=b""):
    """Returns (status, content type, body bytes, extra headers)."""
    path = urlparse(raw_path).path
    if method == "GET":
        if path == "/api/config":
            return _json(200, public_config())
        if path == "/api/example":
            with open(os.path.join(ROOT, "ExampleTranscriptomics.csv"), "rb") as fh:
                return 200, "text/csv", fh.read(), {"Content-Disposition": 'attachment; filename="ExampleTranscriptomics.csv"'}
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.abspath(os.path.join(WEB, rel))
        if not full.startswith(WEB + os.sep) or not os.path.isfile(full):
            return _json(404, {"error": "Not found"})
        with open(full, "rb") as fh:
            return 200, mimetypes.guess_type(full)[0] or "application/octet-stream", fh.read(), {}
    if method != "POST":
        return _json(405, {"error": "Method not allowed"})
    try:
        if path == "/api/upload":
            df = ENGINE.orient(ENGINE.read_table(body))
            if len(df) > 500:
                raise ValueError(f"{len(df)} samples found; the limit is 500 per upload.")
            found = sum(c in ENGINE.fpos for c in df.columns)
            bid = uuid.uuid4().hex
            with LOCK:
                BATCHES[bid] = df
                while len(BATCHES) > MAX_BATCHES:
                    BATCHES.popitem(last=False)
            return _json(200, {"batch": bid, "samples": [str(s) for s in df.index], "genes_found": found, "n_features": len(ENGINE.features)})
        if path == "/api/predict":
            req = json.loads(body or b"{}")
            with LOCK:
                df = BATCHES.get(req.get("batch"))
            if df is None:
                return _json(410, {"error": "This upload has expired. Please upload the file again."})
            i = int(req.get("index", 0))
            return _json(200, ENGINE.predict(df.index[i], df.iloc[i], req.get("sample_type", "cancer")))
        return _json(404, {"error": "Not found"})
    except (ValueError, IndexError) as exc:
        return _json(400, {"error": str(exc)})
    except Exception as exc:
        return _json(500, {"error": f"Analysis failed: {exc}"})


def _cache(ctype):
    return "no-store" if ctype == "application/json" else "no-cache"


class Handler(BaseHTTPRequestHandler):
    server_version = "TCUP/2"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _reply(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            code, ctype, data, extra = _json(413, {"error": "File is larger than 200 MB."})
        else:
            code, ctype, data, extra = route(method, self.path, self.rfile.read(n) if n else b"")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", _cache(ctype))
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply("GET")

    def do_POST(self):
        self._reply("POST")


def wsgi_app(environ, start_response):
    """The same routes as a WSGI application, for gunicorn."""
    n = int(environ.get("CONTENT_LENGTH") or 0)
    if n > MAX_UPLOAD:
        code, ctype, data, extra = _json(413, {"error": "File is larger than 200 MB."})
    else:
        path = environ.get("PATH_INFO", "/") + (("?" + environ["QUERY_STRING"]) if environ.get("QUERY_STRING") else "")
        code, ctype, data, extra = route(environ["REQUEST_METHOD"], path, environ["wsgi.input"].read(n) if n else b"")
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed", 410: "Gone", 413: "Payload Too Large", 500: "Internal Server Error"}.get(code, "OK")
    headers = [("Content-Type", ctype), ("Content-Length", str(len(data))), ("Cache-Control", _cache(ctype))] + list(extra.items())
    start_response(f"{code} {reason}", headers)
    return [data]


if __name__ == "__main__":
    host = "127.0.0.1" if os.environ.get("TCUP_LOCAL", "1") == "1" else "0.0.0.0"
    httpd = ThreadingHTTPServer((host, PORT), Handler)
    print(f"TCUP running at http://{host}:{PORT}", flush=True)
    httpd.serve_forever()
