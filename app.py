import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from server import wsgi_app as server  # noqa: E402

if __name__ == "__main__":
    import runpy

    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py"), run_name="__main__")
