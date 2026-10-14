"""Compatibility entry point for the local Seen It Before dashboard.

Run ``python app.py`` or ``python server.py`` to launch the same web application.
"""

from server import run_server


if __name__ == "__main__":
    run_server()
