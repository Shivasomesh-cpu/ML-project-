"""
Standalone, high-speed Python web server for Brain-FM UI.
Zero external dependencies, zero websocket failures, instant load times.
Strict Black & White aesthetic.
"""

import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 5000
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

class BrainFMRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        # Enable caching-control and CORS for flawless local development
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        # Clean terminal output
        sys.stderr.write(f"[BRAIN-FM WEB] {self.address_string()} - {format % args}\n")


def find_available_port(start_port=5000):
    import socket
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
            port += 1
    return start_port


def run_server():
    port = find_available_port(PORT)
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, BrainFMRequestHandler)
    url = f"http://localhost:{port}"

    print("=" * 65)
    print("  BRAIN-FM // BLACK & WHITE WEB INTERFACE")
    print("=" * 65)
    print(f"  Local URL : {url}")
    print(f"  Directory : {WEB_DIR}")
    print("  Status    : ACTIVE (Press Ctrl+C to stop)")
    print("=" * 65)

    # Open in user browser automatically
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[BRAIN-FM WEB] Server stopped by user.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
